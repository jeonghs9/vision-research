#!/usr/bin/env python3
"""`crossing_trigger_bottomline.py`와 동일한 파이프라인을 구간별로 시간을
재면서 실행한다(1회성 프로파일링 스크립트 — production 코드 아님, 발화
로직은 원본과 완전히 동일하고 시간 측정 코드만 추가했다).

구간 정의:
  차량탐지+추적   - det_model.track() 제너레이터에서 다음 프레임 결과를
                    받는 시간(비디오 디코딩 포함, YOLO det+ByteTrack)
  차량트래커후처리 - VehicleTracker.process()
  에고모션추정     - ORB 리사이즈/그레이스케일 변환 + estimate_homography
  차선검출         - GANet: 캐시 JSONL 읽기 / YOLO-pose: lane_model.predict()
  차선추적         - LaneTracker.step() + solid_tracks/tid_to_track 구성
  판정로직         - 차량별 교차 판정 루프(기하 계산, 에피소드 상태머신)
  그리기           - cv2 박스/폴리라인/분홍 게이지 드로잉
  번호판OCR        - VIOLATION 확정 차량만 대상(crossing_trigger_bottomline.py
                    에서 그대로 import) — PaddleOCR 추론 포함, --no-ocr로 생략 가능
  인코딩           - ffmpeg 파이프 write

주의: 판정 로직 자체는 이 스크립트가 만들어진 시점(박스-크기 1.5배 가드)
기준이라 현재 원본(ratio-jump 가드)과 살짝 다르다 — 순수 시간 측정
목적이라 두 가드 다 O(1) 수준이라 측정치에 미치는 영향은 무시할 만하다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
    profile_stages.py \
    --det-model .../det-best.pt [--lane-source cache --lane-cache ... | --lane-model ...] \
    --input 원본클립_0816_앞1분.mp4 --out-video /dev/null --out-csv /dev/null --device 0
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/hsjeong/workspace/applications/traffic-video-analytics/scripts/02-실선위반판정")
from visualize_solid_lane_crossing_polyline import (  # noqa: E402
    BORDER_MARGIN_PX,
    GATE_REFERENCE_WIDTH,
    ORB_WORK_WIDTH,
    LaneTracker,
    VehicleTracker,
    estimate_homography,
)
from ultralytics import YOLO  # noqa: E402
from crossing_trigger_bottomline import (  # noqa: E402 — 번호판 탐지+OCR 재사용
    KhoiPaddleOCR,
    PlateTextStabilizer,
    ASSOCIATION_METRIC,
    ASSOCIATION_THRESHOLD,
    COLOR_PLATE,
    clip_box,
    draw_plate_text,
    normalize_paddle_device,
    overlap_score,
    run_ocr,
    _prepare_paddleocr_environment,
)


def point_to_polyline_signed_distance(point, polyline):
    best_dist = float("inf")
    best_sign = 0.0
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            continue
        t = np.clip(np.dot(point - a, ab) / ab_len2, 0.0, 1.0)
        closest = a + t * ab
        diff = point - closest
        dist = float(np.hypot(diff[0], diff[1]))
        if dist < best_dist:
            best_dist = dist
            cross_z = ab[0] * diff[1] - ab[1] * diff[0]
            best_sign = 1.0 if cross_z >= 0 else -1.0
    return best_dist, best_sign


def segment_to_polyline_relation(p_left, p_right, polyline):
    d_l, s_l = point_to_polyline_signed_distance(p_left, polyline)
    d_r, s_r = point_to_polyline_signed_distance(p_right, polyline)
    dist = min(d_l, d_r)
    sign = s_l if s_l == s_r else None
    return dist, sign


def crossing_ratio(p_left, p_right, polyline, before_sign):
    d_l, s_l = point_to_polyline_signed_distance(p_left, polyline)
    d_r, s_r = point_to_polyline_signed_distance(p_right, polyline)
    if s_l == s_r:
        return 0.0 if s_l == before_sign else 1.0
    if s_l == before_sign:
        d_same, d_other = d_l, d_r
    else:
        d_same, d_other = d_r, d_l
    denom = d_same + d_other
    if denom < 1e-6:
        return 0.5
    return float(np.clip(d_other / denom, 0.0, 1.0))


def _segments_cross(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def bottom_edge_touches_polyline(p_left, p_right, polyline):
    for i in range(len(polyline) - 1):
        if _segments_cross(p_left, p_right, polyline[i], polyline[i + 1]):
            return True
    return False


class FfmpegWriter:
    def __init__(self, out_path, w, h, fps, crf=23, preset="fast"):
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


class LaneCacheReader:
    def __init__(self, path):
        self._f = open(path, "r")
        self._next = None
        self._advance()

    def _advance(self):
        line = self._f.readline()
        self._next = json.loads(line) if line else None

    def get_dets(self, frame_index):
        if self._next is None or self._next["frame"] != frame_index:
            return []
        record = self._next
        self._advance()
        dets = []
        for lane in record["lanes"]:
            pts = np.array(lane["points"], dtype=np.float32)
            vis = np.ones(len(pts), dtype=bool)
            dets.append((pts, vis, int(lane["cls"]), 1.0))
        return dets

    def close(self):
        self._f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model", required=True)
    ap.add_argument("--lane-source", choices=("model", "cache"), default="model")
    ap.add_argument("--lane-model", default=None)
    ap.add_argument("--lane-cache", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--output-scale", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--car-class", type=int, default=1)
    ap.add_argument("--kpt-vis-thresh", type=float, default=0.6)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--gate", type=float, default=90.0)
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--hold-seconds", type=float, default=1.5)
    ap.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지")
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-ocr", action="store_true", help="번호판 OCR 생략(이 구간만 빼고 측정하고 싶을 때)")
    ap.add_argument("--plate-class", type=int, default=3)
    ap.add_argument("--paddle-device", default=None)
    ap.add_argument("--orb-features", type=int, default=1500, help="에고모션 ORB 특징점 수")
    a = ap.parse_args()

    input_path = Path(a.input).expanduser().resolve()
    capture = cv2.VideoCapture(str(input_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    hold_frames = max(1, int(round(a.hold_seconds * fps)))

    det_model = YOLO(a.det_model)
    lane_model = YOLO(a.lane_model) if a.lane_source == "model" else None
    lane_cache = LaneCacheReader(a.lane_cache) if a.lane_source == "cache" else None
    tracker = VehicleTracker(width, height, a.car_class)
    gate = a.gate * (width / GATE_REFERENCE_WIDTH)
    lane_tracker = LaneTracker(a.min_hits, a.max_misses, gate, a.ema)

    orb = cv2.ORB_create(a.orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = None
    ego_scale = ORB_WORK_WIDTH / width
    ego_h = max(2, int(round(height * ego_scale)))

    # OCR은 det_model.track() 호출(아래) 전에 초기화한다 — ultralytics가
    # device= 인자를 처리하면서 CUDA_VISIBLE_DEVICES를 그 GPU 하나로 좁혀버리는
    # 경우가 있어서, 그 뒤에 PaddleOCR을 원래 GPU 번호로 초기화하면 "GPU
    # count is: 1"류 에러가 난다(실측: GPU 1에서 재현) — crossing_trigger_
    # bottomline.py와 동일한 순서로 맞춰서 회피한다.
    ocr_engine = stabilizer = None
    if not a.no_ocr:
        _prepare_paddleocr_environment()
        from paddleocr import PaddleOCR
        paddle_engine = PaddleOCR(
            lang="en", text_detection_model_name=None, text_recognition_model_name=None,
            use_textline_orientation=True, use_doc_orientation_classify=False, use_doc_unwarping=False,
            device=normalize_paddle_device(a.paddle_device or a.device),
        )
        ocr_engine = KhoiPaddleOCR(paddle_engine)
        stabilizer = PlateTextStabilizer(min_votes=3, history=30)
    plate_text_by_car = {}

    det_results = det_model.track(
        source=str(input_path), conf=0.25, iou=0.5, imgsz=1024,
        device=a.device, tracker="bytetrack.yaml", stream=True, persist=True, verbose=False,
    )

    writer = None
    out_w = out_h = None
    if a.out_video:
        out_w = max(2, int(round(width * a.output_scale)) // 2 * 2)
        out_h = max(2, int(round(height * a.output_scale)) // 2 * 2)
        writer = FfmpegWriter(a.out_video, out_w, out_h, fps)

    active_episode_by_car = {}
    prev_box_by_car = {}
    last_seen_frame_by_car = {}
    flagged_cars = {}

    st = defaultdict(float)  # stage -> accumulated seconds
    n_frames = 0
    n_cars = 0

    det_iter = iter(det_results)
    t_loop_start = time.perf_counter()
    while True:
        if a.max_frames and n_frames >= a.max_frames:
            break
        t0 = time.perf_counter()
        try:
            result = next(det_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()
        st["1_차량탐지+추적"] += t1 - t0

        frame_index = n_frames
        original = result.orig_img
        frame_draw = original.copy() if writer is not None else None
        per_car, _ = tracker.process(result)
        t2 = time.perf_counter()
        st["2_차량트래커후처리"] += t2 - t1

        small = cv2.resize(original, (ORB_WORK_WIDTH, ego_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        Hm = None
        if prev_gray is not None:
            Hm_small = estimate_homography(prev_gray, gray, orb, bf)
            if Hm_small is not None:
                s = ego_scale
                scale_mat = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
                Hm = np.linalg.inv(scale_mat) @ Hm_small @ scale_mat
        prev_gray = gray
        t3 = time.perf_counter()
        st["3_에고모션추정(ORB)"] += t3 - t2

        if a.lane_source == "cache":
            dets = lane_cache.get_dets(frame_index)
        else:
            pose_result = lane_model.predict(source=original, imgsz=1024, conf=0.25, device=a.device, verbose=False)[0]
            dets = []
            if pose_result.keypoints is not None and len(pose_result.keypoints):
                kxy = pose_result.keypoints.data.cpu().numpy()
                cls_arr = pose_result.boxes.cls.cpu().numpy().astype(int)
                conf_arr = pose_result.boxes.conf.cpu().numpy()
                for i in range(len(kxy)):
                    kp = kxy[i]
                    v = kp[:, 2] > a.kpt_vis_thresh if kp.shape[1] > 2 else np.ones(len(kp), bool)
                    if v.sum() < 2:
                        continue
                    p = kp[:, :2].astype(np.float32)
                    dets.append((p, v, int(cls_arr[i]), float(conf_arr[i])))
        t4 = time.perf_counter()
        st["4_차선검출"] += t4 - t3

        live, coasting = lane_tracker.step(dets, Hm)
        solid_tracks = [t for t in live if t.cls == 0]
        tid_to_track = {int(t.tid): t for t in (live + coasting)}
        t5 = time.perf_counter()
        st["5_차선추적"] += t5 - t4

        if frame_draw is not None:
            for t in live:
                pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
                if len(pts) < 2:
                    continue
                pts_int = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
                color = (0, 220, 0) if t.cls == 0 else (0, 255, 255)
                cv2.polylines(frame_draw, [pts_int], False, color, 2, cv2.LINE_AA)
        t6 = time.perf_counter()
        st["7_그리기"] += t6 - t5

        for car_id, info in per_car.items():
            n_cars += 1
            t_a = time.perf_counter()
            box = info["box"]
            x1, y1, x2, y2 = map(float, box)
            bottom_left = np.array([x1, y2], dtype=np.float64)
            bottom_right = np.array([x2, y2], dtype=np.float64)
            border_clipped = bool(
                BORDER_MARGIN_PX > 0 and (
                    x1 <= BORDER_MARGIN_PX or y1 <= BORDER_MARGIN_PX
                    or x2 >= width - BORDER_MARGIN_PX or y2 >= height - BORDER_MARGIN_PX
                )
            )
            prev_seen_frame = last_seen_frame_by_car.get(car_id)
            true_detection_gap = prev_seen_frame is not None and (frame_index - prev_seen_frame) > 1
            last_seen_frame_by_car[car_id] = frame_index

            episode_tid = None
            episode_ratio = None

            if not border_clipped and car_id not in flagged_cars:
                ep = active_episode_by_car.get(car_id)
                if ep is not None:
                    track = tid_to_track.get(ep["tid"])
                    if track is None:
                        active_episode_by_car.pop(car_id, None)
                    else:
                        pts = track.kp[track.vis] if track.vis is not None and track.vis.any() else track.kp
                        if len(pts) >= 2:
                            pts64 = pts.astype(np.float64)
                            cur_w, cur_h = x2 - x1, y2 - y1
                            ref_w, ref_h = ep["car_size_ref"]
                            size_ratio = max(cur_w / ref_w, ref_w / cur_w, cur_h / ref_h, ref_h / cur_h)
                            ep["car_size_ref"] = (cur_w, cur_h)
                            if true_detection_gap and size_ratio > 1.5:
                                active_episode_by_car.pop(car_id, None)
                            else:
                                still_touching = bottom_edge_touches_polyline(bottom_left, bottom_right, pts64)
                                _, sign = segment_to_polyline_relation(bottom_left, bottom_right, pts64)
                                ratio = crossing_ratio(bottom_left, bottom_right, pts64, ep["before_sign"])
                                episode_tid, episode_ratio = ep["tid"], ratio
                                if still_touching:
                                    ep["last_touch_frame"] = frame_index
                                if sign is None:
                                    pass
                                elif sign == ep["before_sign"]:
                                    active_episode_by_car.pop(car_id, None)
                                elif ep.get("hold_count", 0) == 0 and frame_index - ep["last_touch_frame"] > hold_frames:
                                    active_episode_by_car.pop(car_id, None)
                                else:
                                    ep["hold_count"] = ep.get("hold_count", 0) + 1
                                    if ep["hold_count"] >= hold_frames:
                                        flagged_cars[car_id] = {"frame": frame_index, "lane_tid": ep["tid"]}
                                        active_episode_by_car.pop(car_id, None)
                else:
                    prev_box = prev_box_by_car.get(car_id)
                    for t in solid_tracks:
                        pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
                        if len(pts) < 2:
                            continue
                        pts64 = pts.astype(np.float64)
                        if not bottom_edge_touches_polyline(bottom_left, bottom_right, pts64):
                            continue
                        _, sign = segment_to_polyline_relation(bottom_left, bottom_right, pts64)
                        if sign is not None:
                            continue
                        if prev_box is None:
                            continue
                        px1, py1, px2, py2 = prev_box
                        if true_detection_gap:
                            prev_w, prev_h = px2 - px1, py2 - py1
                            cur_w, cur_h = x2 - x1, y2 - y1
                            if max(cur_w / prev_w, prev_w / cur_w, cur_h / prev_h, prev_h / cur_h) > 1.5:
                                continue
                        prev_bl = np.array([px1, py2], dtype=np.float64)
                        prev_br = np.array([px2, py2], dtype=np.float64)
                        _, before_sign = segment_to_polyline_relation(prev_bl, prev_br, pts.astype(np.float64))
                        if before_sign is None:
                            continue
                        active_episode_by_car[car_id] = {
                            "tid": int(t.tid), "before_sign": before_sign, "last_touch_frame": frame_index,
                            "car_size_ref": (x2 - x1, y2 - y1),
                        }
                        episode_tid = int(t.tid)
                        episode_ratio = crossing_ratio(bottom_left, bottom_right, pts64, before_sign)
                        break

            prev_box_by_car[car_id] = (x1, y1, x2, y2)
            t_b = time.perf_counter()
            st["6_판정로직"] += t_b - t_a

            if frame_draw is not None:
                x1i, y1i, x2i, y2i = map(int, box)
                flagged_now = car_id in flagged_cars
                in_episode = car_id in active_episode_by_car
                box_color = (0, 0, 255) if flagged_now else ((0, 165, 255) if in_episode else (0, 255, 255))
                if in_episode and not flagged_now and episode_ratio is not None:
                    fill_h = int(round(episode_ratio * (y2i - y1i)))
                    if fill_h > 0:
                        fill_top = y2i - fill_h
                        overlay = frame_draw.copy()
                        cv2.rectangle(overlay, (x1i, fill_top), (x2i, y2i), (180, 105, 255), -1)
                        cv2.addWeighted(overlay, 0.35, frame_draw, 0.65, 0, dst=frame_draw)
                cv2.rectangle(frame_draw, (x1i, y1i), (x2i, y2i), box_color, 3 if flagged_now else 2)
                cv2.line(frame_draw, (x1i, y2i), (x2i, y2i), (255, 220, 0), 3, cv2.LINE_AA)
                label = f"#{car_id}" + (" (교차중)" if in_episode else "")
                if flagged_now:
                    label = f"VIOLATION {label}"
                cv2.putText(frame_draw, label, (x1i, max(0, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            box_color, 2, cv2.LINE_AA)
            t_c = time.perf_counter()
            st["7_그리기"] += t_c - t_b

        t_ocr0 = time.perf_counter()
        if ocr_engine is not None and result.boxes is not None and len(result.boxes) > 0:
            p_boxes = result.boxes.xyxy.cpu().numpy()
            p_classes = result.boxes.cls.cpu().numpy().astype(int)
            p_track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None else np.full(len(p_classes), -1)
            )
            car_indices = [idx for idx, cid in enumerate(p_classes) if cid == a.car_class and p_track_ids[idx] >= 0]
            plate_drawn = set()
            for plate_index, cid in enumerate(p_classes):
                if cid != a.plate_class or not car_indices:
                    continue
                score, car_index = max(
                    ((overlap_score(p_boxes[idx], p_boxes[plate_index], ASSOCIATION_METRIC), idx)
                     for idx in car_indices),
                    key=lambda item: item[0],
                )
                if score < ASSOCIATION_THRESHOLD:
                    continue
                pcar_id = int(p_track_ids[car_index])
                if pcar_id not in flagged_cars:
                    continue
                px1, py1, px2, py2 = clip_box(p_boxes[plate_index], width, height)
                if px2 <= px1 or py2 <= py1:
                    continue
                text, confidence = run_ocr(ocr_engine, original[py1:py2, px1:px2])
                stable = stabilizer.update(pcar_id, text, confidence)
                if stable:
                    plate_text_by_car[pcar_id] = stable
                shown = plate_text_by_car.get(pcar_id, stable)
                if frame_draw is not None:
                    cv2.rectangle(frame_draw, (px1, py1), (px2, py2), COLOR_PLATE, 2)
                    draw_plate_text(frame_draw, p_boxes[plate_index], shown)
                plate_drawn.add(pcar_id)
        t_ocr1 = time.perf_counter()
        st["9_번호판OCR"] += t_ocr1 - t_ocr0

        if writer is not None:
            t7 = time.perf_counter()
            if (out_w, out_h) != (width, height):
                frame_draw = cv2.resize(frame_draw, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame_draw)
            t8 = time.perf_counter()
            st["8_인코딩"] += t8 - t7

        n_frames += 1
        if n_frames % 200 == 0:
            print(f"[{a.label}] {n_frames}frame elapsed={time.perf_counter()-t_loop_start:.1f}s", flush=True)

    t_total = time.perf_counter() - t_loop_start
    if writer is not None:
        writer.release()
    if lane_cache is not None:
        lane_cache.close()

    print(f"\n=== [{a.label}] 구간별 처리시간 (총 {n_frames}프레임, {n_cars}회 차량평가, 총 {t_total:.1f}s, {n_frames/t_total:.2f}fps) ===")
    ordered = sorted(st.items())
    for name, sec in ordered:
        print(f"  {name:22s} 총 {sec:7.2f}s  평균 {sec/n_frames*1000:7.3f}ms/frame  비중 {sec/t_total*100:5.1f}%")
    other = t_total - sum(st.values())
    print(f"  {'10_기타(루프오버헤드)':22s} 총 {other:7.2f}s  평균 {other/n_frames*1000:7.3f}ms/frame  비중 {other/t_total*100:5.1f}%")


if __name__ == "__main__":
    main()
