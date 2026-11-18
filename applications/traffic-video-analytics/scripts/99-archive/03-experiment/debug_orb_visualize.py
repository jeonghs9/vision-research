#!/usr/bin/env python3
"""ORB 에고모션 디버깅 시각화 — 특징점/매칭선/RANSAC 인라이어를 화면에
그려서 에고모션 추정이 실제로 뭘 보고 매칭하는지 눈으로 확인한다.
차량 박스·차선(GANet 캐시)은 맥락 참고용으로 같이 그리지만, 발화 판정
로직(에피소드/ratio 등)은 전혀 없는 **순수 디버깅용 스크립트**다.

그리는 것:
  - 노란 점: 이번 프레임에서 뽑힌 ORB 특징점 전체(ORB_WORK_WIDTH=960
    기준으로 뽑은 걸 원본 해상도로 확대해서 표시)
  - 초록 선+점: 직전 프레임 대비 매칭되고 RANSAC에서 인라이어로 채택된
    점(= homography 계산에 실제로 쓰인 "신뢰한" 매칭)
  - 빨강 선+점: 매칭은 됐지만 RANSAC이 아웃라이어로 버린 점(예: 차량처럼
    카메라와 다르게 움직이는 물체 위의 점일 가능성이 높음)
  - 좌상단 텍스트: 특징점 수 / 매칭 수 / 인라이어 수

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
    debug_orb_visualize.py \
    --det-model /home/hsjeong/workspace/applications/traffic-video-analytics/models/det-best.pt \
    --lane-cache GANet차선캐시_0816전체.jsonl \
    --input 원본클립_0816_앞1분.mp4 \
    --out-video orb_debug_0816_1분_GANet_ORB800.mp4 \
    --orb-features 800 --device 0
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/hsjeong/workspace/applications/traffic-video-analytics/scripts/02-실선위반판정")
from visualize_solid_lane_crossing_polyline import ORB_WORK_WIDTH  # noqa: E402
from ultralytics import YOLO  # noqa: E402


class FfmpegWriter:
    def __init__(self, out_path: str, w: int, h: int, fps: float, crf: int = 23, preset: str = "fast"):
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


class LaneCacheReader:
    def __init__(self, path: str):
        self._f = open(path, "r")
        self._next = None
        self._advance()

    def _advance(self) -> None:
        line = self._f.readline()
        self._next = json.loads(line) if line else None

    def get_dets(self, frame_index: int):
        if self._next is None or self._next["frame"] != frame_index:
            return []
        record = self._next
        self._advance()
        return [(np.array(lane["points"], dtype=np.float32), int(lane["cls"])) for lane in record["lanes"]]

    def close(self) -> None:
        self._f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model", required=True)
    ap.add_argument("--lane-cache", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--output-scale", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--car-class", type=int, default=1)
    ap.add_argument("--orb-features", type=int, default=800)
    ap.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지")
    a = ap.parse_args()

    input_path = Path(a.input).expanduser().resolve()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    det_model = YOLO(a.det_model)
    lane_cache = LaneCacheReader(a.lane_cache)

    orb = cv2.ORB_create(a.orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = None
    ego_scale = ORB_WORK_WIDTH / width
    inv_scale = 1.0 / ego_scale
    ego_h = max(2, int(round(height * ego_scale)))

    det_results = det_model.track(
        source=str(input_path), conf=0.25, iou=0.5, imgsz=1024,
        device=a.device, tracker="bytetrack.yaml", stream=True, persist=True, verbose=False,
    )

    out_w = max(2, int(round(width * a.output_scale)) // 2 * 2)
    out_h = max(2, int(round(height * a.output_scale)) // 2 * 2)
    writer = FfmpegWriter(a.out_video, out_w, out_h, fps)

    from tqdm import tqdm
    for frame_index, result in enumerate(tqdm(det_results, desc="orb-debug", unit="frame")):
        if a.max_frames and frame_index >= a.max_frames:
            break
        original = result.orig_img
        frame_draw = original.copy()

        # 차량 박스 (맥락 참고용 — 추적/판정 없이 이번 프레임 검출만)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, cls in zip(boxes, classes):
                if cls != a.car_class:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (255, 180, 0), 1)

        # 차선 (GANet 캐시, 맥락 참고용)
        for pts, cls in lane_cache.get_dets(frame_index):
            color = (0, 220, 0) if cls == 0 else (0, 255, 255)
            pts_int = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_draw, [pts_int], False, color, 2, cv2.LINE_AA)

        # ORB 특징점 + 매칭 + RANSAC 인라이어
        small = cv2.resize(original, (ORB_WORK_WIDTH, ego_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        k2, d2 = orb.detectAndCompute(gray, None)
        n_kp = len(k2) if k2 is not None else 0
        n_matches = n_inliers = 0

        if k2 is not None:
            for kp in k2:
                x, y = kp.pt
                cv2.circle(frame_draw, (int(x * inv_scale), int(y * inv_scale)), 2, (0, 255, 255), 1, cv2.LINE_AA)

        if prev_gray is not None and d2 is not None and n_kp >= 30:
            k1, d1 = orb.detectAndCompute(prev_gray, None)
            if d1 is not None and len(k1) >= 30:
                matches = bf.match(d1, d2)
                if len(matches) >= 25:
                    matches = sorted(matches, key=lambda m: m.distance)[:250]
                    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
                    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
                    _, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
                    n_matches = len(matches)
                    if mask is not None:
                        mask = mask.ravel().astype(bool)
                        n_inliers = int(mask.sum())
                        for (x1, y1), (x2, y2), inl in zip(p1, p2, mask):
                            color = (0, 255, 0) if inl else (0, 0, 255)
                            pt1 = (int(x1 * inv_scale), int(y1 * inv_scale))
                            pt2 = (int(x2 * inv_scale), int(y2 * inv_scale))
                            cv2.line(frame_draw, pt1, pt2, color, 1, cv2.LINE_AA)
                            cv2.circle(frame_draw, pt2, 3, color, -1, cv2.LINE_AA)

        prev_gray = gray

        cv2.rectangle(frame_draw, (12, 12), (560, 70), (0, 0, 0), -1)
        cv2.putText(frame_draw, f"ORB features={a.orb_features}  keypoints={n_kp}  matches={n_matches}  "
                                 f"RANSAC inliers={n_inliers}", (24, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_draw, "yellow=keypoint  green=inlier match  red=outlier match", (24, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        if (out_w, out_h) != (width, height):
            frame_draw = cv2.resize(frame_draw, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame_draw)

    writer.release()
    lane_cache.close()
    print(f"[VIDEO] {a.out_video}")


if __name__ == "__main__":
    main()
