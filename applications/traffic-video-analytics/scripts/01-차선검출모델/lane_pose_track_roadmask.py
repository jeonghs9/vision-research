#!/usr/bin/env python3
"""YOLO-pose 차선 검출을 **도로(road) 세그멘테이션 마스크 안에서만** 인정한다.

`lane_pose_track.py` 의 트래킹(자체움직임 보정 + 매칭 + hold + 평활 + 병합)을 그대로
재사용하되, 차선 후보를 트래커에 넣기 전에 **`car-road-best.pt` 의 road 클래스 마스크
안에 있는지**로 한 번 거른다.

왜 필요한가: pose 모델은 도로 바깥(인도 경계석, 화단 가장자리, 건물 외벽선 등)에서도
가끔 차선처럼 생긴 선을 잘못 낸다. road 세그 마스크로 "도로 위에 있는 선만" 인정하면
그런 오검출을 지운다. 두 모델(pose 차선 검출 + seg 도로 분할)을 프레임마다 같이 돌린다.

판정: 차선 키포인트 12개 중 road 마스크 안에 있는 비율이 --min-inroad 이상이면 통과,
아니면 그 프레임에서 버린다(트래커에 아예 안 들어감 → 확정 트랙도 안 생김).

클래스 번호는 `car-road-best.pt` 학습 시 정의(0=road/1=car/2=plate,
`scripts/03-experiment/lane_hough.py` 의 `CLASS_ROAD` 와 동일)를 따른다.

기본은 **영상 전체**를 처리한다. 구간만 보려면 --start-sec / --dur-sec 를 준다.

실행 (전체, 테스트 영상은 반드시 학습에 안 쓴 것으로):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
    /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/01-차선검출모델/lane_pose_track_roadmask.py \
    --pose-weights models/pose-best.pt \
    --seg-weights models/car-road-best.pt \
    --video /home/hsjeong/workspace/dataset/etc./test_video/19700101_000816_T.mp4 \
    --out tests/lane_pose_roadmask/000816_roadmask_full.mp4 \
    --conf 0.30 --width 1600 --ego --crf 25 --show-mask --mask-alpha 0.12

실행 (구간만): 위에 --start-sec 90 --dur-sec 15 추가
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_pose_track import LaneTracker, draw, estimate_homography  # noqa: E402

ROAD_CLASS = 0             # car-road-best.pt: 0=road 1=car 2=plate
MASK_TINT = (60, 200, 60)  # 도로 마스크 오버레이 색 (BGR, --show-mask 일 때만)


def road_mask(seg_result, hw, dev, road_class=ROAD_CLASS):
    """seg 결과에서 road 클래스 마스크만 뽑아 hw(=oh,ow) 크기 bool 텐서로 합친다."""
    if seg_result.masks is None or len(seg_result.masks) == 0:
        return torch.zeros(hw, dtype=torch.bool, device=dev)
    m = seg_result.masks.data
    if m.shape[-2:] != hw:
        m = F.interpolate(m[None], size=hw, mode="nearest")[0]
    m = m > 0.5
    cls = seg_result.boxes.cls.long()
    sel = m[cls == road_class]
    return sel.any(0) if sel.numel() else torch.zeros(hw, dtype=torch.bool, device=dev)


def inroad_fraction(kp, mask_np, ow, oh):
    """차선 키포인트가 road 마스크 안에 있는 비율."""
    x = np.clip(kp[:, 0], 0, ow - 1).astype(np.int32)
    y = np.clip(kp[:, 1], 0, oh - 1).astype(np.int32)
    return float(mask_np[y, x].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-weights", required=True, help="차선 YOLO-pose 가중치")
    ap.add_argument("--seg-weights", required=True, help="도로 세그 가중치 (car-road-best.pt)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-sec", type=float, default=0, help="시작 위치(초). 기본 0")
    ap.add_argument("--dur-sec", type=float, default=0,
                    help="처리 길이(초). 기본 0 = start-sec 부터 영상 끝까지 전체")
    ap.add_argument("--conf", type=float, default=0.25, help="pose(차선) 모델 conf")
    ap.add_argument("--imgsz", type=int, default=1024, help="pose(차선) 모델 imgsz")
    ap.add_argument("--seg-conf", type=float, default=0.25, help="road seg 모델 conf")
    ap.add_argument("--seg-imgsz", type=int, default=1024, help="road seg 모델 imgsz")
    ap.add_argument("--road-class", type=int, default=ROAD_CLASS)
    ap.add_argument("--min-inroad", type=float, default=0.5,
                    help="차선 키포인트 중 road 마스크 안 비율 임계. 미만이면 그 프레임에서 버림")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--ego", action="store_true", help="드론 자체 움직임 보정 (ORB 호모그래피)")
    ap.add_argument("--no-track", action="store_true", help="트래킹 없이 프레임별 원본 검출만 (비교용)")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--gate", type=float, default=90.0)
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--show-mask", action="store_true", help="road 마스크를 반투명 초록으로 겹쳐 보여줌")
    ap.add_argument("--mask-alpha", type=float, default=0.12,
                    help="road 마스크 틴트 불투명도 0~1 (--show-mask 일 때만, 기본 0.12)")
    ap.add_argument("--crf", type=int, default=26, help="h264 압축률 (클수록 저용량, 23~32 권장)")
    ap.add_argument("--fps-out", type=float, default=0, help="출력 fps (0=원본). 낮추면 용량↓")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pose_model = YOLO(a.pose_weights)
    seg_model = YOLO(a.seg_weights)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow = a.width
    oh = int(round(H * ow / W / 2) * 2)
    sx, sy = ow / W, oh / H

    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, min(int(a.start_sec * fps), total_f))
    end_f = (min(int((a.start_sec + a.dur_sec) * fps), total_f)
             if a.dur_sec > 0 else total_f)
    if end_f <= start_f:
        raise SystemExit(f"처리할 프레임이 없음 (start_f={start_f}, end_f={end_f}, total={total_f})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    thick = max(2, ow // 480)

    tracker = LaneTracker(a.min_hits, a.max_misses, a.gate, a.ema)
    orb = cv2.ORB_create(1500) if a.ego else None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if a.ego else None
    prev_gray = None

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    f, n = start_f, 0
    counts, dropped_counts = [], []
    stage_ms = defaultdict(list)  # 구간별 추론시간 (ms/frame)
    pbar = tqdm(total=end_f - start_f, unit="frame", ncols=90)
    while f < end_f:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        t1 = time.perf_counter()
        stage_ms["decode"].append((t1 - t0) * 1000)

        small = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if a.ego else None
        Hm = estimate_homography(prev_gray, gray, orb, bf) if (a.ego and prev_gray is not None) else None
        prev_gray = gray
        t2 = time.perf_counter()
        stage_ms["preprocess+ego"].append((t2 - t1) * 1000)

        # ── 도로 마스크 (출력 해상도 기준) ──
        sr = seg_model.predict(frame, imgsz=a.seg_imgsz, conf=a.seg_conf,
                               classes=[a.road_class], verbose=False)[0]
        sync()
        t3 = time.perf_counter()
        stage_ms["seg_infer"].append((t3 - t2) * 1000)

        rmask = road_mask(sr, (oh, ow), dev, a.road_class).cpu().numpy()
        if a.show_mask:
            tint = np.zeros_like(small)
            tint[rmask] = MASK_TINT
            small = cv2.addWeighted(tint, a.mask_alpha, small, 1 - a.mask_alpha, 0)
        t4 = time.perf_counter()
        stage_ms["mask_postproc"].append((t4 - t3) * 1000)

        # ── 차선 검출 ──
        r = pose_model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        sync()
        t5 = time.perf_counter()
        stage_ms["pose_infer"].append((t5 - t4) * 1000)

        # ── road 마스크 필터 + 트래킹 ──
        dets, n_drop = [], 0
        if r.keypoints is not None and len(r.keypoints):
            kxy = r.keypoints.data.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for i in range(len(kxy)):
                kp = kxy[i]
                v = kp[:, 2] > 0.3 if kp.shape[1] > 2 else np.ones(len(kp), bool)
                if v.sum() < 2:
                    continue
                p = np.stack([kp[:, 0] * sx, kp[:, 1] * sy], 1)
                if inroad_fraction(p, rmask, ow, oh) < a.min_inroad:
                    n_drop += 1
                    continue
                dets.append((p, int(cls[i]), float(cf[i])))

        if a.no_track:
            for kp, cl, cf_ in dets:
                draw(small, kp, cl, cf_, thick)
            shown = len(dets)
        else:
            live, coasting = tracker.step(dets, Hm)
            for t in coasting:
                draw(small, t.kp, t.cls, t.conf, thick, coasting=True)
            for t in live:
                draw(small, t.kp, t.cls, t.conf, thick)
            shown = len(live) + len(coasting)
        t6 = time.perf_counter()
        stage_ms["filter+track+draw"].append((t6 - t5) * 1000)

        counts.append(shown)
        dropped_counts.append(n_drop)
        tt = (f - start_f) / fps
        cv2.putText(small, f"t={tt:5.1f}s  lanes={shown}  (road-filtered {n_drop})"
                    + ("  ego" if a.ego and not a.no_track else "")
                    + ("  [no-track]" if a.no_track else ""),
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(small, "solid=yellow  dashed=green   ~=coasting   초록음영=road mask",
                    (14, oh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
        vw.write(small)
        t7 = time.perf_counter()
        stage_ms["video_write"].append((t7 - t6) * 1000)
        stage_ms["total/frame"].append((t7 - t0) * 1000)
        n += 1
        f += 1

    pbar.close()
    cap.release()
    vw.release()
    if n == 0:
        raise SystemExit("처리된 프레임 0 — 영상/구간을 확인하세요")

    if shutil.which("ffmpeg"):
        tmp = a.out + ".raw.mp4"
        os.replace(a.out, tmp)
        cmd = ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", str(a.crf),
               "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        if a.fps_out > 0:
            cmd += ["-r", str(a.fps_out)]
        cmd.append(a.out)
        print("h264 압축 중...")
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(tmp)

    c, d = np.array(counts), np.array(dropped_counts)
    mb = os.path.getsize(a.out) / 1e6
    print(f"{a.out}  ({mb:.1f} MB)")
    print(f"  {n} 프레임, {ow}x{oh}, min_inroad={a.min_inroad}, ego={a.ego}")
    print(f"  표시 차선: 평균 {c.mean():.1f}  범위 {c.min()}~{c.max()}")
    print(f"  road 마스크로 걸러낸 선: 평균 {d.mean():.2f}/frame (총 {int(d.sum())}건)")

    order = ["decode", "preprocess+ego", "seg_infer", "mask_postproc",
             "pose_infer", "filter+track+draw", "video_write", "total/frame"]
    print(f"\n  구간별 추론시간 (ms/frame, {n}프레임 평균)")
    for k in order:
        v = np.array(stage_ms[k])
        print(f"    {k:<18} {v.mean():7.2f} ± {v.std():5.2f}   p50 {np.percentile(v,50):6.2f}"
              f"   p90 {np.percentile(v,90):6.2f}")
    print(f"  처리량: {1000.0 / np.mean(stage_ms['total/frame']):.2f} frame/s (인코딩 전, "
          f"디코드~기록 포함)")


if __name__ == "__main__":
    main()
