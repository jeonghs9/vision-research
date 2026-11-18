#!/usr/bin/env python3
"""YOLO-pose 차선 추론 시각화 — 키포인트 12개를 순서대로 이어 폴리라인으로 그린다.

Ultralytics 기본 시각화는 skeleton(연결관계)을 몰라 점만 찍는다.
우리 차선은 kp0→kp1→...→kp11 순서가 곧 선이다.

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
    /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/01-차선검출모델/lane_pose_vis.py \
    --weights test/output/003-train-yolo26n-pose-1024-203f-229ep/weights/best.pt \
    --src <이미지 폴더 또는 pose_ds val> --out test/output/lane_pose_vis_203f --conf 0.25
"""
import argparse
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

YELLOW = (0, 255, 255)   # solid
GREEN = (0, 200, 0)      # dashed
GRAY = (150, 150, 150)
NAME = {0: "solid", 1: "dashed"}
COL = {0: YELLOW, 1: GREEN}


def draw_lane(img, kpts, cls, conf, dashed):
    # 클래스는 색으로만 구분 (solid=노랑, dashed=초록). 선 스타일은 둘 다 실선.
    pts = kpts[:, :2].astype(np.int32)
    vis = kpts[:, 2] > 0.3 if kpts.shape[1] > 2 else np.ones(len(kpts), bool)
    pts = pts[vis]
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts], False, COL[cls], 5, cv2.LINE_AA)
    m = pts[len(pts) // 2]
    cv2.putText(img, f"{NAME[cls]} {conf:.2f}", (int(m[0]) + 5, int(m[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[cls], 2, cv2.LINE_AA)


def gt_polylines(label_path, w, h):
    """pose_ds 라벨 (cls xc yc bw bh  x y v ×12, 정규화) → 픽셀 폴리라인 목록."""
    out = []
    if not os.path.exists(label_path):
        return out
    for ln in open(label_path):
        q = list(map(float, ln.split()))
        if len(q) < 41:
            continue
        kp = np.array(q[5:]).reshape(-1, 3)
        pts = np.stack([kp[:, 0] * w, kp[:, 1] * h], 1).astype(np.int32)
        out.append(pts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", required=True, help="이미지 폴더")
    ap.add_argument("--labels", default=None, help="GT 라벨 폴더 (pose_ds/labels/<split>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.weights)
    files = sorted(glob.glob(os.path.join(a.src, "*.jpg")))
    if a.max:
        files = files[:a.max]

    n_det = 0
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        r = model.predict(fp, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        img = cv2.imread(fp)
        H, W = img.shape[:2]
        vis = cv2.resize(img, (1280, int(1280 * H / W)))
        sx, sy = 1280 / W, (1280 * H / W) / H

        if a.labels:
            for g in gt_polylines(os.path.join(a.labels, stem + ".txt"), W, H):
                cv2.polylines(vis, [(g * [sx, sy]).astype(np.int32)], False, GRAY, 2, cv2.LINE_AA)

        if r.keypoints is not None and len(r.keypoints):
            kxy = r.keypoints.data.cpu().numpy()          # (n, 12, 3)
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for i in range(len(kxy)):
                k = kxy[i].copy()
                k[:, 0] *= sx
                k[:, 1] *= sy
                draw_lane(vis, k, cls[i], cf[i], dashed=(cls[i] == 1))
                n_det += 1
        cv2.putText(vis, f"{stem}  pred lanes: {len(r.keypoints) if r.keypoints is not None else 0}",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out, stem + ".jpg"), vis)

    print(f"{len(files)} 장 -> {a.out}  (conf>{a.conf})  총 검출 선 {n_det}  (평균 {n_det/max(len(files),1):.1f})")
    print("노란 실선=solid, 초록 파선=dashed, 회색=GT")


if __name__ == "__main__":
    main()
