#!/usr/bin/env python3
"""GANet 차선캐시 생성 단계(디코딩/전처리/추론/후처리)별 처리속도 측정.
`vision-det/GANet/scripts/04-eval/export_lane_cache.py`의 모델 로딩/전처리/
추론/재샘플링 로직을 그대로 가져오되 time.perf_counter()로 구간별 시간을
잰다. 발화 로직과 무관한 1회성 프로파일링 스크립트 — 캐시 파일은 쓰지
않는다(직렬화 비용만 후처리 구간에 포함해서 측정).

GANet vs YOLO-pose 전체 파이프라인 비교 시, 기존 `profile_stages.py`는
GANet을 "미리 만든 캐시 읽기"로만 돌려서 캐시 생성 비용이 빠져있었다 —
이 스크립트가 그 빠진 비용(캐시를 처음부터 만드는 전처리+추론+후처리)을
측정해서 공정한 end-to-end 비교를 완성한다.

실행 (ganet_cu101 환경 필수 — paddleocr_gpu_cu126과 바이너리 비호환):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    profile_ganet_cache_gen.py \
    --video 원본클립_0816_앞1분.mp4 --max-frames 1800 --label GANet_0816_1분
"""
import argparse
import json
import sys
import time

GANET_SRC = "/home/hsjeong/workspace/vision-det/GANet/source"
sys.path.insert(0, GANET_SRC)

import cv2
import numpy as np
import torch
import mmcv

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

CFG_PATH = f"{GANET_SRC}/projects/cfgs/drone/res34_s8_full_2cls.py"
CKPT_PATH = "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth"


def build_model(cfg_path, ckpt_path, gpu_id):
    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.cuda(gpu_id)
    model.eval()
    return model, cfg


def preprocess(frame_bgr, img_scale_wh, mean, std):
    resized = cv2.resize(frame_bgr, img_scale_wh, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normed = (rgb - mean) / std
    chw = normed.transpose(2, 0, 1)
    return torch.from_numpy(chw).float()


def resample_polyline(points, n=12):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=0, help="0=전체")
    ap.add_argument("--label", default="GANet_cache_gen")
    a = ap.parse_args()

    stage_order = ["1_디코딩", "2_전처리(resize+정규화+텐서화)", "3_추론(GPU forward)",
                   "4_후처리(재샘플링+직렬화)"]
    stage_totals = {k: 0.0 for k in stage_order}

    t_model0 = time.perf_counter()
    model, cfg = build_model(CFG_PATH, CKPT_PATH, a.gpu_id)
    img_scale_wh = tuple(cfg.img_scale)
    mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)
    torch.cuda.synchronize()
    model_load_sec = time.perf_counter() - t_model0

    cap = cv2.VideoCapture(a.video)
    assert cap.isOpened(), f"cannot open {a.video}"

    n_frames = 0
    n_lanes_total = 0
    t_wall0 = time.perf_counter()

    with torch.no_grad():
        while True:
            if a.max_frames and n_frames >= a.max_frames:
                break
            torch.cuda.synchronize()  # 이전 프레임의 비동기 GPU 작업이 이번 측정에 안 섞이게
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret:
                break
            stage_totals["1_디코딩"] += t1 - t0

            tensor = preprocess(frame, img_scale_wh, mean, std).unsqueeze(0).cuda(a.gpu_id)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            stage_totals["2_전처리(resize+정규화+텐서화)"] += t2 - t1

            meta = dict(
                img_shape=(img_scale_wh[1], img_scale_wh[0], 3),
                ori_shape=(frame.shape[0], frame.shape[1], 3),
                offset_x=0, offset_y=0,
                filename=f"frame_{n_frames:06d}.jpg",
                ori_filename=f"frame_{n_frames:06d}.jpg",
            )
            feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
            pred = model.bbox_head.forward_test(feats)
            result = model.post_processing(pred, meta)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            stage_totals["3_추론(GPU forward)"] += t3 - t2

            lanes, _, _, classes = result["final_dict_list"]
            n_lanes_total += len(lanes)
            record_lanes = []
            for lane, cls in zip(lanes, classes):
                resampled = resample_polyline(lane, n=a.n_points)
                record_lanes.append({"cls": int(cls) - 1, "points": np.round(resampled, 1).tolist()})
            _ = json.dumps({"frame": n_frames, "lanes": record_lanes})
            t4 = time.perf_counter()
            stage_totals["4_후처리(재샘플링+직렬화)"] += t4 - t3

            n_frames += 1
            if n_frames % 200 == 0:
                print(f"[{a.label}] frame {n_frames} done", flush=True)

    cap.release()
    t_wall_total = time.perf_counter() - t_wall0

    print(f"\n=== {a.label} ({a.video}) ===")
    print(f"frames={n_frames}  lanes_total={n_lanes_total}")
    print(f"모델 로딩(1회성, fps 계산에서 제외): {model_load_sec:.2f}s")
    print(f"프레임 처리 total: {t_wall_total:.2f}s  fps={n_frames / t_wall_total:.2f}")
    print(f"{'구간':34s} {'총(s)':>10s} {'ms/frame':>10s} {'%':>8s}")
    for name in stage_order:
        total = stage_totals[name]
        per_frame = total / n_frames * 1000 if n_frames else 0.0
        pct = total / t_wall_total * 100 if t_wall_total else 0.0
        print(f"{name:34s} {total:10.2f} {per_frame:10.3f} {pct:7.1f}%")


if __name__ == "__main__":
    main()
