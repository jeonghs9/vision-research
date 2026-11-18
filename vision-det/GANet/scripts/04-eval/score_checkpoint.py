#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] 체크포인트 하나를 val 전체(163장)에 대해 돌려서 정밀도/재현율을 계산한다.

CULane 공식 C++ 평가 도구가 없어서(안 빌드함) 간단한 점 거리 기반 매칭으로
대체한다 — 정확한 벤치마크 수치는 아니지만, 체크포인트끼리 상대 비교하기엔
충분하다.

방법:
  1. 각 예측 차선을 y값 기준으로 재샘플링(step=20px)
  2. 각 GT 차선도 동일하게 재샘플링
  3. 예측-GT 쌍마다 겹치는 y구간에서 평균 |x오차| 계산
  4. 평균 오차가 MATCH_THR(기본 40px, 원본 3840 기준) 이하면 매칭으로 간주
     (예측 1개는 GT 1개하고만 매칭 - 거리 제일 가까운 것부터 그리디로 배정)
  5. precision = 매칭된 예측 수 / 전체 예측 수
     recall    = 매칭된 GT 수 / 전체 GT 수

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/04-eval/score_checkpoint.py \
    projects/cfgs/drone/res18_s8_full.py \
    test/001-res18-1cls-1280-150ep/epoch_100.pth
"""
import argparse
import sys

sys.path.insert(0, "./")

import mmcv
import numpy as np
import torch

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import compat_cfg, get_device
from mmcv.runner import load_checkpoint
from mmdet.utils import build_dp
from projects.plugin import *  # noqa

MATCH_THR = 40.0  # px, 원본 3840 기준


def resample(pts, step=20.0):
    """[(x,y), ...] -> y 오름차순 재샘플 (step 간격)."""
    pts = np.asarray(pts, dtype=np.float64)
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    if y1 - y0 < step:
        return pts
    grid = np.arange(y0, y1 + 1e-6, step)
    x = np.interp(grid, pts[:, 1], pts[:, 0])
    return np.stack([x, grid], axis=1)


def mean_dist(a, b):
    """두 재샘플된 선(a,b)의 겹치는 y구간에서 평균 |x오차|. 안 겹치면 inf."""
    y0 = max(a[:, 1].min(), b[:, 1].min())
    y1 = min(a[:, 1].max(), b[:, 1].max())
    if y1 - y0 < 20:
        return float("inf")
    grid = np.arange(y0, y1 + 1e-6, 20.0)
    xa = np.interp(grid, a[:, 1], a[:, 0])
    xb = np.interp(grid, b[:, 1], b[:, 0])
    return float(np.abs(xa - xb).mean())


def match_lanes(preds, gts):
    """그리디 매칭. 반환: (매칭된 예측 수, 매칭된 GT 수, 매칭쌍 평균거리 목록)."""
    if not preds or not gts:
        return 0, 0, []
    pr = [resample(p) for p in preds]
    gt = [resample(g) for g in gts]
    dmat = np.full((len(pr), len(gt)), np.inf)
    for i, p in enumerate(pr):
        for j, g in enumerate(gt):
            dmat[i, j] = mean_dist(p, g)

    used_p, used_g, dists = set(), set(), []
    pairs = [(dmat[i, j], i, j) for i in range(len(pr)) for j in range(len(gt))]
    pairs.sort(key=lambda t: t[0])
    for d, i, j in pairs:
        if d > MATCH_THR:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        dists.append(d)
    return len(used_p), len(used_g), dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--gpu-id", type=int, default=0)
    a = ap.parse_args()

    cfg = mmcv.Config.fromfile(a.config)
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [a.gpu_id]
    cfg.device = get_device()
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    dataset.open_check()
    dataset.set_all_scenes()
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model.eval()

    tot_pred = tot_gt = tot_match_p = tot_match_g = 0
    all_dists = []
    n_img = 0
    with torch.no_grad():
        for data in loader:
            result = model(return_loss=False, rescale=True, **data)[0]
            lanes = result["final_dict_list"][0]
            meta = result["img_metas"]
            gt_flat = meta.get("gt_points_for_show", [])
            gts = [[(lane[2 * i], lane[2 * i + 1]) for i in range(len(lane) // 2)] for lane in gt_flat]

            mp, mg, dists = match_lanes(lanes, gts)
            tot_pred += len(lanes)
            tot_gt += len(gts)
            tot_match_p += mp
            tot_match_g += mg
            all_dists.extend(dists)
            n_img += 1

    precision = tot_match_p / tot_pred if tot_pred else 0.0
    recall = tot_match_g / tot_gt if tot_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_dist = float(np.mean(all_dists)) if all_dists else float("nan")

    print(f"RESULT n_img={n_img} pred={tot_pred} gt={tot_gt} "
          f"matched={tot_match_p} precision={precision:.4f} recall={recall:.4f} "
          f"f1={f1:.4f} avg_match_dist_px={avg_dist:.2f}")


if __name__ == "__main__":
    main()
