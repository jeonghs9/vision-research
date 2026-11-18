#!/usr/bin/env python3
"""[OpenLane용] 체크포인트 하나를 OpenLane val 또는 test 전체에 돌려서 정밀도/재현율/F1을 계산한다.

드론용 score_checkpoint.py와 같은 점 거리 기반 매칭(공식 CULane IoU 평가 아님)이고, 차이는 세 가지다.
  - 분할 선택: --split val | test (processed/culane_lane_ganet/list/<split>.txt)
  - GT는 변환 라벨 파일(.lines.txt + .cls.txt, 원본 1920x1280 픽셀)에서 직접 읽는다.
  - 클래스(1=실선, 2=점선)까지 본다.

방법:
  1. 예측·GT 차선을 y 기준으로 20px 간격 재샘플
  2. 예측-GT 쌍마다 겹치는 y 구간(20px 이상)에서 평균 |x 오차| 계산
  3. 평균 오차가 --match-thr(기본 20px, 원본 1920 기준) 이하면 매칭 후보, 거리 가까운 쌍부터 그리디 1:1 배정
     (드론 40px@3840과 같은 비율: 이미지 너비의 약 1%)
  4. 지표
     [위치]   클래스 무시하고 매칭 → P / R / F1, 매칭된 쌍 중 클래스까지 맞은 비율(cls_acc)
     [클래스] 같은 클래스끼리만 매칭 → 실선·점선 각각 P / R / F1, 두 클래스 평균(mF1)
              (드론 2클래스 점수는 [위치] 방식으로만 냈다 — 직접 비교 금지)

출력: 마지막 줄 "RESULT ..." 한 줄 (에폭 스윕 스크립트가 grep), --out-json 주면 상세 JSON 저장

실행 (GANet source 폴더에서):
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/04-eval/openlane_score_checkpoint.py \
    projects/cfgs/openlane/res34_s16_full_2cls_P3del_noSA.py \
    /home/hsjeong/workspace/vision-det/GANet/test/015-openlane-res34-2cls-1280-P3del-noSA-150ep/epoch_150.pth \
    --split val
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "./")

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_dp, compat_cfg, get_device

from projects.plugin import *  # noqa

STEP = 20.0
CLS_NAME = {1: "solid", 2: "dashed"}


def resample(pts, step=STEP):
    """[(x,y), ...] -> y 오름차순 재샘플."""
    pts = np.asarray(pts, dtype=np.float64)
    pts = pts[np.argsort(pts[:, 1])]
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    if y1 - y0 < step:
        return pts
    grid = np.arange(y0, y1 + 1e-6, step)
    return np.stack([np.interp(grid, pts[:, 1], pts[:, 0]), grid], axis=1)


def mean_dist(a, b):
    """겹치는 y 구간에서 평균 |x 오차|. 겹침이 STEP 미만이면 inf."""
    y0, y1 = max(a[:, 1].min(), b[:, 1].min()), min(a[:, 1].max(), b[:, 1].max())
    if y1 - y0 < STEP:
        return float("inf")
    grid = np.arange(y0, y1 + 1e-6, STEP)
    return float(np.abs(np.interp(grid, a[:, 1], a[:, 0]) - np.interp(grid, b[:, 1], b[:, 0])).mean())


def greedy_match(pr, gt, thr):
    """pr, gt: 재샘플된 선 리스트. 반환: [(i, j, dist), ...] 1:1 매칭."""
    if not pr or not gt:
        return []
    pairs = sorted((mean_dist(p, g), i, j) for i, p in enumerate(pr) for j, g in enumerate(gt))
    up, ug, out = set(), set(), []
    for d, i, j in pairs:
        if d > thr:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); out.append((i, j, d))
    return out


def read_gt(data_root, rel_jpg):
    """변환 라벨(.lines.txt, .cls.txt) 읽기. CulaneDataset.load_labels와 같이 점 2개 미만 차선은 버린다."""
    stem = os.path.join(data_root, rel_jpg[:-4])
    lanes, classes = [], []
    lines = open(stem + ".lines.txt").read().splitlines()
    cls = [int(x) for x in open(stem + ".cls.txt").read().split()]
    assert len(lines) == len(cls), stem
    for ln, c in zip(lines, cls):
        v = [float(x) for x in ln.split()]
        if len(v) > 3:
            lanes.append([(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)])
            classes.append(c)
    return lanes, classes


def prf(m, n_pred, n_gt):
    p = m / n_pred if n_pred else 0.0
    r = m / n_gt if n_gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--match-thr", type=float, default=20.0, help="매칭 임계값 px (원본 1920 기준)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (스크립트 점검용, 0=전체)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    cfg = compat_cfg(mmcv.Config.fromfile(a.config))
    cfg.gpu_ids = [a.gpu_id]
    cfg.device = get_device()
    data_root = cfg.data_root
    list_file = os.path.join(data_root, "list", f"{a.split}.txt")
    if a.limit:
        tmp = os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), f".score_{a.split}_limit{a.limit}.txt")
        with open(list_file) as f, open(tmp, "w") as g:
            g.write("".join(f.readlines()[:a.limit]))
        list_file = tmp
    cfg.data.test.data_list = [list_file]
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model.eval()

    n_img = n_pred = n_gt = n_match = n_cls_ok = 0
    dists = []
    per = {c: dict(pred=0, gt=0, match=0) for c in CLS_NAME}
    pred_cls_seen = set()
    with torch.no_grad():
        for data in loader:
            res = model(return_loss=False, rescale=True, **data)[0]
            lanes, _, _, pcls = res["final_dict_list"]
            pcls = [int(c) for c in pcls]
            pred_cls_seen.update(pcls)
            rel = res["img_metas"]["ori_filename"]
            gts, gcls = read_gt(data_root, rel)
            pr = [resample(p) for p in lanes]
            gt = [resample(g) for g in gts]

            # [위치] 클래스 무시
            m = greedy_match(pr, gt, a.match_thr)
            n_pred += len(pr); n_gt += len(gt); n_match += len(m)
            n_cls_ok += sum(pcls[i] == gcls[j] for i, j, _ in m)
            dists += [d for _, _, d in m]
            # [클래스] 같은 클래스끼리
            for c in CLS_NAME:
                pi = [k for k, x in enumerate(pcls) if x == c]
                gi = [k for k, x in enumerate(gcls) if x == c]
                mc = greedy_match([pr[k] for k in pi], [gt[k] for k in gi], a.match_thr)
                per[c]["pred"] += len(pi); per[c]["gt"] += len(gi); per[c]["match"] += len(mc)
            n_img += 1

    assert pred_cls_seen <= set(CLS_NAME), f"예측 클래스 값이 1/2가 아님: {pred_cls_seen}"
    p, r, f = prf(n_match, n_pred, n_gt)
    out = dict(split=a.split, n_img=n_img, match_thr=a.match_thr, checkpoint=a.checkpoint,
               pos=dict(pred=n_pred, gt=n_gt, match=n_match, precision=p, recall=r, f1=f,
                        cls_acc=(n_cls_ok / n_match if n_match else 0.0),
                        avg_dist_px=(float(np.mean(dists)) if dists else float("nan"))))
    fs = []
    for c, name in CLS_NAME.items():
        cp, cr, cf = prf(per[c]["match"], per[c]["pred"], per[c]["gt"])
        out[name] = dict(**per[c], precision=cp, recall=cr, f1=cf)
        fs.append(cf)
    out["mF1"] = float(np.mean(fs))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, ensure_ascii=False)
    s, d = out["solid"], out["dashed"]
    print(f"RESULT split={a.split} n_img={n_img} thr={a.match_thr:g} | pos P={p:.4f} R={r:.4f} F1={f:.4f} "
          f"cls_acc={out['pos']['cls_acc']:.4f} dist={out['pos']['avg_dist_px']:.2f} "
          f"| solid P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
          f"| dashed P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} | mF1={out['mF1']:.4f}")


if __name__ == "__main__":
    main()
