#!/usr/bin/env python3
"""후처리 그룹핑 방식 비교 그림: 기존(중심 후보 기준, cluster_by_center_thr=5) vs
root끼리 직접 묶기(cluster_by_center_thr=None -> update_coords_fast, cluster_thr로 root 체인 묶음).
모델 출력(pred)은 동일하고 post_processing만 바꿔서 두 번 돌린다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_grouping_compare.py
"""
import os, sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "05-cls2-점선미탐원인분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
]
COL = {1: (0, 220, 0), 2: (0, 0, 230)}  # BGR: 실선=초록, 점선=빨강

cfg = mmcv.Config.fromfile("projects/cfgs/drone/res34_s8_full_2cls.py")
cfg.model.train_cfg = None
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth", map_location="cpu")
model = model.cuda(0).eval()
W, H = cfg.img_scale
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
pp = model.post_processing
base_by_center = pp.cluster_by_center_thr


def run(pred, meta, by_center):
    pp.cluster_by_center_thr = by_center
    lanes, _, _, classes = pp(pred, meta)["final_dict_list"]
    return lanes, classes


def same_lane(la_, ca_, lb_, cb_, tol):
    """같은 클래스이고, 겹치는 y 구간에서 두 선의 x 차이 평균이 tol(px) 이내면 같은 선."""
    if ca_ != cb_:
        return False
    a_ = np.array(sorted(la_, key=lambda p: p[1]), float)
    b_ = np.array(sorted(lb_, key=lambda p: p[1]), float)
    y0_, y1_ = max(a_[:, 1].min(), b_[:, 1].min()), min(a_[:, 1].max(), b_[:, 1].max())
    if y1_ - y0_ < 20:
        return False
    ys_ = np.linspace(y0_, y1_, 30)
    return np.abs(np.interp(ys_, a_[:, 1], a_[:, 0]) - np.interp(ys_, b_[:, 1], b_[:, 0])).mean() < tol


def draw(frame, lanes, classes, new_flags, lw):
    img = frame.copy()
    for i, (lane, c, new) in enumerate(zip(lanes, classes, new_flags)):
        pts = np.array(lane, np.int32)
        if new:  # 기존엔 없던 선: 노란 테두리로 강조
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), lw * 3)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, COL[c], lw)
        srt = pts[np.argsort(pts[:, 1])]
        top = srt[min(len(srt) - 1, max(0, int(len(srt) * 0.15)))]
        cv2.putText(img, f"L{i}", (int(top[0]) + 12, int(top[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    lw * 0.6, (255, 255, 255), lw * 2, cv2.LINE_AA)
        cv2.putText(img, f"L{i}", (int(top[0]) + 12, int(top[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    lw * 0.6, (0, 0, 0), max(1, lw // 2), cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def describe(lanes, classes, new_flags):
    out = []
    for i, (l, c, n) in enumerate(zip(lanes, classes, new_flags)):
        p = np.array(l)
        out.append(f"L{i}[{'실선' if c == 1 else '점선'}] {len(l)}점, x={p[:,0].min()}~{p[:,0].max()}, "
                   f"y={p[:,1].min()}~{p[:,1].max()}" + ("  ★새로 생김" if n else ""))
    return out


for prefix, path in JOBS:
    frame = cv2.imread(path)
    h0, w0 = frame.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB).astype(np.float32)
    t = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    pp.crop_bbox = [0, 0, w0, h0]
    meta = dict(img_shape=(H, W, 3), ori_shape=(h0, w0, 3), offset_x=0, offset_y=0)
    with torch.no_grad():
        pred = model.bbox_head.forward_test(model.extract_feat(t))

    la, ca = run(pred, meta, base_by_center)
    lb, cb = run(pred, meta, None)
    tol = w0 * 0.015
    new_b = [not any(same_lane(a_, c_, l, c, tol) for a_, c_ in zip(la, ca)) for l, c in zip(lb, cb)]

    lw = max(2, int(round(w0 / 640)))
    fig, ax = plt.subplots(1, 2, figsize=(22, 8))
    fig.suptitle(f"{prefix}-7. 후처리 그룹핑 방식 비교 -- 모델 출력은 동일, post_processing만 변경", fontsize=16)
    ax[0].imshow(draw(frame, la, ca, [False] * len(la), lw))
    ax[0].set_title(f"기존: 중심 후보 기준 (cluster_by_center_thr={base_by_center})\n"
                    f"dy<{model.bbox_head.root_thr} AND conf>{model.bbox_head.kpt_thr}인 점을 중심으로, "
                    f"root가 5칸 이내인 점만 배정 -> {len(la)}개", fontsize=12)
    ax[1].imshow(draw(frame, lb, cb, new_b, lw))
    ax[1].set_title(f"변경: root끼리 직접 묶기 (cluster_by_center_thr=None)\n"
                    f"root(시작점 예측 위치)가 {pp.cluster_thr}칸 이내로 이어지는 점끼리 묶음 -> {len(lb)}개 "
                    f"(노란 테두리=새로 생긴 선)", fontsize=12)
    for a in ax:
        a.axis("off")
    txt_a = "\n".join(describe(la, ca, [False] * len(la)))
    txt_b = "\n".join(describe(lb, cb, new_b))
    fig.text(0.25, 0.01, "[기존]\n" + txt_a, ha="center", va="bottom", fontsize=10)
    fig.text(0.75, 0.01, "[변경]\n" + txt_b, ha="center", va="bottom", fontsize=10)
    n = max(len(la), len(lb))
    fig.text(0.5, 0.01 + 0.0 , "", ha="center")
    fig.tight_layout(rect=[0, 0.03 + 0.024 * (n + 1), 1, 0.94])
    outp = os.path.join(OUTDIR, f"{prefix}-7. 그룹핑방식_비교.png")
    fig.savefig(outp, dpi=120)
    plt.close(fig)
    print("saved:", outp, "| 기존", len(la), "-> 변경", len(lb), "| 새로 생긴 선", sum(new_b))

pp.cluster_by_center_thr = base_by_center
