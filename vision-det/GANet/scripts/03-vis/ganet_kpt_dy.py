#!/usr/bin/env python3
"""confidence>kpt_thr(NMS 후) 키포인트만 골라 각 점의 dy(자기 차선 시작점까지 y거리)를
색으로 표시. dy<root_thr인 점(=실제 중심 후보)은 별표. 차선(점 묶음)마다 가장 아래 점의
dy를 글자로 표기 -- 반대 차로 점선이 시작점 근처(dy<1)까지 내려오는지 바로 보이게.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_kpt_dy.py
"""
import sys
sys.path.insert(0, "./")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

CFG_PATH = "projects/cfgs/drone/res34_s8_full_2cls.py"
CKPT_PATH = "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth"
IMG_PATH = ("/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
            "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png")
OUT_PATH = ("/home/hsjeong/workspace/vision-det/GANet/output/"
            "05-cls2-점선미탐원인분석/figures/1-6. confidence_dy조건.png")
CLASS_LABEL = {0: "실선(class1)", 1: "점선(class2)"}

cfg = mmcv.Config.fromfile(CFG_PATH)
cfg.model.train_cfg = None
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, CKPT_PATH, map_location="cpu")
model = model.cuda(0).eval()

W, H = cfg.img_scale
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
kpt_thr = cfg.model.head.kpt_thr
root_thr = cfg.model.head.root_thr

frame = cv2.imread(IMG_PATH)
resized = cv2.resize(frame, (W, H))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)

with torch.no_grad():
    hd = model.bbox_head.forward(model.extract_feat(t))
    hm = hd["kpts_hm"]
    # ktdet_decode_fast와 동일한 가로방향 NMS
    hmax = F.max_pool2d(hm, (1, 3), stride=(1, 1), padding=(0, 1))
    hm_nms = (hm * (hmax == hm).float())[0].cpu().numpy()
    off = hd["pts_offset"][0].cpu().numpy()


def split_lanes(xs, ys, rx):
    """root_x(자기 시작점 x)가 가까운 점끼리 묶어 차선 단위로 나눈다(간격>8칸이면 다른 차선)."""
    order = np.argsort(rx)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if rx[b] - rx[a] > 4:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    return groups


fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle(f"confidence>{kpt_thr}인 키포인트만 표시, 색=각 점의 dy(자기 차선 시작점까지 남은 y거리, native칸) "
             f"/ 별표=dy<{root_thr} (실제 중심 후보)", fontsize=13)

vmax = 50
_ys, _xs = np.where(hm_nms.max(0) > kpt_thr)
CROP = (max(_xs.min() * 8 - 160, 0), min(_xs.max() * 8 + 120, W), max(_ys.min() * 8 - 100, 0), H)
for c in range(2):
    ax = axes[c]
    ys, xs = np.where(hm_nms[c] > kpt_thr)
    dy = off[1, ys, xs]
    rx = xs + off[0, ys, xs]

    ax.imshow(rgb)
    sc = ax.scatter(xs * 8 + 4, ys * 8 + 4, c=dy, cmap="turbo_r", vmin=0, vmax=vmax, s=22,
                    edgecolors="black", linewidths=0.3)
    cand = dy < root_thr
    ax.scatter(xs[cand] * 8 + 4, ys[cand] * 8 + 4, marker="*", s=420, c="white",
               edgecolors="black", linewidths=1.5, zorder=5,
               label=f"dy<{root_thr} 키포인트(중심 후보) {cand.sum()}개")

    for g in split_lanes(xs, ys, rx):
        g = np.array(g)
        low = g[np.argmax(ys[g])]  # 그 차선에서 가장 아래(카메라 쪽) 점
        ax.annotate(f"최하단 dy={dy[low]:.1f}", (xs[low] * 8 + 4, ys[low] * 8 + 4),
                    xytext=(12, -40), textcoords="offset points", fontsize=11, weight="bold",
                    color="black", bbox=dict(facecolor="yellow", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))

    ax.set_xlim(CROP[0], CROP[1])
    ax.set_ylim(CROP[3], CROP[2])
    ax.set_title(f"{CLASS_LABEL[c]} -- 키포인트 {len(xs)}개 중 dy<{root_thr} {cand.sum()}개")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=f"dy (0=시작점, {vmax}+ 멀리)")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=150)
print("saved:", OUT_PATH)
