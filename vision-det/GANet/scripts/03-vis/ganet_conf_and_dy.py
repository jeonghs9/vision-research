#!/usr/bin/env python3
"""클래스별 confidence(연속값) + dy(offset_y)<root_thr 칸을 함께 시각화.

중심 후보(별표)는 confidence>kpt_thr '그리고' dy<root_thr 둘 다 만족해야 하는데,
지금까지는 그 교집합(별표)과 confidence>kpt_thr만 통과한 점(작은 점)만 보여줬다.
이번엔 "dy<root_thr인 칸이 애초에 어디 있는지"(confidence 무관하게)를 별도
마커로 얹고, confidence 자체도 0/1 마스킹이 아니라 연속값 히트맵으로 깔아서,
반대 차로 점선이 '거의 다 왔는데 문턱 하나 차이로' 떨어진 건지 아니면 '그 근처에
dy<1인 칸 자체가 없는' 건지 눈으로 구분할 수 있게 한다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_conf_and_dy.py
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

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

CFG_PATH = "projects/cfgs/drone/res34_s8_full_2cls.py"
CKPT_PATH = "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth"
IMG_PATH = ("/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
            "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png")
OUT_PATH = "/home/hsjeong/tmp/ganet_conf_and_dy.png"

CLASS_LABEL = {0: "실선(class1)", 1: "점선(class2)"}

cfg = mmcv.Config.fromfile(CFG_PATH)
cfg.model.train_cfg = None
if "init_cfg" in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, CKPT_PATH, map_location="cpu")
model = model.cuda(0).eval()

img_scale_wh = tuple(cfg.img_scale)
mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)
kpt_thr = cfg.model.head.kpt_thr
root_thr = cfg.model.head.root_thr

frame = cv2.imread(IMG_PATH)
resized = cv2.resize(frame, img_scale_wh, interpolation=cv2.INTER_LINEAR)
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
normed = (rgb - mean) / std
tensor = torch.from_numpy(normed.transpose(2, 0, 1)).float().unsqueeze(0).cuda(0)

with torch.no_grad():
    feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
    head_dict = model.bbox_head.forward(feats)

kpts_hm = head_dict["kpts_hm"][0].cpu().numpy()        # (C,h,w) 연속 confidence(0~1)
pts_offset = head_dict["pts_offset"][0].cpu().numpy()   # (2,h,w) [0]=dx,[1]=dy

resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
W, H = img_scale_wh

# 표시 영역: 두 클래스 전부 conf>0.1인 칸의 bbox + 여유(두 점선 영역이 다 보이게 넉넉히)
any_conf = kpts_hm.max(axis=0) > 0.1
ys, xs = np.where(any_conf)
pad = 6
x0 = max(int(xs.min()) - pad, 0) * 8
x1 = min(int(xs.max()) + pad, kpts_hm.shape[2]) * 8
y0 = max(int(ys.min()) - pad, 0) * 8
y1 = min(int(ys.max()) + pad, kpts_hm.shape[1]) * 8

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle(f"클래스별 confidence(연속값) 히트맵 + dy(offset_y)<{root_thr} 칸 표시 "
             f"(별표=confidence>{kpt_thr} AND dy<{root_thr}, 즉 실제 중심 후보)", fontsize=13)

for c in range(2):
    ax = axes[c]
    conf = kpts_hm[c]
    dy = pts_offset[1]
    dy_lt = dy < root_thr
    center_cand = dy_lt & (conf > kpt_thr)

    conf_up = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)

    ax.imshow(resized_rgb)
    im = ax.imshow(conf_up, cmap="inferno", vmin=0, vmax=1, alpha=0.6)

    # dy<root_thr 칸 전체(confidence 무관) -- 하늘색 '+'
    dy_ys, dy_xs = np.where(dy_lt)
    ax.scatter(dy_xs * 8 + 4, dy_ys * 8 + 4, marker="+", s=60, c="cyan",
               linewidths=1.2, label=f"dy<{root_thr}인 칸 (confidence 무관, {dy_lt.sum()}개)")

    # 실제 중심 후보(두 조건 다 만족) -- 큰 별표
    cc_ys, cc_xs = np.where(center_cand)
    if len(cc_xs) > 0:
        ax.scatter(cc_xs * 8 + 4, cc_ys * 8 + 4, marker="*", s=300, c="lime",
                   edgecolors="black", linewidths=1.2,
                   label=f"중심 후보(dy<{root_thr} AND conf>{kpt_thr}) {len(cc_xs)}개")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(f"{CLASS_LABEL[c]} -- dy<{root_thr}인 칸 {dy_lt.sum()}개 중 "
                 f"confidence>{kpt_thr}도 만족하는 칸 {center_cand.sum()}개")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="confidence (sigmoid, 0~1)")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PATH, dpi=150)
print(f"saved: {OUT_PATH}")
