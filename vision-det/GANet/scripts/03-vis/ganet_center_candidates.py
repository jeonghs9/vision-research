#!/usr/bin/env python3
"""confidence>kpt_thr인 전체 키포인트(작은 점)와, 그중 '클러스터 중심 후보'
(root_center_arr = offset_y<root_thr 까지 동시에 만족하는 점, 큰 별표)를
구분해서 그린다. 반대 차로 점선 영역에 큰 별표가 하나도 없다는 걸
직접 눈으로 확인하기 위한 1회성 진단 스크립트 (scripts/에 저장 안 함, tmp 전용).

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_center_candidates.py
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
OUT_PATH = "/home/hsjeong/tmp/ganet_center_candidates.png"

CLASS_COLOR = {1: "lime", 2: "red"}
CLASS_LABEL = {1: "실선(class1)", 2: "점선(class2)"}

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
    output = model.bbox_head.forward_test(feats)  # [(class_id, root_center_arr, kpt_seeds), ...]

resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# kpt_seeds/root_center_arr는 전부 native(stride 8) grid 좌표다 -- 전체 해상도
# 이미지에 겹쳐 그리려면 8을 곱해야 한다(안 곱하면 사진의 작은 모서리만 보임).
all_xy = []
for class_id, root_center_arr, kpt_seeds in output:
    if len(kpt_seeds) > 0:
        all_xy.append(kpt_seeds[:, 1, :] * 8)  # root 좌표 (align=0, root=1)
        all_xy.append(kpt_seeds[:, 0, :] * 8)  # align 좌표
all_xy = np.concatenate(all_xy, axis=0)
pad = 40 * 8
x0 = max(int(all_xy[:, 0].min()) - pad, 0)
x1 = min(int(all_xy[:, 0].max()) + pad, img_scale_wh[0])
y0 = max(int(all_xy[:, 1].min()) - pad * 3, 0)
y1 = min(int(all_xy[:, 1].max()) + pad // 4, img_scale_wh[1])

fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(resized_rgb)

for class_id, root_center_arr, kpt_seeds in output:
    color = CLASS_COLOR[class_id]
    label = CLASS_LABEL[class_id]
    # 개별 키포인트(작은 점) -- align 좌표(실제 검출 위치), native->전체해상도 8배
    if len(kpt_seeds) > 0:
        align_xy = kpt_seeds[:, 0, :] * 8
        ax.scatter(align_xy[:, 0], align_xy[:, 1], s=8, c=color, alpha=0.5,
                   label=f"{label} 키포인트 {len(kpt_seeds)}개 (confidence>{kpt_thr})")
    # 클러스터 중심 후보(큰 별표) -- root_center_arr은 native grid 좌표라 8배
    if len(root_center_arr) > 0:
        cand = root_center_arr.astype(float) * 8
        ax.scatter(cand[:, 0], cand[:, 1], s=400, c=color, marker="*",
                   edgecolors="black", linewidths=1.5, zorder=5,
                   label=f"{label} 중심 후보 {len(root_center_arr)}개 (+offset_y<{root_thr})")
    else:
        ax.text(0.02, 0.06 if class_id == 2 else 0.1, f"{label}: 중심 후보 0개!",
                transform=ax.transAxes, color=color, fontsize=13, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8))

ax.set_xlim(x0, x1)
ax.set_ylim(y1, y0)  # y축 뒤집기(이미지 좌표계)
ax.set_title("클러스터 중심 후보(별표) vs 전체 키포인트(점) -- 반대 차로 점선엔 별표가 없다",
             fontsize=13)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.axis("off")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"saved: {OUT_PATH}")
