#!/usr/bin/env python3
"""GANet 미탐원인분석 그림 7장(N. / N-1 ~ N-6)을 한 장의 입력 이미지에 대해 한 번에 생성.

  N.   원본프레임_GANet_최종검출결과   (post_processing까지 끝난 polyline)
  N-1. FPN특징맵_히트맵                 (aux_feat 64채널 L2-norm)
  N-2. 키포인트_컨피던스맵_히트맵       (keypts_head sigmoid, 클래스별)
  N-3. 오프셋맵_히트맵                  (pts_offset 크기/x/y, confidence 마스킹)
  N-4. 오프셋맵_XY벡터_히트맵           (x/y 성분, 도로영역 확대)
  N-5. 시작점후보                       (키포인트 vs 중심 후보 root_center_arr)
  N-6. confidence_dy조건                (키포인트별 dy, dy<root_thr 별표)

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_analysis_all.py \
    --image "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg" \
    --prefix 2
"""
import argparse
import os
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

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--prefix", required=True)
ap.add_argument("--config", default="projects/cfgs/drone/res34_s8_full_2cls.py")
ap.add_argument("--checkpoint", default="/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth")
ap.add_argument("--outdir", default="/home/hsjeong/workspace/vision-det/GANet/output/"
                                     "05-cls2-점선미탐원인분석/figures")
a = ap.parse_args()
P = a.prefix
out = lambda name: os.path.join(a.outdir, f"{P}{name}")

cfg = mmcv.Config.fromfile(a.config)
cfg.model.train_cfg = None
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, a.checkpoint, map_location="cpu")
model = model.cuda(0).eval()

W, H = cfg.img_scale
S = 8
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
kpt_thr = cfg.model.head.kpt_thr
root_thr = cfg.model.head.root_thr
crop_cfg = list(cfg.model.post_processing.crop_bbox)

frame = cv2.imread(a.image)
assert frame is not None, a.image
h0, w0 = frame.shape[:2]
resized = cv2.resize(frame, (W, H))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)

with torch.no_grad():
    feats = model.extract_feat(t)
    hd = model.bbox_head.forward(feats)
    pred = model.bbox_head.forward_test(feats)
    model.post_processing.crop_bbox = [0, 0, w0, h0]  # 학습 config 값(드론 3840x2160) 대신 실제 해상도
    meta = dict(img_shape=(H, W, 3), ori_shape=(h0, w0, 3), offset_x=0, offset_y=0,
                filename=a.image, ori_filename=a.image)
    result = model.post_processing(pred, meta)
    aux = feats["aux_feat"][0].cpu().numpy()            # (64,h,w)
    hm_t = hd["kpts_hm"]
    hmax = F.max_pool2d(hm_t, (1, 3), stride=(1, 1), padding=(0, 1))
    hm_nms = (hm_t * (hmax == hm_t).float())[0].cpu().numpy()
    hm = hm_t[0].cpu().numpy()                           # (C,h,w)
    off = hd["pts_offset"][0].cpu().numpy()              # (2,h,w)

fh, fw = hm.shape[1:]
base_caption = (f"입력: {os.path.basename(a.image)} ({w0}x{h0}) → 모델 입력 리사이즈 {W}x{H} → "
                f"feature map {fw}x{fh} (stride {S})")
up = lambda m, interp=cv2.INTER_NEAREST: cv2.resize(m.astype(np.float32), (W, H), interpolation=interp)
conf_all = hm.max(0)
mask = conf_all > kpt_thr
CLS = {1: "실선(class1)", 2: "점선(class2)"}
COL_BGR = {1: (0, 220, 0), 2: (0, 0, 220)}

# ---------------- N. 최종 검출 결과 ----------------
lanes, _, _, classes = result["final_dict_list"]
det = frame.copy()
info = []
n_out = 0
lw = max(2, int(round(w0 / 640)))
for i, (lane, c) in enumerate(zip(lanes, classes)):
    pts = np.array(lane, dtype=np.int32)
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] < w0) & (pts[:, 1] >= 0) & (pts[:, 1] < h0)).sum()
    n_out += len(pts) - inside
    cv2.polylines(det, [pts.reshape(-1, 1, 2)], False, COL_BGR.get(c, (255, 255, 255)), lw)
    info.append(f"- lane{i}[{'실선' if c == 1 else '점선'}]: {len(pts)}점, x={pts[:,0].min()}~{pts[:,0].max()}, "
                f"y={pts[:,1].min()}~{pts[:,1].max()}, 프레임 내부 {inside}/{len(pts)}")
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle(f"{P}. 원본 프레임에 대한 GANet 최종 검출 결과", fontsize=16)
ax[0].imshow(orig_rgb); ax[0].set_title("원본 프레임", fontsize=14); ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(det, cv2.COLOR_BGR2RGB))
ax[1].set_title("GANet 최종 검출 결과\n(post_processing까지 끝난 polyline)", fontsize=14); ax[1].axis("off")
n_pts = sum(len(l) for l in lanes)
fig.text(0.5, 0.01,
         f"입력: {os.path.basename(a.image)} ({w0}x{h0}) | 모델 입력 리사이즈: {W}x{H}\n"
         f"crop_bbox: 학습 config 원값={crop_cfg} -> 실제 입력 해상도로 보정=[0, 0, {w0}, {h0}]\n"
         f"검출 차선 수: {len(lanes)}, 좌표가 프레임 밖으로 나간 점: {n_out}/{n_pts}개\n"
         + "\n".join(info) + "\n초록=실선(class1), 빨강=점선(class2).",
         ha="center", va="bottom", fontsize=10)
fig.tight_layout(rect=[0, 0.12 + 0.022 * len(lanes), 1, 0.95])
fig.savefig(out(". 원본프레임_GANet_최종검출결과.png"), dpi=120); plt.close(fig)

# ---------------- N-1. FPN feature ----------------
l2 = np.sqrt((aux ** 2).sum(0))
l2n = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-9)
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"{P}-1. FPN 출력 feature[0] 히트맵 (aux_feat)", fontsize=16)
ax[0].imshow(rgb); ax[0].set_title("원본(리사이즈)", fontsize=14); ax[0].axis("off")
ax[1].imshow(rgb)
im = ax[1].imshow(up(l2n, cv2.INTER_LINEAR), cmap="jet", vmin=0, vmax=1, alpha=0.55)
ax[1].set_title("FPN feature[0] 활성화 강도\n(64채널 L2-norm, LFA·offset head 진입 전)", fontsize=14); ax[1].axis("off")
fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02, label="정규화된 활성화 강도")
fig.text(0.5, 0.02, f"{base_caption} | 채널별 L2-norm 범위: {l2.min():.2f} ~ {l2.max():.2f} (이미지 내에서 0~1로 정규화)\n"
         "이 feature는 top-down 융합이 끝난 직후, LFA(deformable conv)와 offset_head 양쪽 모두에 들어가기 전의 "
         "공통 입력이다 (dcn_fpn.py의 aux_feat).", ha="center", fontsize=11)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])
fig.savefig(out("-1. FPN특징맵_히트맵.png"), dpi=120); plt.close(fig)

# ---------------- N-2. confidence map ----------------
dark = (rgb * 0.35).astype(np.uint8)
fig, ax = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle(f"{P}-2. Keypoint Confidence Map 히트맵 (클래스별)", fontsize=16)
ax[0].imshow(rgb); ax[0].set_title("원본(리사이즈)", fontsize=14); ax[0].axis("off")
for c in range(2):
    ax[c + 1].imshow(dark)
    im = ax[c + 1].imshow(up(hm[c], cv2.INTER_LINEAR), cmap="inferno", vmin=0, vmax=1,
                          alpha=np.clip(up(hm[c], cv2.INTER_LINEAR) * 1.5, 0, 1))
    ax[c + 1].set_title(f"confidence map: {CLS[c + 1]}\n(keypts_head 출력, sigmoid 확률)", fontsize=14)
    ax[c + 1].axis("off")
    fig.colorbar(im, ax=ax[c + 1], fraction=0.035, pad=0.02, label="키포인트 확률 (sigmoid)")
fig.text(0.5, 0.02, f"{base_caption}\n클래스별 확률 범위(0~1 스케일, 절대값): 실선(class1): "
         f"{hm[0].min():.3f}~{hm[0].max():.3f} | 점선(class2): {hm[1].min():.3f}~{hm[1].max():.3f}\n"
         "색 스케일은 두 클래스 모두 0~1 절대값으로 고정.", ha="center", fontsize=11)
fig.tight_layout(rect=[0, 0.08, 1, 0.95])
fig.savefig(out("-2. 키포인트_컨피던스맵_히트맵.png"), dpi=120); plt.close(fig)

# ---------------- N-3. offset map (masked) ----------------
mag = np.sqrt(off[0] ** 2 + off[1] ** 2)
def masked(v):
    m = v.copy(); m[~mask] = np.nan; return m
panels = [("offset magnitude (confidence 마스킹됨)\n√(x²+y²)", mag, "jet", "정규화된 offset 크기"),
          ("x-offset (채널0, confidence 마스킹됨)", off[0], "coolwarm", "정규화된 x-offset"),
          ("y-offset (채널1, confidence 마스킹됨)", off[1], "coolwarm", "정규화된 y-offset")]
fig, ax = plt.subplots(1, 4, figsize=(30, 7))
fig.suptitle(f"{P}-3. Offset Map 히트맵 (offset_head 출력, aux_feat 입력, confidence로 마스킹)", fontsize=16)
ax[0].imshow(rgb); ax[0].set_title("원본(리사이즈)", fontsize=14); ax[0].axis("off")
for i, (title, v, cmap, lab) in enumerate(panels):
    vm = v[mask]
    lo, hi = (vm.min(), vm.max()) if vm.size else (0, 1)
    norm = (masked(v) - lo) / (hi - lo + 1e-9)
    ax[i + 1].imshow(rgb)
    im = ax[i + 1].imshow(up(norm), cmap=cmap, vmin=0, vmax=1)
    ax[i + 1].set_title(f"{title}, 마스크 안쪽 실제값 {lo:.2f}~{hi:.2f}", fontsize=13); ax[i + 1].axis("off")
    fig.colorbar(im, ax=ax[i + 1], fraction=0.035, pad=0.02, label=lab)
fig.text(0.5, 0.02, f"{base_caption}\nconfidence(실선/점선 중 최댓값) > kpt_thr({kpt_thr})인 칸만 색칠함 "
         f"({mask.sum()}/{mask.size}칸, {mask.mean()*100:.2f}%) -- 나머지는 원본 그대로.\n"
         "ganetloss.py의 offset loss는 키포인트 칸에서만 계산되고 디코딩도 이 칸에서만 offset을 읽으므로 마스크 밖 값은 표시하지 않는다.",
         ha="center", fontsize=11)
fig.tight_layout(rect=[0, 0.09, 1, 0.95])
fig.savefig(out("-3. 오프셋맵_히트맵.png"), dpi=120); plt.close(fig)

# ---------------- 공통 표시영역(도로 영역 확대) ----------------
ys_m, xs_m = np.where(mask)
pad = 4
CX0 = max(int(xs_m.min()) - pad, 0) * S; CX1 = min(int(xs_m.max()) + pad + 1, fw) * S
CY0 = max(int(ys_m.min()) - pad, 0) * S; CY1 = min(int(ys_m.max()) + pad + 1, fh) * S

# ---------------- N-4. offset x/y ----------------
ox, oy = up(masked(off[0])), up(masked(off[1]))
vx, vy = np.nanmax(np.abs(ox)), np.nanmax(np.abs(oy))
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(f"{P}-4. 오프셋맵(pts_offset) x/y 성분 히트맵 (offset_head 출력, confidence>{kpt_thr} 위치만, "
             "LFA 이전 feature 기반)", fontsize=14)
ax[0].imshow(rgb[CY0:CY1, CX0:CX1]); ax[0].set_title("원본 프레임 (도로 영역만 확대)"); ax[0].axis("off")
for i, (v, vmx, ttl) in enumerate([(ox, vx, "offset x성분 (coord_mat[0]+이 값 = root의 x좌표)\n빨강=+x(오른쪽을 가리킴), 파랑=-x(왼쪽을 가리킴)"),
                                   (oy, vy, "offset y성분 (coord_mat[1]+이 값 = root의 y좌표)\n빨강=+y(아래를 가리킴), 파랑=-y(위를 가리킴)")]):
    ax[i + 1].imshow(rgb[CY0:CY1, CX0:CX1])
    im = ax[i + 1].imshow(v[CY0:CY1, CX0:CX1], cmap="coolwarm", vmin=-vmx, vmax=vmx, alpha=0.75)
    ax[i + 1].set_title(ttl); ax[i + 1].axis("off")
    fig.colorbar(im, ax=ax[i + 1], fraction=0.046, pad=0.04)
fig.text(0.5, 0.02, f"입력: {os.path.basename(a.image)} | native feature map stride 8 -> nearest-neighbor로 {W}x{H} 확대 표시 | "
         f"마스킹: confidence>{kpt_thr} (전체 {mask.size}칸 중 {mask.sum()}칸={mask.mean()*100:.2f}%만 표시) | "
         f"표시 영역: 모델입력 기준 x={CX0}~{CX1}, y={CY0}~{CY1}", ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(out("-4. 오프셋맵_XY벡터_히트맵.png"), dpi=150); plt.close(fig)

# ---------------- N-5. 시작점 후보 ----------------
fig, ax = plt.subplots(figsize=(14, 10))
ax.imshow(rgb)
COL = {1: "lime", 2: "red"}
for class_id, cand, seeds in pred:
    if len(seeds):
        al = seeds[:, 0, :] * S
        ax.scatter(al[:, 0], al[:, 1], s=8, c=COL[class_id], alpha=0.6,
                   label=f"{CLS[class_id]} 키포인트 {len(seeds)}개 (confidence>{kpt_thr})")
    if len(cand):
        cc = cand.astype(float) * S
        ax.scatter(cc[:, 0], cc[:, 1], s=400, c=COL[class_id], marker="*", edgecolors="black",
                   linewidths=1.5, zorder=5, label=f"{CLS[class_id]} 중심 후보 {len(cand)}개 (+offset_y<{root_thr})")
ax.set_xlim(CX0, CX1); ax.set_ylim(CY1, CY0)
ax.set_title(f"{P}-5. 클러스터 중심 후보(별표) vs 전체 키포인트(점)", fontsize=13)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9); ax.axis("off")
fig.tight_layout(); fig.savefig(out("-5. 시작점후보.png"), dpi=150); plt.close(fig)

# ---------------- N-6. 키포인트별 dy ----------------
def split_lanes(rx):
    order = np.argsort(rx); groups, cur = [], [order[0]]
    for p_, q_ in zip(order[:-1], order[1:]):
        if rx[q_] - rx[p_] > 4:
            groups.append(cur); cur = []
        cur.append(q_)
    groups.append(cur)
    return groups

fig, ax = plt.subplots(1, 2, figsize=(22, 9.5))
fig.suptitle(f"{P}-6. confidence>{kpt_thr}인 키포인트만 표시, 색=각 점의 dy(자기 차선 시작점까지 남은 y거리, native칸) "
             f"/ 별표=dy<{root_thr} (실제 중심 후보)", fontsize=13)
for c in range(2):
    ys, xs = np.where(hm_nms[c] > kpt_thr)
    axc = ax[c]
    axc.imshow(rgb)
    if len(xs):
        dy = off[1, ys, xs]; rx = xs + off[0, ys, xs]
        sc = axc.scatter(xs * S + 4, ys * S + 4, c=dy, cmap="turbo_r", vmin=0, vmax=50, s=22,
                         edgecolors="black", linewidths=0.3)
        cd = dy < root_thr
        axc.scatter(xs[cd] * S + 4, ys[cd] * S + 4, marker="*", s=420, c="white", edgecolors="black",
                    linewidths=1.5, zorder=5, label=f"dy<{root_thr} 키포인트(중심 후보) {cd.sum()}개")
        for g in split_lanes(rx):
            g = np.array(g); low = g[np.argmax(ys[g])]
            right = xs[low] * S > CX0 + 0.75 * (CX1 - CX0)
            axc.annotate(f"최하단 dy={dy[low]:.1f}", (xs[low] * S + 4, ys[low] * S + 4),
                         xytext=(-130, -40) if right else (12, -40),
                         textcoords="offset points", fontsize=10, weight="bold",
                         bbox=dict(facecolor="yellow", alpha=0.85),
                         arrowprops=dict(arrowstyle="->", color="yellow", lw=1.5))
        fig.colorbar(sc, ax=axc, fraction=0.035, pad=0.02, label="dy (0=시작점, 50+ 멀리)")
        axc.set_title(f"{CLS[c + 1]} -- 키포인트 {len(xs)}개 중 dy<{root_thr} {cd.sum()}개")
        axc.legend(loc="upper left", fontsize=10, framealpha=0.9)
    else:
        axc.set_title(f"{CLS[c + 1]} -- 키포인트 0개")
    axc.set_xlim(CX0, CX1); axc.set_ylim(CY1, CY0); axc.axis("off")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out("-6. confidence_dy조건.png"), dpi=150); plt.close(fig)

print("lanes:", len(lanes), "classes:", list(classes))
for class_id, cand, seeds in pred:
    print(f"class{class_id}: 키포인트 {len(seeds)}개, 중심후보 {len(cand)}개 {cand.tolist()}")
print("saved prefix", P, "->", a.outdir)
