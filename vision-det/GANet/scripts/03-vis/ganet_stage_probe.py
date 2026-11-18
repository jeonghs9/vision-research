#!/usr/bin/env python3
"""단계별 특징 비교: 반대 차로 점선의 confidence가 아래쪽에서 사라지는 문제가 파이프라인 어느 단계에서 시작되는지.

단계: (1) backbone C3(stride8) (2) lateral0(자기 레벨만) (3) aux_feat(top-down 융합 후, LFA 전)
      (4) LFA 후(d_feat+aux) (5) features[0](fpn_conv 후, keypts_head 입력) (6) keypts_head 은닉(ReLU 후)
      (7) 최종 confidence
지표: 각 행(native y)에서 선 경로 위 칸의 특징과, 같은 선 위쪽(confidence 확실 구간, y45~60) 평균 특징의
      코사인 유사도. 경로 오차를 감안해 경로 x ±1칸 중 최댓값. 대조: 근접 차로 점선(끝까지 잘 잡힘),
      기준선: 같은 행에서 차선이 아닌 칸(최대 conf<0.05) 평균.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_stage_probe.py
"""
import sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

IMG = ("/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
       "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png")
OUT = ("/home/hsjeong/workspace/vision-det/GANet/output/"
       "05-cls2-점선미탐원인분석/figures/1-10. 단계별특징_비교.png")
# 선 경로(native 좌표 꺾은선)
FAR = [(103, 45), (77, 67), (64, 78), (49.8, 90)]  # 실제 페인트: 점선 키포인트 -> 실선으로 잡힌 조각 끝 -> 같은 기울기로 연장
NEAR = [(111, 45), (105.5, 91)]   # 근접 차로 점선 키포인트 실측 경로
Y0, Y1 = 45, 90
REF = (45, 60)                     # 프로토타입(같은 선 위쪽 확실 구간)

cfg = mmcv.Config.fromfile("projects/cfgs/drone/res34_s8_full_2cls.py")
cfg.model.train_cfg = None
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth", map_location="cpu")
model = model.cuda(0).eval()
W, H = cfg.img_scale
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)

cap = {}
neck, head = model.neck, model.bbox_head
hooks = [
    neck.lateral_convs[0].register_forward_hook(lambda m, i, o: cap.__setitem__("lateral", o.clone())),
    neck.def_convs[0].register_forward_hook(lambda m, i, o: cap.__setitem__("dfeat", o[0].clone())),
    head.keypts_head.hm[1].register_forward_hook(lambda m, i, o: cap.__setitem__("hidden", o.clone())),
]
fr = cv2.imread(IMG)
rgb = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB)
t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
with torch.no_grad():
    bb = model.backbone(t)
    nk = neck(bb)
    hd = head.forward(nk)
for h in hooks:
    h.remove()
assert cfg.model.neck.deconv_layer[0] and not cfg.model.neck.deconv_before
stages = [
    ("① backbone C3 (stride8, 128ch)", bb[1]),
    ("② lateral (자기 레벨만, 64ch)", cap["lateral"]),
    ("③ aux_feat (FPN 융합 후, LFA 전)", nk["aux_feat"]),
    ("④ LFA 후 (deform + aux_feat)", cap["dfeat"] + nk["aux_feat"]),
    ("⑤ features[0] (keypoint head 입력)", nk["features"][0]),
    ("⑥ keypoint head 은닉층 (ReLU 후)", cap["hidden"]),
]
feats = [(n, F.normalize(f[0].float(), dim=0).cpu().numpy()) for n, f in stages]  # (C,h,w) 단위벡터
conf = hd["kpts_hm"][0].cpu().numpy()
fh, fw = conf.shape[1:]
for n, f in feats:
    assert f.shape[1:] == (fh, fw), (n, f.shape)


def path_x(path, y):
    for (xa, ya), (xb, yb) in zip(path[:-1], path[1:]):
        if ya <= y <= yb:
            return xa + (xb - xa) * (y - ya) / (yb - ya)
    return path[-1][0]


def best_cell(fn, proto, y, x):
    xs = [int(np.clip(round(x) + d, 0, fw - 1)) for d in (-1, 0, 1)]
    sims = [float(fn[:, y, xx] @ proto) for xx in xs]
    k = int(np.argmax(sims))
    return sims[k], xs[k]


rows = np.arange(Y0, Y1 + 1)
bg_mask = conf.max(0) < 0.05
res = {}
for n, fn in feats:
    out = {}
    for tag, path in [("far", FAR), ("near", NEAR)]:
        ref_cells = [(y, int(round(path_x(path, y)))) for y in range(REF[0], REF[1] + 1)]
        proto = np.mean([fn[:, y, x] for y, x in ref_cells], 0)
        proto /= np.linalg.norm(proto) + 1e-9
        out[tag] = [best_cell(fn, proto, y, path_x(path, y))[0] for y in rows]
        if tag == "far":
            out["bg"] = [float((fn[:, y, bg_mask[y]].T @ proto).mean()) if bg_mask[y].any() else np.nan for y in rows]
    res[n] = out
conf_far = [max(conf[:, y, int(np.clip(round(path_x(FAR, y)) + d, 0, fw - 1))].max() for d in (-1, 0, 1)) for y in rows]
conf_far_d = [max(conf[1, y, int(np.clip(round(path_x(FAR, y)) + d, 0, fw - 1))] for d in (-1, 0, 1)) for y in rows]
conf_far_s = [max(conf[0, y, int(np.clip(round(path_x(FAR, y)) + d, 0, fw - 1))] for d in (-1, 0, 1)) for y in rows]
conf_near = [max(conf[1, y, int(np.clip(round(path_x(NEAR, y)) + d, 0, fw - 1))] for d in (-1, 0, 1)) for y in rows]

# ---- 그림 ----
fig = plt.figure(figsize=(22, 13))
gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 1, 1, 1])
ax_img = fig.add_subplot(gs[:, 0])
crop = rgb[300:736, 350:1000]
ax_img.imshow(crop, extent=(350, 1000, 736, 300))
for path, col, lab in [(FAR, "#e6194b", "반대 차로 점선 경로"), (NEAR, "#3cb44b", "근접 차로 점선 경로(대조)")]:
    ys_ = np.arange(Y0, Y1 + 1)
    ax_img.plot([path_x(path, y) * 8 + 4 for y in ys_], ys_ * 8 + 4, c=col, lw=2, label=lab)
for yy, ls, lab in [(67, "--", "y=67 점선 conf 끊김"), (77, ":", "y=77 실선 conf도 끊김")]:
    ax_img.axhline(yy * 8 + 4, c="yellow", ls=ls, lw=1.5, label=lab)
ax_img.set_xlim(350, 1000); ax_img.set_ylim(736, 300); ax_img.axis("off")
ax_img.legend(loc="upper right", fontsize=9)
ax_img.set_title("측정 경로 (모델 입력 1280x736 좌표, 일부 확대)", fontsize=11)

for k, (n, out) in enumerate(res.items()):
    ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
    ax.plot(rows, out["far"], c="#e6194b", lw=2, label="반대 차로 점선")
    ax.plot(rows, out["near"], c="#3cb44b", lw=2, label="근접 차로 점선(대조)")
    ax.plot(rows, out["bg"], c="gray", lw=1.5, ls="--", label="차선 아닌 칸 평균(기준선)")
    for yy, ls in [(67, "--"), (77, ":")]:
        ax.axvline(yy, c="orange", ls=ls, lw=1)
    ax.axvspan(*REF, color="#e6194b", alpha=0.07)
    ax.set_title(n, fontsize=11)
    ax.set_xlabel("native y (아래로 갈수록 카메라 쪽)"); ax.set_ylabel("위쪽 구간 평균과의 코사인 유사도")
    ax.set_ylim(-0.2, 1.02); ax.grid(alpha=0.3)
    if k == 0:
        ax.legend(fontsize=8, loc="lower left")
ax = fig.add_subplot(gs[2, 1:])
ax.plot(rows, conf_far_d, c="#e6194b", lw=2, label="반대 차로: 점선 conf")
ax.plot(rows, conf_far_s, c="#e6194b", lw=2, ls=":", label="반대 차로: 실선 conf")
ax.plot(rows, conf_near, c="#3cb44b", lw=2, label="근접 차로: 점선 conf")
ax.axhline(0.4, c="k", lw=1, ls="--", label="kpt_thr 0.4")
for yy, ls in [(67, "--"), (77, ":")]:
    ax.axvline(yy, c="orange", ls=ls, lw=1)
ax.set_title("⑦ 최종 confidence (경로 ±1칸 최대)", fontsize=11)
ax.set_xlabel("native y"); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=4)
fig.suptitle("1-10. 단계별 특징 비교 -- 반대 차로 점선의 '차선다움'이 파이프라인 어느 단계에서 약해지는가 "
             f"(분홍 음영 y{REF[0]}~{REF[1]} = 기준 구간)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=130)
print("saved:", OUT)

# 수치 요약: 구간별 평균
def seg(v, a, b):
    v = np.array(v); m = (rows >= a) & (rows <= b)
    return np.nanmean(v[m])
print(f"{'단계':38s} | 반대 y45-60 | y61-67 | y68-77 | y78-90 || 근접 y78-90 | 기준선 y78-90")
for n, out in res.items():
    print(f"{n:38s} | {seg(out['far'],45,60):.2f} | {seg(out['far'],61,67):.2f} | {seg(out['far'],68,77):.2f} | "
          f"{seg(out['far'],78,90):.2f} || {seg(out['near'],78,90):.2f} | {seg(out['bg'],78,90):.2f}")
print(f"{'⑦ conf(반대 점선/실선, 근접 점선)':38s} | {seg(conf_far_d,45,60):.2f} | {seg(conf_far_d,61,67):.2f} | "
      f"{seg(conf_far_d,68,77):.2f}/{seg(conf_far_s,68,77):.2f} | {seg(conf_far_d,78,90):.2f}/{seg(conf_far_s,78,90):.2f} || "
      f"{seg(conf_near,78,90):.2f}")
