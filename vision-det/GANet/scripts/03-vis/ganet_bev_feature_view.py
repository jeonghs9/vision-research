"""0단계: 백본 특징맵을 BEV(조감도)로 펴서 원본 시점과 비교 (활성 크기 = 채널 L2-norm).

1줄(원본 시점): 입력(+BEV 기준 사다리꼴), C3, C4, C5, AC5(=SA(C5))
2줄(BEV):       BEV 입력, BC3, BC4, BC5, B(AC5)(=AC5를 편 것), BAC5(=SA(BC5), 지금 가중치 그대로)
BEV 변환: 프레임마다 5방식으로 검출된 선 중 맨 왼쪽/맨 오른쪽 선을 직선 맞춤 -> 두 선이 겹치는 y 구간의
          사다리꼴 4점을 BEV 에서 세로 평행선(x=0.3W, 0.7W, y=0~H)으로 보냄. 출력 크기 = 모델 입력 크기
          (BC5 가 23x40 으로 SA 위치 임베딩 크기와 같아야 SA 를 그대로 적용 가능).
특징맵은 각 해상도(stride 8/16/32)에 맞춘 호모그래피로 grid_sample 해서 편다. 원본 밖 영역은 회색.
모델 코드는 건드리지 않는다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_bev_feature_view.py
"""
import os, sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa
from projects.plugin.core.post_process import group_points_fast

OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "06-cls2-특징맵BEV무학습분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
S = 8  # heatmap stride

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
neck = model.neck

feats = {}
model.backbone.register_forward_hook(lambda m, i, o: feats.update(C3=o[1], C4=o[2], C5=o[3]))
# 아래에서 SA(BC5)를 따로 부를 때 원래 AC5 가 덮어써지지 않게 처음 값만 보관 (프레임마다 feats.clear()).
# forward hook 이 값을 반환하면 모듈 출력이 그 값으로 바뀌므로 반드시 None 을 반환해야 한다.
def _keep_ac5(m, i, o):
    if "AC5" not in feats:
        feats["AC5"] = o
neck.trans_head.register_forward_hook(_keep_ac5)


def lanes_px(pred):
    """5방식으로 묶인 선들(점 > points_thr)의 점을 입력 픽셀 좌표로."""
    out = []
    for _, cand, seeds in pred:
        if len(seeds) == 0:
            continue
        groups, _ = group_points_fast(seeds, cand, pp.cluster_thr, pp.cluster_by_center_thr)
        for g in groups:
            if len(g) > pp.points_thr:
                out.append(np.array([p[1] for p in g], float) * S + S / 2)
    return out


def bev_homography(lanes):
    fits = []
    for pts in lanes:
        a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)  # x = a*y + b
        fits.append((a, b, pts[:, 1].min(), pts[:, 1].max()))
    xb = [a * H * 0.9 + b for a, b, _, _ in fits]
    L, R = fits[int(np.argmin(xb))], fits[int(np.argmax(xb))]
    width = lambda y: (R[0] * y + R[1]) - (L[0] * y + L[1])
    y_bot = H - 1
    # 위쪽은 두 직선을 연장: 더 높이 올라간 선의 끝까지, 단 간격이 아래쪽의 25% 아래로 좁아지기 전까지(소실점 근처 제외)
    y_top = min(L[2], R[2])
    while width(y_top) < 0.25 * width(y_bot) and y_top < y_bot - 40:
        y_top += 4
    src = np.float32([[L[0] * y_top + L[1], y_top], [R[0] * y_top + R[1], y_top],
                      [R[0] * y_bot + R[1], y_bot], [L[0] * y_bot + L[1], y_bot]])
    dst = np.float32([[0.3 * W, 0], [0.7 * W, 0], [0.7 * W, H], [0.3 * W, H]])
    return cv2.getPerspectiveTransform(src, dst), src


def warp_feat(f, Hm, s):
    """특징맵 f (1,C,h,w) 를 입력 픽셀 호모그래피 Hm 에 맞춰 같은 크기 BEV 로 편다. 칸 중심 = s*i + s/2."""
    _, _, h, w = f.shape
    Sm = np.array([[s, 0, s / 2], [0, s, s / 2], [0, 0, 1]], float)
    Hinv = np.linalg.inv(Sm) @ np.linalg.inv(Hm) @ Sm  # BEV 칸 -> 원본 칸
    yy, xx = np.mgrid[0:h, 0:w]
    p = Hinv @ np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    sx, sy = p[0] / p[2], p[1] / p[2]
    grid = np.stack([sx / (w - 1) * 2 - 1, sy / (h - 1) * 2 - 1], -1).reshape(1, h, w, 2)
    grid_t = torch.from_numpy(grid).float().to(f.device)
    valid = ((np.abs(grid[0, ..., 0]) <= 1) & (np.abs(grid[0, ..., 1]) <= 1))
    return F.grid_sample(f, grid_t, mode="bilinear", padding_mode="zeros", align_corners=True), valid


def l2map(f, valid=None):
    m = f[0].norm(dim=0).detach().cpu().numpy()
    if valid is not None:
        m = np.where(valid, m, np.nan)
    lo, hi = np.nanmin(m), np.nanmax(m)
    return (m - lo) / (hi - lo + 1e-8)


def show(ax, bg, m, title):
    ax.imshow(bg)
    if m is not None:
        mm = cv2.resize(np.nan_to_num(m, nan=-1).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mm = np.ma.masked_less(mm, 0)
        ax.imshow(mm, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])


os.makedirs(OUTDIR, exist_ok=True)
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    feats.clear()
    with torch.no_grad():
        pred = model.bbox_head.forward_test(model.extract_feat(t))
        lanes = lanes_px(pred)
        Hm, src = bev_homography(lanes)
        bev = {}
        for k, s in [("C3", 8), ("C4", 16), ("C5", 32), ("AC5", 32)]:
            bev[k] = warp_feat(feats[k], Hm, s)
        bac5 = neck.trans_head(bev["C5"][0])  # SA(BC5), 지금 가중치 그대로
    print(f"[프레임{name}] 선 {len(lanes)}개, 사다리꼴 {np.round(src).astype(int).tolist()}, "
          f"BC5 {tuple(bev['C5'][0].shape[2:])}, 유효칸 {bev['C5'][1].mean()*100:.0f}%")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    bev_rgb = cv2.warpPerspective(rgb, Hm, (W, H), borderValue=(128, 128, 128))
    bev_gray = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2GRAY)
    bev_bg = np.stack([bev_gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2

    fig, axes = plt.subplots(2, 6, figsize=(30, 6.9), gridspec_kw=dict(hspace=0.12, wspace=0.04))
    ax = axes[0, 0]
    show(ax, rgb / 255.0, None, "입력 (원본 시점)")
    for pts in lanes:
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color="#ffe119")
    ax.add_patch(plt.Polygon(src, fill=False, ec="#00e5ff", lw=2.5))
    ax.plot(src[:, 0], src[:, 1], "o", color="#00e5ff", ms=8)
    show(axes[0, 1], bg, l2map(feats["C3"]), "C3 · 1/8")
    show(axes[0, 2], bg, l2map(feats["C4"]), "C4 · 1/16")
    show(axes[0, 3], bg, l2map(feats["C5"]), "C5 · 1/32")
    show(axes[0, 4], bg, l2map(feats["AC5"]), "AC5 = SA(C5)")
    axes[0, 5].axis("off")

    ax = axes[1, 0]
    show(ax, bev_rgb / 255.0, None, "BEV 입력")
    ax.add_patch(plt.Rectangle((0.3 * W, 0), 0.4 * W, H, fill=False, ec="#00e5ff", lw=2.5))
    show(axes[1, 1], bev_bg, l2map(*bev["C3"]), "BC3")
    show(axes[1, 2], bev_bg, l2map(*bev["C4"]), "BC4")
    show(axes[1, 3], bev_bg, l2map(*bev["C5"]), "BC5")
    show(axes[1, 4], bev_bg, l2map(*bev["AC5"]), "B(AC5) = AC5를 편 것")
    show(axes[1, 5], bev_bg, l2map(bac5, bev["C5"][1]), "BAC5 = SA(BC5)\n(지금 가중치)")
    axes[0, 0].set_ylabel("원본 시점", fontsize=16)
    axes[1, 0].set_ylabel("BEV", fontsize=16)
    fig.suptitle(f"{name}-2. 특징맵 BEV 비교 (활성 크기, 하늘색 = BEV 기준 사다리꼴 ↔ 직사각형)", fontsize=18)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
    fig.savefig(f"{OUTDIR}/{name}-2. BEV_특징맵비교.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
