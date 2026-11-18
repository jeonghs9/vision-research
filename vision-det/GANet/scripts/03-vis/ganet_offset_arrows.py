#!/usr/bin/env python3
"""키포인트별 offset 벡터(점 -> 그 점이 가리키는 시작점 root)를 화살표로 시각화 + 시작점 자리의 confidence 수치 표기.

  N-8 (기존 방식, cluster_by_center_thr=5):
    - 별표: 시작점 후보(dy<root_thr AND conf>kpt_thr), 원: 합쳐진 후보 중심에서 5칸 반경
    - 색: 배정된 선, 회색: 어느 원에도 못 들어가 버려진 점
    - 라벨: 선마다 '시작점(후보 중심) 자리 conf', 버려진 점 묶음마다 '그 점들이 가리키는 시작점 자리 conf'
  N-9 (None 방식, root끼리 cluster_thr 이내로 이어지면 묶음):
    - 색: 묶음, 큰 원 마커: 묶음이 가리키는 시작점(root 중앙값), 라벨: 그 자리 conf
conf 수치 = 그 선 클래스의 sigmoid confidence. '칸' = 시작점이 걸친 native 칸 값, '주변최대' = 주변 3x3칸 최댓값.
배정 결과는 post_process.py의 group_points_fast를 그대로 호출해서 얻는다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_offset_arrows.py
"""
import os, sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa
from projects.plugin.core.post_process import group_points_fast

OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "05-cls2-점선미탐원인분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
CLS = {0: "실선(class1)", 1: "점선(class2)"}
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#bfef45", "#9a6324", "#000075", "#808000", "#ffe119"]
DROP = "#7f7f7f"
S = 8

cfg = mmcv.Config.fromfile("projects/cfgs/drone/res34_s8_full_2cls.py")
cfg.model.train_cfg = None
cfg.model.backbone.init_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth", map_location="cpu")
model = model.cuda(0).eval()
W, H = cfg.img_scale
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
kpt_thr = model.bbox_head.kpt_thr
root_thr = model.bbox_head.root_thr
pp = model.post_processing
BY_CENTER = pp.cluster_by_center_thr
CLUSTER_THR = pp.cluster_thr


def P(v):  # native 좌표 -> 모델 입력 픽셀(칸 중심)
    return np.asarray(v, float) * S + 4


def conf_at(hm_c, x, y):
    h, w = hm_c.shape
    xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
    nb = hm_c[max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2]
    return hm_c[yi, xi], nb.max()


def conf_label(hm_c, x, y):
    v, m = conf_at(hm_c, x, y)
    ok = "통과" if m > kpt_thr else "미달"
    return f"시작점 자리 conf={v:.2f} (주변최대 {m:.2f}, {kpt_thr} {ok})"


PENDING = []


def label(ax, xy, text, col, dx=10, dy=14, fg="black"):
    PENDING.append((ax, xy, text, col, fg))


def flush_labels(ax, xlo, xhi):
    items = sorted([it for it in PENDING if it[0] is ax], key=lambda it: it[1][0])
    for r, (_, xy, text, col, fg) in enumerate(items):
        left = xy[0] < xlo + 0.4 * (xhi - xlo)
        _label(ax, xy, text, col, dx=(-40 if left else 10), dy=-(30 + 40 * (r % 3)), fg=fg)
    PENDING[:] = [it for it in PENDING if it[0] is not ax]


def _label(ax, xy, text, col, dx=10, dy=14, fg="black"):
    ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", fontsize=9, weight="bold", color=fg,
                bbox=dict(facecolor=col, alpha=0.75, edgecolor="none"), zorder=8,
                arrowprops=dict(arrowstyle="-", color=col, lw=1))


def draw_points(ax, idx, xs, ys, rx, ry, col):
    idx = idx[np.argsort(ys[idx])]
    ax.scatter(P(xs[idx]), P(ys[idx]), s=10, c=col, edgecolors="none", alpha=0.9, zorder=3)
    for k in idx[::3]:
        ax.annotate("", xy=(P(rx[k]), P(ry[k])), xytext=(P(xs[k]), P(ys[k])),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.75), zorder=2)
    ax.scatter(P(rx[idx]), P(ry[idx]), s=26, c=col, marker="x", linewidths=1.2, zorder=4)


def make(prefix, path, mode):
    frame = cv2.imread(path)
    rgb = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    with torch.no_grad():
        feats = model.extract_feat(t)
        hd = model.bbox_head.forward(feats)
        pred = model.bbox_head.forward_test(feats)
        hmt = hd["kpts_hm"]
        hmax = F.max_pool2d(hmt, (1, 3), stride=(1, 1), padding=(0, 1))
        hm_nms = (hmt * (hmax == hmt).float())[0].cpu().numpy()
        hm = hmt[0].cpu().numpy()
        off = hd["pts_offset"][0].cpu().numpy()

    allx, ally, per_class = [], [], []
    for c in range(2):
        ys, xs = np.where(hm_nms[c] > kpt_thr)
        rx, ry = xs + off[0, ys, xs], ys + off[1, ys, xs]
        per_class.append((xs, ys, rx, ry))
        allx += list(xs) + list(rx); ally += list(ys) + list(ry)
    pad = 6
    x0 = max(min(allx) - pad, 0) * S; x1 = min(max(allx) + pad, W / S) * S
    y0 = max(min(ally) - pad, 0) * S; y1 = min(max(ally) + pad, H / S) * S

    fig, axes = plt.subplots(1, 2, figsize=(22, 10.5))
    if mode == "center":
        fig.suptitle(f"{prefix}-8. offset 방향 + 시작점 자리 confidence (기존 방식, cluster_by_center_thr={BY_CENTER}) -- "
                     f"별표=시작점 후보, 원=후보 중심 {BY_CENTER}칸 반경, 회색=버려진 점", fontsize=12)
    else:
        fig.suptitle(f"{prefix}-9. offset 방향 + 시작점 자리 confidence (None 방식, root끼리 {CLUSTER_THR}칸 이내로 "
                     f"이어지면 묶음) -- 큰 원 마커=묶음이 가리키는 시작점(root 중앙값)", fontsize=12)
    for c in range(2):
        ax = axes[c]
        ax.imshow(rgb)
        xs, ys, rx, ry = per_class[c]
        cand = pred[c][1]
        n_lane = n_drop = 0
        if len(xs):
            seeds = np.stack([np.stack([xs, ys], 1), np.stack([rx, ry], 1)], 1).astype(np.float32)
            if mode == "center":
                groups, centers = group_points_fast(seeds, cand, CLUSTER_THR, BY_CENTER)
                gid = np.full(len(xs), -1)
                for gi, g in enumerate(groups):
                    for p in g:
                        gid[p[0]] = gi
                # 버려진 점: root끼리 다시 묶어서 묶음별로 '가리키는 시작점 자리' conf 표기
                d = np.where(gid < 0)[0]
                n_drop = len(d)
                if n_drop:
                    draw_points(ax, d, xs, ys, rx, ry, DROP)
                    dseeds = seeds[d]
                    dgroups, _ = group_points_fast(dseeds, cand, CLUSTER_THR, None)
                    for dg in dgroups:
                        if len(dg) < 3:
                            continue
                        di = d[[p[0] for p in dg]]
                        mx, my = np.median(rx[di]), np.median(ry[di])
                        ax.scatter(P(mx), P(my), s=140, facecolors="none", edgecolors="black", linewidths=2, zorder=7)
                        label(ax, (P(mx), P(my)), f"버려진 {len(di)}점이 가리키는\n{conf_label(hm[c], mx, my)}",
                              DROP, dx=-150, dy=-48, fg="white")
                for gi, g in enumerate(groups):
                    idx = np.where(gid == gi)[0]
                    col = PALETTE[gi % len(PALETTE)]
                    cx, cy = centers[gi]
                    ax.add_patch(Circle((P(cx), P(cy)), BY_CENTER * S, fill=False, ec=col, lw=2, ls="--", zorder=5))
                    if len(idx) > pp.points_thr:
                        n_lane += 1
                    if len(idx):
                        draw_points(ax, idx, xs, ys, rx, ry, col)
                    tail = "" if len(idx) > pp.points_thr else f" (≤{pp.points_thr}점, 탈락)"
                    label(ax, (P(cx), P(cy)), f"선{gi}: {len(idx)}점{tail}\n{conf_label(hm[c], cx, cy)}", col,
                          dx=10, dy=24 + 30 * (gi % 2))
                if len(cand):
                    ax.scatter(P(cand[:, 0]), P(cand[:, 1]), marker="*", s=380, c="white", edgecolors="black",
                               linewidths=1.5, zorder=6)
                title = f"{CLS[c]} -- 최종 선 {n_lane}개, 버려진 점 {n_drop}개"
            else:
                groups, _ = group_points_fast(seeds, cand, CLUSTER_THR, None)
                for gi, g in enumerate(groups):
                    idx = np.array([p[0] for p in g])
                    if len(idx) <= 1:
                        draw_points(ax, idx, xs, ys, rx, ry, DROP)
                        continue
                    col = PALETTE[gi % len(PALETTE)]
                    draw_points(ax, idx, xs, ys, rx, ry, col)
                    if len(idx) > pp.points_thr:
                        n_lane += 1
                    mx, my = np.median(rx[idx]), np.median(ry[idx])
                    ax.scatter(P(mx), P(my), s=160, c=col, edgecolors="black", linewidths=1.5, zorder=7)
                    tail = "" if len(idx) > pp.points_thr else f" (≤{pp.points_thr}점, 탈락)"
                    label(ax, (P(mx), P(my)), f"묶음{gi}: {len(idx)}점{tail}\n{conf_label(hm[c], mx, my)}", col,
                          dx=10, dy=24 + 30 * (gi % 2))
                title = f"{CLS[c]} -- 최종 선 {n_lane}개 (RANSAC 전 기준)"
        else:
            title = f"{CLS[c]} -- 키포인트 0개"
        ax.scatter([], [], s=10, c="k", label=f"키포인트 (confidence>{kpt_thr}) {len(xs)}개")
        ax.scatter([], [], s=26, c="k", marker="x", label="root = 그 점이 가리키는 시작점")
        if mode == "center":
            ax.scatter([], [], marker="*", s=200, c="white", edgecolors="black",
                       label=f"시작점 후보 (dy<{root_thr} AND conf>{kpt_thr}) {len(cand)}개")
            ax.scatter([], [], s=26, c=DROP, marker="x", label=f"버려진 점 {n_drop}개")
        else:
            ax.scatter([], [], s=120, c="w", edgecolors="black", label="묶음이 가리키는 시작점 (root 중앙값)")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        flush_labels(ax, x0, x1)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    fig.text(0.5, 0.01, f"입력: {os.path.basename(path)} | 모델 입력 {W}x{H} 좌표 | 화살표 3개 중 1개만 표시 | "
             f"conf = 그 선 클래스의 confidence, '시작점 자리 conf'는 그 칸 값, 괄호는 주변 3x3칸 최댓값 | "
             f"점 수는 RANSAC 적용 전", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    num = "8" if mode == "center" else "9"
    name = "오프셋방향_시각화" if mode == "center" else "오프셋방향_None방식"
    out = os.path.join(OUTDIR, f"{prefix}-{num}. {name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


for prefix, path in JOBS:
    make(prefix, path, "center")
    make(prefix, path, "none")
