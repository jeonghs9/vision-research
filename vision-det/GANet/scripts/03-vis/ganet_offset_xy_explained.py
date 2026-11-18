#!/usr/bin/env python3
"""1-4-2 / 2-4-2 / 3-4-2: 오프셋 x/y 히트맵 + 차선별 예시 화살표(점 -> 시작점) + 말로 쓴 컬러바.

실행 (GANet/source 에서):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_offset_xy_explained.py
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

CFG = "projects/cfgs/drone/res34_s8_full_2cls.py"
CKPT = "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth"
OUT = "/home/hsjeong/workspace/vision-det/GANet/output/05-cls2-점선미탐원인분석/figures"
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
S = 8  # stride


def build():
    cfg = mmcv.Config.fromfile(CFG)
    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu")
    return model.cuda().eval(), cfg


def pick_examples(mask, off):
    """마스크 연결영역(=대략 차선 하나)마다 대표 점 1개: 위쪽 20% 지점(시작점에서 먼 쪽)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ex = []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        if len(ys) < 12:
            continue
        ty = int(np.percentile(ys, 20))
        row = xs[ys == ty]
        if len(row) == 0:
            continue
        tx = int(np.median(row))
        if not mask[ty, tx]:
            tx = int(row[np.argmin(np.abs(row - tx))])
        dx, dy = float(off[0, ty, tx]), float(off[1, ty, tx])
        ex.append((tx, ty, dx, dy))
    ex.sort(key=lambda e: e[0])
    return ex


def choose3(ex):
    """설명용으로 3개만: 가장 왼쪽을 가리키는 점, 거의 바로 아래, 가장 오른쪽."""
    if not ex:
        return []
    picks = [min(ex, key=lambda e: e[2]), min(ex, key=lambda e: abs(e[2])), max(ex, key=lambda e: e[2])]
    out = []
    for e in picks:
        if e not in out:
            out.append(e)
    return out


def panel(ax, fig, bg, val_up, vmax, ex, which, box):
    ax.imshow(bg)
    im = ax.imshow(val_up, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.9)
    x0, x1, y0, y1 = box
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    for tx, ty, dx, dy in ex:
        px, py = tx * S + S / 2, ty * S + S / 2
        rx, ry = px + dx * S, py + dy * S
        ax.annotate("", xy=(rx, ry), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6, mutation_scale=14))
        ax.plot(px, py, "o", ms=7, mfc="yellow", mec="black", mew=1.2)
        ax.plot(rx, ry, "*", ms=13, mfc="lime", mec="black", mew=1.0)
        if which == "x":
            txt = "바로 아래" if abs(dx) < 1.5 else f"{'왼쪽' if dx < 0 else '오른쪽'} {abs(dx):.0f}칸"
        else:
            txt = f"아래 {dy:.0f}칸"
        ax.text(px, py - 14, txt, ha="center", va="bottom", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    v = int(round(vmax))
    cb.set_ticks([-vmax, 0, vmax])
    if which == "x":
        cb.set_ticklabels([f"왼쪽 {v}칸", "0 (바로 아래)", f"오른쪽 {v}칸"])
    else:
        cb.set_ticklabels([f"위 {v}칸", "0 (시작점)", f"아래 {v}칸"])
    cb.ax.tick_params(labelsize=11)
    cb.outline.set_visible(False)


def make(model, cfg, prefix, path):
    W, H = tuple(cfg.img_scale)
    mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
    std = np.array(cfg.img_norm_cfg["std"], np.float32)
    thr = cfg.model.head.kpt_thr
    frame = cv2.imread(path)
    assert frame is not None, path
    rs = cv2.resize(frame, (W, H))
    x = torch.from_numpy(((cv2.cvtColor(rs, cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std)
                         .transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
        hd = model.bbox_head.forward(model.extract_feat(x))
    conf = hd["kpts_hm"][0].cpu().numpy().max(0)
    off = hd["pts_offset"][0].cpu().numpy()
    mask = conf > thr
    ex = choose3(pick_examples(mask, off))

    ox, oy = off[0].copy(), off[1].copy()
    ox[~mask] = np.nan; oy[~mask] = np.nan
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    vmx = float(np.nanmax(np.abs(ox))); vmy = float(np.nanmax(np.abs(oy)))

    ys, xs = np.where(mask)
    xs_all = list(xs * S) + [(e[0] + e[2]) * S for e in ex]
    ys_all = list(ys * S) + [(e[1] + e[3]) * S for e in ex]
    pad = 30
    box = (max(min(xs_all) - pad, 0), min(max(xs_all) + S + pad, W),
           max(min(ys_all) - pad - 20, 0), min(max(ys_all) + S + pad, H))
    rgb = cv2.cvtColor(rs, cv2.COLOR_BGR2RGB)
    # 배경은 흐린 흑백 -> 오프셋 색만 눈에 띄게
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.repeat((gray * 0.5 + 255 * 0.5).astype(np.uint8)[..., None], 3, axis=2)

    bw, bh = box[1] - box[0], box[3] - box[2]
    fig_w = 22
    fig_h = max(5.5, min(12, fig_w / 3 * bh / bw + 1.6))
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
    fig.suptitle(f"{prefix}-4-2. 오프셋맵: 각 점에서 \"내 차선 시작점\"까지 몇 칸?", fontsize=17)
    axes[0].imshow(rgb)
    axes[0].set_xlim(box[0], box[1]); axes[0].set_ylim(box[3], box[2]); axes[0].axis("off")
    axes[0].set_title("원본", fontsize=14)
    panel(axes[1], fig, bg, up(ox), vmx, ex, "x", box)
    axes[1].set_title("x: 시작점이 옆으로 몇 칸?\n파랑 = 왼쪽   흰색 = 바로 아래   빨강 = 오른쪽", fontsize=14)
    panel(axes[2], fig, bg, up(oy), vmy, ex, "y", box)
    axes[2].set_title("y: 시작점이 아래로 몇 칸?\n진한 빨강 = 멀다   흰색 = 여기가 시작점", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{OUT}/{prefix}-4-2. 오프셋맵_XY_쉽게보기.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out, [(round(e[2], 1), round(e[3], 1)) for e in ex])


if __name__ == "__main__":
    m, c = build()
    for p, img in JOBS:
        make(m, c, p, img)
