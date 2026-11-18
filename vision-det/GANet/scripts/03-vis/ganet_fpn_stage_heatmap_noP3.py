"""P3 단계를 제거한 GANet의 FPN 단계별 히트맵 (1-1 그림의 P3 제거 버전, 학습 없이 추론만 변경).

경로: C4 -> lateral, C5 -> attention(AC5) -> lateral -> 업샘플해 더함 = P4
      -> (기존 P3 자리 그대로) DCN(def_convs[0]) -> fpn_convs[0] -> head(conf, offset).  C3는 쓰지 않음.
      offset 헤드 입력(aux_feat) = DCN 전 P4. 출력 해상도 1/16 (기존 1/8).
GANet 소스는 건드리지 않고 neck.forward 만 이 스크립트 안에서 바꿔 끼운다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap_noP3.py
"""
import sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "07-cls2-P3제거특징맵분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
STAGES = ["C3", "C4", "C5", "C5+SA", "P4_pre", "P4"]
TITLES = ["C3\n백본 1/8 · 128ch (neck에선 안 씀)", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+SA\nattention 후 · 64ch",
          "P4 (DCN 전)\noffset 헤드 입력 · 1/16", "P4 (최종)\nconf 헤드 입력 · 1/16"]

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
neck = model.neck

feats = {}
def keep(name, t):
    t.retain_grad()
    feats[name] = t

import torch.nn.functional as F


def forward_noP3(inputs):  # DeformFPN.forward 에서 P3(1/8) 단계만 뺀 것
    C3, C4, C5 = list(inputs)[1:]
    keep("C3", C3); keep("C4", C4); keep("C5", C5)
    ac5 = neck.trans_head(C5)
    keep("C5+SA", ac5)
    p4 = neck.lateral_convs[1](C4)
    p4 = p4 + F.interpolate(neck.lateral_convs[2](ac5), size=p4.shape[2:], **neck.upsample_cfg)
    keep("P4_pre", p4)
    d_feat, points = neck.def_convs[0](p4)  # 기존 P3 자리의 DCN(1x7) 그대로
    out = neck.fpn_convs[0](d_feat)
    keep("P4", out)
    return dict(features=(out,), aux_feat=p4, deform_points=(points,))
neck.forward = forward_noP3


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)


def up(a):
    return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    feats.clear()
    with torch.enable_grad():
        hm_all = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"][0]
        A_list = [feats[st] for st in STAGES]
        cams, info = [], []
        for c in range(2):  # 0=실선, 1=점선
            sel = hm_all[c] > kpt_thr
            n_sel = int(sel.sum())
            info.append(n_sel)
            if n_sel == 0:
                cams.append(None)
                continue
            grads = torch.autograd.grad(hm_all[c][sel].sum(), A_list, retain_graph=True)
            cams.append([torch.relu((g[0].mean(dim=(1, 2))[:, None, None] * a[0]).sum(0)).detach().cpu().numpy()
                         for a, g in zip(A_list, grads)])
    print(f"[프레임{name}] conf>{kpt_thr} 칸: 실선 {info[0]}개, 점선 {info[1]}개")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.4 * 0.5
    fig, axes = plt.subplots(3, 6, figsize=(30, 9.3), gridspec_kw=dict(hspace=0.04, wspace=0.04))
    for j, st in enumerate(STAGES):
        act = A_list[j][0].detach().norm(dim=0).cpu().numpy()
        rows = [act] + [None if cams[c] is None else cams[c][j] for c in range(2)]
        for r, m in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(bg)
            if m is not None:
                ax.imshow(up(norm01(m)), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            elif j == 3:
                ax.text(0.5, 0.5, f"{'실선' if r == 1 else '점선'} conf {kpt_thr} 넘는 칸 없음", transform=ax.transAxes, ha="center", va="center",
                        fontsize=16, color="white", weight="bold", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(TITLES[j], fontsize=14)
    axes[0, 0].set_ylabel("활성 크기", fontsize=16)
    axes[1, 0].set_ylabel("기여도\n(실선 conf)", fontsize=16)
    axes[2, 0].set_ylabel("기여도\n(점선 conf)", fontsize=16)
    fig.suptitle(f"{name}-1. P3 제거 시 FPN 단계별 히트맵 (P4가 최종 출력, 1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)", fontsize=18)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
    import os; os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(f"{OUTDIR}/{name}-1. FPN단계별_히트맵_P3제거.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
