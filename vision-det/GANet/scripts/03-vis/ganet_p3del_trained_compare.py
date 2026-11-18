"""P3del로 다시 학습한 res34 모델(006) vs 기존 res34 2cls(004) — 프레임별 그림 2종. (2026-11-10 사용자 요청으로 res34 고정, res18 줄 뺌)

{n}-3. 최종검출결과_P3del학습.png : 줄 = res34 기존 / res34 P3del,
                                    칸 = 실선 conf, 점선 conf, 최종 검출
{n}-4. FPN단계별_히트맵_P3del학습.png : res34 P3del 모델의 C3 · C4 · C5 · C5+SA · P4(DCN 전) · P4(최종)
                                        (1-1 그림과 같은 방식: 활성 크기 + 실선·점선 conf Grad-CAM)
체크포인트: 기존은 models/ 채택본(res34 ep130), P3del은 epoch_150.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare.py
"""
import os
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
from projects.plugin.core.post_process import group_points_fast

G = "/home/hsjeong/workspace/vision-det/GANet"
OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "07-cls2-P3제거특징맵분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
MODELS = [
    ("res34 기존 (P3, 1/8)", "res34_s8_full_2cls", f"{G}/models/drone-ganet-res34-2cls-v1.pth"),
    ("res34 P3del (P4, 1/16)", "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth"),
]
CLS_NAME = ["실선", "점선"]
CLS_COLOR = ["#00dc00", "#dc0000"]


def build(cfg_name, ckpt):
    cfg = mmcv.Config.fromfile(f"projects/cfgs/drone/{cfg_name}.py")
    cfg.model.train_cfg = None
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(m, ckpt, map_location="cpu")
    return m.cuda(0).eval(), cfg


models = [(tag, *build(c, k)) for tag, c, k in MODELS]
_, _, cfg0 = models[0]
W, H = cfg0.img_scale
mean = np.array(cfg0.img_norm_cfg["mean"], np.float32)
std = np.array(cfg0.img_norm_cfg["std"], np.float32)


def detect(model, t):
    pp = model.post_processing
    S = model.post_processing.hm_down_scale
    with torch.no_grad():
        x = model.extract_feat(t)
        hm = model.bbox_head.forward(x)["kpts_hm"][0].cpu().numpy()
        pred = model.bbox_head.forward_test(x)
    rows, lanes = [], []
    for c, (_, cand, seeds) in enumerate(pred):
        n = len(seeds)
        if n == 0:
            rows.append((0, len(cand), [], 0)); continue
        groups, _ = group_points_fast(seeds, cand, pp.cluster_thr, pp.cluster_by_center_thr)
        kept = [g for g in groups if len(g) > pp.points_thr]
        rows.append((n, len(cand), [len(g) for g in kept], n - sum(len(g) for g in groups)))
        for g in kept:
            pts = np.array([p[1] for p in g], float) * S + S / 2
            lanes.append((c, pts[np.argsort(pts[:, 1])]))
    return hm, rows, lanes


# ---- res34 P3del 히트맵용 hook ----
hm_model = models[1][1]
kpt_thr = hm_model.bbox_head.kpt_thr
feats = {}
def keep(name, t):
    t.retain_grad()
    feats[name] = t
def bb_hook(m, i, o):  # out_indices=(1,2,3) -> (C3, C4, C5)
    if feats.get("_on"):
        keep("C3", o[0]); keep("C4", o[1]); keep("C5", o[2])
hm_model.backbone.register_forward_hook(bb_hook)
def _hook(name, pre=False):
    def f(m, i, o=None):
        if feats.get("_on"):
            keep(name, i[0] if pre else o)  # 반환값 None -> 출력 안 바꿈
    return f
hm_model.neck.trans_head.register_forward_hook(_hook("C5+SA"))
hm_model.neck.def_convs[0].register_forward_pre_hook(_hook("P4_pre", pre=True))
hm_model.neck.fpn_convs[0].register_forward_hook(_hook("P4"))
STAGES = ["C3", "C4", "C5", "C5+SA", "P4_pre", "P4"]
TITLES = ["C3\n백본 1/8 · 128ch (neck에선 안 씀)", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+SA\nattention 후 · 64ch",
          "P4 (DCN 전)\noffset 헤드 입력 · 1/16", "P4 (최종)\nconf 헤드 입력 · 1/16"]


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)


os.makedirs(OUTDIR, exist_ok=True)
for name, path in JOBS:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # ---- 3. 최종 검출 ----
    res = [(tag, *detect(m, t)) for tag, m, _ in models]
    print(f"[프레임{name}]")
    for c in range(2):
        for tag, hm, rows, _ in res:
            r = rows[c]
            print(f"   {CLS_NAME[c]} {tag:24s}: 키포인트 {r[0]:3d}, 후보 {r[1]:2d}, 선 {len(r[2])}개 {r[2]}, "
                  f"버려진 점 {r[3]}, 0.4 넘는 칸 {int((hm[c] > kpt_thr).sum())}")
    n = len(res)
    fig, axes = plt.subplots(n, 3, figsize=(21, 4.2 * n), gridspec_kw=dict(hspace=0.1, wspace=0.03))
    bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
    for r, (tag, hm, _, lanes) in enumerate(res):
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(bg)
            ax.imshow(cv2.resize(hm[c], (W, H), interpolation=cv2.INTER_NEAREST), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{CLS_NAME[c]} conf", fontsize=15)
        axes[r, 0].set_ylabel(tag, fontsize=15)
        ax = axes[r, 2]
        ax.imshow(rgb)
        for c, pts in lanes:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
    fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))
    fig.suptitle(f"{name}-3. P3del 구조로 다시 학습한 모델의 최종 검출 결과 vs 기존", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes[:, 0:2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
    cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
    fig.savefig(f"{OUTDIR}/{name}-3. 최종검출결과_P3del학습.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # ---- 4. FPN 단계별 히트맵 (res34 P3del) ----
    feats.clear(); feats["_on"] = True
    with torch.enable_grad():
        hm_all = hm_model.bbox_head.forward(hm_model.extract_feat(t))["kpts_hm"][0]
        A_list = [feats[st] for st in STAGES]
        cams = []
        for c in range(2):
            sel = hm_all[c] > kpt_thr
            if int(sel.sum()) == 0:
                cams.append(None); continue
            grads = torch.autograd.grad(hm_all[c][sel].sum(), A_list, retain_graph=True)
            cams.append([torch.relu((g[0].mean(dim=(1, 2))[:, None, None] * a[0]).sum(0)).detach().cpu().numpy()
                         for a, g in zip(A_list, grads)])
    feats["_on"] = False
    bg2 = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.4 * 0.5
    fig, axes = plt.subplots(3, 6, figsize=(30, 9.3), gridspec_kw=dict(hspace=0.04, wspace=0.04))
    for j in range(len(STAGES)):
        act = A_list[j][0].detach().norm(dim=0).cpu().numpy()
        rows = [act] + [None if cams[c] is None else cams[c][j] for c in range(2)]
        for r, m in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(bg2)
            if m is not None:
                ax.imshow(cv2.resize(norm01(m).astype(np.float32), (W, H)), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            elif j == 3:
                ax.text(0.5, 0.5, f"{CLS_NAME[r - 1]} conf {kpt_thr} 넘는 칸 없음", transform=ax.transAxes, ha="center",
                        va="center", fontsize=16, color="white", weight="bold",
                        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(TITLES[j], fontsize=14)
    axes[0, 0].set_ylabel("활성 크기", fontsize=16)
    axes[1, 0].set_ylabel("기여도\n(실선 conf)", fontsize=16)
    axes[2, 0].set_ylabel("기여도\n(점선 conf)", fontsize=16)
    fig.suptitle(f"{name}-4. P3del 구조로 다시 학습한 res34 모델의 FPN 단계별 히트맵 "
                 "(1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)", fontsize=18)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
    fig.savefig(f"{OUTDIR}/{name}-4. FPN단계별_히트맵_P3del학습.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
