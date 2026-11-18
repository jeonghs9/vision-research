"""FPN 단계별 히트맵 (7단계 x 2줄): 윗줄 = 활성 크기(채널 L2-norm), 2·3줄 = 기여도(Grad-CAM, 실선·점선 클래스 conf 기준).

단계: C3, C4, C5 (backbone) / C5+SA (trans_head 출력) / P4 (top-down 합친 뒤, fpn_convs[1] 입력)
      / P3 DCN 전 (aux_feat = def_convs[0] 입력, offset 헤드 입력) / P3 최종 (fpn_convs[0] 출력 = features[0], conf 헤드 입력)
Grad-CAM 목표값: 점선 클래스 confidence 가 kpt_thr 를 넘는 칸들의 합. 단계마다 0~1 로 따로 정규화.
모델 코드는 건드리지 않고 forward hook 으로만 중간 특징을 잡는다.
cluster_by_center_thr(5/None)는 후처리라 이 그림과 무관.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap.py
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
          "06-cls2-특징맵BEV무학습분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
STAGES = ["C3", "C4", "C5", "C5+SA", "P4", "P3_pre", "P3"]
TITLES = ["C3\n백본 1/8 · 128ch", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+SA\nattention 후 · 64ch",
          "P4\n위층 합친 뒤 · 1/16", "P3 (DCN 전)\noffset 헤드 입력 · 1/8", "P3 (최종)\nconf 헤드 입력 · 1/8"]

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

def bb_hook(m, i, o):  # backbone 출력 (C2, C3, C4, C5)
    keep("C3", o[1]); keep("C4", o[2]); keep("C5", o[3])
model.backbone.register_forward_hook(bb_hook)
neck.trans_head.register_forward_hook(lambda m, i, o: keep("C5+SA", o))
neck.fpn_convs[1].register_forward_pre_hook(lambda m, i: keep("P4", i[0]))
neck.def_convs[0].register_forward_pre_hook(lambda m, i: keep("P3_pre", i[0]))
neck.fpn_convs[0].register_forward_hook(lambda m, i, o: keep("P3", o))


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
    fig, axes = plt.subplots(3, 7, figsize=(35, 9.3), gridspec_kw=dict(hspace=0.04, wspace=0.04))
    for j, st in enumerate(STAGES):
        act = A_list[j][0].detach().norm(dim=0).cpu().numpy()
        rows = [act] + [None if cams[c] is None else cams[c][j] for c in range(2)]
        for r, m in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(bg)
            if m is not None:
                ax.imshow(up(norm01(m)), cmap="jet", alpha=0.55, vmin=0, vmax=1)
            elif j == 3:
                ax.text(0.5, 0.5, f"실선 conf {kpt_thr} 넘는 칸 없음", transform=ax.transAxes, ha="center", va="center",
                        fontsize=16, color="white", weight="bold", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(TITLES[j], fontsize=14)
    axes[0, 0].set_ylabel("활성 크기", fontsize=16)
    axes[1, 0].set_ylabel("기여도\n(실선 conf)", fontsize=16)
    axes[2, 0].set_ylabel("기여도\n(점선 conf)", fontsize=16)
    fig.suptitle(f"{name}-1. FPN 단계별 히트맵 (1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)", fontsize=18)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
    import os; os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(f"{OUTDIR}/{name}-1. FPN단계별_히트맵.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
