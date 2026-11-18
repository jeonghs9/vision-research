"""FPN 단계별 히트맵 — SA(C5 attention) 적용/미적용 비교. 원본 입력(12번 그림)·BEV 입력(13번 그림).
블록 4개: 기존 / 기존−SA(test/007 ep140) / P3del(test/006 ep150) / P3del−SA(test/008 ep110), 모두 res34. SA 없는 모델은 C5+SA 칸이 빔.
10·11번 그림(ganet_fpn_heatmap_4way.py)과 같은 방식. 블록마다 3줄(활성 크기, 실선 conf 기여도, 점선 conf 기여도), 1-1·4번 그림과 같은 방식.
칸은 두 구조를 맞춰 7칸: C3 · C4 · C5 · C5+SA · P4 · P3(DCN 전) · 최종.  P3del 은 P3 칸이 비고, P4 칸 = DCN 전 P4, 최종 = DCN 뒤 P4.
neck.forward 는 DeformFPN.forward 를 그대로 옮겨 중간값만 꺼내도록 바꿔 끼운 것(원래 forward 와 conf 차이를 실행 때 assert).
BEV 사다리꼴은 5·6·7번 그림과 같음.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_sa.py
"""
import os
import sys
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

G = "/home/hsjeong/workspace/vision-det/GANet"
OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "07-cls2-P3제거특징맵분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
MODELS = [("기존", "res34_s8_full_2cls", f"{G}/models/drone-ganet-res34-2cls-v1.pth"),
          ("기존−SA", "res34_s8_full_2cls_noSA", f"{G}/test/007-res34-2cls-1280-noSA-150ep/epoch_140.pth"),
          ("P3del", "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth"),
          ("P3del−SA", "res34_s16_full_2cls_P3del_noSA", f"{G}/test/008-res34-2cls-1280-P3del-noSA-150ep/epoch_110.pth")]
COLS = ["C3", "C4", "C5", "C5+SA", "P4", "P3_pre", "OUT"]
TITLES = ["C3\n백본 1/8 · 128ch", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+SA\nattention 후 · 64ch (SA 모델만)",
          "P4 (1/16)\n기존: 위층 합친 뒤 / P3del: DCN 전", "P3 (DCN 전, 1/8)\n기존만", "최종 (conf 헤드 입력)\n기존: P3 1/8 / P3del: P4 1/16"]
CLS_NAME = ["실선", "점선"]


def build(cfg_name, ckpt):
    cfg = mmcv.Config.fromfile(f"projects/cfgs/drone/{cfg_name}.py")
    cfg.model.train_cfg = None
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(m, ckpt, map_location="cpu")
    return m.cuda(0).eval(), cfg


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-8)


def attach(model):
    neck = model.neck
    feats = {}
    orig_forward = neck.forward

    def keep(name, t):
        t.retain_grad()
        feats[name] = t

    def forward_keep(inputs):  # DeformFPN.forward 와 같은 계산 (deconv_before=False, use_res=False, trans_mode=replace)
        inputs = list(inputs)
        if len(inputs) > len(neck.in_channels):
            del inputs[0]
        if len(inputs) == 2:
            keep("C3", feats["_bb"][0])
        for nm, x in zip(["C3", "C4", "C5"][-len(inputs):], inputs):
            keep(nm, x)
        if neck.trans_cfg is not None:
            inputs[neck.trans_idx] = neck.trans_head(inputs[neck.trans_idx])
            keep("C5+SA", inputs[neck.trans_idx])
        lats = [lc(inputs[i + neck.start_level]) for i, lc in enumerate(neck.lateral_convs)]
        for i in range(len(lats) - 1, 0, -1):
            lats[i - 1] = lats[i - 1] + F.interpolate(lats[i], size=lats[i - 1].shape[2:], **neck.upsample_cfg)
        if len(lats) == 3:
            keep("P4", lats[1]); keep("P3_pre", lats[0])
        else:
            keep("P4", lats[0])
        d_feat, points = neck.def_convs[0](lats[0])
        out = neck.fpn_convs[0](d_feat)
        keep("OUT", out)
        return dict(features=(out,), aux_feat=lats[0], deform_points=(points,))

    model.backbone.register_forward_hook(lambda m, i, o: feats.__setitem__("_bb", o) or None)
    return feats, orig_forward, forward_keep


models = []
for tag, c, k in MODELS:
    m, cfg = build(c, k)
    models.append((tag, m, *attach(m)))
W, H = cfg.img_scale
mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
std = np.array(cfg.img_norm_cfg["std"], np.float32)
M_bev = cv2.getPerspectiveTransform(np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]]),
                                    np.float32([[0, 0], [W, 0], [W, H], [0, H]]))

os.makedirs(OUTDIR, exist_ok=True)
for name, path in JOBS:
    orig = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    blocks = []
    for itag, rgb in [("원본", orig), ("BEV", cv2.warpPerspective(orig, M_bev, (W, H)))]:
        t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
        for tag, model, feats, orig_forward, forward_keep in models:
            kpt_thr = model.bbox_head.kpt_thr
            with torch.no_grad():
                model.neck.forward = orig_forward
                hm_ref = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"][0]
            model.neck.forward = forward_keep
            feats.clear()
            with torch.enable_grad():
                hm = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"][0]
                assert float((hm - hm_ref).abs().max()) < 1e-5, "forward_keep 가 원래 forward 와 다름"
                cols = [c for c in COLS if c in feats]
                A = {c: feats[c] for c in cols}
                cams, n_sel = [], []
                for k in range(2):
                    sel = hm[k] > kpt_thr
                    n_sel.append(int(sel.sum()))
                    if n_sel[-1] == 0:
                        cams.append(None); continue
                    gr = torch.autograd.grad(hm[k][sel].sum(), [A[c] for c in cols], retain_graph=True)
                    cams.append({c: torch.relu((g[0].mean(dim=(1, 2))[:, None, None] * A[c][0]).sum(0)).detach().cpu().numpy()
                                 for c, g in zip(cols, gr)})
            act = {c: A[c][0].detach().norm(dim=0).cpu().numpy() for c in cols}
            print(f"[프레임{name}] {itag}·{tag}: conf>{kpt_thr} 칸 실선 {n_sel[0]}, 점선 {n_sel[1]}")
            blocks.append((f"{itag}·{tag}", rgb, act, cams))

    for itag, fno in [("원본", "12"), ("BEV", "13")]:
        sub = [bk for bk in blocks if bk[0].startswith(itag)]
        nr, nc = 3 * len(sub), len(COLS)
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.05 * nr), gridspec_kw=dict(hspace=0.04, wspace=0.04))
        for b, (btag, rgb, act, cams) in enumerate(sub):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
            for j, c in enumerate(COLS):
                for r in range(3):
                    ax = axes[3 * b + r, j]
                    ax.set_xticks([]); ax.set_yticks([])
                    if c not in act:  # P3del 의 P3 칸, SA 없는 모델의 C5+SA 칸
                        ax.axis("off")
                        if r == 1:
                            ax.text(0.5, 0.5, "SA 없음" if c == "C5+SA" else "P3 없음", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="gray")
                        continue
                    ax.imshow(bg)
                    m = act[c] if r == 0 else (None if cams[r - 1] is None else cams[r - 1][c])
                    if m is not None:
                        ax.imshow(cv2.resize(norm01(m).astype(np.float32), (W, H)), cmap="jet", alpha=0.55, vmin=0, vmax=1)
                    elif j == 1:
                        ax.text(0.5, 0.5, f"{CLS_NAME[r - 1]} conf 0.4 넘는 칸 없음", transform=ax.transAxes, ha="center",
                                va="center", fontsize=14, color="white", weight="bold",
                                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
                    if b == 0 and r == 0:
                        ax.set_title(TITLES[j], fontsize=14)
            for r, lab in enumerate(["활성 크기", "실선 conf\n기여도", "점선 conf\n기여도"]):
                axes[3 * b + r, 0].set_ylabel(f"{btag}\n{lab}", fontsize=15, rotation=0, ha="right", va="center", labelpad=12)
        fig.subplots_adjust(top=1 - 1.6 / (3.05 * nr))
        fig.suptitle(f"{name}-{fno}. {itag} 입력 FPN 단계별 히트맵 — SA 적용 vs 미적용 (res34 기존·P3del, 블록마다 1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)",
                     fontsize=20)
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axes, fraction=0.01, pad=0.01, ticks=[0, 1])
        cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
        fig.savefig(f"{OUTDIR}/{name}-{fno}. FPN단계별_히트맵_{itag}입력_SA적용vs미적용.png", dpi=70, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
