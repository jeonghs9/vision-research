"""FPN 단계별 히트맵 — APT(Adaptive Patch) 토큰 병합 3종: P3del+SA+APT / P3del+DA+APT / P3del−SA+APT. 원본 입력(20번)·BEV 입력(21번).
블록 3개: test/011 ep110 / test/012 ep110 / test/013 ep150 (각 실험 최고 F1 epoch), 모두 res34.
ganet_fpn_heatmap_da.py(16·17번 그림)에서 모델만 바꾸고, neck 대신 쓰는 forward 에 APT 병합(apt_cfg)을 넣은 것.
C5+attention 칸: SA·DA+APT 는 attention 뒤(64ch), −SA+APT 는 병합 뒤 C5(512ch).
원래 forward 와 conf 차이를 실행 때 assert.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_apt.py
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
from projects.plugin.models.dcn_fpn import apt_merge_mask

G = "/home/hsjeong/workspace/vision-det/GANet"
OUTDIR = ("/home/hsjeong/workspace/vision-det/GANet/output/"
          "07-cls2-P3제거특징맵분석/figures")
JOBS = [
    ("1", "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
          "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"),
    ("2", "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg"),
    ("3", "/home/hsjeong/workspace/vision-det/GANet/test/tmp/f33/19700101_000816_T_f990(33s).png"),
]
MODELS = [("P3del+SA+APT", "res34_s16_full_2cls_P3del_APT", f"{G}/test/011-res34-2cls-1280-P3del-APT-150ep/epoch_110.pth"),
          ("P3del+DA+APT\n(DA는 query)", "res34_s16_full_2cls_P3del_DAq_APT", f"{G}/test/012-res34-2cls-1280-P3del-DAq-APT-150ep/epoch_110.pth"),
          ("P3del−SA+APT\n(SA 제거, 병합만)", "res34_s16_full_2cls_P3del_noSA_APT", f"{G}/test/013-res34-2cls-1280-P3del-noSA-APT-150ep/epoch_150.pth")]
COLS = ["C3", "C4", "C5", "C5+SA", "P4", "OUT"]
TITLES = ["C3\n백본 1/8 · 128ch (neck에선 안 씀)", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+attention (APT)\nSA·DA 후 64ch / −SA는 병합 후 512ch",
          "P4 (DCN 전, 1/16)\noffset 헤드 입력", "최종 (DCN 뒤 P4, 1/16)\nconf 헤드 입력"]
CLS_NAME = ["실선", "점선"]


def build(cfg_name, ckpt):
    # noSA/DA config 는 기준 config 모듈의 model dict 를 직접 고쳐 쓰는데, 그 모듈이 sys.modules 에 캐시되어
    # 다음 config 에 고친 값이 넘어간다(noSA 다음 DA 를 읽으면 trans_cfg=None). 읽을 때마다 캐시를 비움.
    for k in [k for k in sys.modules if k.startswith("projects.cfgs")]:
        del sys.modules[k]
    cfg = mmcv.Config.fromfile(f"projects/cfgs/drone/{cfg_name}.py")
    cfg.model.train_cfg = None
    cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    r = m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=True)  # 구조가 config 와 다르면 여기서 멈춤
    print(cfg_name, r)
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

    def forward_keep(inputs, img=None):  # DeformFPN.forward 와 같은 계산 (deconv_before=False, use_res=False, trans_mode=replace)
        inputs = list(inputs)
        if len(inputs) > len(neck.in_channels):
            del inputs[0]
        if len(inputs) == 2:
            keep("C3", feats["_bb"][0])
        for nm, x in zip(["C3", "C4", "C5"][-len(inputs):], inputs):
            keep(nm, x)
        ac = neck.apt_cfg
        merged = apt_merge_mask(img, ac.get("base_patch", 32), ac["threshold"], ac["mean"], ac["std"], ac.get("bins", 512))
        feats["_merged"] = merged
        if neck.apt_merge is not None:  # −SA+APT: attention 없이 C5 병합만
            inputs[neck.trans_idx] = neck.apt_merge(inputs[neck.trans_idx], merged)
            keep("C5+SA", inputs[neck.trans_idx])
        if neck.trans_cfg is not None:
            inputs[neck.trans_idx] = neck.trans_head(inputs[neck.trans_idx], merged)
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
            mc, mf = feats["_merged"]
            print(f"[프레임{name}] {itag}·{tag}: conf>{kpt_thr} 칸 실선 {n_sel[0]}, 점선 {n_sel[1]}, 병합 2x2 {int(mc.sum())}개 (C5 토큰 {int((~mf).sum() + mc.sum())}/920)")
            blocks.append((f"{itag}·{tag}", rgb, act, cams))

    for itag, fno in [("원본", "20"), ("BEV", "21")]:
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
                            ax.text(0.5, 0.5, "attention 없음", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="gray")
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
        fig.suptitle(f"{name}-{fno}. {itag} 입력 FPN 단계별 히트맵 — P3del+SA+APT / P3del+DA+APT / P3del−SA+APT (res34, APT τ=6.5, 블록마다 1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)",
                     fontsize=20)
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axes, fraction=0.01, pad=0.01, ticks=[0, 1])
        cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
        fig.savefig(f"{OUTDIR}/{name}-{fno}. FPN단계별_히트맵_{itag}입력_APT3종.png", dpi=70, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
