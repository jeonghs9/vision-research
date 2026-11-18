"""실선 점수 − 점선 점수 기준 기여도(구분 근거) 히트맵 — res34 기존 / res34 P3del 학습 모델, 원본 입력 (2단계 8·9번 그림).

목표값 S = Σ_(차선 칸) (실선 conf − 점선 conf),  차선 칸 = 실선 또는 점선 conf 가 0.4 를 넘는 칸.
Grad-CAM 을 ReLU 없이 계산해서 부호를 둘로 나눔:
  2줄 "실선 쪽 근거" = 실선−점선을 키운 곳 (+),  3줄 "점선 쪽 근거" = 실선−점선을 줄인 곳 (−)
2·3줄은 같은 칸(층) 안에서 같은 최댓값으로 맞춰 칠하므로, 두 줄의 세기를 서로 비교할 수 있다.
neck.forward 는 DeformFPN.forward 를 그대로 옮겨 중간값만 꺼내도록 바꿔 끼운 것(원래 forward 와 conf 차이 0 을 실행 때 확인).

{n}-8. 실선점선구분근거_히트맵_P3del학습.png : res34 P3del (test/006 epoch_150)
{n}-9. 실선점선구분근거_히트맵_기존.png       : res34 기존 (models/drone-ganet-res34-2cls-v1.pth)

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_solid_vs_dashed_cam.py
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
MODELS = [  # (파일 번호, 파일 이름, 제목, config, 체크포인트, 단계, 칸 제목)
    ("8", "실선점선구분근거_히트맵_P3del학습", "P3del로 다시 학습한 res34",
     "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth",
     ["C3", "C4", "C5", "C5+SA", "P4_pre", "P4"],
     ["C3\n백본 1/8 · 128ch (neck에선 안 씀)", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch",
      "C5+SA\nattention 후 · 64ch", "P4 (DCN 전)\noffset 헤드 입력 · 1/16", "P4 (최종)\nconf 헤드 입력 · 1/16"]),
    ("9", "실선점선구분근거_히트맵_기존", "기존 res34",
     "res34_s8_full_2cls", f"{G}/models/drone-ganet-res34-2cls-v1.pth",
     ["C3", "C4", "C5", "C5+SA", "P4", "P3_pre", "P3"],
     ["C3\n백본 1/8 · 128ch", "C4\n백본 1/16 · 256ch", "C5\n백본 1/32 · 512ch", "C5+SA\nattention 후 · 64ch",
      "P4\n위층 합친 뒤 · 1/16", "P3 (DCN 전)\noffset 헤드 입력 · 1/8", "P3 (최종)\nconf 헤드 입력 · 1/8"]),
]


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


os.makedirs(OUTDIR, exist_ok=True)
for fno, fname, mtitle, cfg_name, ckpt, STAGES, TITLES in MODELS:
    model, cfg = build(cfg_name, ckpt)
    W, H = cfg.img_scale
    mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
    std = np.array(cfg.img_norm_cfg["std"], np.float32)
    kpt_thr = model.bbox_head.kpt_thr
    neck = model.neck
    orig_forward = neck.forward
    feats = {}

    def keep(name, t):
        t.retain_grad()
        feats[name] = t

    def forward_keep(inputs):  # DeformFPN.forward 와 같은 계산 (deconv_before=False, use_res=False, trans_mode=replace)
        inputs = list(inputs)
        if len(inputs) > len(neck.in_channels):
            del inputs[0]
        names = ["C3", "C4", "C5"][-len(inputs):]
        for nm, x in zip(names, list(inputs)):
            keep(nm, x)
        if "C3" not in names:
            keep("C3", feats["_bb"][0])
        inputs[neck.trans_idx] = neck.trans_head(inputs[neck.trans_idx])
        keep("C5+SA", inputs[neck.trans_idx])
        lats = [lc(inputs[i + neck.start_level]) for i, lc in enumerate(neck.lateral_convs)]
        for i in range(len(lats) - 1, 0, -1):
            lats[i - 1] = lats[i - 1] + F.interpolate(lats[i], size=lats[i - 1].shape[2:], **neck.upsample_cfg)
        if len(lats) == 3:
            keep("P4", lats[1]); keep("P3_pre", lats[0])
        else:
            keep("P4_pre", lats[0])
        d_feat, points = neck.def_convs[0](lats[0])
        out = neck.fpn_convs[0](d_feat)
        keep(STAGES[-1], out)
        return dict(features=(out,), aux_feat=lats[0], deform_points=(points,))

    def bb_hook(m, i, o):
        feats["_bb"] = o  # P3del 은 out_indices=(1,2,3) -> o[0] 이 C3
    model.backbone.register_forward_hook(bb_hook)

    for name, path in JOBS:
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
        with torch.no_grad():
            neck.forward = orig_forward
            hm_ref = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"][0]
        neck.forward = forward_keep
        feats.clear()
        with torch.enable_grad():
            hm = model.bbox_head.forward(model.extract_feat(t))["kpts_hm"][0]
            assert float((hm - hm_ref).abs().max()) < 1e-5, "forward_keep 가 원래 forward 와 다름"
            mask = torch.maximum(hm[0], hm[1]) > kpt_thr
            n_lane = int(mask.sum())
            A_list = [feats[st] for st in STAGES]
            S = (hm[0] - hm[1])[mask].sum()
            grads = torch.autograd.grad(S, A_list)
            cams = [(g[0].mean(dim=(1, 2))[:, None, None] * a[0]).sum(0).detach().cpu().numpy()
                    for a, g in zip(A_list, grads)]
        n_sol = int(((hm[0] >= hm[1]) & mask).sum())
        shares = [float(np.clip(c, 0, None).sum() / (np.abs(c).sum() + 1e-8)) for c in cams]
        print(f"[{mtitle} 프레임{name}] 차선 칸 {n_lane}개 (실선 우세 {n_sol}, 점선 우세 {n_lane - n_sol}), "
              "실선 쪽 비율 " + ", ".join(f"{st} {s:.2f}" for st, s in zip(STAGES, shares)))

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
        nc = len(STAGES)
        fig, axes = plt.subplots(3, nc, figsize=(5 * nc, 9.3), gridspec_kw=dict(hspace=0.04, wspace=0.04))
        for j in range(nc):
            act = norm01(A_list[j][0].detach().norm(dim=0).cpu().numpy())
            c = cams[j]
            vmax = np.abs(c).max() + 1e-8  # 2·3줄 같은 기준
            rows = [act, np.clip(c, 0, None) / vmax, np.clip(-c, 0, None) / vmax]
            for r, mm in enumerate(rows):
                ax = axes[r, j]
                ax.imshow(bg)
                ax.imshow(cv2.resize(mm.astype(np.float32), (W, H)), cmap="jet", alpha=0.55, vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(TITLES[j], fontsize=14)
        axes[0, 0].set_ylabel("활성 크기", fontsize=16)
        axes[1, 0].set_ylabel("실선 쪽 근거\n(실선−점선 ↑)", fontsize=16)
        axes[2, 0].set_ylabel("점선 쪽 근거\n(실선−점선 ↓)", fontsize=16)
        fig.suptitle(f"{name}-{fno}. {mtitle} 모델의 실선·점선 구분 근거 히트맵 "
                     "(1줄: 반응 크기, 2·3줄: '실선 점수 − 점선 점수'를 키운 곳 / 줄인 곳, 2·3줄은 같은 기준)", fontsize=18)
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01, ticks=[0, 1])
        cb.ax.set_yticklabels(["낮음", "높음"], fontsize=13)
        fig.savefig(f"{OUTDIR}/{name}-{fno}. {fname}.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    del model
    torch.cuda.empty_cache()
