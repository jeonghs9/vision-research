#!/usr/bin/env python3
"""GANet 미탐 사례 분석 — 중간 feature/confidence map/offset map 히트맵.

aihub_CCTV고속도로/1 영상의 첫 프레임을 실제 배포 추론 코드
(traffic-video-analytics/.../profile_ganet_cache_gen.py)와 동일한 모델
(res34_s8_full_2cls.py + drone-ganet-res34-2cls-v1.pth)·전처리로 통과시켜
아래 3개 중간 산출물을 히트맵으로 저장한다.

  1-1. FPN 출력(aux_feat, LFA/offset head 진입 전 feature[0])
  1-2. keypoints_head 이후 confidence map (클래스별)
  1-3. offset_head 이후 offset map (magnitude + x/y 채널) -- confidence map으로 마스킹

1-3 마스킹 근거: ganetloss.py의 offset loss는 `output*mask, target*mask`로
계산되고(mask=offset_mask), 주석에 "pts_offset和int_offset的监督，只有关键点才有"
(offset 지도학습은 키포인트 칸에서만 있다)라고 명시돼 있다 -- 즉 confidence가
낮은 칸(배경)의 offset 출력은 학습 신호를 받은 적이 없는 값이라 의미가 없고,
실제 디코딩(ktdet_decode_fast)도 kpts_hm이 kpt_thr을 넘는 칸에서만 offset을
읽는다. 그래서 1-3은 confidence(실선+점선 중 최댓값)가 kpt_thr을 넘는 칸만
색을 칠하고, 나머지는 원본 그대로 둔다.

실행 (ganet_cu101 환경 필수):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_miss_detection_viz.py
"""
import sys

GANET_SRC = "/home/hsjeong/workspace/vision-det/GANet/source"
sys.path.insert(0, GANET_SRC)

import cv2
import numpy as np
import torch
import mmcv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT_PATH)
KFONT = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.family"] = KFONT
plt.rcParams["axes.unicode_minus"] = False

CFG_PATH = f"{GANET_SRC}/projects/cfgs/drone/res34_s8_full_2cls.py"
CKPT_PATH = "/home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth"
IMG_PATH = (
    "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/"
    "Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png"
)
OUT_DIR = (
    "/home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/"
    "signed-distance-validation/미탐사례분석"
)
GPU_ID = 0

CLASS_NAMES = {1: "실선(class1)", 2: "점선(class2)"}


def build_model(cfg_path, ckpt_path, gpu_id):
    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.cuda(gpu_id)
    model.eval()
    return model, cfg


def preprocess(frame_bgr, img_scale_wh, mean, std):
    resized_bgr = cv2.resize(frame_bgr, img_scale_wh, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    normed = (rgb - mean) / std
    chw = normed.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).float()
    resized_rgb_u8 = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return tensor, resized_rgb_u8


def upsample_map(m_hw, out_hw, interp=cv2.INTER_CUBIC):
    """(h,w) float map -> (H,W) via upsample (feature map 해상도를 원본 프레임 크기로)."""
    out_w, out_h = out_hw
    return cv2.resize(m_hw.astype(np.float32), (out_w, out_h), interpolation=interp)


def colorize(norm01, cmap="jet"):
    cm = matplotlib.colormaps.get_cmap(cmap)
    rgba = cm(np.clip(norm01, 0.0, 1.0))
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(base_rgb_u8, heat01, cmap="jet", alpha=0.55, mask01=None):
    """mask01(0~1, HxW)이 주어지면 그 비율만큼만 섞는다 -- 0인 칸은 base 그대로."""
    heat_rgb = colorize(heat01, cmap=cmap)
    eff_alpha = alpha if mask01 is None else alpha * mask01[..., None]
    blended = base_rgb_u8.astype(np.float32) * (1 - eff_alpha) + heat_rgb.astype(np.float32) * eff_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_panel(panels, titles, suptitle, caption, out_path, cbar_specs=None):
    """panels: list of HxWx3 uint8 images. cbar_specs: list of (vmin,vmax,cmap,label) or None per panel."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.4))
    if n == 1:
        axes = [axes]
    for ax, img, title, cbs in zip(axes, panels, titles, cbar_specs or [None] * n):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontfamily=KFONT)
        ax.axis("off")
        if cbs is not None:
            vmin, vmax, cmap, label = cbs
            sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(label, fontsize=10, fontfamily=KFONT)
    fig.suptitle(suptitle, fontsize=16, fontfamily=KFONT, y=0.99)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10.5, fontfamily=KFONT, wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    mmcv.mkdir_or_exist(OUT_DIR)

    model, cfg = build_model(CFG_PATH, CKPT_PATH, GPU_ID)
    img_scale_wh = tuple(cfg.img_scale)  # (1280, 736) = (W,H)
    mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)
    hm_idx = cfg.model.head.hm_idx
    kpt_thr = cfg.model.head.kpt_thr
    out_w, out_h = img_scale_wh

    frame_bgr = cv2.imread(IMG_PATH)
    assert frame_bgr is not None, f"cannot read {IMG_PATH}"
    ori_h, ori_w = frame_bgr.shape[:2]

    tensor, base_rgb_u8 = preprocess(frame_bgr, img_scale_wh, mean, std)
    tensor = tensor.unsqueeze(0).cuda(GPU_ID)

    with torch.no_grad():
        feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
        head_dict = model.bbox_head.forward(feats)

    aux_feat = feats["aux_feat"]            # (1,64,h,w) FPN 출력, LFA/offset head 진입 전
    kpts_hm = head_dict["kpts_hm"]          # (1,num_classes,h,w) sigmoid+clamp 완료
    pts_offset = head_dict["pts_offset"]    # (1,2,h,w)

    feat_h, feat_w = aux_feat.shape[-2:]
    common_caption_res = (
        f"입력 프레임: {ori_w}x{ori_h} → 모델 입력 리사이즈 {out_w}x{out_h} "
        f"→ feature map 해상도 {feat_w}x{feat_h} (stride {out_h // feat_h})"
    )

    # ---------------- 1-1. FPN feature[0] (aux_feat, LFA/offset head 진입 전) ----------------
    feat_np = aux_feat[0].detach().cpu().numpy()  # (64,h,w)
    feat_energy = np.linalg.norm(feat_np, axis=0)  # 채널별 L2-norm -> (h,w) 활성화 강도
    e_min, e_max = float(feat_energy.min()), float(feat_energy.max())
    feat_energy_norm = (feat_energy - e_min) / max(e_max - e_min, 1e-6)
    feat_up = upsample_map(feat_energy_norm, (out_w, out_h))
    feat_overlay = overlay(base_rgb_u8, feat_up, cmap="jet", alpha=0.55)
    save_panel(
        panels=[base_rgb_u8, feat_overlay],
        titles=["원본(리사이즈)", "FPN feature[0] 활성화 강도\n(64채널 L2-norm, LFA·offset head 진입 전)"],
        suptitle="1-1. FPN 출력 feature[0] 히트맵 (aux_feat)",
        caption=(
            common_caption_res
            + f" | 채널별 L2-norm 범위: {e_min:.2f} ~ {e_max:.2f} (이미지 내에서 0~1로 정규화)"
            + "\n이 feature는 top-down 융합이 끝난 직후, LFA(deformable conv)와 offset_head 양쪽 모두에 들어가기 "
            + "전의 공통 입력이다 (dcn_fpn.py의 aux_feat)."
        ),
        out_path=f"{OUT_DIR}/1-1. FPN특징맵_히트맵.png",
        cbar_specs=[None, (0, 1, "jet", "정규화된 활성화 강도")],
    )

    # ---------------- 1-2. confidence map (keypoints_head 이후, 클래스별) ----------------
    num_classes = kpts_hm.shape[1]
    kpts_np = kpts_hm[0].detach().cpu().numpy()  # (C,h,w), 이미 sigmoid 적용된 0~1 확률
    panels = [base_rgb_u8]
    titles = ["원본(리사이즈)"]
    cbar_specs = [None]
    per_class_minmax = []
    for c in range(num_classes):
        cls_map = kpts_np[c]
        cls_up = upsample_map(cls_map, (out_w, out_h))
        panels.append(overlay(base_rgb_u8, cls_up, cmap="inferno", alpha=0.6))
        cname = CLASS_NAMES.get(c + 1, f"class{c + 1}")
        titles.append(f"confidence map: {cname}\n(keypts_head 출력, sigmoid 확률)")
        cbar_specs.append((0, 1, "inferno", "키포인트 확률 (sigmoid)"))
        per_class_minmax.append((cname, float(cls_map.min()), float(cls_map.max())))
    minmax_txt = " | ".join(f"{n}: {mn:.3f}~{mx:.3f}" for n, mn, mx in per_class_minmax)
    save_panel(
        panels=panels,
        titles=titles,
        suptitle="1-2. Keypoint Confidence Map 히트맵 (클래스별)",
        caption=(
            common_caption_res + f"\n클래스별 확률 범위(0~1 스케일, 절대값): {minmax_txt}\n"
            "색 스케일은 두 클래스 모두 0~1 절대값으로 고정 — 왼쪽 구간의 색이 전반적으로 어둡다면 "
            "그 영역의 키포인트 확률 자체가 낮다는(=모델이 애초에 낮게 예측) 뜻이다."
        ),
        out_path=f"{OUT_DIR}/1-2. 키포인트_컨피던스맵_히트맵.png",
        cbar_specs=cbar_specs,
    )

    # ---------------- 1-3. offset map (offset_head 이후) -- confidence로 마스킹 ----------------
    # offset_head의 학습 loss(ganetloss.py:68,111)는 "mask"(=실제 키포인트 칸)에서만
    # 계산된다 -- 배경 칸(하늘/바다 등)의 offset 출력은 학습 신호를 받은 적이 없어
    # 의미가 없고, 디코딩도 confidence>kpt_thr인 칸에서만 offset을 읽는다. 그래서
    # confidence(실선/점선 중 최댓값)가 kpt_thr을 넘는 칸만 보여주고 나머지는 원본
    # 그대로 둔다 -- "실제로 쓰이는 값"만 눈에 보이게.
    conf_max_hw = kpts_np.max(axis=0)  # (h,w), 클래스 무관 최댓값
    conf_mask_hw = (conf_max_hw > kpt_thr).astype(np.float32)
    mask_up = upsample_map(conf_mask_hw, (out_w, out_h), interp=cv2.INTER_NEAREST)
    n_masked = int((mask_up > 0.5).sum())
    n_total = mask_up.size
    print(f"confidence>{kpt_thr} 마스크: {n_masked}/{n_total}칸 "
          f"({n_masked / n_total * 100:.2f}%)만 offset 값을 표시")

    off_np = pts_offset[0].detach().cpu().numpy()  # (2,h,w): channel0=x-offset, channel1=y-offset
    off_x, off_y = off_np[0], off_np[1]
    off_mag = np.sqrt(off_x ** 2 + off_y ** 2)

    # 정규화 범위도 마스크 안쪽 값만 기준으로 잡는다(마스크 밖 노이즈가 스케일을 왜곡하지 않게)
    def masked_minmax(arr):
        vals = arr[conf_mask_hw > 0.5]
        if vals.size == 0:
            return float(arr.min()), float(arr.max())
        return float(vals.min()), float(vals.max())

    mag_min, mag_max = masked_minmax(off_mag)
    mag_norm = (off_mag - mag_min) / max(mag_max - mag_min, 1e-6)
    mag_up = upsample_map(mag_norm, (out_w, out_h))

    x_min, x_max = masked_minmax(off_x)
    x_norm = (off_x - x_min) / max(x_max - x_min, 1e-6)
    x_up = upsample_map(x_norm, (out_w, out_h))

    y_min, y_max = masked_minmax(off_y)
    y_norm = (off_y - y_min) / max(y_max - y_min, 1e-6)
    y_up = upsample_map(y_norm, (out_w, out_h))

    panels = [
        base_rgb_u8,
        overlay(base_rgb_u8, mag_up, cmap="jet", alpha=0.65, mask01=mask_up),
        overlay(base_rgb_u8, x_up, cmap="coolwarm", alpha=0.65, mask01=mask_up),
        overlay(base_rgb_u8, y_up, cmap="coolwarm", alpha=0.65, mask01=mask_up),
    ]
    titles = [
        "원본(리사이즈)",
        f"offset magnitude (confidence 마스킹됨)\n√(x²+y²), 마스크 안쪽 실제값 {mag_min:.2f}~{mag_max:.2f}",
        f"x-offset (채널0, confidence 마스킹됨)\n마스크 안쪽 실제값 {x_min:.2f}~{x_max:.2f}",
        f"y-offset (채널1, confidence 마스킹됨)\n마스크 안쪽 실제값 {y_min:.2f}~{y_max:.2f}",
    ]
    cbar_specs = [
        None,
        (0, 1, "jet", "정규화된 offset 크기"),
        (0, 1, "coolwarm", "정규화된 x-offset"),
        (0, 1, "coolwarm", "정규화된 y-offset"),
    ]
    save_panel(
        panels=panels,
        titles=titles,
        suptitle="1-3. Offset Map 히트맵 (offset_head 출력, aux_feat 입력, confidence로 마스킹)",
        caption=(
            common_caption_res
            + f"\nconfidence(실선/점선 중 최댓값) > kpt_thr({kpt_thr})인 칸만 색칠함 "
            f"({n_masked}/{n_total}칸, {n_masked / n_total * 100:.2f}%) -- 나머지는 원본 그대로.\n"
            "ganetloss.py의 offset loss는 'mask'(키포인트 칸)에서만 계산되고(주석: \"offset 지도학습은 "
            "키포인트에서만 있다\"), 디코딩도 이 칸에서만 offset을 읽는다 -- 마스크 밖 값은 학습된 적이 "
            "없어 의미가 없으므로 애초에 표시하지 않는다.\n"
            "offset_head는 feature[0](post-LFA)이 아니라 1-1의 aux_feat(pre-LFA)을 입력으로 받는다. "
            "각 픽셀이 차선 '루트(root)' 위치까지 가져야 할 (x,y) 변위를 예측한 값이다."
        ),
        out_path=f"{OUT_DIR}/1-3. 오프셋맵_히트맵.png",
        cbar_specs=cbar_specs,
    )


if __name__ == "__main__":
    main()
