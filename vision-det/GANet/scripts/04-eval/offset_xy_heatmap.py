#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] offset_head 출력(pts_offset)의 x성분/y성분을 각각 히트맵으로 그린다.

`ganet_head.py`의 `ktdet_decode_fast`를 보면 `pts_offset`은 2채널
(offset_split[0]=x, offset_split[1]=y)이고, `root_mat = coord_mat +
pts_offset`로 각 키포인트가 "자기 차선의 시작점" 방향을 가리키는 벡터를
만든다(`coord_mat`도 x,y 순서로 쌓여 있어 채널 순서가 맞는다). 반대 차로
점선 미탐 원인분석(`docs/01-차선검출모델/11. GANet_미탐원인분석.md`)에서
L2-norm 합산 히트맵(1-3)만으로는 "두 점선의 root가 실제로 가까운 좌표로
수렴하는지"를 방향별로 볼 수 없어서, x/y 성분을 따로 시각화한다.

confidence 마스킹은 1-3과 동일한 기준(`kpts_hm > kpt_thr`, 클래스는
max로 통합) 적용 -- offset 손실이 키포인트 위치에만 걸리므로 배경 영역의
offset 값은 노이즈다(ganetloss.py 참고).

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/04-eval/offset_xy_heatmap.py \
    projects/cfgs/drone/res34_s8_full_2cls.py \
    /home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth \
    "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png" \
    --out "/home/hsjeong/workspace/vision-det/GANet/output/05-cls2-점선미탐원인분석/figures/1-4. 오프셋맵_XY벡터_히트맵.png" \
    --gpu-id 0
"""
import argparse
import sys

sys.path.insert(0, "./")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # ttc에 한글 글리프도 포함돼 있음
plt.rcParams["axes.unicode_minus"] = False
import mmcv
import numpy as np
import torch

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa


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
    resized = cv2.resize(frame_bgr, img_scale_wh, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normed = (rgb - mean) / std
    chw = normed.transpose(2, 0, 1)
    return torch.from_numpy(chw).float(), resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    a = ap.parse_args()

    model, cfg = build_model(a.config, a.checkpoint, a.gpu_id)
    img_scale_wh = tuple(cfg.img_scale)  # (W, H) = (1280, 736)
    mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)
    kpt_thr = cfg.model.head.kpt_thr

    frame = cv2.imread(a.image)
    assert frame is not None, f"cannot read {a.image}"
    tensor, resized_bgr = preprocess(frame, img_scale_wh, mean, std)
    tensor = tensor.unsqueeze(0).cuda(a.gpu_id)

    with torch.no_grad():
        feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
        head_dict = model.bbox_head.forward(feats)

    kpts_hm = head_dict["kpts_hm"][0].cpu().numpy()       # (C, h, w), sigmoid 완료
    pts_offset = head_dict["pts_offset"][0].cpu().numpy()  # (2, h, w) -- [0]=x, [1]=y

    conf = kpts_hm.max(axis=0)  # 클래스 통합 confidence (실선/점선 중 더 높은 쪽)
    mask = conf > kpt_thr
    n_total = mask.size
    n_kept = int(mask.sum())
    print(f"native feature map: {mask.shape[1]}x{mask.shape[0]} (W x H), "
          f"kpt_thr={kpt_thr}, masked-in: {n_kept}/{n_total} ({n_kept / n_total * 100:.2f}%)")

    off_x = pts_offset[0].copy()
    off_y = pts_offset[1].copy()
    off_x[~mask] = np.nan
    off_y[~mask] = np.nan

    W, H = img_scale_wh
    off_x_up = cv2.resize(off_x, (W, H), interpolation=cv2.INTER_NEAREST)
    off_y_up = cv2.resize(off_y, (W, H), interpolation=cv2.INTER_NEAREST)

    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    vmax_x = np.nanmax(np.abs(off_x_up)) if n_kept > 0 else 1.0
    vmax_y = np.nanmax(np.abs(off_y_up)) if n_kept > 0 else 1.0

    # 도로 영역(색칠된 칸)만 보이게 확대 -- 하늘/바다 여백을 빼면 평행한
    # 줄무늬 패턴이 훨씬 잘 보인다. stride 8칸 기준 여유를 좀 더 줌(패딩 4칸=32px).
    ys, xs = np.where(mask)
    pad = 4
    x0 = max(int(xs.min()) - pad, 0) * 8
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1]) * 8
    y0 = max(int(ys.min()) - pad, 0) * 8
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0]) * 8
    crop = lambda a: a[y0:y1, x0:x1]
    resized_rgb_c = crop(resized_rgb)
    off_x_c = crop(off_x_up)
    off_y_c = crop(off_y_up)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    fig.suptitle("1-4. 오프셋맵(pts_offset) x/y 성분 히트맵 "
                 "(offset_head 출력, confidence>%.1f 위치만, LFA 이전 feature 기반)" % kpt_thr,
                 fontsize=14)

    axes[0].imshow(resized_rgb_c)
    axes[0].set_title(f"원본 프레임 (도로 영역만 확대, 모델 입력 {W}x{H} 기준)")
    axes[0].axis("off")

    im1 = axes[1].imshow(resized_rgb_c)
    im1 = axes[1].imshow(off_x_c, cmap="coolwarm", vmin=-vmax_x, vmax=vmax_x, alpha=0.75)
    axes[1].set_title("offset x성분 (coord_mat[0]+이 값 = root의 x좌표)\n"
                       "빨강=+x(오른쪽을 가리킴), 파랑=-x(왼쪽을 가리킴)")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(resized_rgb_c)
    im2 = axes[2].imshow(off_y_c, cmap="coolwarm", vmin=-vmax_y, vmax=vmax_y, alpha=0.75)
    axes[2].set_title("offset y성분 (coord_mat[1]+이 값 = root의 y좌표)\n"
                       "빨강=+y(아래를 가리킴), 파랑=-y(위를 가리킴)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.text(0.5, 0.02,
              f"입력: {a.image.split('/')[-1]} | native feature map stride 8 -> "
              f"nearest-neighbor로 {W}x{H} 확대 표시 | "
              f"마스킹: confidence>{kpt_thr} (전체 {n_total}칸 중 {n_kept}칸={n_kept / n_total * 100:.2f}%만 표시) | "
              f"표시 영역: 마스킹된 칸의 bbox+여유 4칸(모델입력 기준 x={x0}~{x1}, y={y0}~{y1})",
              ha="center", fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(a.out, dpi=150)
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
