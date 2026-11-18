#!/usr/bin/env python3
"""GANet 최종 검출 결과(후처리까지 끝난 polyline)를 원본 프레임 위에 그린다.

scripts/04-eval/infer_video_2cls.py와 완전히 동일한 추론/후처리 경로
(extract_feat -> bbox_head.forward_test -> model.post_processing)를
그대로 사용한다 -- 실제로 결과영상을 만든 코드와 같은 경로여야 비교가
의미있다.

주의: PostProcessor.crop_bbox는 학습 config(res34_s8_full_2cls.py)에
[0,0,3840,2160]로 박혀있고, adjust_result()는 crop_bbox가 있으면
ori_shape(진짜 입력 해상도)를 무시하고 ratio_x=(crop_bbox[2]-crop_bbox[0])/img_w,
ratio_y=(crop_bbox[3]-crop_bbox[1])/img_h 로만 좌표를 역변환한다
(post_process.py:358-373). 즉 "원본이 항상 3840x2160"라고 가정하고 좌표를
되돌린다 -- 이 AIhub 프레임은 실제로 1920x1080이라 이 가정이 깨진다.
이 스크립트는 바로 그 실제 결과(좌표가 깨지는지 여부)를 눈으로 보기 위한 것.

실행 (ganet_cu101 환경 필수):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_final_result_viz.py
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
LINE_WIDTH = 6  # infer_video_2cls.py 기본값과 동일

CLASS_COLORS_BGR = {
    1: (0, 220, 0),   # 실선 - 초록
    2: (0, 0, 220),   # 점선 - 빨강
}
CLASS_NAMES = {1: "실선", 2: "점선"}


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
    return torch.from_numpy(chw).float()


def main():
    mmcv.mkdir_or_exist(OUT_DIR)

    model, cfg = build_model(CFG_PATH, CKPT_PATH, GPU_ID)
    img_scale_wh = tuple(cfg.img_scale)  # (1280, 736)
    mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)
    crop_bbox_trained = cfg.model.post_processing.get("crop_bbox")

    frame_bgr = cv2.imread(IMG_PATH)
    assert frame_bgr is not None, f"cannot read {IMG_PATH}"
    ori_h, ori_w = frame_bgr.shape[:2]

    # crop_bbox가 학습 config 값(드론 3840x2160)으로 박혀있어 실제 입력(1920x1080,
    # AIhub CCTV)과 안 맞는다 -- adjust_result()가 crop_bbox 기준으로만 좌표를
    # 역변환하기 때문에(ori_shape는 무시) 그대로 두면 좌표가 전부 프레임 밖으로
    # 나간다(직접 실행해서 확인: 186점 중 0점만 프레임 내부). 실제 입력 해상도로
    # 덮어써서 traffic-video-analytics의 "_보정" 캐시와 동일한 보정을 적용한다.
    crop_bbox_fixed = [0, 0, ori_w, ori_h]
    model.post_processing.crop_bbox = crop_bbox_fixed

    tensor = preprocess(frame_bgr, img_scale_wh, mean, std).unsqueeze(0).cuda(GPU_ID)

    meta = dict(
        img_shape=(img_scale_wh[1], img_scale_wh[0], 3),
        ori_shape=(ori_h, ori_w, 3),
        offset_x=0, offset_y=0,
        filename="frame_000000.jpg",
        ori_filename="frame_000000.jpg",
    )

    with torch.no_grad():
        feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
        pred = model.bbox_head.forward_test(feats)
        result = model.post_processing(pred, meta)

    lanes, _, _, classes = result["final_dict_list"]

    canvas = frame_bgr.copy()
    n_pts_total = 0
    n_pts_in_frame = 0
    x_range_all, y_range_all = [], []
    per_lane_info = []
    for lane_idx, (lane, cls) in enumerate(zip(lanes, classes)):
        pts = np.array(lane, dtype=np.int32)
        n_pts_total += len(pts)
        in_frame = (pts[:, 0] >= 0) & (pts[:, 0] < ori_w) & (pts[:, 1] >= 0) & (pts[:, 1] < ori_h)
        n_pts_in_frame += int(in_frame.sum())
        x_range_all.append((pts[:, 0].min(), pts[:, 0].max()))
        y_range_all.append((pts[:, 1].min(), pts[:, 1].max()))
        per_lane_info.append(
            f"  - lane{lane_idx}[{CLASS_NAMES.get(cls, cls)}]: {len(pts)}점, "
            f"x={pts[:,0].min()}~{pts[:,0].max()}, y={pts[:,1].min()}~{pts[:,1].max()}, "
            f"프레임 내부 {int(in_frame.sum())}/{len(pts)}"
        )
        color = CLASS_COLORS_BGR.get(cls, (255, 255, 255))
        pts_draw = pts.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_draw], isClosed=False, color=color, thickness=LINE_WIDTH)

    print(f"검출된 차선 수: {len(lanes)}")
    for info in per_lane_info:
        print(info)
    print(f"전체 점 {n_pts_total}개 중 프레임 내부 {n_pts_in_frame}개 "
          f"({n_pts_in_frame / max(n_pts_total,1) * 100:.1f}%)")

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    ori_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    lane_summary = "\n".join(per_lane_info) if per_lane_info else "  (검출된 차선 없음)"
    caption = (
        f"입력: {IMG_PATH.split('/')[-1]} ({ori_w}x{ori_h}) | 모델 입력 리사이즈: {img_scale_wh[0]}x{img_scale_wh[1]}\n"
        f"crop_bbox: 학습 config 원값={crop_bbox_trained}(드론 3840x2160 기준) -> 실제 입력 해상도로 보정={crop_bbox_fixed} "
        "(보정 전 그대로 쓰면 186점 중 0점만 프레임 내부 -- 전부 밖으로 나감, traffic-video-analytics의 \"_보정\" 캐시와 동일한 보정)\n"
        f"검출 차선 수: {len(lanes)}, 좌표가 프레임 밖으로 나간 점: {n_pts_total - n_pts_in_frame}/{n_pts_total}개\n"
        f"{lane_summary}\n"
        "초록=실선(class1), 빨강=점선(class2)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    axes[0].imshow(ori_rgb)
    axes[0].set_title("원본 프레임", fontsize=14, fontfamily=KFONT)
    axes[0].axis("off")
    axes[1].imshow(canvas_rgb)
    axes[1].set_title("GANet 최종 검출 결과\n(post_processing까지 끝난 polyline)", fontsize=14, fontfamily=KFONT)
    axes[1].axis("off")
    fig.suptitle("1. 원본 프레임에 대한 GANet 최종 검출 결과", fontsize=16, fontfamily=KFONT, y=0.995)
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, fontfamily=KFONT, wrap=True)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))

    out_path = f"{OUT_DIR}/1. 원본프레임_GANet_최종검출결과.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")

    cv2.imwrite("/home/hsjeong/tmp/_raw_result_canvas.png", canvas)
    print("saved raw canvas for cropping")


if __name__ == "__main__":
    main()
