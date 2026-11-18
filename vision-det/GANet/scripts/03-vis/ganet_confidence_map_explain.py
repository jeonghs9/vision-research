"""GANet 학습 타겟(confidence map) 생성 과정 설명 그림.

왼쪽: 라벨링 원본(.lines.txt) 폴리라인을 그대로 이은 GT
오른쪽: 실제 학습에 쓰이는 CollectLanePoints.get_targets()가 만든
        (1) 스플라인 리샘플링된 점들 (2) 그 점에 가우시안을 찍어 만든 confidence map

파이프라인 재현 자체(ploy_fitting_cube, draw_umich_gaussian, CollectLanePoints)는
GANet 원본 코드를 직접 import해서 그대로 호출한다 (별도 재구현 없음).
"""
import importlib.util
import os
from functools import cmp_to_key

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE_DIR = "/home/hsjeong/workspace/vision-det/GANet/source"
DATA_ROOT = "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/culane_lane_ganet"
STEM = "images/19700101_000157_T__000290"
OUT_PATH = "/home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/signed-distance-validation/GANet_confidence_map_설명.png"

# ---- res34_s8_full_2cls.py(=res18_s8_full.py 상속) 기준 실제 config 값 ----
IMG_SCALE = (1280, 736)          # (width, height)
FPN_DOWN_SCALE = [8, 16, 32]
HM_IDX = 0
HM_DOWN_SCALE = FPN_DOWN_SCALE[HM_IDX]   # 8
GAUSSIAN_RADIUS = 2
NUM_CLASSES = 2
CLASS_COLORS_BGR = {1: (0, 220, 0), 2: (0, 0, 220)}   # 1=실선(초록) 2=점선(빨강), ganet.py와 동일
CLASS_NAMES = {1: "실선", 2: "점선"}

# ---- GANet 원본 target-생성 코드를 패키지 __init__ 체인(컴파일된 DCN .so 의존) 없이 직접 로드 ----
spec = importlib.util.spec_from_file_location(
    "lane_target_collect",
    os.path.join(SOURCE_DIR, "projects/plugin/datasets/pipelines/lane_target_collect.py"),
)
ltc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ltc)


def load_lane_labels(stem):
    lines_path = os.path.join(DATA_ROOT, stem + ".lines.txt")
    cls_path = os.path.join(DATA_ROOT, stem + ".cls.txt")
    lanes = []
    with open(lines_path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split(' ') if v]
            if len(vals) > 3:
                lanes.append(vals)  # flat [x0,y0,x1,y1,...], 원본 3840x2160 좌표
    with open(cls_path) as f:
        classes = [int(v) for v in f.read().split()]
    if len(classes) != len(lanes):
        classes = [1] * len(lanes)
    return lanes, classes


def main():
    img = cv2.imread(os.path.join(DATA_ROOT, STEM + ".jpg"))
    ori_h, ori_w = img.shape[:2]
    assert (ori_w, ori_h) == (3840, 2160), (ori_w, ori_h)

    lanes_raw, classes = load_lane_labels(STEM)

    # --- 학습 때와 동일하게: Crop(전체 프레임, no-op) + Resize(1280x736) ---
    out_w, out_h = IMG_SCALE
    img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    sx, sy = out_w / ori_w, out_h / ori_h

    def scale_flat(flat, sx, sy):
        pts = [(flat[2 * i] * sx, flat[2 * i + 1] * sy) for i in range(len(flat) // 2)]
        return pts

    lanes_resized = [scale_flat(l, sx, sy) for l in lanes_raw]

    # =========================================================
    # 왼쪽 패널: 라벨링 원본 폴리라인 (resize된 좌표계로만 옮김, 점 개수/위치는 라벨 그대로)
    # =========================================================
    left = img_resized.copy()
    for pts, cls in zip(lanes_resized, classes):
        color = CLASS_COLORS_BGR.get(cls, (255, 255, 0))
        pts_i = np.array(pts, dtype=np.int32)
        cv2.polylines(left, [pts_i], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        for (x, y) in pts_i:
            cv2.circle(left, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)

    # =========================================================
    # 오른쪽 패널: 실제 CollectLanePoints.get_targets() 호출 결과
    # =========================================================
    collector = ltc.CollectLanePoints(
        max_lane_num=13,
        lane_extend=False,
        gaussian_hyper=dict(radius=GAUSSIAN_RADIUS, root_radius=4),
        dis_weights=[1, 0.4, 0.2],
        fpn_cfg=dict(hm_idx=HM_IDX, fpn_down_scale=FPN_DOWN_SCALE,
                     sample_per_lane=[41, 21, 11], deconv_layer=[True, False, False]),
        num_classes=NUM_CLASSES,
    )
    gt_points_flat = [[c for xy in pts for c in xy] for pts in lanes_resized]  # 다시 flat list로
    results = dict(
        img_shape=[out_h, out_w, 3],
        gt_points=gt_points_flat,
        id_classes=classes,
    )
    targets = collector.get_targets(results)
    kpts_hm = targets['gt_kpts_hm'].data.numpy()  # (num_classes, hm_h, hm_w) -- 실제 학습 타겟 그 자체
    hm_h, hm_w = kpts_hm.shape[1:]
    assert (hm_h, hm_w) == (out_h // HM_DOWN_SCALE, out_w // HM_DOWN_SCALE)

    # confidence map 컬러 오버레이용: 클래스별 색을 heatmap 값으로 가중합
    heat_color = np.zeros((hm_h, hm_w, 3), dtype=np.float32)
    for cls in (1, 2):
        cls_idx = cls - 1
        color = np.array(CLASS_COLORS_BGR[cls], dtype=np.float32)
        heat_color += kpts_hm[cls_idx][..., None] * color[None, None, :]
    heat_color = np.clip(heat_color, 0, 255).astype(np.uint8)
    heat_color_up = cv2.resize(heat_color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    heat_alpha = np.clip(kpts_hm.max(axis=0), 0, 1)
    heat_alpha_up = cv2.resize(heat_alpha, (out_w, out_h), interpolation=cv2.INTER_NEAREST)[..., None]

    right = img_resized.copy().astype(np.float32)
    right = right * (1 - 0.65 * heat_alpha_up) + heat_color_up.astype(np.float32) * (0.65 * heat_alpha_up)
    right = np.clip(right, 0, 255).astype(np.uint8)

    # --- 실제 get_targets() 내부와 '동일한' 리샘플링 호출을 그대로 재현해서, 쓰인 점 좌표 자체를 표시 ---
    n_resampled_total = 0
    for pts_flat, cls in zip(gt_points_flat, classes):
        cls_idx = min(max(cls - 1, 0), NUM_CLASSES - 1)
        pts = ltc.convert_scale(pts_flat, HM_DOWN_SCALE)
        if len(pts) < 2:
            continue
        pts = ltc.ploy_fitting_cube(pts, hm_h, hm_w, int(360 / HM_DOWN_SCALE))
        if pts is None:
            continue
        pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))
        pts = ltc.clamp_line(pts, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)
        if pts is None or len(pts) <= 1:
            continue
        color = CLASS_COLORS_BGR.get(cls, (255, 255, 0))
        for (x, y) in pts:
            # heatmap 좌표(92x160) -> 화면 표시 좌표(1280x736)로 역변환
            X, Y = int(round(x * HM_DOWN_SCALE)), int(round(y * HM_DOWN_SCALE))
            cv2.circle(right, (X, Y), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(right, (X, Y), 2, color, -1, lineType=cv2.LINE_AA)
        n_resampled_total += len(pts)

    # =========================================================
    # 합치기 + 한글 캡션 (Noto Sans CJK)
    # =========================================================
    gap = 14
    pad_top, pad_bottom = 130, 150
    canvas_w = out_w * 2 + gap
    canvas_h = out_h + pad_top + pad_bottom
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    left_pil = Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    right_pil = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    canvas.paste(left_pil, (0, pad_top))
    canvas.paste(right_pil, (out_w + gap, pad_top))

    draw = ImageDraw.Draw(canvas)
    cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    f_title = ImageFont.truetype(cjk_path, 26)
    f_sub = ImageFont.truetype(cjk_path, 18)
    f_small = ImageFont.truetype(cjk_path, 15)

    n_raw_pts = sum(len(p) // 2 for p in lanes_raw)
    draw.text((canvas_w // 2, 18), "GANet confidence map 학습 타겟 생성 과정", font=f_title,
              fill=(20, 20, 20), anchor="ma")
    draw.text((canvas_w // 2, 54), f"샘플: {STEM}.jpg   원본 {ori_w}x{ori_h} → 학습 입력 {out_w}x{out_h}"
              f"   (차선 {len(lanes_raw)}개, 실선 {classes.count(1)} / 점선 {classes.count(2)})",
              font=f_sub, fill=(70, 70, 70), anchor="ma")

    draw.text((out_w // 2, pad_top - 32), "① 내가 라벨링한 GT (.lines.txt 폴리라인 그대로)",
              font=f_sub, fill=(20, 20, 20), anchor="ma")
    draw.text((out_w + gap + out_w // 2, pad_top - 32),
              "② 학습에 쓰이는 confidence map + 리샘플링된 점",
              font=f_sub, fill=(20, 20, 20), anchor="ma")

    y0 = pad_top + out_h + 14
    draw.text((out_w // 2, y0),
              f"원본 라벨 점 개수 합계: {n_raw_pts}개 (라벨링 vertex 그대로, 점 사이 간격 불균일)",
              font=f_small, fill=(60, 60, 60), anchor="ma")
    draw.text((out_w + gap + out_w // 2, y0),
              f"리샘플링 점 개수 합계: {n_resampled_total}개 "
              f"(스플라인 피팅 후 y축 기준 균등 간격 재배치, 겹치는 가우시안은 max로 합성)",
              font=f_small, fill=(60, 60, 60), anchor="ma")
    draw.text((out_w + gap + out_w // 2, y0 + 26),
              f"heatmap 해상도 {hm_w}x{hm_h} (=입력/{HM_DOWN_SCALE}), 가우시안 반경(radius)={GAUSSIAN_RADIUS}px"
              f" (heatmap 기준) → 표시용으로 {HM_DOWN_SCALE}배 확대",
              font=f_small, fill=(60, 60, 60), anchor="ma")
    draw.text((canvas_w // 2, y0 + 60),
              "초록=실선(class 1)  빨강=점선(class 2)   |   두 패널 모두 같은 1280x736 입력 위에 표시",
              font=f_small, fill=(90, 90, 90), anchor="ma")
    draw.text((canvas_w // 2, y0 + 86),
              "코드 출처: projects/plugin/datasets/pipelines/lane_target_collect.py "
              "(ploy_fitting_cube, draw_umich_gaussian, CollectLanePoints.get_targets — 원본 그대로 호출)",
              font=f_small, fill=(130, 130, 130), anchor="ma")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    canvas.save(OUT_PATH)
    print("saved:", OUT_PATH)
    print("raw points per lane:", [len(p) // 2 for p in lanes_raw])
    print("resampled total:", n_resampled_total)


if __name__ == "__main__":
    main()
