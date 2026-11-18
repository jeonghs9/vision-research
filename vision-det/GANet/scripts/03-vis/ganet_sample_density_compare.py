"""리샘플링 점 개수(45)를 바꾸면 confidence map이 어떻게 달라지는지 비교.

같은 함수(convert_scale/ploy_fitting_cube/draw_umich_gaussian)를 그대로 쓰되
점 개수(sample_num)만 바꿔서 세 가지 경우를 나란히 렌더링한다.
"""
import importlib.util
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE_DIR = "/home/hsjeong/workspace/vision-det/GANet/source"
DATA_ROOT = "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/culane_lane_ganet"
STEM = "images/19700101_000157_T__000290"
OUT_PATH = ("/home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/"
            "signed-distance-validation/GANet_점개수_비교.png")

IMG_SCALE = (1280, 736)
HM_DOWN_SCALE = 8
GAUSSIAN_RADIUS = 2
COLOR_BGR = (0, 0, 220)  # 점선(class2) 빨강 하나로 통일해서 비교

spec = importlib.util.spec_from_file_location(
    "lane_target_collect",
    os.path.join(SOURCE_DIR, "projects/plugin/datasets/pipelines/lane_target_collect.py"),
)
ltc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ltc)


def main():
    img = cv2.imread(os.path.join(DATA_ROOT, STEM + ".jpg"))
    ori_h, ori_w = img.shape[:2]
    out_w, out_h = IMG_SCALE
    img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    sx, sy = out_w / ori_w, out_h / ori_h

    with open(os.path.join(DATA_ROOT, STEM + ".lines.txt")) as f:
        raw_lanes = [[float(v) for v in line.strip().split(' ') if v] for line in f if line.strip()]

    # 제일 긴(세로로 긴) 차선 하나만 골라서 비교에 사용
    def y_span(flat):
        ys = flat[1::2]
        return max(ys) - min(ys)

    lane = max(raw_lanes, key=y_span)
    pts_resized = [(lane[2 * i] * sx, lane[2 * i + 1] * sy) for i in range(len(lane) // 2)]
    gt_flat = [c for xy in pts_resized for c in xy]

    hm_h, hm_w = out_h // HM_DOWN_SCALE, out_w // HM_DOWN_SCALE

    # 비교할 크롭 영역 (이 차선 주변만)
    xs = [p[0] for p in pts_resized]
    ys = [p[1] for p in pts_resized]
    x0, x1 = max(int(min(xs)) - 60, 0), min(int(max(xs)) + 60, out_w)
    y0, y1 = max(int(min(ys)) - 20, 0), min(int(max(ys)) + 20, out_h)

    def render(sample_num):
        kpts_hm = np.zeros((hm_h, hm_w), np.float32)
        pts = ltc.convert_scale(gt_flat, HM_DOWN_SCALE)
        pts = ltc.ploy_fitting_cube(pts, hm_h, hm_w, sample_num)
        pts_used = []
        if pts is not None:
            pts = ltc.clamp_line(pts, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)
        if pts is not None:
            for p in pts:
                pt_int = (int(p[0]), int(p[1]))
                ltc.draw_umich_gaussian(kpts_hm, pt_int, radius=GAUSSIAN_RADIUS)
                pts_used.append(pt_int)

        heat_up = cv2.resize(kpts_hm, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        base = img_resized.copy().astype(np.float32)
        color = np.array(COLOR_BGR, dtype=np.float32)
        alpha = np.clip(heat_up, 0, 1)[..., None]
        blended = base * (1 - 0.7 * alpha) + color[None, None, :] * (0.7 * alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        for (px, py) in pts_used:
            X, Y = int(px * HM_DOWN_SCALE), int(py * HM_DOWN_SCALE)
            cv2.circle(blended, (X, Y), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(blended, (X, Y), 2, COLOR_BGR, -1, lineType=cv2.LINE_AA)
        crop = blended[y0:y1, x0:x1]
        return crop, len(pts_used)

    cases = [(8, "점 8개 (너무 적음)"), (45, "점 45개 (현재 코드)"), (150, "점 150개 (너무 많음)")]
    crops = []
    counts = []
    for n, _ in cases:
        c, used = render(n)
        crops.append(c)
        counts.append(used)

    gap = 24
    ch, cw = crops[0].shape[:2]
    pad_top, pad_bottom = 120, 70
    min_cw = 330  # 라벨 텍스트가 들어갈 최소 폭
    disp_cw = max(cw, min_cw)
    canvas_w = disp_cw * 3 + gap * 2
    canvas_h = ch + pad_top + pad_bottom
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    for i, c in enumerate(crops):
        pil_c = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        x_off = i * (disp_cw + gap) + (disp_cw - cw) // 2
        canvas.paste(pil_c, (x_off, pad_top))

    draw = ImageDraw.Draw(canvas)
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    f_title = ImageFont.truetype(cjk, 24)
    f_sub = ImageFont.truetype(cjk, 16)
    f_small = ImageFont.truetype(cjk, 14)

    draw.text((canvas_w // 2, 18), "리샘플링 점 개수를 바꾸면 confidence map이 어떻게 달라지나",
              font=f_title, fill=(20, 20, 20), anchor="ma")

    labels = [
        [f"점 {counts[0]}개 (너무 적음)", "간격 > 가우시안 지름(5px)", "→ 중간이 끊긴 점선처럼 됨"],
        [f"점 {counts[1]}개 (현재 코드값)", "간격 < 가우시안 지름", "→ 끊김 없이 이어진 띠"],
        [f"점 {counts[2]}개 (너무 많음)", "같은 정수 픽셀에 중복 적중", "→ 결과는 동일, 연산만 낭비"],
    ]
    for i, lines in enumerate(labels):
        cx = i * (disp_cw + gap) + disp_cw // 2
        draw.text((cx, pad_top - 72), lines[0], font=f_sub, fill=(20, 20, 20), anchor="ma")
        draw.text((cx, pad_top - 48), lines[1], font=f_small, fill=(90, 90, 90), anchor="ma")
        draw.text((cx, pad_top - 28), lines[2], font=f_small, fill=(180, 30, 30), anchor="ma")

    draw.text((canvas_w // 2, pad_top + ch + 20),
              f"같은 차선(가장 긴 점선 차선), 같은 gaussian radius=2px(heatmap 기준) 기준  |  "
              f"heatmap 해상도 {hm_w}x{hm_h}, 표시는 {HM_DOWN_SCALE}배 확대", font=f_small,
              fill=(90, 90, 90), anchor="ma")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    canvas.save(OUT_PATH)
    print("saved:", OUT_PATH)
    print("point counts used:", counts)


if __name__ == "__main__":
    main()
