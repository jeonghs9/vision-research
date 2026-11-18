"""[OpenLane용] OpenLane 분할 목록에서 seed 고정 무작위 N장을 뽑아 2D 차선 라벨 vis 생성.

패널: [원본 | 원본 + 차선 라벨(uv)] 2분할 (전처리 없음), 파일명은 원본 stem 그대로.
 - category 별 색, uv 점을 가공 없이 직선으로 연결, 4px 굵게 (visibility 는 xyz 기준이라 미사용)
 - manifest.txt: 원본 stem <TAB> segment
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

CATS = {
    1: ("white-dash", (120, 210, 255)), 2: ("white-solid", (255, 255, 255)),
    3: ("double-white-dash", (150, 150, 255)), 4: ("double-white-solid", (30, 70, 255)),
    5: ("white-ldash-rsolid", (0, 200, 170)), 6: ("white-lsolid-rdash", (120, 255, 190)),
    7: ("yellow-dash", (255, 235, 0)), 8: ("yellow-solid", (255, 165, 0)),
    9: ("double-yellow-dash", (255, 200, 130)), 10: ("double-yellow-solid", (230, 100, 0)),
    11: ("yellow-ldash-rsolid", (255, 180, 150)), 12: ("yellow-lsolid-rdash", (150, 90, 30)),
    20: ("left-curbside", (255, 0, 0)), 21: ("right-curbside", (0, 220, 0)),
    0: ("unknown", (150, 150, 150)),
}
THICK, SHIFT = 4, 4

# 2클래스 모드: 실선/점선으로 합침, 연석(20/21)·unknown(0)은 제외(그리지 않음). 색은 드론 GANet 2클래스(문서 8)와 동일
TWO_CLS = {**{c: "solid" for c in (2, 4, 8, 10, 5, 6, 11, 12)}, **{c: "dashed" for c in (1, 3, 7, 9)}}
TWO_COLOR = {"solid": (0, 220, 0), "dashed": (220, 0, 0)}  # RGB: 실선 초록, 점선 빨강


def bgr(c):
    return (c[2], c[1], c[0])


def draw_lane(img, lane, mode="cat"):
    u, v = lane["uv"]
    if len(u) < 2:
        return
    if mode == "2cls":
        cls = TWO_CLS.get(lane["category"])
        if cls is None:  # 연석·unknown 제외
            return
        color = TWO_COLOR[cls]
    else:
        color = CATS[lane["category"]][1]
    pts = np.round(np.stack([u, v], 1) * (1 << SHIFT)).astype(np.int32)
    cv2.polylines(img, [pts], False, bgr(color), THICK, cv2.LINE_AA, SHIFT)


def put_label(img, text, org):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, 1.1, 2)
    x, y = org
    cv2.rectangle(img, (x - 8, y - h - 10), (x + w + 8, y + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


def legend(img, counts):
    font = cv2.FONT_HERSHEY_SIMPLEX
    keys = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 0)
    row, pad, sw = 28, 10, 70
    w, h = 360 + sw, row * (len(keys) + 1) + 2 * pad
    H, W = img.shape[:2]
    x0, y0 = W - w - 12, H - h - 12
    img[y0:y0 + h, x0:x0 + w] = (img[y0:y0 + h, x0:x0 + w] * 0.3).astype(np.uint8)
    for i, k in enumerate(keys):
        y = y0 + pad + row * i + row // 2
        cv2.line(img, (x0 + pad, y), (x0 + pad + sw, y), bgr(CATS[k][1]), THICK, cv2.LINE_AA)
        cv2.putText(img, f"{k:>2} {CATS[k][0]} (x{counts.get(k, 0)})", (x0 + 2 * pad + sw, y + 6), font, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    y = y0 + pad + row * len(keys) + row // 2
    cv2.putText(img, f"lines drawn {THICK} px thick", (x0 + pad, y + 6), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def legend_2cls(img, counts, n_excluded):
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = [("solid (실선: 2,4,8,10 + 혼합 5,6,11,12)", "solid", counts.get("solid", 0)),
            ("dashed (점선: 1,3,7,9)", "dashed", counts.get("dashed", 0))]
    row, pad, sw = 30, 10, 70
    w, h = 560, row * (len(rows) + 2) + 2 * pad
    H, W = img.shape[:2]
    x0, y0 = W - w - 12, H - h - 12
    img[y0:y0 + h, x0:x0 + w] = (img[y0:y0 + h, x0:x0 + w] * 0.3).astype(np.uint8)
    for i, (name, k, n) in enumerate(rows):
        y = y0 + pad + row * i + row // 2
        cv2.line(img, (x0 + pad, y), (x0 + pad + sw, y), bgr(TWO_COLOR[k]), THICK, cv2.LINE_AA)
        cv2.putText(img, f"{name} x{n}", (x0 + 2 * pad + sw, y + 6), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    y = y0 + pad + row * len(rows) + row // 2
    cv2.putText(img, f"excluded, not drawn (curb 20/21, unknown 0): x{n_excluded}", (x0 + pad, y + 6), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += row
    cv2.putText(img, f"lines drawn {THICK} px thick", (x0 + pad, y + 6), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OpenLane raw/ 폴더")
    ap.add_argument("--split-list", required=True, help="분할 목록(예: splits/test.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["cat", "2cls"], default="cat", help="cat=category별 색, 2cls=실선/점선 2클래스(연석 제외)")
    a = ap.parse_args()

    lines = sorted(x.strip() for x in open(a.split_list) if x.strip())
    picks = random.Random(a.seed).sample(lines, min(a.n, len(lines)))
    os.makedirs(a.out, exist_ok=True)
    for old in os.listdir(a.out):
        if old.endswith(".jpg") or old == "manifest.txt":
            os.remove(os.path.join(a.out, old))
    manifest, used_seg = [], set()
    for p in sorted(picks):
        _, folder, seg, fn = p.split("/")
        stem = fn[:-4]
        img = cv2.imread(os.path.join(a.root, p))
        lab = json.load(open(os.path.join(a.root, "lane3d_1000", folder, seg, stem + ".json")))
        right, counts = img.copy(), {}
        n_drawn = n_excl = 0
        for ln in lab["lane_lines"]:
            if a.mode == "2cls":
                k = TWO_CLS.get(ln["category"])
                if k is None:
                    n_excl += 1
                else:
                    counts[k] = counts.get(k, 0) + 1
                    n_drawn += 1
            else:
                counts[ln["category"]] = counts.get(ln["category"], 0) + 1
            draw_lane(right, ln, a.mode)
        if a.mode == "2cls":
            legend_2cls(right, counts, n_excl)
        else:
            legend(right, counts)
        left = img.copy()
        put_label(left, "image", (24, 52))
        if a.mode == "2cls":
            put_label(right, f"lane label 2cls ({n_drawn} lanes, {n_excl} excluded)", (24, 52))
        else:
            put_label(right, f"lane label uv ({len(lab['lane_lines'])} lanes)", (24, 52))
        panel = np.concatenate([left, np.full((img.shape[0], 10, 3), 255, np.uint8), right], 1)
        cv2.imwrite(os.path.join(a.out, stem + ".jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append(f"{stem}\t{seg}")
        used_seg.add(seg)
    with open(os.path.join(a.out, "manifest.txt"), "w") as f:
        f.write("frame_stem\tsegment\n" + "\n".join(manifest) + "\n")
    print(f"saved {len(picks)} (segments {len(used_seg)}) -> {a.out}")


if __name__ == "__main__":
    main()
