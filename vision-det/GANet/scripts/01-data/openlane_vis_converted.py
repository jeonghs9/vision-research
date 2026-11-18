"""[OpenLane용] 변환 결과(processed/culane_lane_ganet)의 .lines.txt/.cls.txt를 이미지에 다시 그려 확인하는 vis.

패널: [원본 | 변환 라벨] 2분할 (이미지는 변환하지 않으므로 2분할), 파일명은 원본 stem 그대로.
실선(cls 1) 초록, 점선(cls 2) 빨강, 4px 굵게. 목록에서 seed 고정 무작위 N장 (openlane_vis_split.py와 같은 규칙 → 같은 프레임).

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/clrkdnet_cu113/bin/python -I \
    /home/hsjeong/workspace/vision-det/GANet/scripts/01-data/openlane_vis_converted.py \
    --data-root /home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet \
    --split test --out /home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/vis --n 30 --seed 0
"""
import argparse
import os
import random

import cv2
import numpy as np

COLOR = {1: (0, 220, 0), 2: (0, 0, 220)}  # BGR: 실선 초록, 점선 빨강
THICK, SHIFT = 4, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lines = sorted(x.strip() for x in open(os.path.join(a.data_root, "list", f"{a.split}.txt")) if x.strip())
    picks = sorted(random.Random(a.seed).sample(lines, min(a.n, len(lines))))
    os.makedirs(a.out, exist_ok=True)
    for old in os.listdir(a.out):
        if old.endswith(".jpg"):
            os.remove(os.path.join(a.out, old))
    stats = []
    for p in picks:
        stem = os.path.join(a.data_root, p[:-4])
        img = cv2.imread(stem + ".jpg")
        right = img.copy()
        lanes = [x.split() for x in open(stem + ".lines.txt") if x.strip()]
        cls = [int(x) for x in open(stem + ".cls.txt").read().split()]
        assert len(lanes) == len(cls), p
        cnt = {1: 0, 2: 0}
        for q, c in zip(lanes, cls):
            pts = np.round(np.asarray(q, float).reshape(-1, 2) * (1 << SHIFT)).astype(np.int32)
            cv2.polylines(right, [pts], False, COLOR[c], THICK, cv2.LINE_AA, SHIFT)
            cnt[c] += 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        for im, t in ((img, "image"), (right, f"converted lines.txt (solid {cnt[1]}, dashed {cnt[2]})")):
            (w, h), _ = cv2.getTextSize(t, font, 1.1, 2)
            cv2.rectangle(im, (16, 52 - h - 10), (32 + w, 62), (0, 0, 0), -1)
            cv2.putText(im, t, (24, 52), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(right, f"solid=green dashed=red | lines drawn {THICK} px thick", (24, img.shape[0] - 24), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panel = np.concatenate([img, np.full((img.shape[0], 10, 3), 255, np.uint8), right], 1)
        cv2.imwrite(os.path.join(a.out, os.path.basename(stem) + ".jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        stats.append((cnt[1], cnt[2]))
    print(f"saved {len(picks)} -> {a.out} | solid {sum(s for s,_ in stats)} dashed {sum(d for _,d in stats)} | frames with no lanes {sum(1 for s,d in stats if s+d==0)}")


if __name__ == "__main__":
    main()
