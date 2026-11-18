#!/usr/bin/env python3
"""SAM3 6클래스 라벨을 이미지 위에 그려 검수용 시각화를 만든다.

계획: docs/05-오토바이헬멧및다인승/

워크스페이스 규칙상 분할·라벨을 만들 때마다 눈으로 확인할 시각화가 필요하다.
라벨 변환은 조용히 틀리고(클래스 ID 밀림, 좌표 뒤집힘) 학습도 그대로 돌아간다.

**볼 때 확인할 것**
  - 클래스 ID 가 맞는가 (helmet 이 머리에, motorcycle 이 오토바이에)
  - 2인 탑승에서 person 이 2개 잡혔는가
  - 눈에 보이는데 라벨이 없는 객체는 없는가 (SAM3 부분 실패)

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_safety_04_vis_labels.py \
      --images /home/hsjeong/workspace/dataset/det/rider-safety/processed/04-drone-frames/images \
      --labels /home/hsjeong/workspace/dataset/det/rider-safety/processed/05-sam3-6cls/labels \
      --out    /home/hsjeong/workspace/dataset/det/rider-safety/processed/05-sam3-6cls/vis \
      --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate", 4: "helmet", 5: "head"}
COLORS = {
    0: (0, 255, 95),      # person      연두
    1: (0, 200, 255),     # car         하늘
    2: (255, 176, 0),     # motorcycle  주황
    3: (255, 60, 224),    # plate       자홍
    4: (255, 45, 85),     # helmet      빨강 (굵게)
    5: (0, 212, 255),     # head        청록 (굵게)
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100, help="시각화할 장수 (무작위)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=2, help="박스 선 두께 (helmet/head 는 +1)")
    p.add_argument("--label_text", action="store_true", help="클래스 이름을 박스 위에 쓴다")
    p.add_argument("--scale", type=float, default=1.0, help="출력 크기 배율 (0.5 면 절반)")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir, lbl_dir, out_dir = Path(args.images), Path(args.labels), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 라벨이 있는 이미지만 대상으로 한다
    pairs = []
    for ip in sorted(img_dir.iterdir()):
        if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lp = lbl_dir / f"{ip.stem}.txt"
        if lp.exists():
            pairs.append((ip, lp))
    if not pairs:
        raise SystemExit(f"[ERROR] 라벨이 있는 이미지를 찾지 못했다: {lbl_dir}")

    rng = random.Random(args.seed)
    picked = rng.sample(pairs, min(args.n, len(pairs)))
    print(f"[INFO] 라벨 보유 {len(pairs):,}장 중 {len(picked)}장 시각화 (seed {args.seed})")

    total = collections.Counter()
    per_src = collections.Counter()
    empty = 0
    for ip, lp in picked:
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        n_box = 0
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            box = ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H)
            lw = args.width + 1 if cid in (4, 5) else args.width
            d.rectangle(box, outline=COLORS[cid], width=lw)
            if args.label_text:
                d.text((box[0], max(12, box[1] - 13)), NAMES[cid], fill=COLORS[cid])
            total[cid] += 1
            n_box += 1
        if n_box == 0:
            empty += 1
        per_src[ip.stem.split("_")[0]] += 1
        if args.scale != 1.0:
            im = im.resize((int(W * args.scale), int(H * args.scale)), Image.LANCZOS)
        im.save(out_dir / f"{ip.stem}_vis.jpg", quality=92)

    n = len(picked)
    print(f"\n■ 시각화 {n}장 클래스별 박스")
    print(f"{'클래스':<14}{'박스':>9}{'장당':>9}")
    print("-" * 32)
    for cid in sorted(NAMES):
        print(f"{NAMES[cid]:<14}{total[cid]:>9,}{total[cid] / n:>9.1f}")
    print("-" * 32)
    print(f"{'합계':<14}{sum(total.values()):>9,}{sum(total.values()) / n:>9.1f}")
    print(f"\n라벨 0개인 이미지: {empty}장")
    print("출처별: " + "  ".join(f"{k} {v}" for k, v in sorted(per_src.items())))
    print(f"\n출력: {out_dir}")


if __name__ == "__main__":
    main()
