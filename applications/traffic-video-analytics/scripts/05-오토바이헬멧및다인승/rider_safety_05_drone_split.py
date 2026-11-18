#!/usr/bin/env python3
"""드론 SAM3 6클래스 데이터를 영상 단위로 train/val/test 분할한다.

계획: docs/05-오토바이헬멧및다인승/

**영상 단위로 나누는 이유**
  같은 영상에서 프레임을 무작위로 나누면 인접 프레임(0.2초 차이)이 train 과 test 에
  흩어져 test 가 사실상 train 이 된다. 02 문서에서 겪은 augmentation 누수와 같은 원리다.
  영상 자체를 통째로 홀드아웃해야 "처음 보는 장면에서 몇 점인가" 를 잴 수 있다.

  001 학습의 근본 문제가 val/test 를 학습과 같은 도메인(Vehicle v3)에서 뽑은 것이었다
  (06 문서). 이번에는 실제 배포 대상 영상을 val/test 로 쓴다.

배정
  train  162756 · 163257 · 164259   4,505장
  val    163758                     1,504장   ← best.pt 선택 기준
  test   164800                       342장   ← 최종 평가, 한 번만 본다

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_safety_05_drone_split.py \
      --images /home/hsjeong/workspace/dataset/det/rider-safety/processed/04-drone-frames/images \
      --labels /home/hsjeong/workspace/dataset/det/rider-safety/processed/05-sam3-6cls/labels \
      --out    /home/hsjeong/workspace/dataset/det/rider-safety/splits/002-drone-6cls
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate_number", 4: "helmet", 5: "head"}
ASSIGN = {
    "162756": "train", "163257": "train", "164259": "train",
    "163758": "val",
    "164800": "test",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require_all", action="store_true", help="라벨이 없는 이미지가 있으면 중단")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir, lbl_dir, out = Path(args.images), Path(args.labels), Path(args.out)

    pairs, missing = [], []
    for ip in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / f"{ip.stem}.txt"
        (pairs if lp.exists() else missing).append((ip, lp))
    if missing:
        msg = f"[{'ERROR' if args.require_all else 'WARN'}] 라벨 없는 이미지 {len(missing):,}장"
        print(msg)
        if args.require_all:
            raise SystemExit(1)

    for sp in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    stats = {sp: collections.Counter() for sp in ("train", "val", "test")}
    counts = collections.Counter()
    rows = []
    for ip, lp in pairs:
        vid = ip.stem.split("_")[0]
        sp = ASSIGN.get(vid)
        if sp is None:
            continue
        for sub, src in (("images", ip), ("labels", lp)):
            dst = out / sub / sp / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                stats[sp][int(parts[0])] += 1
        counts[sp] += 1
        rows.append({"stem": ip.stem, "video": vid, "split": sp})

    (out / "data.yaml").write_text(
        "# 드론 SAM3 6클래스 — 영상 단위 분할\n"
        "# train 162756·163257·164259 / val 163758 / test 164800\n"
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 6\nnames:\n" + "".join(f"  {c}: {NAMES[c]}\n" for c in sorted(NAMES))
    )
    with (out / "groups.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "video", "split"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'split':<8}{'이미지':>8}" + "".join(f"{NAMES[c]:>13}" for c in sorted(NAMES)))
    print("-" * 94)
    for sp in ("train", "val", "test"):
        print(f"{sp:<8}{counts[sp]:>8,}" + "".join(f"{stats[sp][c]:>13,}" for c in sorted(NAMES)))
    print(f"\n출력: {out}\n  images/ labels/ data.yaml groups.csv")


if __name__ == "__main__":
    main()
