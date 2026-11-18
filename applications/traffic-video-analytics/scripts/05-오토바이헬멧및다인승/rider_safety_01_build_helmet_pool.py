#!/usr/bin/env python3
"""Roboflow 원본 3종에서 helmet 라벨만 뽑아 단일 풀로 통합한다.

계획 문서: docs/05-오토바이헬멧및다인승/01. 계획.md

수행하는 일
  1. raw/ 아래 각 Roboflow 데이터셋의 train/valid/test 를 모두 순회한다
  2. 라벨에서 helmet 클래스만 남기고 우리 스키마 ID(4)로 리맵한다
     - driver / bicyclist / motorbike / car 등 나머지 원본 라벨은 전부 버린다.
       이들은 "사람 + 오토바이" 를 통째로 감싼 박스라 우리 정의와 다르며,
       person / car / motorcycle / plate_number 는 이후 SAM3 가 일괄 생성한다
  3. helmet 박스가 없는 이미지는 제외한다 (--keep-no-helmet 으로 유지 가능)
  4. Roboflow 분할(train/valid/test)을 해체해 하나의 풀로 합친다
     - 우리 분할은 이후 splits/ 에서 새로 만든다
  5. 이미지는 심볼릭 링크로 연결한다 (raw/ 원본 보존, 용량 절약)

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_safety_01_build_helmet_pool.py \
      --raw_dir /home/hsjeong/workspace/dataset/det/rider-safety/raw \
      --out_dir /home/hsjeong/workspace/dataset/det/rider-safety/processed/01-helmet-pool
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml

HELMET_ID = 4  # 우리 스키마: 0 person, 1 car, 2 motorcycle, 3 plate_number, 4 helmet
SPLITS = ("train", "valid", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", required=True, help="Roboflow 데이터셋들이 있는 raw 디렉터리")
    p.add_argument("--out_dir", required=True, help="통합 풀 출력 디렉터리")
    p.add_argument("--exclude", nargs="*", default=[], help="제외할 raw 하위 폴더명 (부분 일치)")
    p.add_argument(
        "--keep-no-helmet",
        action="store_true",
        help="helmet 박스가 없는 이미지도 유지한다 (negative 로 활용). 기본은 제외",
    )
    p.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 이미지를 복사한다")
    p.add_argument("--dry-run", action="store_true", help="집계만 하고 파일을 만들지 않는다")
    return p.parse_args()


def read_labels(path: Path) -> tuple[list[tuple[int, str]], int]:
    """YOLO 라벨 파일을 (class_id, "cx cy w h") 목록으로 읽는다.

    helmet-detection v18 에는 bbox(5필드) 와 폴리곤(클래스 + 좌표쌍, 홀수 필드)이
    섞여 있다. 폴리곤은 버리지 않고 **외접 사각형으로 변환**한다.
    반환값의 두 번째 항목은 변환한 폴리곤 개수.
    """
    rows: list[tuple[int, str]] = []
    n_poly = 0
    if not path.exists():
        return rows, n_poly
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        if len(parts) == 5:
            rows.append((cid, " ".join(parts[1:])))
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2:  # 좌표쌍이 맞지 않으면 신뢰할 수 없다
            continue
        xs, ys = coords[0::2], coords[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        rows.append((cid, f"{(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"))
        n_poly += 1
    return rows, n_poly


def main():
    args = parse_args()
    # 심볼릭 링크는 링크 위치 기준으로 해석되므로 반드시 절대 경로로 건다
    raw_dir, out_dir = Path(args.raw_dir).resolve(), Path(args.out_dir).resolve()
    img_out, lbl_out = out_dir / "images", out_dir / "labels"

    datasets = sorted(d for d in raw_dir.iterdir() if d.is_dir() and (d / "data.yaml").exists())
    datasets = [d for d in datasets if not any(x.lower() in d.name.lower() for x in args.exclude)]
    if not datasets:
        raise SystemExit(f"[ERROR] {raw_dir} 에서 data.yaml 을 가진 데이터셋을 찾지 못했다")

    if not args.dry_run:
        for d in (img_out, lbl_out):
            d.mkdir(parents=True, exist_ok=True)

    manifest = []
    totals = {"seen": 0, "kept": 0, "dropped_no_helmet": 0, "helmet_boxes": 0, "dropped_boxes": 0, "poly": 0}

    for ds in datasets:
        names = yaml.safe_load((ds / "data.yaml").read_text())["names"]
        if "helmet" not in names:
            print(f"[SKIP] {ds.name}: helmet 클래스가 없다 (names={names})")
            continue
        src_id = names.index("helmet")

        stat = {"seen": 0, "kept": 0, "dropped": 0, "boxes": 0, "dropped_boxes": 0, "poly": 0}
        for split in SPLITS:
            img_dir, lbl_dir = ds / split / "images", ds / split / "labels"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                stat["seen"] += 1
                rows, n_poly = read_labels(lbl_dir / f"{img.stem}.txt")
                helmets = [coord for cid, coord in rows if cid == src_id]
                stat["dropped_boxes"] += len(rows) - len(helmets)
                stat["poly"] += n_poly

                if not helmets and not args.keep_no_helmet:
                    stat["dropped"] += 1
                    continue

                stat["kept"] += 1
                stat["boxes"] += len(helmets)
                manifest.append(
                    {
                        "stem": img.stem,
                        "source_dataset": ds.name,
                        "source_split": split,
                        "helmet_boxes": len(helmets),
                        "source_image": str(img),
                    }
                )
                if args.dry_run:
                    continue

                dst_img = img_out / img.name
                if dst_img.is_symlink() or dst_img.exists():
                    dst_img.unlink()
                if args.copy:
                    shutil.copy2(img, dst_img)
                else:
                    dst_img.symlink_to(img)
                text = "".join(f"{HELMET_ID} {coord}\n" for coord in helmets)
                (lbl_out / f"{img.stem}.txt").write_text(text)

        print(
            f"[{ds.name}]\n"
            f"    이미지 {stat['seen']:,} → 유지 {stat['kept']:,} / 제외 {stat['dropped']:,}\n"
            f"    helmet 박스 {stat['boxes']:,} · 버린 원본 박스 {stat['dropped_boxes']:,}"
            + (f"\n    폴리곤 → bbox 변환 {stat['poly']:,}" if stat["poly"] else "")
        )
        totals["seen"] += stat["seen"]
        totals["kept"] += stat["kept"]
        totals["dropped_no_helmet"] += stat["dropped"]
        totals["helmet_boxes"] += stat["boxes"]
        totals["dropped_boxes"] += stat["dropped_boxes"]
        totals["poly"] += stat["poly"]

    print(
        f"\n=== 합계 ===\n"
        f"  원본 이미지      {totals['seen']:,}\n"
        f"  풀에 담긴 이미지  {totals['kept']:,}\n"
        f"  helmet 없어 제외 {totals['dropped_no_helmet']:,}\n"
        f"  helmet 박스      {totals['helmet_boxes']:,}\n"
        f"  버린 원본 박스    {totals['dropped_boxes']:,}  (SAM3 가 대체 생성)\n"
        f"  폴리곤 → bbox     {totals['poly']:,}"
    )

    if args.dry_run:
        print("\n[DRY-RUN] 파일을 만들지 않았다.")
        return

    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "source_dataset", "source_split", "helmet_boxes", "source_image"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\n출력: {out_dir}\n  images/ labels/ manifest.csv")


if __name__ == "__main__":
    main()
