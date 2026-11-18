#!/usr/bin/env python3
"""helmet 라벨(클래스 4)과 SAM3 자동 라벨(클래스 0~3)을 합쳐 5클래스 데이터셋을 만든다.

계획 문서: docs/05-오토바이헬멧및다인승/01. 계획.md

두 라벨은 클래스가 겹치지 않으므로 단순 연결로 충분하다.
  helmet 풀 : 4 helmet                                  (사람이 만든 정답)
  SAM3      : 0 person / 1 car / 2 motorcycle / 3 plate  (자동 생성)

검증 항목
  - 두 입력의 stem 집합이 일치하는지
  - 좌표가 [0,1] 범위이고 폭·높이가 양수인지
  - 클래스 ID 가 0~4 밖으로 벗어나지 않는지

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_safety_02_merge_labels.py \
      --helmet_dir /home/hsjeong/workspace/dataset/det/rider-safety/processed/01-helmet-pool \
      --sam3_dir   /home/hsjeong/workspace/dataset/det/rider-safety/processed/02-sam3-auto \
      --out_dir    /home/hsjeong/workspace/dataset/det/rider-safety/processed/03-merged
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate_number", 4: "helmet"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--helmet_dir", required=True, help="01-helmet-pool (images/ labels/ manifest.csv)")
    p.add_argument("--sam3_dir", required=True, help="02-sam3-auto (labels/)")
    p.add_argument("--out_dir", required=True, help="병합 결과 출력 디렉터리")
    p.add_argument("--dry-run", action="store_true", help="집계만 하고 파일을 만들지 않는다")
    return p.parse_args()


def read_boxes(path: Path, expect: set[int]) -> tuple[list[str], list[str]]:
    """라벨 파일을 읽어 (정상 라인, 문제 라인) 으로 나눈다."""
    good, bad = [], []
    if not path.exists():
        return good, bad
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            if line.strip():
                bad.append(f"{path.name}: 필드 {len(parts)}개 — {line[:50]}")
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        if cid not in expect:
            bad.append(f"{path.name}: 예상 밖 클래스 {cid}")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad.append(f"{path.name}: 좌표 범위 이탈 — {line[:50]}")
            continue
        good.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return good, bad


def main():
    args = parse_args()
    helmet_dir, sam3_dir, out_dir = (Path(p).resolve() for p in (args.helmet_dir, args.sam3_dir, args.out_dir))
    img_out, lbl_out = out_dir / "images", out_dir / "labels"

    manifest = list(csv.DictReader((helmet_dir / "manifest.csv").open()))
    stems = [r["stem"] for r in manifest]

    # 입력 정합성부터 확인한다
    h_stems = {p.stem for p in (helmet_dir / "labels").glob("*.txt")}
    s_stems = {p.stem for p in (sam3_dir / "labels").glob("*.txt")}
    if h_stems != s_stems:
        raise SystemExit(
            f"[ERROR] 두 라벨 집합이 다르다. helmet 전용 {len(h_stems - s_stems)}개 / SAM3 전용 {len(s_stems - h_stems)}개"
        )
    if set(stems) != h_stems:
        raise SystemExit(f"[ERROR] manifest({len(stems)})와 라벨({len(h_stems)})이 일치하지 않는다")

    if not args.dry_run:
        for d in (img_out, lbl_out):
            d.mkdir(parents=True, exist_ok=True)

    box = collections.Counter()
    per_ds = collections.defaultdict(collections.Counter)
    problems: list[str] = []
    no_helmet = no_sam3 = 0

    for r in manifest:
        stem, ds = r["stem"], r["source_dataset"].split(".")[0]
        h_lines, h_bad = read_boxes(helmet_dir / "labels" / f"{stem}.txt", {4})
        s_lines, s_bad = read_boxes(sam3_dir / "labels" / f"{stem}.txt", {0, 1, 2, 3})
        problems += h_bad + s_bad
        if not h_lines:
            no_helmet += 1
        if not s_lines:
            no_sam3 += 1

        merged = s_lines + h_lines  # 클래스 순서대로 person→helmet
        for line in merged:
            cid = int(line.split()[0])
            box[cid] += 1
            per_ds[ds][cid] += 1

        if args.dry_run:
            continue

        src = helmet_dir / "images" / f"{stem}.jpg"
        dst = img_out / f"{stem}.jpg"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        (lbl_out / f"{stem}.txt").write_text("\n".join(merged) + "\n")

    total = len(manifest)
    print(f"■ 병합 이미지 {total:,}장\n")
    print("■ 클래스별 박스")
    for c in sorted(box):
        print(f"   {c} {NAMES[c]:<14}{box[c]:>10,}")
    print(f"   {'합계':<16}{sum(box.values()):>10,}   장당 {sum(box.values()) / total:.1f}")

    print(f"\n■ 출처별 장당 평균")
    header = f"{'출처':<28}" + "".join(f"{NAMES[c]:>13}" for c in sorted(NAMES))
    print(header)
    ds_imgs = collections.Counter(r["source_dataset"].split(".")[0] for r in manifest)
    for ds in sorted(ds_imgs, key=lambda k: -ds_imgs[k]):
        n = ds_imgs[ds]
        print(f"{ds:<28}" + "".join(f"{per_ds[ds][c] / n:>13.1f}" for c in sorted(NAMES)))

    print(f"\n■ 검증")
    print(f"   문제 라인          {len(problems)}건")
    print(f"   helmet 없는 이미지 {no_helmet}장")
    print(f"   SAM3 없는 이미지   {no_sam3}장")
    for p in problems[:5]:
        print(f"     - {p}")

    if args.dry_run:
        print("\n[DRY-RUN] 파일을 만들지 않았다.")
        return

    yaml_text = (
        f"# rider-safety 5클래스 병합 데이터셋\n"
        f"# helmet = 사람이 만든 Roboflow 정답 / 나머지 = SAM3 자동 생성\n"
        f"path: {out_dir}\n"
        f"train: images\nval: images\n\n"
        f"nc: 5\nnames:\n" + "".join(f"  {c}: {NAMES[c]}\n" for c in sorted(NAMES))
    )
    (out_dir / "data.yaml").write_text(yaml_text)
    print(f"\n출력: {out_dir}\n  images/ labels/ data.yaml")
    print("  ※ data.yaml 의 train/val 은 임시값이다. 정식 분할은 splits/ 에서 만든다")


if __name__ == "__main__":
    main()
