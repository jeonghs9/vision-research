#!/usr/bin/env python3
"""Prepare AIHUB vehicle category data for image classification.

Output layout:
  split/
    train/<class_name>/...
    val/<class_name>/...
    test/<class_name>/...

The existing AIHUB Validation split is used as test for classes that have it.
Validation images are sampled from Training. Classes with no original
Validation data are split from Training into 7:2:1 train/val/test.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


DEFAULT_SOURCE = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_CATEGORY/original")
DEFAULT_OUTPUT = Path("/home/hsjeong/workspace/dataset/AIHUB/CAR_CATEGORY/split")

EXCLUDED_CLASSES = {"이륜차"}
CLASSES = [
    "SUV",
    "버스",
    "세단",
    "승합",
    "컨버터블",
    "쿠페",
    "트럭",
    "특수",
    "해치백",
    "화물",
]

CLASS_NAME_MAP = {
    "SUV": "suv",
    "버스": "bus",
    "세단": "sedan",
    "승합": "van",
    "컨버터블": "convertible",
    "쿠페": "coupe",
    "트럭": "truck",
    "특수": "special",
    "해치백": "hatchback",
    "화물": "cargo",
}


@dataclass(frozen=True)
class ImageItem:
    src: Path
    rel: Path

    @property
    def flat_name(self) -> str:
        return "__".join(self.rel.parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test classification split for AIHUB vehicle categories."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Original AIHUB root directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output split directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--val-ratio-from-train",
        type=float,
        default=0.20,
        help="For classes with original Validation data, sample this ratio from Training into val.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing train/val/test directories in output before copying.",
    )
    parser.add_argument(
        "--preserve-subdirs",
        action="store_true",
        help="Keep original brand/model subdirectories under each class. Do not use this for Ultralytics classification.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned counts without copying files.")
    return parser.parse_args()


def source_class_dir(source: Path, split: str, class_name: str) -> Path:
    return source / split / f"[원천]{class_name}" / class_name


def collect_images(class_dir: Path) -> list[ImageItem]:
    if not class_dir.exists():
        return []
    images = []
    for path in sorted(class_dir.rglob("*.jpg")):
        images.append(ImageItem(src=path, rel=path.relative_to(class_dir)))
    return images


def ensure_output_ready(output: Path, overwrite: bool, dry_run: bool) -> None:
    split_dirs = [output / "train", output / "val", output / "test"]
    existing = [p for p in split_dirs if p.exists() and any(p.iterdir())]
    if existing and not overwrite:
        names = ", ".join(str(p) for p in existing)
        raise SystemExit(
            f"Output split directories are not empty: {names}\n"
            "Use --overwrite to remove train/val/test and recreate them."
        )
    if dry_run:
        return
    if overwrite:
        for p in split_dirs:
            if p.exists():
                shutil.rmtree(p)
        for cache in output.glob("*.cache"):
            cache.unlink()
    for split in ("train", "val", "test"):
        (output / split).mkdir(parents=True, exist_ok=True)


def split_items(
    train_items: list[ImageItem],
    valid_items: list[ImageItem],
    rng: random.Random,
    val_ratio_from_train: float,
) -> tuple[list[ImageItem], list[ImageItem], list[ImageItem]]:
    train_pool = train_items[:]
    rng.shuffle(train_pool)

    if valid_items:
        val_count = round(len(train_pool) * val_ratio_from_train)
        val = train_pool[:val_count]
        train = train_pool[val_count:]
        test = valid_items[:]
        return train, val, test

    total = len(train_pool)
    test_count = round(total * 0.10)
    val_count = round(total * 0.20)
    test = train_pool[:test_count]
    val = train_pool[test_count : test_count + val_count]
    train = train_pool[test_count + val_count :]
    return train, val, test


def copy_items(
    items: list[ImageItem],
    output: Path,
    split: str,
    class_name: str,
    preserve_subdirs: bool,
) -> None:
    progress = tqdm(items, desc=f"{split}/{class_name}", unit="img", leave=False)
    for item in progress:
        rel = item.rel if preserve_subdirs else Path(item.flat_name)
        dst = output / split / class_name / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.src, dst)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")

    classes = [class_name for class_name in CLASSES if class_name not in EXCLUDED_CLASSES]
    rng = random.Random(args.seed)
    ensure_output_ready(output, args.overwrite, args.dry_run)

    plan: dict[str, dict[str, list[ImageItem]]] = {}
    print("Planning split...")
    for class_name in classes:
        train_src = collect_images(source_class_dir(source, "Training", class_name))
        valid_src = collect_images(source_class_dir(source, "Validation", class_name))
        train, val, test = split_items(train_src, valid_src, rng, args.val_ratio_from_train)
        plan[class_name] = {"train": train, "val": val, "test": test}
        print(
            f"{class_name}->{CLASS_NAME_MAP[class_name]}: train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"source_train={len(train_src):,}, source_valid={len(valid_src):,}"
        )

    totals = {
        split: sum(len(plan[class_name][split]) for class_name in classes)
        for split in ("train", "val", "test")
    }
    total = sum(totals.values())
    print(
        f"TOTAL: train={totals['train']:,}, val={totals['val']:,}, test={totals['test']:,}, "
        f"all={total:,}"
    )

    if args.dry_run:
        print("Dry run only. No files copied.")
        return

    print(f"Copying files to: {output}")
    for class_name in tqdm(classes, desc="classes", unit="class"):
        for split in ("train", "val", "test"):
            copy_items(plan[class_name][split], output, split, CLASS_NAME_MAP[class_name], args.preserve_subdirs)

    print("Done.")


if __name__ == "__main__":
    main()
