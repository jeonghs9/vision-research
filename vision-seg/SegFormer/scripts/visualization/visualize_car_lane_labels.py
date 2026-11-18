#!/usr/bin/env python3
"""Overlay AIHUB CAR_LANE JSON annotations, including vertex order."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    # Visualization colors identify the line pattern, independently of the
    # annotated lane paint color: solid=green, dotted=yellow (BGR).
    ("traffic_lane", None, "solid"): (0, 255, 0),
    ("traffic_lane", None, "dotted"): (0, 255, 255),
    ("stop_line", None, None): (0, 0, 255),
    ("crosswalk", None, None): (0, 255, 0),
}


def annotation_style(annotation):
    attrs = {item["code"]: item["value"] for item in annotation.get("attributes", [])}
    label_class = annotation["class"]
    color = COLORS.get(
        (label_class, None, attrs.get("lane_type")),
        COLORS.get((label_class, None, None), (255, 0, 255)),
    )
    if label_class == "traffic_lane":
        label = f"{attrs.get('lane_color', '?')} {attrs.get('lane_type', '?')}"
    else:
        label = label_class
    return color, label


def draw_annotation(image, annotation):
    points = np.asarray([(item["x"], item["y"]) for item in annotation["data"]], dtype=np.int32)
    if len(points) < 2:
        return
    color, label = annotation_style(annotation)
    closed = annotation["category"] == "polygon"
    cv2.polylines(image, [points], closed, color, thickness=4, lineType=cv2.LINE_AA)

    # Orange-red dots and 0-based indices make the serialized connection order explicit.
    for index, (x, y) in enumerate(points):
        cv2.circle(image, (int(x), int(y)), 6, (30, 30, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image, str(index), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, str(index), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    x, y = points[0]
    cv2.putText(image, label, (int(x) + 8, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (int(x) + 8, int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0,
                        help="0-based index of the first sorted label to process")
    parser.add_argument("--limit", type=int,
                        help="Maximum number of sorted labels to process")
    args = parser.parse_args()

    # AIHUB archives use both the conventional ``원천데이터/라벨링데이터``
    # layout and extracted archive names such as ``[원천]c_.../[라벨]c_...``.
    # Locate the roots by their role so either layout works without staging files.
    directories = [path for path in args.input.iterdir() if path.is_dir()]
    image_root = next(
        (path for path in directories if path.name == "원천데이터" or "원천" in path.name),
        None,
    )
    label_root = next(
        (path for path in directories if path.name == "라벨링데이터" or "라벨" in path.name),
        None,
    )
    if image_root is None or label_root is None:
        raise RuntimeError(f"Could not locate image/label roots under {args.input}")
    all_labels = sorted(label_root.rglob("*.json"))
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    labels = all_labels[args.start:]
    if args.limit is not None:
        labels = labels[:args.limit]
    # Extracted source archives may contain extra directory levels, while labels
    # are flat. Index source images by stem to pair them deterministically.
    image_by_stem = {}
    for candidate in image_root.rglob("*"):
        if candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if candidate.stem in image_by_stem:
                raise RuntimeError(f"Duplicate image stem: {candidate.stem}")
            image_by_stem[candidate.stem] = candidate
    written = 0
    for label_path in labels:
        relative = label_path.relative_to(label_root)
        data = json.loads(label_path.read_text(encoding="utf-8"))
        image_path = image_by_stem.get(label_path.stem)
        if image_path is None:
            print(f"SKIP (missing image): {label_path.stem}")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP (missing image): {image_path}")
            continue
        for annotation in data.get("annotations", []):
            # Composite each annotation at 55% opacity so the road image
            # remains visible beneath lane lines, vertex markers, and labels.
            overlay = image.copy()
            draw_annotation(overlay, annotation)
            image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
        output_path = args.output / relative.with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {output_path}")
        written += 1
    print(f"Wrote {written}/{len(labels)} visualizations to {args.output} "
          f"(labels {args.start}..{args.start + len(labels) - 1} of {len(all_labels)})")


if __name__ == "__main__":
    main()
