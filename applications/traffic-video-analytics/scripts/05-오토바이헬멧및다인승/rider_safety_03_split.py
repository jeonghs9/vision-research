#!/usr/bin/env python3
"""병합 데이터셋을 train/val/test 로 나눈다 (A안: val/test 는 드론 유사 뷰에서만).

계획 문서: docs/05-오토바이헬멧및다인승/01. 계획.md

분할 방침
  - val / test 는 `--eval_source` (기본 Vehicle v3, 고각 하향 뷰) 에서만 뽑는다.
    지상 뷰가 섞인 val 로는 드론 영상 성능을 잴 수 없기 때문이다.
  - 나머지 데이터셋은 전량 train 으로 보낸다.

**클립 그룹 단위로 나눈다 (누수 방지)**
  Roboflow 데이터에는 두 종류의 근접 중복이 있다.
    1. augmentation 복제본  — 같은 원본을 회전·밝기 변경한 것 (`<원본>.rf.<해시>`)
    2. 같은 영상의 연속 프레임 — `3J4A7819_frame_0028`, `Video9_4`, `00-302-`
  이들이 train 과 test 에 흩어지면 test 가 사실상 train 이 되어 지표가 부풀려진다.
  같은 그룹은 반드시 같은 split 으로 보낸다.

산출물
  images/{train,val,test}/  labels/{train,val,test}/  (심볼릭 링크)
  vis/{train,val,test}/     각 30장 라벨 시각화 (워크스페이스 필수 규칙)
  data.yaml  groups.csv     그룹 배정 기록 (되돌리기·재현용)

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_safety_03_split.py \
      --merged_dir /home/hsjeong/workspace/dataset/det/rider-safety/processed/03-merged \
      --manifest   /home/hsjeong/workspace/dataset/det/rider-safety/processed/01-helmet-pool/manifest.csv \
      --out_dir    /home/hsjeong/workspace/dataset/det/rider-safety/splits/001-5cls-droneval
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
import re
from pathlib import Path

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate_number", 4: "helmet"}
COLORS = {0: "#00ff5f", 1: "#00c8ff", 2: "#ffb000", 3: "#ff3ce0", 4: "#ff2d55"}
SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--eval_source", default="Vehicle", help="val/test 를 뽑을 데이터셋 (이름 앞부분)")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--vis_n", type=int, default=30, help="split 당 시각화 장수")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clip_key(stem: str, ds: str) -> str:
    """augmentation 복제본과 같은 영상의 프레임을 하나의 그룹으로 묶는다."""
    n = stem.split(".rf.")[0]
    for _ in range(2):  # `_png_jpg` 처럼 확장자가 두 번 붙는 경우가 있다
        n = re.sub(r"_(jpg|jpeg|png)$", "", n)
    m = re.match(r"^(.+?)_frame[_-]?\d+$", n)
    if m:
        return f"{ds}|{m.group(1)}"
    m = re.match(r"^(.+?)[-_]0*\d{1,5}[-_]?$", n)
    if m and len(m.group(1)) >= 2:
        return f"{ds}|{m.group(1)}"
    return f"{ds}|{n}"


def assign_groups(groups: dict[str, list], val_ratio: float, test_ratio: float, rng: random.Random):
    """큰 그룹부터 목표 이미지 수에 가장 모자란 split 에 넣는다.

    그룹을 무작위로 뿌리면 큰 그룹 하나가 비율을 무너뜨린다.
    """
    total = sum(len(v) for v in groups.values())
    target = {"val": total * val_ratio, "test": total * test_ratio, "train": total * (1 - val_ratio - test_ratio)}
    cur = {s: 0 for s in SPLITS}
    out = {s: [] for s in SPLITS}
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    rng.shuffle(keys[: max(1, len(keys) // 10)])  # 동률 방지용 약간의 무작위성
    for k in sorted(keys, key=lambda k: -len(groups[k])):
        s = max(SPLITS, key=lambda s: (target[s] - cur[s]) / max(target[s], 1))
        out[s].append(k)
        cur[s] += len(groups[k])
    return out


def draw_vis(img_path: Path, lbl_path: Path, dst: Path):
    from PIL import Image, ImageDraw

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        x, y, w, h = (float(v) for v in p[1:])
        box = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
        d.rectangle(box, outline=COLORS[c], width=4 if c == 4 else 2)
        d.text((box[0], max(0, box[1] - 11)), NAMES[c], fill=COLORS[c])
    im.save(dst, quality=90)


def main():
    args = parse_args()
    merged, out_dir = Path(args.merged_dir).resolve(), Path(args.out_dir).resolve()
    rng = random.Random(args.seed)

    man = list(csv.DictReader(Path(args.manifest).open()))
    groups: dict[str, list] = collections.defaultdict(list)
    for r in man:
        ds = r["source_dataset"].split(".")[0]
        groups[clip_key(r["stem"], ds)].append(r)

    eval_groups = {k: v for k, v in groups.items() if k.startswith(args.eval_source)}
    rest_groups = {k: v for k, v in groups.items() if not k.startswith(args.eval_source)}
    if not eval_groups:
        raise SystemExit(f"[ERROR] '{args.eval_source}' 로 시작하는 데이터셋이 없다")

    assigned = assign_groups(eval_groups, args.val_ratio, args.test_ratio, rng)
    assigned["train"] += list(rest_groups)  # 나머지 데이터셋은 전량 train

    # 누수 검사 — 같은 그룹이 두 split 에 걸치면 안 된다
    seen: dict[str, str] = {}
    for s in SPLITS:
        for k in assigned[s]:
            if k in seen:
                raise SystemExit(f"[ERROR] 그룹 {k} 이 {seen[k]} 와 {s} 에 중복 배정됐다")
            seen[k] = s

    for s in SPLITS:
        for sub in ("images", "labels", "vis"):
            (out_dir / sub / s).mkdir(parents=True, exist_ok=True)

    stats = {s: collections.Counter() for s in SPLITS}
    ds_stats = {s: collections.Counter() for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    rows = []
    for s in SPLITS:
        for k in assigned[s]:
            for r in groups[k]:
                stem = r["stem"]
                for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                    src = merged / sub / f"{stem}{ext}"
                    dst = out_dir / sub / s / f"{stem}{ext}"
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
                for line in (merged / "labels" / f"{stem}.txt").read_text().splitlines():
                    if line.strip():
                        stats[s][int(line.split()[0])] += 1
                ds_stats[s][r["source_dataset"].split(".")[0]] += 1
                counts[s] += 1
                rows.append({"stem": stem, "group": k, "split": s, "source_dataset": r["source_dataset"]})

    # 시각화 — split 마다 seed 고정 무작위 N 장
    for s in SPLITS:
        pool = sorted((out_dir / "images" / s).iterdir())
        pick = random.Random(args.seed).sample(pool, min(args.vis_n, len(pool)))
        for p in pick:
            draw_vis(p, out_dir / "labels" / s / f"{p.stem}.txt", out_dir / "vis" / s / f"{p.stem}.jpg")

    with (out_dir / "groups.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "group", "split", "source_dataset"])
        w.writeheader()
        w.writerows(rows)

    (out_dir / "data.yaml").write_text(
        f"# rider-safety 5클래스 — val/test 는 드론 유사 뷰({args.eval_source}) 전용\n"
        f"path: {out_dir}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: 5\nnames:\n" + "".join(f"  {c}: {NAMES[c]}\n" for c in sorted(NAMES))
    )

    total = sum(counts.values())
    print(f"■ 분할 결과 (그룹 단위, seed={args.seed})\n")
    print(f"{'split':<8}{'이미지':>9}{'비율':>8}{'그룹':>8}" + "".join(f"{NAMES[c]:>13}" for c in sorted(NAMES)))
    for s in SPLITS:
        print(
            f"{s:<8}{counts[s]:>9,}{counts[s] / total * 100:>7.1f}%{len(assigned[s]):>8,}"
            + "".join(f"{stats[s][c]:>13,}" for c in sorted(NAMES))
        )
    print(f"\n■ split 별 출처 구성")
    for s in SPLITS:
        print(f"   {s:<6} " + "  ".join(f"{k}={v:,}" for k, v in sorted(ds_stats[s].items())))
    print(f"\n■ 그룹 누수 검사 : 통과 (공유 그룹 0개)")
    print(f"\n출력: {out_dir}\n  images/ labels/ vis/ data.yaml groups.csv")


if __name__ == "__main__":
    main()
