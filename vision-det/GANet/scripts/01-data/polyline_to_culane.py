#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] si_vietnam_drone 폴리라인 라벨을 GANet이 쓰는 CULane 형식으로 변환한다.

GANet(`projects/plugin/datasets/culane_dataset.py`, mmdet2_25 브랜치)이 요구하는
구조를 그대로 만든다. CLRKDNet 프로젝트의 `scripts/01-data/polyline_to_culane.py`와
점 변환 로직(정규화→픽셀, y 방향 재샘플)은 같지만, GANet은 세그멘테이션 마스크가
필요 없고 `list/*.txt`도 이미지 경로만 담는 단순한 형식이라 그 부분만 다르다.

## GANet이 요구하는 것 (culane_dataset.py, tusimple_dataset.py 를 읽고 확인)

```text
data_root/
├── list/train.txt   이미지 상대경로 한 줄씩 (앞에 '/' 있어도 없어도 됨, 코드가 벗겨줌)
├── list/val.txt
├── list/test.txt
├── images/<stem>.jpg          (원본 심볼릭 링크 — 복사 안 함)
└── images/<stem>.lines.txt    차선 1개 = 1줄, "x1 y1 x2 y2 ..." (픽셀 좌표)
```
세그 마스크는 필요 없다(CLRNet 계열과 다른 점).

## 원본 라벨과 분할 기준

라벨: `cls x1 y1 ... xn yn` (정규화, cls 0=실선/1=점선) — 화면 하나에
`labels/train/*.txt` 594장이 전부 모여 있고, train/val 분할은 파일명이 아니라
**영상(클립) 단위**로 한다(프레임 단위로 섞으면 같은 클립이 양쪽에 섞여 누수).
YOLinO(`source/src/yolino/dataset/drone_pytorch.py`)가 쓰는 것과 **동일한 분할**
(`VAL_VIDEOS = ["19700101_001037_T"]`, 431 train / 163 val)을 기본값으로 써서
나중에 모델 간 비교가 공정하게 되도록 했다.

점 순서도 YOLinO와 맞췄다: 원본 라벨은 사람이 그린 순서 그대로라 선마다 방향이
제각각인데, y 오름차순(위→아래)으로 통일한다(YOLinO가 `fit_lines` 이어붙이기
방향 불일치 문제로 이미 겪고 고친 것과 같은 이유).

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/01-data/polyline_to_culane.py \
    --src /home/hsjeong/workspace/dataset/etc./si_vietnam_drone/yolo-pose \
    --out /home/hsjeong/workspace/dataset/etc./si_vietnam_drone/culane_lane_ganet \
    --val_videos 19700101_001037_T
"""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


def read_polylines(lp, W, H):
    """라벨 파일(정규화 폴리라인)을 픽셀 좌표 목록으로. (cls, (N,2) xy pixel) 리스트."""
    out = []
    for line in open(lp, encoding="utf-8"):
        q = line.split()
        if len(q) < 5 or (len(q) - 1) % 2:
            continue
        cls = int(q[0])
        if cls not in (0, 1):
            continue
        co = np.asarray(list(map(float, q[1:])), dtype=np.float64).reshape(-1, 2)
        if len(co) < 2:
            continue
        pts = np.stack([co[:, 0] * W, co[:, 1] * H], 1)
        out.append((cls, pts))
    return out


def densify(pts, step):
    """y 오름차순으로 정렬하고 `step` 픽셀 간격으로 다시 뽑는다.

    - 방향 통일: YOLinO가 겪은 것과 같은 이유(원본 라벨 방향이 제각각)로 y
      오름차순(화면 위→아래)으로 맞춘다.
    - 재샘플: 원본은 2~6점뿐이라 GANet의 keypoint 샘플링(`sample_per_lane`)
      밀도에 비해 성기다. 촘촘하게 보간해서 채운다.
    """
    P = pts[np.argsort(pts[:, 1])]
    ys, xs = P[:, 1], P[:, 0]
    y0, y1 = ys.min(), ys.max()
    if y1 - y0 < step:
        return P
    grid = np.arange(y0, y1 + 1e-6, step)
    return np.stack([np.interp(grid, ys, xs), grid], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="폴리라인 라벨 루트 (images|labels/train 을 가짐)")
    ap.add_argument("--out", required=True, help="CULane 형식 출력 루트")
    ap.add_argument("--val_videos", nargs="*", default=["19700101_001037_T"],
                    help="검증으로 뺄 영상 이름(클립 단위 홀드아웃). YOLinO와 동일 기본값")
    ap.add_argument("--step", type=float, default=20.0, help="y 방향 점 간격(px)")
    ap.add_argument("--keep", type=int, nargs="+", default=[0, 1],
                    help="유지할 클래스(0=실선,1=점선). GANet 기본 num_lane_classes=1이라 구분 없이 합쳐진다")
    ap.add_argument("--write-class", action="store_true",
                    help="차선마다 <stem>.cls.txt에 클래스(1=실선,2=점선, GANet id_classes 1-indexed 관례)도 같이 기록. "
                         "1클래스 config는 이 파일을 안 읽으므로 기존 데이터에 덮어써도 안전")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (out / "list").mkdir(parents=True, exist_ok=True)
    keep = set(a.keep)

    rows = {"train": [], "val": []}
    stats = {"train": [], "val": []}
    W = H = None

    for lp in sorted(glob.glob(str(src / "labels" / "train" / "*.txt"))):
        stem = Path(lp).stem
        img_src = src / "images" / "train" / f"{stem}.jpg"
        if not img_src.exists():
            print(f"  ⚠ {stem}: 이미지 없음 — 건너뜀")
            continue

        vid = stem.split("__")[0]
        split = "val" if vid in a.val_videos else "train"

        if W is None:
            img = cv2.imread(str(img_src))
            H, W = img.shape[:2]

        lanes = [(c, p) for c, p in read_polylines(lp, W, H) if c in keep]
        if not lanes:
            continue

        link = img_dir / f"{stem}.jpg"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img_src.resolve())

        with open(img_dir / f"{stem}.lines.txt", "w") as f:
            for _c, pts in lanes:
                d = densify(pts, a.step)
                f.write(" ".join(f"{x:.3f} {y:.3f}" for x, y in d) + "\n")

        if a.write_class:
            # 원본 cls: 0=실선/1=점선 -> GANet id_classes(1-indexed): 1=실선/2=점선
            with open(img_dir / f"{stem}.cls.txt", "w") as f:
                f.write("\n".join(str(c + 1) for c, _pts in lanes) + "\n")

        rows[split].append(f"images/{stem}.jpg")
        stats[split].append(len(lanes))

    with open(out / "list" / "train.txt", "w") as f:
        f.write("\n".join(rows["train"]) + "\n")
    for name in ("val.txt", "test.txt"):
        with open(out / "list" / name, "w") as f:
            f.write("\n".join(rows["val"]) + "\n")

    print(f"출력 {out}")
    for k in ("train", "val"):
        if stats[k]:
            print(f"  {k:>5}: {len(stats[k])}장, 차선 {sum(stats[k])}개 "
                  f"(장당 최소 {min(stats[k])} 최대 {max(stats[k])})")
    print(f"이미지 규격 {W}x{H}")


if __name__ == "__main__":
    main()
