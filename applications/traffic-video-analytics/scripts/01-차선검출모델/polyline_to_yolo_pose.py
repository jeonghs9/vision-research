#!/usr/bin/env python3
"""차선 폴리라인 라벨을 **YOLO-pose 형식**으로 변환한다.

## 왜 pose 인가

차선은 개수가 프레임마다 크게 변하고(실측 0~24개) 화면상 각도도 6~90도로 넓다.
행 앵커 방식(UFLD 등)은 "각 가로줄에 차선이 한 번씩 나온다" 를 출력 구조에
박아둬서 누운 차선을 표현하지 못한다. **물체 검출 기반인 pose 는 개수·각도에
제약이 없다.**

    차선 1개 = 물체 1개
      클래스   : 0 실선 / 1 점선
      박스     : 그 차선을 감싸는 사각형
      키포인트 : 선을 따라 균등 간격으로 뽑은 K개 점

## 입력 (라벨링 도구 출력)

    cls x1 y1 x2 y2 ... xn yn      # 정규화, 점 개수 자유

## 출력 (YOLO-pose)

    cls xc yc w h  px1 py1 v1  ...  pxK pyK vK      # 전부 정규화

## 변환에서 처리하는 것 세 가지

1. **점 개수 통일** — 호 길이 기준 균등 재샘플링. 사람이 몇 점을 찍든 K개가 된다.
2. **방향 통일** — 키포인트는 슬롯이라 순서가 뒤섞이면 모델이 "0번은 위? 아래?"
   를 배우지 못하고 전부 선 한가운데를 예측한다. 주축 방향을 아래쪽(수평이면
   오른쪽)으로 맞춘다. `y` 로만 정렬하면 누운 차선에서 깨진다.
3. **박스 최소 크기** — 수직에 가까운 선은 박스 폭이 0 에 가깝다. 학습이
   불안정해지므로 최소 폭·높이를 준다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/01-차선검출모델/polyline_to_yolo_pose.py \
    --src /home/hsjeong/workspace/dataset/etc./si_vietnam_drone/yolo-pose \
    --kpt 12
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

CLASS_NAMES = {0: "solid", 1: "dashed"}


def resample(pts, k):
    """호 길이 기준으로 k개 점을 균등하게 다시 뽑는다."""
    P = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-9:                       # 길이 0 이면 같은 점 반복
        return np.repeat(P[:1], k, axis=0)
    t = np.linspace(0.0, total, k)
    return np.stack([np.interp(t, cum, P[:, 0]),
                     np.interp(t, cum, P[:, 1])], axis=1)


def orient(pts):
    """주축이 아래쪽(거의 수평이면 오른쪽)을 향하도록 점 순서를 맞춘다."""
    P = np.asarray(pts, dtype=np.float64)
    d = P[-1] - P[0]
    if abs(d[1]) > abs(d[0]):               # 세로에 가까움 -> y 증가 방향
        return P[::-1] if d[1] < 0 else P
    return P[::-1] if d[0] < 0 else P       # 가로에 가까움 -> x 증가 방향


def convert_file(lp, k, min_box, keep):
    out, dropped = [], []
    for ln, line in enumerate(open(lp, encoding="utf-8"), 1):
        q = line.split()
        if not q:
            continue
        if len(q) < 5 or (len(q) - 1) % 2:
            dropped.append(f"{ln}행 형식 이상"); continue
        cls = int(q[0])
        if cls not in keep:
            dropped.append(f"{ln}행 클래스 {cls} 제외"); continue
        co = np.asarray(list(map(float, q[1:])), dtype=np.float64)
        pts = co.reshape(-1, 2)
        if len(pts) < 2:
            dropped.append(f"{ln}행 점 부족"); continue

        P = resample(orient(pts), k)
        P = np.clip(P, 0.0, 1.0)

        x0, y0 = P[:, 0].min(), P[:, 1].min()
        x1, y1 = P[:, 0].max(), P[:, 1].max()
        w = max(x1 - x0, min_box)           # 수직선은 폭이 0 에 가깝다
        h = max(y1 - y0, min_box)
        xc = np.clip((x0 + x1) / 2, w / 2, 1 - w / 2)
        yc = np.clip((y0 + y1) / 2, h / 2, 1 - h / 2)

        vals = [f"{cls}", f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"]
        for (px, py) in P:
            vals += [f"{px:.6f}", f"{py:.6f}", "2"]   # 2 = 보임
        out.append(" ".join(vals))
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="labels/<split> 를 담은 루트")
    ap.add_argument("--out", default=None,
                    help="학습용 데이터셋 루트. 기본: <src>/pose_ds\n"
                         "Ultralytics 는 이미지 경로의 /images/ 를 /labels/ 로 바꿔 "
                         "라벨을 찾는다. 원본 폴리라인이 <src>/labels 에 있으므로 "
                         "**별도 루트**를 만들어야 충돌하지 않는다")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--kpt", type=int, default=12)
    ap.add_argument("--min_box", type=float, default=0.004,
                    help="박스 최소 폭·높이(정규화). 수직선 대비")
    ap.add_argument("--keep", type=int, nargs="+", default=[0, 1],
                    help="남길 클래스. 기본은 실선(0)·점선(1) 만")
    ap.add_argument("--val_videos", nargs="*", default=[],
                    help="검증셋으로 뺄 **영상 이름**. 예: 19700101_001037_T\n"
                         "프레임 단위 무작위 분할은 인접 프레임이 거의 같아 누수가 "
                         "생기고 성능이 부풀려진다. 반드시 영상(클립) 단위로 나눈다.")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else src / "pose_ds"
    keep = set(a.keep)
    print(f"입력 {src}/labels/<split>  +  {src}/images/<split>")
    print(f"출력 {out}/{{images,labels}}/<split>")
    print(f"키포인트 {a.kpt}개, 남길 클래스 {sorted(keep)}\n")

    total = tdrop = 0
    for sp in a.splits:
        d = src / "labels" / sp
        if not d.is_dir():
            continue
        for _sp in ({sp} | ({"val"} if a.val_videos and sp == "train" else set())):
            (out / "labels" / _sp).mkdir(parents=True, exist_ok=True)
            (out / "images" / _sp).mkdir(parents=True, exist_ok=True)
        n = files = linked = 0
        nval = 0
        for lp in sorted(glob.glob(str(d / "*.txt"))):
            stem = Path(lp).stem
            # 클립 단위 분할: 영상 이름이 --val_videos 에 있으면 val 로 보낸다
            vid = stem.split("__")[0]
            tgt = "val" if (sp == "train" and vid in a.val_videos) else sp
            od = out / "labels" / tgt
            oi = out / "images" / tgt
            # 라벨에 대응하는 이미지가 없으면 학습에 못 쓴다. 건너뛴다.
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = src / "images" / sp / f"{stem}{ext}"
                if cand.exists():
                    img = cand.resolve(); break
            if img is None:
                print(f"  ⚠ {stem}: 이미지 없음 — 건너뜀"); continue

            lines, dropped = convert_file(lp, a.kpt, a.min_box, keep)
            (od / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            # 링크 이름은 **라벨의 stem** 을 따라야 한다. resolve() 로 원본을
            # 따라가면 프레임 파일명(000001.jpg)이 되어 라벨과 짝이 안 맞고,
            # Ultralytics 가 "라벨 없음 = 배경" 으로 조용히 처리한다.
            link = oi / f"{stem}{img.suffix}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(img)            # 파일 단위 링크 (폴더 링크는 피한다)
            n += len(lines); files += 1; linked += 1; tdrop += len(dropped)
            if tgt == "val" and sp == "train":
                nval += 1
            for msg in dropped:
                print(f"  ⚠ {stem} {msg}")
        if nval:
            print(f"{sp:>6}: 파일 {files}개 (train {files-nval} / val {nval}), 선 {n}개")
            print(f"        검증 홀드아웃 영상: {', '.join(a.val_videos)}")
        else:
            print(f"{sp:>6}: 파일 {files}개, 이미지 링크 {linked}개, 선 {n}개")
        total += n

    # data.yaml
    yml = out / "lane_pose.yaml"
    names = "\n".join(f"  {i}: {CLASS_NAMES.get(i, str(i))}" for i in sorted(keep))
    # 좌우 반전 시 키포인트 순서는 그대로다 (위->아래 순서가 반전과 무관)
    flip = ", ".join(str(i) for i in range(a.kpt))
    yml.write_text(
        f"""# 차선 폴리라인 -> YOLO-pose (자동 생성: polyline_to_yolo_pose.py)
path: {out}
train: images/train
val: images/val

kpt_shape: [{a.kpt}, 3]
# 좌우 반전해도 "선을 따라가는 순서" 는 바뀌지 않으므로 항등 사상이다
flip_idx: [{flip}]

names:
{names}
""", encoding="utf-8")
    print(f"\n총 {total}개 선 변환, 제외 {tdrop}건")
    print(f"data.yaml: {yml}")


if __name__ == "__main__":
    main()
