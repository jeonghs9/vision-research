#!/usr/bin/env python3
"""[OpenLane용] OpenLane 2D 차선 라벨(uv)을 GANet이 읽는 CULane 형식(2클래스)으로 변환한다.

입력: 분할 목록(splits/{train,val,test}.txt, 한 줄 = raw 기준 상대 이미지 경로
      images/<training|validation>/<segment>/<stem>.jpg) + raw/lane3d_1000/<...>/<stem>.json
출력(--out):
  images/<training|validation>/<segment>/<stem>.jpg         raw 이미지 심볼릭 링크 (복사 안 함, raw 수정 없음)
  images/<...>/<stem>.lines.txt   차선 1줄 = "x1 y1 x2 y2 ..." (원본 1920x1280 기준 픽셀, y 오름차순)
  images/<...>/<stem>.cls.txt     차선과 같은 순서, 한 줄에 정수 하나 (GANet id_classes 1-indexed: 1=실선, 2=점선)
  list/{train,val,test}.txt       분할 목록 그대로 (GANet 데이터셋이 읽는 목록)
  convert_report.txt              변환 통계

클래스 매핑(문서 docs/02-openlane/1. … §5, 확정):
  점선 2 : category 1, 3, 7, 9
  실선 1 : category 2, 4, 8, 10  + 혼합선 5, 6, 11, 12
  제외   : 연석 20, 21 / unknown 0

점 처리 (GANet이 y 기준으로 정렬한 뒤 x=f(y) 스플라인을 맞추기 때문 — lane_target_collect.py 참고):
  - OpenLane uv는 점이 촘촘(레인당 중앙값 ~125개)하지만 y가 단조롭지 않고 흔들린다(레인의 97%).
    → y로 정렬, 같은 y는 x 평균, y --step px 간격으로 다시 뽑는다(선형 보간).
  - y 범위가 --min-y-extent px 미만인 레인(거의 수평 — x=f(y)로 표현 불가)은 제외한다.
  - 재샘플한 점이 2개 미만이면 제외한다.
  - 차선이 하나도 안 남은 프레임은 기본적으로 **유지**(빈 .lines.txt). --drop-empty 로 목록에서 뺄 수 있다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/clrkdnet_cu113/bin/python -I \
    /home/hsjeong/workspace/vision-det/GANet/scripts/01-data/openlane_to_culane.py \
    --root /home/hsjeong/workspace/dataset/det/OpenLane/raw \
    --splits /home/hsjeong/workspace/dataset/det/OpenLane/splits \
    --out /home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet
"""
import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np

SOLID = {2, 4, 8, 10, 5, 6, 11, 12}
DASHED = {1, 3, 7, 9}
CLS_ID = {**{c: 1 for c in SOLID}, **{c: 2 for c in DASHED}}  # GANet 1-indexed: 1=실선, 2=점선
W, H = 1920, 1280


def resample(u, v, step):
    """y 오름차순 정렬 → 같은 y는 x 평균 → step px 간격 선형 보간. 점 2개 미만이면 None."""
    o = np.argsort(v, kind="stable")
    u, v = u[o], v[o]
    uy, inv = np.unique(np.round(v, 2), return_inverse=True)
    ux = np.bincount(inv, weights=u) / np.bincount(inv)
    if len(uy) < 2:
        return None
    grid = np.arange(uy[0], uy[-1] + 1e-6, step)
    if len(grid) < 2:
        return None
    pts = np.stack([np.interp(grid, uy, ux), grid], 1)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OpenLane raw/ 폴더")
    ap.add_argument("--splits", required=True, help="분할 목록 폴더 (train.txt, val.txt, test.txt)")
    ap.add_argument("--out", required=True, help="CULane 형식 출력 루트")
    ap.add_argument("--step", type=float, default=10.0, help="y 방향 재샘플 간격(px, 원본 1280 높이 기준)")
    ap.add_argument("--min-y-extent", type=float, default=30.0, help="이보다 y 범위가 작은 레인은 제외(px)")
    ap.add_argument("--drop-empty", action="store_true", help="변환 후 차선이 0개인 프레임을 목록에서 뺀다")
    a = ap.parse_args()

    # 생성물만 지운다 (out 안의 images/, list/) — 재분할 후 다시 돌릴 때 잔여물 방지
    for sub in ("images", "list"):
        shutil.rmtree(os.path.join(a.out, sub), ignore_errors=True)
    os.makedirs(os.path.join(a.out, "list"), exist_ok=True)

    report = []
    for split in ("train", "val", "test"):
        lines_in = [x.strip() for x in open(os.path.join(a.splits, f"{split}.txt")) if x.strip()]
        st = Counter()
        n_lane_frame = []
        kept_list = []
        for p in lines_in:
            lab = os.path.join(a.root, p.replace("images/", "lane3d_1000/", 1)[:-4] + ".json")
            j = json.load(open(lab))
            lanes, classes = [], []
            for ln in j["lane_lines"]:
                st["lanes_raw"] += 1
                c = ln["category"]
                if c not in CLS_ID:
                    st["excluded_curb_or_unknown"] += 1
                    continue
                u, v = (np.asarray(x, float) for x in ln["uv"])
                if len(u) < 2 or v.max() - v.min() < a.min_y_extent:
                    st["dropped_short_or_horizontal"] += 1
                    continue
                pts = resample(u, v, a.step)
                if pts is None:
                    st["dropped_resample"] += 1
                    continue
                lanes.append(pts)
                classes.append(CLS_ID[c])
                st["solid" if CLS_ID[c] == 1 else "dashed"] += 1
            n_lane_frame.append(len(lanes))
            if not lanes:
                st["frames_empty"] += 1
                if a.drop_empty:
                    continue
            dst_dir = os.path.join(a.out, os.path.dirname(p))
            os.makedirs(dst_dir, exist_ok=True)
            stem = os.path.join(a.out, p[:-4])
            link = stem + ".jpg"
            os.symlink(os.path.realpath(os.path.join(a.root, p)), link)
            with open(stem + ".lines.txt", "w") as f:
                for pts in lanes:
                    f.write(" ".join(f"{x:.3f} {y:.3f}" for x, y in pts) + "\n")
            with open(stem + ".cls.txt", "w") as f:
                f.write("".join(f"{c}\n" for c in classes))
            kept_list.append(p)
        with open(os.path.join(a.out, "list", f"{split}.txt"), "w") as f:
            f.write("\n".join(kept_list) + "\n")
        mx = max(n_lane_frame) if n_lane_frame else 0
        report.append(
            f"{split}: 목록 {len(lines_in)}장 -> 유지 {len(kept_list)}장 | 원본 차선 {st['lanes_raw']}개 -> 실선 {st['solid']} · 점선 {st['dashed']} "
            f"| 제외: 연석/unknown {st['excluded_curb_or_unknown']}, 짧음·수평(y범위<{a.min_y_extent:g}px) {st['dropped_short_or_horizontal']}, "
            f"재샘플 실패 {st['dropped_resample']} | 차선 0개 프레임 {st['frames_empty']} | 프레임당 최대 차선 {mx}")
    txt = "\n".join(report)
    print(txt)
    with open(os.path.join(a.out, "convert_report.txt"), "w") as f:
        f.write(f"step={a.step} min_y_extent={a.min_y_extent} drop_empty={a.drop_empty}\n" + txt + "\n")


if __name__ == "__main__":
    main()
