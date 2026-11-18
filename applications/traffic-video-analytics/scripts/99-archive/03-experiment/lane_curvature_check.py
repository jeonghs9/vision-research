#!/usr/bin/env python3
"""곡선 차선에서 하프 변환(직선 모델)이 얼마나 손해를 보는지 측정한다.

하프 변환은 **원리상 직선만** 찾는다. 차선이 휘어 있으면 짧은 직선 여러 개로
쪼개지고, 8단계 병합은 각도가 달라 그 조각들을 하나로 못 묶는다. RANSAC 곡선
피팅으로 넘어갈 값어치가 있는지를 **근거 있는 수치로** 판단하기 위한 스크립트다.

지표는 **새그(sagitta)** 하나다.

**새그** — 차선 덩어리가 직선에서 벗어난 최대 거리(px).
   양 끝을 이은 시위에서 가장 멀리 떨어진 점까지의 거리다.
이 값이 병합 허용치(`--merge_rho`, 기본 25px)를 넘으면 하프는 그 차선을
**하나로 묶지 못한다.** 파라미터와 직접 연결돼 있어 손해가 바로 읽힌다.

한때 "설명률"(검출된 선 3px 안에 든 세선 픽셀의 비율)도 같이 쟀으나 **지웠다.**
차로를 가로지르는 가짜 선이 여러 차선의 점선을 스치면서 점수를 크게 올려,
**틀린 선일수록 높아지는** 지표였다. 검출 품질 판단은 패널을 눈으로 본다.

새그는 **덩어리 단위**로 잰다. 프레임 단위로 평균내면 직선 차선이 다수라
곡선 신호가 희석된다 (07 문서 4.3절과 같은 실수).

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_curvature_check.py --seed 0 --n 30
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load_pipeline():
    """lane_hough.py 의 전처리 단계를 그대로 재사용한다 (중복 구현 금지)."""
    spec = importlib.util.spec_from_file_location("lane_hough", HERE / "lane_hough.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sagitta(pts):
    """덩어리 픽셀의 주축 기준 새그와 길이를 구한다.

    PCA 로 주축(t)·수직축(u)을 잡고, u 를 t 의 **2차식**으로 맞춘 뒤 양 끝을 잇는
    시위에서의 최대 이탈을 잰다. 잡음에 흔들리지 않도록 원시 최대값이 아니라
    적합된 곡선에서 계산한다.

    반환: (길이 px, 새그 px, 직선 RMS, 2차 RMS)
    """
    c = pts.mean(0)
    q = pts - c
    # 주축 = 공분산의 최대 고유벡터
    w, v = np.linalg.eigh(np.cov(q.T))
    axis = v[:, np.argmax(w)]
    perp = np.array([-axis[1], axis[0]])
    t = q @ axis
    u = q @ perp
    length = float(t.max() - t.min())
    if length < 1:
        return length, 0.0, 0.0, 0.0

    rms_line = float(np.sqrt((u ** 2).mean()))          # 주축 자체가 최적 직선
    if len(t) < 8:
        return length, 0.0, rms_line, rms_line
    co = np.polyfit(t, u, 2)
    rms_quad = float(np.sqrt(((u - np.polyval(co, t)) ** 2).mean()))

    # 적합된 2차곡선의 양 끝을 잇는 시위에서 최대 이탈 = 새그
    t0, t1 = t.min(), t.max()
    ts = np.linspace(t0, t1, 64)
    us = np.polyval(co, ts)
    chord = (us[-1] - us[0]) / (t1 - t0) * (ts - t0) + us[0]
    sag = float(np.abs(us - chord).max())
    return length, sag, rms_line, rms_quad


def seg_distance(pts, segs):
    """각 점에서 가장 가까운 선분까지의 거리."""
    if not segs:
        return np.full(len(pts), 1e9)
    best = np.full(len(pts), 1e9)
    for x1, y1, x2, y2 in segs:
        a = np.array([x1, y1], float)
        b = np.array([x2, y2], float)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-6:
            d = np.linalg.norm(pts - a, axis=1)
        else:
            s = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
            d = np.linalg.norm(pts - (a + s[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--width", type=int, default=1920)
    # lane_hough 의 현재 기준 설정과 동일하게 둔다 (07 문서 5절)
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.10)
    ap.add_argument("--min_elong", type=float, default=4.0)
    ap.add_argument("--max_width", type=float, default=60.0)
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--car_dilate", type=int, default=15)
    ap.add_argument("--hough_thresh", type=int, default=25)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_gap", type=int, default=60)
    ap.add_argument("--angle_min", type=float, default=20.0)
    ap.add_argument("--merge_deg", type=float, default=4.0)
    ap.add_argument("--merge_rho", type=float, default=25.0)
    ap.add_argument("--merge_min_len", type=float, default=120.0)
    # 측정 조건
    ap.add_argument("--min_blob_len", type=float, default=150.0,
                    help="이 길이 이상인 덩어리만 새그를 잰다 (짧은 조각은 곡률 무의미)")
    ap.add_argument("--band", type=float, default=15.0,
                    help="병합 선 주변 이 거리 안의 세선 픽셀을 그 차선의 증거로 본다")
    a = ap.parse_args()

    lh = load_pipeline()
    from ultralytics import YOLO

    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames, sel = lh.decode(lh.VIDEO, a.n, a.stride, W, H, a.seed)
    model = YOLO(lh.WEIGHTS)
    print(f"seed {a.seed} 무작위 {len(frames)}장, {W}x{H}")
    print(f"새그 측정 대상: 길이 >= {a.min_blob_len}px 덩어리")
    print(f"병합 허용치(merge_rho) = {a.merge_rho}px  <- 새그가 이걸 넘으면 하프가 못 묶는다\n")

    print(f"{'f':>3} {'긴덩어리':>8} {'새그중앙':>9} {'새그최대':>9} "
          f"{'초과':>5} {'2차개선':>8} {'선분':>5}")
    print("-" * 68)

    rows = []
    blob_all = []
    lane_all = []
    for i, f in enumerate(frames):
        det = model.predict(f[:, :, ::-1], imgsz=1024, conf=0.25,
                            device=0, verbose=False)[0]
        m = lh.class_masks(det, (H, W), dev)
        car = m[lh.CLASS_CAR]
        if a.car_dilate > 1:
            car = F.max_pool2d(car[None, None].float(), a.car_dilate, 1,
                               a.car_dilate // 2)[0, 0] > 0.5
        roi = m[lh.CLASS_ROAD] & (~car) & (~m[lh.CLASS_PLATE])
        if roi.sum() < 100:
            continue

        rgb = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()
        mx, _ = rgb.max(0)
        mn, _ = rgb.min(0)
        s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))
        thr = torch.quantile(mx[roi].float(), 1.0 - a.pct / 100.0)
        cand = ((mx >= thr) & (s <= a.sat_max) & roi).to(torch.uint8).cpu().numpy()

        narrow, _, _ = lh.shape_filter(cand, a.min_elong, a.max_width, a.min_area)
        skel = lh.skeletonize(narrow)

        lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, a.hough_thresh,
                                minLineLength=a.min_len, maxLineGap=a.max_gap)
        segs = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                ang = min(ang, 180 - ang)
                if ang >= a.angle_min:
                    segs.append((int(x1), int(y1), int(x2), int(y2)))
        merged = lh.merge_segments(segs, a.merge_deg, a.merge_rho)
        msegs = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, ln in merged
                 if ln >= a.merge_min_len]

        # ── 지표 1: 덩어리별 새그
        n_cc, lab, st, _ = cv2.connectedComponentsWithStats(narrow, connectivity=8)
        sags, imps = [], []
        for k in range(1, n_cc):
            if st[k, cv2.CC_STAT_AREA] < a.min_area:
                continue
            x, y = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
            w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
            ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
            pts = np.stack([xs + x, ys + y], 1).astype(np.float64)
            ln, sag, rl, rq = sagitta(pts)
            if ln < a.min_blob_len:
                continue
            sags.append(sag)
            imps.append(0.0 if rl < 1e-6 else max(0.0, 1.0 - rq / rl))
            blob_all.append((i, ln, sag))

        # ── 지표 1b: **병합된 선 단위** 새그
        # 덩어리 단위로만 재면 점선이 조각마다 짧게 끊겨 있어 차선 전체가 휘어도
        # 조각은 직선으로 나온다. 병합 선 주변(`--band`)의 세선 픽셀을 모두 모아
        # 차선 전 구간에 걸친 곡률을 잰다.
        sy, sx = np.nonzero(skel)
        skel_pts = np.stack([sx, sy], 1).astype(np.float64)
        for seg in msegs:
            d1 = seg_distance(skel_pts, [seg])
            near = skel_pts[d1 <= a.band]
            if len(near) < 30:
                continue
            ln, sag, rl, rq = sagitta(near)
            if ln < a.min_blob_len:
                continue
            lane_all.append((i, ln, sag))

        if sags:
            sa = np.array(sags)
            over = int((sa > a.merge_rho).sum())
            rows.append((i, len(sa), float(np.median(sa)), float(sa.max()),
                         over, float(np.median(imps)), len(msegs)))
            print(f"{i:3d} {len(sa):8d} {np.median(sa):9.1f} {sa.max():9.1f} "
                  f"{over:5d} {np.median(imps)*100:7.0f}% {len(msegs):5d}")
        else:
            rows.append((i, 0, 0, 0, 0, 0, len(msegs)))
            print(f"{i:3d} {0:8d} {'-':>9} {'-':>9} {'-':>5} {'-':>8} {len(msegs):5d}")

    R = np.array([r for r in rows if r[1] > 0], dtype=float)
    B = np.array(blob_all, dtype=float)
    L = np.array(lane_all, dtype=float)
    print("\n" + "=" * 68)
    print(f"긴 덩어리 총 {len(B)}개 (길이 >= {a.min_blob_len}px)")
    print(f"  새그 중앙값        {np.median(B[:,2]):6.1f} px")
    print(f"  새그 상위 25%      {np.percentile(B[:,2],75):6.1f} px")
    print(f"  새그 상위 10%      {np.percentile(B[:,2],90):6.1f} px")
    print(f"  **{a.merge_rho}px 초과**   {100*(B[:,2]>a.merge_rho).mean():5.1f} %"
          f"  <- 하프가 하나로 못 묶는 비율")
    print(f"\n병합 선 단위 (점선 포함, 차선 전 구간) {len(L)}개")
    print(f"  구간 길이 중앙값    {np.median(L[:,1]):6.0f} px")
    print(f"  새그 중앙값        {np.median(L[:,2]):6.1f} px")
    print(f"  새그 상위 25%      {np.percentile(L[:,2],75):6.1f} px")
    print(f"  새그 상위 10%      {np.percentile(L[:,2],90):6.1f} px")
    print(f"  새그 최대          {L[:,2].max():6.1f} px")
    print(f"  **{a.merge_rho}px 초과**   {100*(L[:,2]>a.merge_rho).mean():5.1f} %")

    # 곡선이 심한 프레임과 직선 프레임의 새그를 나눠 본다
    fr_sag = {}
    for fi, ln, sag in blob_all:
        fr_sag.setdefault(int(fi), []).append(sag)
    med = {k: float(np.median(v)) for k, v in fr_sag.items()}
    seg_of = {int(r[0]): r[6] for r in rows}
    order = sorted(med, key=lambda k: -med[k])
    curved, straight = order[:8], order[-8:]
    print(f"\n{'':14}{'프레임':>28}{'새그중앙':>10}{'선분':>7}")
    for name, grp in (("곡선 상위 8장", curved), ("직선 하위 8장", straight)):
        ms = np.mean([med[k] for k in grp])
        sg = np.mean([seg_of[k] for k in grp])
        print(f"{name:14}{str(sorted(grp)):>28}{ms:9.1f}px{sg:7.1f}")
    print("\n판단 기준")
    print(f"  - 새그 {a.merge_rho}px 초과가 20% 넘으면 RANSAC 곡선 피팅 값어치 있음")
    print("  - 0% 면 차선이 사실상 직선이라는 뜻이고, 곡선 모델은 필요 없다")


if __name__ == "__main__":
    main()
