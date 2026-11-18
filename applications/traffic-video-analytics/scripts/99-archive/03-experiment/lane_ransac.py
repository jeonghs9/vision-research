#!/usr/bin/env python3
"""RANSAC 으로 차선을 뽑아 하프 변환과 나란히 비교한다.

①~⑤ 단계(road ROI, 밝기 후보, 세장비 필터, 세선화)는 `lane_hough.py` 를 그대로
불러 쓴다. **⑥번 "점 -> 선" 자리만** 하프에서 RANSAC 으로 바꿔 끼운 것이다.
둘은 순서로 이어지는 단계가 아니라 같은 자리를 대체하는 방법이다.

RANSAC 은 scikit-learn 의 `RANSACRegressor` 를 쓴다 (직접 구현하지 않는다).

## 순차 RANSAC

RANSAC 은 한 번에 **선 하나만** 찾는다. 차선은 여러 개이므로 이렇게 반복한다.

```text
1. 남은 점들로 RANSAC -> 가장 표를 많이 받은 선 하나
2. 그 선에 속한 점(inlier)들을 제거
3. 남은 점이 충분하면 1로
```

## 세로선 문제

`RANSACRegressor` 는 `y = f(x)` 를 맞춘다. 차선이 화면에서 **세로에 가까우면**
한 x 에 y 가 여럿이라 맞출 수 없다. 그래서 매번 `y=f(x)` 와 `x=f(y)` 를 **둘 다**
시도하고 inlier 가 많은 쪽을 택한다.

## 직선인가 곡선인가 — 직선으로 결론남

`--degree 1` 은 직선, `--degree 2` 는 포물선이다. 란삭이 하프와 다른 유일한 점이
**곡선을 맞출 수 있다**는 것이라, 처음에는 둘 다 그려 비교했다.

결과는 **곡선이 더 나빴다.** 자유도가 높은 만큼 도로를 가로질러 휘감는 호가
생긴다. `lane_curvature_check.py` 로 잰 차선의 실제 휨(새그 최대 16.5px, 병합
허용치 25px 미만)과도 일치한다 — 직선인 대상에 곡선 모델을 주면 그 자유도로
잡음을 맞추러 간다. **기본값을 1(직선)로 두고 2차는 옵션으로만 남긴다.**

## 지표를 두지 않는다

한때 "설명률"(검출된 선 3px 안에 든 흰 점의 비율)을 찍었으나 **지웠다.**
차로를 가로지르는 가짜 선이 여러 차선의 점선을 스치면서 점수를 크게 올려,
**틀린 선일수록 높아지는** 지표였다. 판단은 패널을 눈으로 보고 한다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_ransac.py --seed 0 --n 30 \
    --out test/tmp/01-lane-postprocess-runs/lane_ransac
"""
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("lane_hough", HERE / "lane_hough.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_once(pts, degree, tol, trials, seed):
    """점 집합에서 가장 표를 많이 받는 곡선 하나를 찾는다.

    `y=f(x)` 와 `x=f(y)` 를 둘 다 시도해 inlier 가 많은 쪽을 반환한다.
    반환: (inlier 불린 배열, 그리기용 polyline 점들) 또는 None
    """
    best = None
    for swap in (False, True):
        A = pts[:, 1] if swap else pts[:, 0]      # 독립변수
        B = pts[:, 0] if swap else pts[:, 1]      # 종속변수
        if A.max() - A.min() < 10:
            continue
        try:
            est = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            r = RANSACRegressor(estimator=est, min_samples=degree + 1,
                                residual_threshold=tol, max_trials=trials,
                                random_state=seed)
            r.fit(A.reshape(-1, 1), B)
        except Exception:
            continue
        inl = r.inlier_mask_
        if best is None or inl.sum() > best[0].sum():
            a0, a1 = A[inl].min(), A[inl].max()
            ts = np.linspace(a0, a1, 80)
            bs = r.predict(ts.reshape(-1, 1))
            poly = np.stack([bs, ts], 1) if swap else np.stack([ts, bs], 1)
            best = (inl, poly)
    return best


def sequential_ransac(pts, degree, tol, trials, max_lines, min_inliers,
                      min_span, seed):
    """inlier 를 제거해가며 여러 선을 차례로 찾는다."""
    remain = pts.copy()
    out = []
    for k in range(max_lines):
        if len(remain) < min_inliers:
            break
        got = fit_once(remain, degree, tol, trials, seed + k)
        if got is None:
            break
        inl, poly = got
        if inl.sum() < min_inliers:
            break
        span = float(np.hypot(*(poly[-1] - poly[0])))
        if span >= min_span:
            out.append((poly, int(inl.sum()), span))
        remain = remain[~inl]
    return out


def draw_poly(img, poly, color, thick):
    p = poly.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], False, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--out", default="test/tmp/01-lane-postprocess-runs/lane_ransac")
    # ①~⑤ (lane_hough 현재 기준과 동일)
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.10)
    ap.add_argument("--min_elong", type=float, default=4.0)
    ap.add_argument("--max_width", type=float, default=60.0)
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--car_dilate", type=int, default=15)
    # 하프 (비교군)
    ap.add_argument("--hough_thresh", type=int, default=25)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_gap", type=int, default=60)
    ap.add_argument("--angle_min", type=float, default=20.0)
    ap.add_argument("--merge_deg", type=float, default=4.0)
    ap.add_argument("--merge_rho", type=float, default=25.0)
    ap.add_argument("--merge_min_len", type=float, default=120.0)
    # RANSAC
    ap.add_argument("--degree", type=int, default=1,
                    help="1=직선, 2=포물선. 곡선은 이 영상에서 더 나쁘다 "
                         "(도로를 휘감는 호가 생김). 기본은 직선")
    ap.add_argument("--tol", type=float, default=3.0, help="inlier 허용 오차 px")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--max_lines", type=int, default=12)
    ap.add_argument("--min_inliers", type=int, default=60)
    ap.add_argument("--min_span", type=float, default=120.0)
    ap.add_argument("--max_pts", type=int, default=20000, help="속도용 표본 상한")
    ap.add_argument("--crop_margin", type=int, default=40)
    ap.add_argument("--panel_width", type=int, default=960)
    a = ap.parse_args()

    lh = load_pipeline()
    from ultralytics import YOLO

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    frames, _ = lh.decode(lh.VIDEO, a.n, a.stride, W, H, a.seed)
    model = YOLO(lh.WEIGHTS)
    print(f"seed {a.seed} 무작위 {len(frames)}장, {W}x{H}")
    print(f"RANSAC tol={a.tol}px trials={a.trials} 최대 {a.max_lines}개, "
          f"inlier>={a.min_inliers}, 길이>={a.min_span}px")
    print(f"출력 {outdir}\n")
    print(f"{'f':>3} {'세선점':>8} {'하프':>6} {'란삭':>8}")
    print("-" * 40)

    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default()

    rows = []
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

        # ── 비교군: 하프
        lines = cv2.HoughLinesP(skel * 255, 1, np.pi / 180, a.hough_thresh,
                                minLineLength=a.min_len, maxLineGap=a.max_gap)
        segs = []
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if min(ang, 180 - ang) >= a.angle_min:
                    segs.append((int(x1), int(y1), int(x2), int(y2)))
        mg = lh.merge_segments(segs, a.merge_deg, a.merge_rho)
        hough = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, ln in mg
                 if ln >= a.merge_min_len]

        # ── RANSAC
        sy, sx = np.nonzero(skel)
        pts = np.stack([sx, sy], 1).astype(np.float64)
        if len(pts) > a.max_pts:
            pts = pts[rng.choice(len(pts), a.max_pts, replace=False)]
        ran = sequential_ransac(pts, a.degree, a.tol, a.trials, a.max_lines,
                                a.min_inliers, a.min_span, a.seed)

        rows.append((len(sx), len(hough), len(ran)))
        print(f"{i:3d} {len(sx):8d} {len(hough):6d} {len(ran):8d}")

        # ── 패널 [원본 | 하프 | 란삭]
        base = f.copy()
        pa = base.copy()
        for x1, y1, x2, y2 in hough:
            cv2.line(pa, (x1, y1), (x2, y2), (255, 40, 40), 4)      # 빨강 (RGB)
        pb = base.copy()
        for poly, _, _ in ran:
            draw_poly(pb, poly, (255, 200, 40), 4)                  # 노랑 (RGB)

        rn = m[lh.CLASS_ROAD].cpu().numpy()
        ys, xs = np.nonzero(rn)
        if len(xs) == 0:
            continue
        mgn = a.crop_margin
        x0 = max(int(xs.min()) - mgn, 0); x1c = min(int(xs.max()) + mgn, W)
        y0 = max(int(ys.min()) - mgn, 0); y1c = min(int(ys.max()) + mgn, H)
        panels = [p[y0:y1c, x0:x1c] for p in (base, pa, pb)]
        kind = "직선" if a.degree == 1 else f"{a.degree}차곡선"
        titles = ["1 원본", f"2 하프 {len(hough)}개",
                  f"3 란삭({kind}) {len(ran)}개"]
        chh, cww = y1c - y0, x1c - x0
        pw = a.panel_width
        ph = max(int(round(chh * pw / cww)), 1)
        canvas = Image.new("RGB", (pw * 3, ph), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for j, (p, t) in enumerate(zip(panels, titles)):
            # decode() 는 ffmpeg -pix_fmt rgb24 로 읽으므로 프레임은 **RGB** 다.
            # PIL 도 RGB 를 기대하니 채널을 뒤집으면 안 된다 (뒤집으면 빨강<->파랑).
            im = Image.fromarray(p.clip(0, 255).astype(np.uint8)).resize((pw, ph),
                                                                         Image.LANCZOS)
            cx, cy = j * pw, 0
            canvas.paste(im, (cx, cy))
            d.rectangle([cx, cy, cx + 420, cy + 42], fill=(0, 0, 0))
            d.text((cx + 8, cy + 6), t, fill=(255, 255, 0), font=font)
        canvas.save(outdir / f"{i:02d}.jpg", quality=88)

    R = np.array(rows, dtype=float)
    kind = "직선" if a.degree == 1 else f"{a.degree}차곡선"
    print("\n" + "=" * 60)
    print(f"패널 {len(rows)}장: {outdir}   [원본 | 하프 | 란삭]\n")
    print(f"{'':16}{'검출 선 수':>12}")
    print(f"{'하프 변환':16}{R[:,1].mean():12.1f}")
    print(f"{'RANSAC ' + kind:16}{R[:,2].mean():12.1f}")
    print("\n선 개수는 품질이 아니다. 반드시 패널을 눈으로 볼 것 —")
    print("RANSAC 은 차로를 가로지르는 가짜 선을 만들면서 개수만 늘린다.")


if __name__ == "__main__":
    main()
