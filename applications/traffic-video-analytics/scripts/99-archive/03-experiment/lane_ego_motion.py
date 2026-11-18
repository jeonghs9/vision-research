#!/usr/bin/env python3
"""드론 자체 움직임(ego-motion)을 추정하고 **정렬이 실제로 맞는지 검증**한다.

시간축 누적의 전제는 "이전 프레임을 현재 좌표로 옮기면 차선이 겹친다" 이다.
그게 안 맞으면 누적은 차선을 번지게 만들 뿐이다. 그래서 누적을 붙이기 전에
**정렬 품질부터 잰다.**

## 지표 — 정렬 IoU

이전 프레임의 차선 후보(④번 결과)를 현재 좌표로 옮긴 뒤, 현재 프레임의 후보와
얼마나 겹치는지를 IoU 로 잰다.

**이 지표는 조작할 수 없다.** 정렬이 틀리면 무조건 낮아진다.
(07 문서 8.3절의 "설명률" 은 틀린 선일수록 높아지는 지표라 폐기했다.
그 실수를 피하려고 결과가 아니라 **전제**를 재는 지표를 골랐다.)

보정 없이 그냥 겹쳤을 때(항등변환)를 같이 재서 **보정이 실제로 얼마나 버는지**
비교한다.

## 검증하는 갈림길 두 가지

1. **아핀 vs 호모그래피**
   도로는 평면이므로 원리적으로는 호모그래피(8 자유도)가 맞다. 아핀 부분추정
   (`estimateAffinePartial2D`, 4 자유도)은 원근 변화를 표현하지 못한다.
   그러나 자유도가 높으면 잡음에 흔들리기도 한다. 실측으로 고른다.

2. **특징점을 road 마스크로 제한할 것인가**
   건물·나무는 도로와 다른 평면(다른 높이)이라 시차(parallax)로 다르게 움직인다.
   road 안으로 제한하면 도로 평면에 맞춰진다. 우리는 road 마스크가 있으니 공짜다.

## k 를 늘리면

t-k 를 t 로 옮기려면 변환을 k 번 곱해야 하고 **오차가 누적된다.**
k=1,3,5,9 를 재서 몇 장까지 안전한지 본다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_ego_motion.py
"""
import argparse
import importlib.util
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grab(video, start, n, w, h):
    """start 프레임부터 연속 n 장 (stride 1)."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video,
         "-vf", f"select='gte(n\\,{start})',scale={w}:{h}:flags=bilinear",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    nb, out = w * h * 3, []
    for _ in range(n):
        b = p.stdout.read(nb)
        if len(b) < nb:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(h, w, 3).copy())
    p.stdout.close(); p.kill()
    return out


def to3x3(M):
    if M is None:
        return None
    if M.shape == (2, 3):
        return np.vstack([M, [0, 0, 1]]).astype(np.float64)
    return M.astype(np.float64)


def estimate(prev_gray, cur_gray, mask, model, scale, n_feat=300,
             win=15, levels=2):
    """prev -> cur 변환을 구한다. 반환: dict(방법이름 -> 3x3, 1920 기준)

    `mask` 는 특징점을 제한할 영역 (None 이면 전체).

    특징점 수와 추적 창은 속도에 직결된다. 실측(구간 4500, 이동 중)에서
    점 800/win21/lvl3 은 6.4ms, 점 200/win15/lvl2 는 3.8ms 였고 **추정 이동량
    차이는 0.2px**(전체 이동 17px 의 1 %) 였다. 정확도 손실 없이 줄일 수 있다.
    """
    p0 = cv2.goodFeaturesToTrack(prev_gray, n_feat, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 20:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                         winSize=(win, win), maxLevel=levels)
    ok = st.ravel() == 1
    if ok.sum() < 20:
        return None, int(ok.sum())
    a, b = p0[ok], p1[ok]

    out = {}
    if model in ("affine", "both"):
        M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        out["affine"] = to3x3(M)
    if model in ("homography", "both"):
        if len(a) >= 8:
            Hm, _ = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
            out["homography"] = to3x3(Hm)
        else:
            out["homography"] = None

    # 작은 해상도에서 구한 변환을 1920 좌표계로 환산: S H S^-1
    S = np.diag([scale, scale, 1.0])
    Si = np.diag([1 / scale, 1 / scale, 1.0])
    for k in list(out):
        out[k] = None if out[k] is None else S @ out[k] @ Si
    return out, int(ok.sum())


def iou(a, b):
    u = np.count_nonzero(a | b)
    return 0.0 if u == 0 else np.count_nonzero(a & b) / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--starts", type=int, nargs="+",
                    default=[500, 2500, 4500, 6500, 8000])
    ap.add_argument("--n", type=int, default=24, help="구간마다 연속 프레임 수")
    ap.add_argument("--small", type=int, default=480, help="자체 움직임 추정 해상도 가로")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 9])
    # ①~④ (lane_hough 현재 기준)
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.10)
    ap.add_argument("--min_elong", type=float, default=4.0)
    ap.add_argument("--max_width", type=float, default=60.0)
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--car_dilate", type=int, default=15)
    a = ap.parse_args()

    lh = load("lane_hough")
    from ultralytics import YOLO
    video = a.video or lh.VIDEO
    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    SW = a.small
    SH = int(round(H * SW / W))
    scale = W / SW
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(lh.WEIGHTS)

    print(f"자체 움직임 추정 + 정렬 검증   {W}x{H} (추정은 {SW}x{SH})")
    print(f"구간 {a.starts}, 구간당 {a.n}프레임 (stride 1)\n")

    methods = ["affine_all", "affine_road", "homog_all", "homog_road"]
    # 결과: [방법][k] -> IoU 목록,  그리고 보정 없음(baseline)
    acc = {m: {k: [] for k in a.ks} for m in methods}
    acc["none"] = {k: [] for k in a.ks}
    per_seg = {}

    for start in a.starts:
        frames = grab(video, start, a.n, W, H)
        if len(frames) < max(a.ks) + 2:
            continue
        cands, grays, roads = [], [], []
        for f in frames:
            det = model.predict(f[:, :, ::-1], imgsz=1024, conf=0.25,
                                device=0, verbose=False)[0]
            m = lh.class_masks(det, (H, W), dev)
            car = F.max_pool2d(m[lh.CLASS_CAR][None, None].float(),
                               a.car_dilate, 1, a.car_dilate // 2)[0, 0] > 0.5
            roi = m[lh.CLASS_ROAD] & (~car) & (~m[lh.CLASS_PLATE])
            rgb = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()
            mx, _ = rgb.max(0); mn, _ = rgb.min(0)
            s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6),
                            torch.zeros_like(mx))
            if roi.sum() < 100:
                cands.append(np.zeros((H, W), np.uint8))
            else:
                thr = torch.quantile(mx[roi].float(), 1.0 - a.pct / 100.0)
                cand = ((mx >= thr) & (s <= a.sat_max) & roi).to(torch.uint8).cpu().numpy()
                nar, _, _ = lh.shape_filter(cand, a.min_elong, a.max_width, a.min_area)
                cands.append(nar)
            grays.append(cv2.cvtColor(cv2.resize(f, (SW, SH)), cv2.COLOR_RGB2GRAY))
            rn = m[lh.CLASS_ROAD].cpu().numpy().astype(np.uint8)
            roads.append(cv2.resize(rn, (SW, SH), interpolation=cv2.INTER_NEAREST) * 255)

        # 연속 프레임 변환 (i-1 -> i)
        steps = {m: [None] * len(frames) for m in methods}
        for i in range(1, len(frames)):
            for tag, msk in (("all", None), ("road", roads[i - 1])):
                got, _ = estimate(grays[i - 1], grays[i], msk, "both", scale)
                if got is None:
                    continue
                steps[f"affine_{tag}"][i] = got.get("affine")
                steps[f"homog_{tag}"][i] = got.get("homography")

        seg = {m: {k: [] for k in a.ks} for m in methods}
        seg["none"] = {k: [] for k in a.ks}
        for t in range(max(a.ks), len(frames)):
            cur = cands[t].astype(bool)
            if cur.sum() < 200:
                continue
            for k in a.ks:
                seg["none"][k].append(iou(cands[t - k].astype(bool), cur))
                for m in methods:
                    Mc = np.eye(3)
                    bad = False
                    for j in range(t - k + 1, t + 1):
                        if steps[m][j] is None:
                            bad = True; break
                        Mc = steps[m][j] @ Mc
                    if bad:
                        continue
                    warped = cv2.warpPerspective(cands[t - k], Mc, (W, H),
                                                 flags=cv2.INTER_NEAREST)
                    seg[m][k].append(iou(warped.astype(bool), cur))
        per_seg[start] = seg
        for m in list(methods) + ["none"]:
            for k in a.ks:
                acc[m][k].extend(seg[m][k])

    def fmt(v):
        return f"{np.mean(v)*100:6.1f}" if v else "   -  "

    print("=" * 78)
    print("정렬 IoU (%) — 이전 프레임 후보를 현재로 옮겨 현재 후보와 겹친 정도\n")
    hdr = "".join(f"{'k='+str(k):>9}" for k in a.ks)
    print(f"{'방법':<16}{hdr}")
    print("-" * 78)
    print(f"{'보정 없음':<16}" + "".join(fmt(acc['none'][k]) + "   " for k in a.ks))
    for m in methods:
        print(f"{m:<16}" + "".join(fmt(acc[m][k]) + "   " for k in a.ks))

    print("\n" + "=" * 78)
    print("구간별 (k=5 기준) — 드론 정지/이동 구간 비교\n")
    print(f"{'구간':>8}{'보정없음':>10}" + "".join(f"{m:>14}" for m in methods))
    print("-" * 78)
    for st, seg in per_seg.items():
        row = f"{st:>8}{fmt(seg['none'][5]):>10}"
        for m in methods:
            row += f"{fmt(seg[m][5]):>14}"
        print(row)

    print("\n판단 기준")
    print("  - 보정 후 IoU 가 보정 전보다 뚜렷이 높아야 누적이 의미 있다")
    print("  - k 가 커져도 IoU 가 급락하지 않는 지점까지가 안전한 창 크기")
    print("  - 이동 구간에서 보정 전/후 차이가 크게 벌어져야 한다 (정지 구간은 원래 잘 맞음)")


if __name__ == "__main__":
    main()
