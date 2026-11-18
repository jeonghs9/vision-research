#!/usr/bin/env python3
"""시간축 누적 — 자체 움직임을 보정해 여러 프레임의 차선 증거를 겹쳐 쌓는다.

## 왜

차선 도색은 **지면에 고정**돼 있고 드론만 움직인다. 자체 움직임을 보정해 정렬하면
여러 프레임의 차선이 같은 자리에 겹친다. 반대로 **차량은 움직이므로 번져서 표를
못 받는다.** 트럭에 가려진 구간도 앞뒤 프레임이 메운다.

`lane_ego_motion.py` 로 잰 근거:

| | k=5 정렬 IoU |
|---|---|
| 보정 없음 | 16.1 % |
| 보정 후(호모그래피) | 44.1 % |

**보정 없이 5장을 쌓으면 84 % 가 어긋난 채 겹쳐진다.** 번지기만 한다.

또 하나: 드론이 거의 안 움직이는 구간(2.1px/frame)에서도 k=1 IoU 가 65.7 % 다.
**후보 마스크 자체가 프레임마다 흔들린다** (이동 구간은 후보 픽셀 수가 52 % 출렁).
그게 "절반 미검출" 의 원인이고, **여러 프레임 투표가 바로 그 약**이다.

## 표본을 어떻게 뽑나

누적은 **연속 프레임**이 필요한데 기존 검증은 무작위 30장이었다. 무작위로 뽑은
프레임끼리는 몇 분씩 떨어져 겹칠 수 없다.

**무작위 30장을 기준점(anchor)으로 두고, 각 기준점의 직전 N-1 장을 같이 가져온다.**
표본의 공정성(07 문서 4.1절)을 유지하면서 연속 창을 확보한다. 패널은 기준점
프레임에 대해 그리므로 기존 대조표와 **직접 비교**된다.

## 설정 근거

| 항목 | 값 | 근거 (lane_ego_motion.py) |
|---|---|---|
| 변환 모델 | **호모그래피** | k=1·3·5 에서 모두 최고 |
| 특징점 범위 | **화면 전체** | road 로 제한하면 6 % 손해 — 아스팔트에 무늬가 없다 |
| 창 크기 N | **5** | k=9 에서 44.1 -> 27.7 로 급락 |
| 채택 표수 M | 2 | 5장 중 2장 이상. 실측으로 조정 |

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_accum.py --seed 0 --n 30 --win 5 --vote 2
"""
import argparse
import importlib.util
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_windows(video, anchors, win, w, h):
    """기준점마다 [anchor-win+1 .. anchor] 를 한 번의 ffmpeg 호출로 받는다."""
    need = []
    for a in anchors:
        need.extend(range(max(a - win + 1, 0), a + 1))
    need = sorted(set(need))
    vf = "select='" + "+".join(f"eq(n\\,{i})" for i in need) + "'"
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video,
         "-vf", f"{vf},scale={w}:{h}:flags=bilinear",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    nb, got = w * h * 3, {}
    for i in need:
        b = p.stdout.read(nb)
        if len(b) < nb:
            break
        got[i] = np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
    p.stdout.close(); p.kill()
    return got


def lines_from(skel, a, lh):
    """⑥ 하프 -> ⑦ 각도 -> ⑧ 병합."""
    L = cv2.HoughLinesP(skel, 1, np.pi / 180, a.hough_thresh,
                        minLineLength=a.min_len, maxLineGap=a.max_gap)
    segs = []
    if L is not None:
        for x1, y1, x2, y2 in np.asarray(L).reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if min(ang, 180 - ang) >= a.angle_min:
                segs.append((int(x1), int(y1), int(x2), int(y2)))
    mg = lh.merge_segments(segs, a.merge_deg, a.merge_rho)
    return [m for m in mg if m[5] >= a.merge_min_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--win", type=int, default=5, help="누적 창 크기 N")
    ap.add_argument("--vote", type=int, default=2, help="채택 표수 M (N 중 M)")
    ap.add_argument("--refilter", action="store_true",
                    help="누적 후 세장비 필터를 **다시** 적용한다. 정렬 잔차 때문에 "
                         "누적하면 선이 옆으로 번져 굵어지는데, 필터를 누적 전에만 "
                         "걸면 그 굵어짐을 아무도 거르지 못한다")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--small", type=int, default=480)
    ap.add_argument("--out", default="test/tmp/01-lane-postprocess-runs/lane_accum")
    # ①~④
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.10)
    ap.add_argument("--min_elong", type=float, default=4.0)
    ap.add_argument("--max_width", type=float, default=60.0)
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--car_dilate", type=int, default=15)
    # ⑥⑦⑧
    ap.add_argument("--hough_thresh", type=int, default=25)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_gap", type=int, default=60)
    ap.add_argument("--angle_min", type=float, default=20.0)
    ap.add_argument("--merge_deg", type=float, default=4.0)
    ap.add_argument("--merge_rho", type=float, default=25.0)
    ap.add_argument("--merge_min_len", type=float, default=120.0)
    ap.add_argument("--crop_margin", type=int, default=40)
    ap.add_argument("--panel_width", type=int, default=960)
    a = ap.parse_args()

    lh = load("lane_hough")
    lem = load("lane_ego_motion")
    from ultralytics import YOLO
    import random

    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    SW = a.small; SH = int(round(H * SW / W)); sc = W / SW
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)

    N = lh.total_frames(lh.VIDEO)
    anchors = sorted(random.Random(a.seed).sample(range(N), min(a.n, N)))
    anchors = [max(x, a.win - 1) for x in anchors]
    print(f"기준점 {len(anchors)}개 (seed {a.seed}) — 각 기준점의 직전 {a.win-1}장 포함")
    print(f"창 N={a.win}, 채택 M={a.vote}, 호모그래피(화면 전체 특징점)")
    print(f"출력 {outdir}\n")

    frames = decode_windows(lh.VIDEO, anchors, a.win, W, H)
    model = YOLO(lh.WEIGHTS)
    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default()

    def stage14(f):
        """①~④ -> (후보 픽셀, road 마스크)"""
        det = model.predict(f[:, :, ::-1], imgsz=1024, conf=0.25,
                            device=0, verbose=False)[0]
        m = lh.class_masks(det, (H, W), dev)
        car = F.max_pool2d(m[lh.CLASS_CAR][None, None].float(),
                           a.car_dilate, 1, a.car_dilate // 2)[0, 0] > 0.5
        roi = m[lh.CLASS_ROAD] & (~car) & (~m[lh.CLASS_PLATE])
        rn = m[lh.CLASS_ROAD].cpu().numpy()
        if roi.sum() < 100:
            return np.zeros((H, W), np.uint8), rn
        rgb = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()
        mx, _ = rgb.max(0); mn, _ = rgb.min(0)
        s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))
        thr = torch.quantile(mx[roi].float(), 1.0 - a.pct / 100.0)
        cand = ((mx >= thr) & (s <= a.sat_max) & roi).to(torch.uint8).cpu().numpy()
        nar, _, _ = lh.shape_filter(cand, a.min_elong, a.max_width, a.min_area)
        return nar, rn

    print(f"{'f':>3}{'기준프레임':>10}{'단일 px%':>10}{'누적 px%':>10}"
          f"{'단일 선':>8}{'누적 선':>8}{'정렬성공':>9}{'번짐비율':>10}")
    print("-" * 74)
    stats = []
    for i, anc in enumerate(anchors):
        idxs = [j for j in range(anc - a.win + 1, anc + 1) if j in frames]
        if len(idxs) < 2:
            continue
        masks, grays = [], []
        for j in idxs:
            nar, rn = stage14(frames[j])
            masks.append(nar)
            grays.append(cv2.cvtColor(cv2.resize(frames[j], (SW, SH)),
                                      cv2.COLOR_RGB2GRAY))
            if j == anc:
                road_cur = rn
        cur = masks[-1]

        # 각 과거 프레임을 현재(기준점) 좌표로 정렬해 투표
        votes = cur.astype(np.int32).copy()
        ok_cnt = 0
        for t in range(len(idxs) - 2, -1, -1):
            Mc = np.eye(3)
            bad = False
            for j in range(t + 1, len(idxs)):          # t -> 현재까지 누적 곱
                got, _ = lem.estimate(grays[j - 1], grays[j], None, "homography", sc)
                Hm = got.get("homography") if got else None
                if Hm is None:
                    bad = True; break
                Mc = Hm @ Mc
            if bad:
                continue
            votes += cv2.warpPerspective(masks[t], Mc, (W, H),
                                         flags=cv2.INTER_NEAREST).astype(np.int32)
            ok_cnt += 1
        accum = (votes >= a.vote).astype(np.uint8)
        if a.refilter:
            accum, _, _ = lh.shape_filter(accum, a.min_elong, a.max_width, a.min_area)

        # 진단: 새로 생긴 픽셀이 기존 선에 붙어 있나(번짐), 떨어져 있나(진짜 회복)
        newly_m = (accum > 0) & (cur == 0)
        if newly_m.any():
            near = cv2.dilate(cur, np.ones((7, 7), np.uint8)) > 0
            thick_ratio = float((newly_m & near).sum()) / float(newly_m.sum())
        else:
            thick_ratio = 0.0

        sk_s = lh.skeletonize(cur)
        sk_a = lh.skeletonize(accum)
        ln_s = lines_from(sk_s, a, lh)
        ln_a = lines_from(sk_a, a, lh)
        stats.append((cur.mean() * 100, accum.mean() * 100,
                      len(ln_s), len(ln_a), ok_cnt, thick_ratio * 100))
        print(f"{i:3d}{anc:10d}{cur.mean()*100:10.3f}{accum.mean()*100:10.3f}"
              f"{len(ln_s):8d}{len(ln_a):8d}{ok_cnt:6d}/{len(idxs)-1}"
              f"{thick_ratio*100:9.0f}%")

        # 패널
        base = frames[anc]
        p2 = np.dstack([cur * 255] * 3)
        p3 = np.dstack([accum * 255] * 3)
        # 누적으로 새로 생긴 부분을 초록, 원래 있던 곳은 흰색
        p4 = np.dstack([cur * 255] * 3)
        newly = (accum > 0) & (cur == 0)
        p4[newly] = (60, 255, 90)
        p5 = base.copy()
        for x1, y1, x2, y2, _, _ in ln_s:
            cv2.line(p5, (x1, y1), (x2, y2), (255, 40, 40), 4, cv2.LINE_AA)
        p6 = base.copy()
        for x1, y1, x2, y2, _, _ in ln_a:
            cv2.line(p6, (x1, y1), (x2, y2), (60, 255, 90), 4, cv2.LINE_AA)

        ys, xs = np.nonzero(road_cur)
        if len(xs) == 0:
            continue
        mg = a.crop_margin
        x0 = max(int(xs.min()) - mg, 0); x1c = min(int(xs.max()) + mg, W)
        y0 = max(int(ys.min()) - mg, 0); y1c = min(int(ys.max()) + mg, H)
        panels = [p[y0:y1c, x0:x1c] for p in (base, p2, p3, p4, p5, p6)]
        titles = ["1 원본", "2 단일 프레임 후보", f"3 누적 후보 ({a.win}중 {a.vote}표)",
                  "4 새로 채워진 곳(초록)", f"5 하프 단일 {len(ln_s)}개",
                  f"6 하프 누적 {len(ln_a)}개"]
        ch, cw = y1c - y0, x1c - x0
        pw = a.panel_width; ph = max(int(round(ch * pw / cw)), 1)
        canvas = Image.new("RGB", (pw * 3, ph * 2), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for j, (p, t) in enumerate(zip(panels, titles)):
            im = Image.fromarray(p.clip(0, 255).astype(np.uint8)).resize(
                (pw, ph), Image.LANCZOS)
            cx, cy = (j % 3) * pw, (j // 3) * ph
            canvas.paste(im, (cx, cy))
            d.rectangle([cx, cy, cx + 470, cy + 42], fill=(0, 0, 0))
            d.text((cx + 8, cy + 6), t, fill=(255, 255, 0), font=font)
        canvas.save(outdir / f"{i:02d}.jpg", quality=88)

    S = np.array(stats, dtype=float)
    print("\n" + "=" * 62)
    print(f"패널 {len(stats)}장: {outdir}")
    print("  [원본 | 단일후보 | 누적후보 | 새로채워진곳 | 하프단일 | 하프누적]\n")
    print(f"{'':18}{'단일':>10}{'누적':>10}{'변화':>10}")
    print("-" * 50)
    print(f"{'후보 픽셀 %':<18}{S[:,0].mean():10.3f}{S[:,1].mean():10.3f}"
          f"{(S[:,1].mean()/max(S[:,0].mean(),1e-9)-1)*100:+9.1f}%")
    print(f"{'검출 선 수':<18}{S[:,2].mean():10.2f}{S[:,3].mean():10.2f}"
          f"{S[:,3].mean()-S[:,2].mean():+10.2f}")
    print(f"{'선 수 표준편차':<18}{S[:,2].std():10.2f}{S[:,3].std():10.2f}"
          f"{S[:,3].std()-S[:,2].std():+10.2f}")
    print(f"{'선 0개 프레임':<18}{int((S[:,2]==0).sum()):10d}{int((S[:,3]==0).sum()):10d}")
    print(f"\n정렬 성공률 {S[:,4].mean():.2f}/{a.win-1} "
          f"({S[:,4].mean()/(a.win-1)*100:.0f}%)")
    print(f"**번짐 비율 {S[:,5].mean():.0f} %** — 새로 생긴 픽셀 중 기존 선 7px 안에 "
          f"붙은 것의 비율")
    print("  높으면 = 선이 굵어지기만 함 (정렬 잔차).  낮으면 = 떨어진 곳이 새로 회복됨")
    print("\n※ 선 개수는 품질이 아니다. 반드시 패널 4번(새로 채워진 곳)을 눈으로 볼 것 —")
    print("  차선이 채워졌는지, 아니면 엉뚱한 곳이 번졌는지는 그림으로만 판단된다.")


if __name__ == "__main__":
    main()
