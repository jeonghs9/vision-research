#!/usr/bin/env python3
"""road 마스크 안의 밝은 픽셀을 **하프 변환으로 직선으로 묶어** 차선이 되는지 본다.

앞선 실험(`lane_from_road_mask.py`)에서 밝기 임계까지는 했지만 **덩어리만 나오고
선으로 묶이지 않았다** (세장비 중앙값 1.7 — 거의 정사각형). 그 빠진 단계를 채운다.

파이프라인:

    ① road 마스크                seg 모델 (기존)
    ② - car - plate             흰 차량·번호판 제외 (기존)
    ③ 밝기 임계 (ROI 내 백분위)  차선 후보 픽셀 (기존)
    ─────────────────────────── 여기까지 기존 ───────────────────────
    ④ 폭 필터                    두꺼운 덩어리 제거      <- 분리대를 거르는 핵심
    ⑤ 세선화                     선의 중심선만 남김
    ⑥ HoughLinesP                점 -> 직선 선분
    ⑦ 각도·길이 필터             도로 방향과 나란한 것만

**④ 가 핵심이다.** 앞선 실험에서 가장 강하게 잡힌 것이 차선이 아니라 **중앙분리대와
콘크리트 포장**이었다. 각도로는 못 거른다(차선과 방향이 같다). 폭으로 거른다.

    차선 도색     1080p 화면에서 약  5~15 px
    분리대·연석                약 30~100 px

**⑤ 세선화가 필요한 이유** — 차선 도색은 '면'이라 하프에 그대로 넣으면 한 차선에서
양쪽 경계로 선이 2개 나온다. 1픽셀 중심선으로 줄여야 차선 하나당 선 하나가 된다.

판정 기준 (셋 다 통과해야 성공):
  1. 차선이 직선으로 묶이는가       프레임당 4~12개의 긴 선분
  2. 분리대와 구분되는가            폭 필터 후 분리대 선분이 사라짐
  3. 프레임 간 안정적인가           선 개수·각도의 표준편차가 작음

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_hough.py --n 30
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

VIDEO = "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000157_T.mp4"
WEIGHTS = ("/home/hsjeong/workspace/vision-seg/Yolo26-seg/test/output/"
           "yolo26s_seg_1024_b32/weights/best.pt")
OUT = ("/home/hsjeong/workspace/applications/traffic-video-analytics/"
       "test/tmp/01-lane-postprocess-runs/lane_hough")

CLASS_ROAD, CLASS_CAR, CLASS_PLATE = 0, 1, 2


def total_frames(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def decode(video, n, stride, w, h, seed=None):
    """seed 가 None 이면 앞에서부터 stride 간격으로, 주어지면 **영상 전체에서 무작위**로 뽑는다.

    앞부분만 보면 표본이 편향된다. 실제로 이 영상은 앞 30프레임의 도로 면적이 8.8% 인데
    전체 평균은 38% 다 — 앞부분은 도로가 멀리 작게 보이는 구간이다.
    """
    if seed is None:
        vf = f"select='not(mod(n\\,{stride}))'"
        idx = None
    else:
        import random
        N = total_frames(video)
        idx = sorted(random.Random(seed).sample(range(N), min(n, N)))
        # ffmpeg select 는 eq(n,f) 를 + 로 이어 여러 프레임을 고를 수 있다
        vf = "select='" + "+".join(f"eq(n\\,{i})" for i in idx) + "'"
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video,
         "-vf", f"{vf},scale={w}:{h}:flags=bilinear",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    nb, fr = w * h * 3, []
    for _ in range(n):
        b = p.stdout.read(nb)
        if len(b) < nb:
            break
        fr.append(np.frombuffer(b, np.uint8).reshape(h, w, 3).copy())
    p.stdout.close(); p.kill()
    return fr, idx


def class_masks(det, hw, dev):
    import torch.nn.functional as F
    out = {c: torch.zeros(hw, dtype=torch.bool, device=dev) for c in (0, 1, 2)}
    if det.masks is None or len(det.masks) == 0:
        return out
    m = det.masks.data
    if m.shape[-2:] != hw:
        m = F.interpolate(m[None], size=hw, mode="nearest")[0]
    m = m > 0.5
    cls = det.boxes.cls.long()
    for c in (0, 1, 2):
        sel = m[cls == c]
        if sel.numel():
            out[c] = sel.any(0)
    return out


def shape_filter(bin_u8, min_elong, max_width, min_area):
    """④ 연결요소를 **세장비**로 거른다 (폭 대신).

    처음에는 평균 폭(px)에 고정 상한을 뒀는데 실패했다. **드론 고도가 바뀌면 화면상
    차선 폭도 바뀐다** — 멀리서는 5~15px, 가까이서는 30~50px 이다. 고정 18px 상한을
    두니 드론이 낮게 나는 프레임에서 차선이 통째로 걸러졌다(프레임 12: 11개 버림 -> 선 0개).

    **세장비(긴변/짧은변)는 고도가 바뀌어도 유지된다.** 차선은 가늘고 길다는 성질이
    스케일에 무관하기 때문이다. 폭 상한은 아주 큰 값(분리대·포장면 제거용)만 남긴다.

    단, 세장비를 **바운딩 박스**로 재면 대각선 차선에서 무너진다. 이 영상의 차선은
    화면상 30~60도 기울어져 있고, 45도 선의 박스는 정사각형이라 세장비가 1.0 이 된다.
    폭도 `면적/박스변` 이라 sqrt(2) 배 부풀려진다. 그래서 두 임계를 모두 무력화
    (elong 1.0, width 60) 해야 했고, 그 결과 맞은편 차도의 밝은 포장면 덩어리가
    통과해 하프 단계에서 부챗살 오검출을 만들었다.

    `cv2.minAreaRect` 로 **기울어진 최소 외접 사각형**을 구하면 회전에 무관하게
    참 길이·참 폭이 나온다. 대각 차선도 세장비가 제대로 나온다.
    """
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    keep = np.zeros_like(bin_u8)
    kept = dropped = 0
    for k in range(1, n):
        area = st[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        # 덩어리 bbox 안에서만 좌표를 뽑아 minAreaRect 에 넘긴다 (전체 스캔 회피)
        x, y = st[k, cv2.CC_STAT_LEFT], st[k, cv2.CC_STAT_TOP]
        w, h = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == k)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        (rw, rh) = cv2.minAreaRect(pts)[1]
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        elong = long_side / short_side
        mean_w = short_side          # 회전 사각형의 짧은 변 = 참 폭
        if elong >= min_elong and mean_w <= max_width:
            keep[lab == k] = 1
            kept += 1
        else:
            dropped += 1
    return keep, kept, dropped


def skeletonize(bin_u8):
    """⑤ 형태학적 세선화 (Zhang-Suen 대신 cv2 반복 침식 방식).

    scipy/skimage 가 이 환경에 없으므로 OpenCV 만으로 구현한다.
    """
    import cv2
    img = bin_u8.copy()
    skel = np.zeros_like(img)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(60):                      # 폭 상한이 있으므로 60회면 충분
        op = cv2.morphologyEx(img, cv2.MORPH_OPEN, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, op))
        img = cv2.erode(img, el)
        if cv2.countNonZero(img) == 0:
            break
    return skel


def merge_segments(segs, tol_deg, tol_rho):
    """⑧ 같은 직선 위에 놓인 선분 조각들을 하나로 병합한다.

    하프 변환은 점선 차선 하나를 여러 조각으로 내놓는다(실측 40.8개). 조각 수가 아니라
    **차선 수**를 세려면 병합이 필요하다.

    각 선분을 무한직선의 극좌표 (theta, rho) 로 표현해 묶는다.
      theta : 선분의 방향각 [0, 180)
      rho   : 원점에서 그 직선까지의 수직거리 (부호 있음)
    같은 차선의 조각은 theta·rho 가 거의 같고, 다른 차선은 rho 가 크게 다르다.

    묶은 뒤에는 방향벡터에 투영해 양 끝점을 구해 하나의 긴 선분으로 만든다.
    """
    if not segs:
        return []
    feats = []
    for x1, y1, x2, y2 in segs:
        t = np.arctan2(y2 - y1, x2 - x1) % np.pi          # [0, pi)
        r = -x1 * np.sin(t) + y1 * np.cos(t)              # 수직거리
        feats.append((t, r, (x1, y1, x2, y2)))

    tol_t = np.radians(tol_deg)
    clusters = []                                          # [[theta, rho, [segs]], ...]
    for t, r, seg in feats:
        hit = None
        for c in clusters:
            dt = abs(t - c[0]); dt = min(dt, np.pi - dt)   # 각도는 원형이라 179도와 1도가 가깝다
            if dt <= tol_t and abs(r - c[1]) <= tol_rho:
                hit = c; break
        if hit is None:
            clusters.append([t, r, [seg]])
        else:
            n = len(hit[2])
            hit[0] = (hit[0] * n + t) / (n + 1)            # 평균으로 갱신
            hit[1] = (hit[1] * n + r) / (n + 1)
            hit[2].append(seg)

    merged = []
    for t, r, ss in clusters:
        d = np.array([np.cos(t), np.sin(t)])               # 방향벡터
        pts = np.array([[x, y] for x1, y1, x2, y2 in ss for x, y in ((x1, y1), (x2, y2))],
                       dtype=np.float64)
        proj = pts @ d
        p0, p1 = pts[proj.argmin()], pts[proj.argmax()]
        merged.append((int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), len(ss),
                       float(np.hypot(*(p1 - p0)))))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=1920)
    # ③ 밝기
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.35)
    ap.add_argument("--car_dilate", type=int, default=15,
                    help="car 마스크를 이만큼 부풀려 지운다. 마스크 경계가 딱 맞지 "
                         "않아 차 테두리의 밝은 픽셀이 남는다. lane_ransac.py·"
                         "lane_curvature_check.py 와 같은 값으로 맞춘 것")
    # ④ 폭 필터
    ap.add_argument("--min_elong", type=float, default=2.5,
                    help="세장비(긴변/짧은변) 하한. 차선은 가늘고 길다 — 고도에 무관")
    ap.add_argument("--max_width", type=float, default=60.0,
                    help="평균 폭 상한 px. 분리대·포장면 제거용이라 넉넉히 둔다")
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--no_skel", action="store_true",
                    help="⑤ 세선화를 건너뛴다. 하프는 세선화를 요구하지 않지만, 굵은 "
                         "띠에는 직선을 여러 개 그을 수 있어 한 차선에서 평행선이 "
                         "여러 개 나온다. ⑧ 병합이 그걸 대신 묶어줄 수 있는지 실측용")
    # ⑥⑦ 하프
    ap.add_argument("--hough_thresh", type=int, default=40)
    ap.add_argument("--min_len", type=int, default=60)
    ap.add_argument("--max_gap", type=int, default=30)
    ap.add_argument("--angle_min", type=float, default=20.0,
                    help="이 각도(도)보다 수평에 가까운 선은 버린다. 5도에서 20도로 "
                         "올렸다 — 무작위 30장 중 26장이 완전히 동일했고, 바뀐 4장에서 "
                         "사라진 7개는 전부 차로를 가로지르는 잡선이었다 "
                         "(output/01-차선검출모델/figures/angle_compare.jpg). 30도는 8개를 더 지우지만 "
                         "그게 잡선인지 미검증이라 올리지 않는다")
    ap.add_argument("--panel_width", type=int, default=640)
    # 패널을 가로로 6개 이어붙이면 각 칸이 640x360 이라 차선처럼 가는 것이 안 보인다.
    # --crop_road 를 주면 **road 마스크의 경계상자로 잘라** 하늘·건물을 버리고,
    # 2x3 격자로 배치해 칸당 해상도를 크게 만든다.
    ap.add_argument("--crop_road", action="store_true",
                    help="도로 영역만 잘라 2x3 격자로 크게 저장한다")
    ap.add_argument("--crop_margin", type=int, default=40)
    # ⑧ 선분 병합
    ap.add_argument("--merge", action="store_true", help="같은 직선의 조각들을 병합한다")
    ap.add_argument("--merge_deg", type=float, default=4.0, help="병합 각도 허용치(도)")
    ap.add_argument("--merge_rho", type=float, default=25.0, help="병합 수직거리 허용치(px)")
    ap.add_argument("--merge_min_len", type=float, default=0.0,
                    help="병합 후 이 길이 미만 선분은 버린다")
    ap.add_argument("--seed", type=int, default=None,
                    help="지정하면 영상 전체에서 무작위로 n장을 뽑는다 (앞부분 편향 방지)")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO
    from PIL import Image
    import torch.nn.functional as F

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"영상 {Path(a.video).name}  처리 {W}x{H}")
    print(f"③ 밝기 상위 {a.pct}%, 채도<={a.sat_max}")
    print(f"④ 세장비 >= {a.min_elong}, 평균 폭 <= {a.max_width}px, 면적 >= {a.min_area}")
    print(f"⑥ Hough thresh={a.hough_thresh} minLen={a.min_len} maxGap={a.max_gap}")
    print(f"⑦ |각도| >= {a.angle_min}도")
    print(f"출력 {outdir}\n")

    frames, sel_idx = decode(a.video, a.n, a.stride, W, H, a.seed)
    if sel_idx is not None:
        print(f"무작위 표본 (seed {a.seed}): 전체 {total_frames(a.video):,}프레임 중 {len(frames)}장")
        print(f"  프레임 번호: {sel_idx[:8]} ...\n")
    else:
        print(f"앞에서부터 stride {a.stride} 로 {len(frames)}장 (편향 주의)\n")
    model = YOLO(a.weights)

    print(f"{'f':>3} {'후보px%':>8} {'덩어리':>7} {'폭통과':>7} {'폭버림':>7} "
          f"{'선분':>6} {'각도필터후':>10} {'병합후':>7} {'평균길이':>9}")
    print("-" * 80)

    stats = []
    for i, f in enumerate(frames):
        det = model.predict(f[:, :, ::-1], imgsz=a.imgsz, conf=a.conf,
                            device=0, verbose=False)[0]
        m = class_masks(det, (H, W), dev)

        car = m[CLASS_CAR]
        if a.car_dilate > 1:
            car = F.max_pool2d(car[None, None].float(), a.car_dilate, 1,
                               a.car_dilate // 2)[0, 0] > 0.5
        roi = m[CLASS_ROAD] & (~car) & (~m[CLASS_PLATE])

        # ③ 밝기 임계
        rgb_t = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()
        if roi.sum() < 100:
            continue
        mx, _ = rgb_t.max(0); mn, _ = rgb_t.min(0)
        s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))
        thr = torch.quantile(mx[roi].float(), 1.0 - a.pct / 100.0)
        cand = (roi & (mx >= thr) & (s <= a.sat_max)).cpu().numpy().astype(np.uint8)

        # ④ 폭 필터
        narrow, kept, dropped = shape_filter(cand, a.min_elong, a.max_width, a.min_area)
        # ⑤ 세선화 (--no_skel 이면 건너뛴다)
        skel = narrow if a.no_skel else skeletonize(narrow)
        # ⑥ 하프
        lines = cv2.HoughLinesP(skel, 1, np.pi / 180, a.hough_thresh,
                                minLineLength=a.min_len, maxLineGap=a.max_gap)
        # HoughLinesP 는 OpenCV 버전에 따라 (N,1,4) 또는 (N,4) 를 준다. 항상 (N,4) 로 맞춘다.
        lines = np.empty((0, 4), np.int32) if lines is None else np.asarray(lines).reshape(-1, 4)
        n_raw = len(lines)
        # ⑦ 각도 필터 (수평에 가까운 선 제거)
        keep = []
        for x1, y1, x2, y2 in lines:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            ang = min(ang, 180 - ang)
            if ang >= a.angle_min:
                keep.append((x1, y1, x2, y2, np.hypot(x2 - x1, y2 - y1)))
        n_keep = len(keep)
        if a.merge:
            mg = merge_segments([(k[0], k[1], k[2], k[3]) for k in keep],
                                a.merge_deg, a.merge_rho)
            mg = [m for m in mg if m[5] >= a.merge_min_len]
            keep = [(m[0], m[1], m[2], m[3], m[5]) for m in mg]
        mlen = float(np.mean([k[4] for k in keep])) if keep else 0.0
        cp = cand.mean() * 100
        stats.append((cp, kept + dropped, kept, dropped, n_raw, len(keep), mlen, n_keep))
        print(f"{i:>3} {cp:8.3f} {kept+dropped:7d} {kept:7d} {dropped:7d} "
              f"{n_raw:6d} {n_keep:10d} {len(keep):7d} {mlen:9.0f}")

        # 패널 생성
        base = f.astype(np.float32)
        p2 = base.copy()
        rn = m[CLASS_ROAD].cpu().numpy(); cn = car.cpu().numpy()
        p2[rn] = p2[rn] * .5 + np.array([150, 60, 190]) * .5
        p2[cn] = p2[cn] * .5 + np.array([30, 230, 70]) * .5
        p3 = np.dstack([cand * 255] * 3).astype(np.float32)
        p4 = np.dstack([narrow * 255] * 3).astype(np.float32)
        p5 = np.dstack([skel * 255] * 3).astype(np.float32)
        p6 = base.copy().astype(np.uint8)
        for x1, y1, x2, y2, _ in keep:
            cv2.line(p6, (x1, y1), (x2, y2), (255, 40, 40), 4 if a.merge else 3)
        panels = [base, p2, p3, p4, p5, p6.astype(np.float32)]
        titles = ["1 원본", "2 road-car", "3 밝기후보", "4 폭필터", "5 세선화", "6 하프선분"]

        if a.crop_road:
            ys, xs = np.nonzero(rn)
            if len(xs) == 0:
                continue
            m_ = a.crop_margin
            x0 = max(int(xs.min()) - m_, 0); x1 = min(int(xs.max()) + m_, W)
            y0 = max(int(ys.min()) - m_, 0); y1 = min(int(ys.max()) + m_, H)
            panels = [p[y0:y1, x0:x1] for p in panels]
            ch, cw = y1 - y0, x1 - x0
            pw = 960; ph = max(int(round(ch * pw / cw)), 1)
            canvas = Image.new("RGB", (pw * 3, ph * 2), (20, 20, 20))
            for j, p in enumerate(panels):
                im = Image.fromarray(p.clip(0, 255).astype(np.uint8)).resize((pw, ph), Image.LANCZOS)
                canvas.paste(im, (pw * (j % 3), ph * (j // 3)))
        else:
            pw = a.panel_width; ph = int(round(H * pw / W))
            canvas = Image.new("RGB", (pw * 6, ph))
            for j, p in enumerate(panels):
                canvas.paste(Image.fromarray(p.clip(0, 255).astype(np.uint8))
                             .resize((pw, ph), Image.BILINEAR), (pw * j, 0))
        canvas.save(outdir / f"{i:02d}.jpg", quality=92)

    s = np.array(stats)
    print(f"\n패널 {len(stats)}장: {outdir}")
    print("  [원본 | road-car | ③밝기후보 | ④폭필터 | ⑤세선화 | ⑥⑦하프선분]\n")
    print(f"후보 픽셀 비율      {s[:,0].mean():7.3f} %")
    print(f"덩어리(면적>={a.min_area})  {s[:,1].mean():7.1f} 개")
    print(f"  폭 통과           {s[:,2].mean():7.1f} 개")
    print(f"  폭 버림(두꺼움)   {s[:,3].mean():7.1f} 개")
    print(f"하프 선분 (각도필터 전) {s[:,4].mean():7.1f} 개")
    print(f"하프 선분 (각도필터 후) {s[:,7].mean():7.1f} 개")
    if a.merge:
        print(f"**병합 후**            {s[:,5].mean():7.1f} 개  표준편차 {s[:,5].std():.1f}"
              f"   (조각 {s[:,7].mean():.1f} -> {s[:,5].mean():.1f})")
    else:
        print(f"  (병합 미적용 — --merge 로 켠다)")
    print(f"선분 평균 길이       {s[:,6].mean():7.0f} px")
    print("\n판정:")
    print(f"  1. 직선으로 묶이는가   선분 {s[:,5].mean():.1f}개 (기준 4~12개)")
    print(f"  2. 분리대 구분되는가   패널 4번(폭필터)에서 분리대가 사라졌는지 눈으로 확인")
    print(f"  3. 프레임간 안정적인가 선분 수 표준편차 {s[:,5].std():.1f} (작을수록 안정)")


if __name__ == "__main__":
    main()
