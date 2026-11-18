#!/usr/bin/env python3
"""차선 검출 결과를 **영상**으로 뽑는다. 하프와 란삭을 각각 따로 만든다.

`lane_hough.py` 의 ①~⑤ 단계와 `lane_ransac.py` 의 순차 RANSAC 을 그대로 불러 쓴다.
정지 프레임 패널(`lane_hough.py`, `lane_ransac.py`)과 달리 **연속 구간**을 처리해
프레임 간 흔들림·놓침이 시간축에서 어떻게 보이는지 확인하는 것이 목적이다.

방법마다 별도 파일을 만든다 (`--method hough` / `--method ransac`).

## 속도

란삭은 프레임당 최대 `max_lines x 2방향` 번 적합해서 하프보다 훨씬 느리다.
`--max_pts` 와 `--trials` 로 조절한다. 실측값은 실행 시 표시된다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_video.py --method hough  --start 60 --dur 20
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_video.py --method ransac --start 60 --dur 20
"""
import argparse
import importlib.util
import subprocess
import time
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


def stream_frames(video, start, dur, w, h):
    """구간을 순서대로 디코딩해 프레임을 하나씩 내보낸다 (전부 메모리에 올리지 않는다)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video]
    if dur > 0:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"scale={w}:{h}:flags=bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    nb = w * h * 3
    while True:
        b = p.stdout.read(nb)
        if len(b) < nb:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3).copy()
    p.stdout.close()
    p.wait()


def open_writer(path, w, h, fps):
    """RGB 프레임을 받아 mp4 로 인코딩하는 ffmpeg 파이프."""
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, bufsize=10 ** 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hough", "ransac"], required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="시작 초")
    ap.add_argument("--dur", type=float, default=20.0, help="길이 초 (0=끝까지)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--limit", type=int, default=0, help="프레임 수 상한 (속도 측정용)")
    # 시간축 누적 (lane_accum.py 와 같은 방식, 연속 프레임이므로 창이 자연스럽게 채워진다)
    ap.add_argument("--accum", type=int, default=0,
                    help="누적 창 크기 N (0=끄기). 이전 N-1 장을 호모그래피로 현재 "
                         "좌표에 정렬해 픽셀별로 투표한다")
    ap.add_argument("--vote", type=int, default=2, help="(구버전 호환) 미사용")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="누적 지도에 현재 프레임을 섞는 비율. 작을수록 과거를 오래 "
                         "기억한다. 0.35 면 t-k 프레임의 가중치가 0.35*0.65^k 로 "
                         "약 5프레임 창에 해당한다")
    ap.add_argument("--acc_thr", type=int, default=90,
                    help="누적 지도에서 이 값 이상이면 차선 후보로 채택 (0~255). "
                         "90 은 '최근 5장 중 2장에서 보임' 에 대응한다")
    ap.add_argument("--small", type=int, default=480, help="자체 움직임 추정 해상도")
    # ①~⑤ — lane_hough.py 현재 기준과 동일
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--sat_max", type=float, default=0.10)
    ap.add_argument("--min_elong", type=float, default=4.0)
    ap.add_argument("--max_width", type=float, default=60.0)
    ap.add_argument("--min_area", type=int, default=40)
    ap.add_argument("--car_dilate", type=int, default=15)
    # 하프
    ap.add_argument("--hough_thresh", type=int, default=25)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_gap", type=int, default=60)
    ap.add_argument("--angle_min", type=float, default=20.0)
    ap.add_argument("--merge_deg", type=float, default=4.0)
    ap.add_argument("--merge_rho", type=float, default=25.0)
    ap.add_argument("--merge_min_len", type=float, default=120.0)
    # 란삭
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--max_lines", type=int, default=10)
    ap.add_argument("--min_inliers", type=int, default=60)
    ap.add_argument("--min_span", type=float, default=120.0)
    ap.add_argument("--max_pts", type=int, default=6000)
    a = ap.parse_args()

    lh = load("lane_hough")
    lr = load("lane_ransac")
    lem = load("lane_ego_motion")
    from ultralytics import YOLO

    video = a.video or lh.VIDEO
    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    outdir = Path("/home/hsjeong/workspace/applications/traffic-video-analytics/output/01-차선검출모델/videos")
    outdir.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else outdir / f"lane_{a.method}_{int(a.start)}s_{int(a.dur)}s.mp4"

    model = YOLO(lh.WEIGHTS)
    # 누적용 상태 — 아래 출력문에서도 쓰므로 여기서 먼저 정의한다
    SW = a.small
    SH = int(round(H * SW / W))
    sc = W / SW
    hist = []               # 현재 좌표로 정렬해 보관 중인 최근 N 장의 후보 마스크
    prev_small = None       # 직전 프레임의 축소 회색조 (자체 움직임 추정용)
    acc = None              # 누적 지도 (uint8). 현재 좌표계로 계속 따라온다
    A = {k: 0.0 for k in ("축소", "움직임추정", "워핑", "투표")}   # 누적 단계 세부 계측
    color = (255, 40, 40) if a.method == "hough" else (255, 200, 40)   # 빨강 / 노랑 (RGB)
    if a.accum > 1:
        color = (60, 255, 90)                                          # 누적은 초록
    print(f"방법 {a.method} | 구간 {a.start}s 부터 {a.dur}s | {W}x{H} @ {a.fps}fps")
    if a.accum > 1:
        print(f"  누적: 지수감쇠 alpha={a.alpha}, 임계 {a.acc_thr}/255, "
              f"호모그래피 {SW}x{SH}")
    if a.method == "ransac":
        print(f"  란삭: degree={a.degree} trials={a.trials} max_pts={a.max_pts} "
              f"max_lines={a.max_lines}")
    print(f"출력 {out}\n")

    wr = open_writer(out, W, H, a.fps)
    t0 = time.time()
    n = 0
    n_lines = []
    # 구간별 누적 시간. GPU 는 비동기라 각 구간 끝에서 synchronize 해야 정확하다.
    T = {k: 0.0 for k in ("디코드", "모델추론", "마스크·ROI", "③밝기임계",
                          "④형태필터", "자체움직임·누적", "⑤세선화", "⑥선검출",
                          "그리기·인코딩")}
    cuda = dev.type == "cuda"

    def tick():
        if cuda:
            torch.cuda.synchronize()
        return time.time()

    tprev = tick()
    for f in stream_frames(video, a.start, a.dur, W, H):
        t = tick(); T["디코드"] += t - tprev; tprev = t

        det = model.predict(f[:, :, ::-1], imgsz=1024, conf=0.25,
                            device=0, verbose=False)[0]
        t = tick(); T["모델추론"] += t - tprev; tprev = t
        m = lh.class_masks(det, (H, W), dev)
        car = m[lh.CLASS_CAR]
        if a.car_dilate > 1:
            car = F.max_pool2d(car[None, None].float(), a.car_dilate, 1,
                               a.car_dilate // 2)[0, 0] > 0.5
        roi = m[lh.CLASS_ROAD] & (~car) & (~m[lh.CLASS_PLATE])

        t = tick(); T["마스크·ROI"] += t - tprev; tprev = t

        vis = f.copy()
        k = 0
        if roi.sum() >= 100:
            rgb = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()
            mx, _ = rgb.max(0)
            mn, _ = rgb.min(0)
            s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))
            thr = torch.quantile(mx[roi].float(), 1.0 - a.pct / 100.0)
            cand = ((mx >= thr) & (s <= a.sat_max) & roi).to(torch.uint8).cpu().numpy()
            t = tick(); T["③밝기임계"] += t - tprev; tprev = t

            narrow, _, _ = lh.shape_filter(cand, a.min_elong, a.max_width, a.min_area)
            t = tick(); T["④형태필터"] += t - tprev; tprev = t

            if a.accum > 1:
                ta = time.time()
                small = cv2.cvtColor(cv2.resize(f, (SW, SH)), cv2.COLOR_RGB2GRAY)
                A["축소"] += time.time() - ta; ta = time.time()

                # 이전 프레임 -> 현재 변환을 구해, 보관 중인 마스크들을 현재로 끌어온다
                Hm = None
                if prev_small is not None:
                    got, _ = lem.estimate(prev_small, small, None, "homography", sc)
                    Hm = got.get("homography") if got else None
                A["움직임추정"] += time.time() - ta; ta = time.time()

                if prev_small is not None and Hm is None:
                    hist.clear(); acc = None      # 추적 실패 -> 누적 중단
                A["워핑"] += time.time() - ta; ta = time.time()

                prev_small = small
                # 마스크를 N장 보관해 매번 N-1번 워핑하면 비싸다(실측 14.5ms).
                # **누적 지도 한 장만** 유지하고 그것만 워핑하면 1번으로 끝난다.
                # 창(N중 M)이 지수 감쇠로 바뀌지만, 최근 프레임에 가중치를 더 주는
                # 쪽이라 목적에 어긋나지 않는다.
                cur255 = narrow * 255
                if acc is None:
                    acc = cur255.copy()
                else:
                    if Hm is not None:
                        acc = cv2.warpPerspective(acc, Hm, (W, H),
                                                  flags=cv2.INTER_NEAREST)
                    cv2.addWeighted(acc, 1.0 - a.alpha, cur255, a.alpha, 0, dst=acc)
                narrow = (acc >= a.acc_thr).astype(np.uint8)
                A["투표"] += time.time() - ta
            t = tick(); T["자체움직임·누적"] += t - tprev; tprev = t

            skel = lh.skeletonize(narrow)
            t = tick(); T["⑤세선화"] += t - tprev; tprev = t

            if a.method == "hough":
                lines = cv2.HoughLinesP(skel, 1, np.pi / 180, a.hough_thresh,
                                        minLineLength=a.min_len, maxLineGap=a.max_gap)
                segs = []
                if lines is not None:
                    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                        if min(ang, 180 - ang) >= a.angle_min:
                            segs.append((int(x1), int(y1), int(x2), int(y2)))
                mg = lh.merge_segments(segs, a.merge_deg, a.merge_rho)
                for x1, y1, x2, y2, _, ln in mg:
                    if ln >= a.merge_min_len:
                        cv2.line(vis, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                        k += 1
            else:
                sy, sx = np.nonzero(skel)
                pts = np.stack([sx, sy], 1).astype(np.float64)
                if len(pts) > a.max_pts:
                    pts = pts[rng.choice(len(pts), a.max_pts, replace=False)]
                if len(pts) >= a.min_inliers:
                    for poly, _, _ in lr.sequential_ransac(
                            pts, a.degree, a.tol, a.trials, a.max_lines,
                            a.min_inliers, a.min_span, 0):
                        lr.draw_poly(vis, poly, color, 4)
                        k += 1

        t = tick(); T["⑥선검출"] += t - tprev; tprev = t

        n_lines.append(k)
        # 좌상단에 방법 이름과 현재 선 개수
        cv2.rectangle(vis, (0, 0), (470, 46), (0, 0, 0), -1)
        tag = f"{a.method.upper()}" + (f"+ACC{a.accum}/{a.vote}" if a.accum > 1 else "")
        cv2.putText(vis, f"{tag}  lines={k}", (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        wr.stdin.write(vis.astype(np.uint8).tobytes())
        t = tick(); T["그리기·인코딩"] += t - tprev; tprev = t

        n += 1
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  {n}프레임  {el:.0f}s  ({el/n:.2f} s/frame)  선 평균 "
                  f"{np.mean(n_lines):.1f}")
        if a.limit and n >= a.limit:
            break

    wr.stdin.close()
    wr.wait()
    el = time.time() - t0
    tot = sum(T.values())
    print(f"\n{'='*62}")
    print(f"구간별 소요 시간 ({n}프레임, stride 1)\n")
    print(f"{'구간':<16}{'합계 s':>10}{'ms/frame':>12}{'비중':>9}")
    print("-" * 62)
    for k2, v in T.items():
        print(f"{k2:<16}{v:10.1f}{v/max(n,1)*1000:12.1f}{v/max(tot,1e-9)*100:8.1f}%")
    print("-" * 62)
    print(f"{'합계':<16}{tot:10.1f}{tot/max(n,1)*1000:12.1f}{100.0:8.1f}%")
    if a.accum > 1:
        print(f"\n  └ 누적 단계 세부")
        for k2, v in A.items():
            print(f"    {k2:<14}{v:10.1f}{v/max(n,1)*1000:12.1f}")

    fps_eff = n / max(el, 1e-9)
    print(f"\n실측 {el:.0f}s = {el/60:.1f}분  ({el/max(n,1):.3f} s/frame, {fps_eff:.1f} fps)")
    try:
        N_all = lh.total_frames(video)
        print(f"전체 영상 {N_all:,}프레임 기준 예상: "
              f"{N_all*el/max(n,1)/60:.1f}분")
    except Exception:
        pass
    nl = np.array(n_lines, dtype=float)
    jitter = np.abs(np.diff(nl)).mean() if len(nl) > 1 else 0.0
    print(f"\n선 개수 평균 {nl.mean():.1f}  최소 {int(nl.min())}  최대 {int(nl.max())}")
    print(f"**인접 프레임 흔들림 {jitter:.2f}개** — 연속한 두 프레임의 선 개수 차이 평균")
    print(f"선 0개 프레임 {int((nl==0).sum())}장 ({(nl==0).mean()*100:.1f}%)")
    print(f"저장: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
