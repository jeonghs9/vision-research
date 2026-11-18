#!/usr/bin/env python3
"""해상도별 추론 속도를 구간별로 분해해 측정한다.

계획: docs/05-오토바이헬멧및다인승/01. 계획.md

측정 구간
    decode      영상 디코딩 (cv2.VideoCapture.read)
    preprocess  letterbox·정규화·GPU 전송   ┐
    inference   모델 순전파                  ├ Ultralytics 가 보고하는 값
    postprocess NMS 등                       ┘
    draw        박스·라벨 그리기
    encode      출력 영상 쓰기 (리사이즈 포함)

GPU 를 독점한 상태에서 돌려야 의미가 있다. 다른 학습·추론이 같은 GPU 를 쓰면
경쟁 때문에 실제보다 느리게 나온다.

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-bench/rider_safety_bench_imgsz.py \
      --video /home/hsjeong/workspace/dataset/etc./test_video/drone/recard_visible_164800.ts \
      --gpu 0 --frames 300
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate", 4: "HELMET"}
COLORS = {0: (80, 255, 80), 1: (255, 200, 0), 2: (0, 180, 255), 3: (255, 0, 255), 4: (0, 60, 255)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument(
        "--model",
        default="/home/hsjeong/workspace/applications/traffic-video-analytics/"
        "test/output/rider-safety/001-helmet-5cls/weights/best.pt",
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 1280, 1600, 1920, 2560, 2880])
    p.add_argument("--frames", type=int, default=300, help="측정에 쓸 프레임 수")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out_width", type=int, default=1920)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--video_fps", type=float, default=25.0, help="실시간 판정 기준")
    return p.parse_args()


def load_frames(path: str, n: int):
    """디코딩 시간을 재면서 프레임을 메모리에 올린다."""
    cap = cv2.VideoCapture(path)
    frames, t = [], 0.0
    while len(frames) < n:
        t0 = time.perf_counter()
        ok, f = cap.read()
        t += time.perf_counter() - t0
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, t / max(len(frames), 1) * 1000  # ms/frame


def main():
    args = parse_args()
    frames, decode_ms = load_frames(args.video, args.frames)
    if not frames:
        raise SystemExit(f"[ERROR] 프레임을 읽지 못했다: {args.video}")
    H, W = frames[0].shape[:2]
    ow = args.out_width or W
    oh = int(round(H * ow / W)) // 2 * 2
    thick = max(2, W // 1400)
    scale = W / 2880 * 0.7

    print(f"영상   : {Path(args.video).name}  {W}x{H}")
    print(f"프레임 : {len(frames)}장 측정 (워밍업 {args.warmup})")
    print(f"디코딩 : {decode_ms:.1f} ms/frame  (해상도와 무관)\n")

    model = YOLO(args.model)
    rows = []
    for sz in args.sizes:
        for f in frames[: args.warmup]:  # 워밍업 — 첫 호출의 커널 컴파일 비용 제외
            model.predict(f, imgsz=sz, conf=args.conf, device=args.gpu, verbose=False)

        pre = inf = post = draw = enc = 0.0
        nbox = nhelmet = 0
        writer = cv2.VideoWriter(
            "/dev/null", cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (ow, oh)
        )
        for f in frames:
            r = model.predict(f, imgsz=sz, conf=args.conf, device=args.gpu, verbose=False)[0]
            pre += r.speed["preprocess"]
            inf += r.speed["inference"]
            post += r.speed["postprocess"]

            canvas = f.copy()
            t0 = time.perf_counter()
            for box, cls, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                c = int(cls)
                nbox += 1
                nhelmet += c == 4
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[c], thick * 2 if c == 4 else thick)
                cv2.putText(canvas, f"{NAMES[c]} {sc:.2f}", (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS[c], thick)
            draw += (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            writer.write(cv2.resize(canvas, (ow, oh)))
            enc += (time.perf_counter() - t0) * 1000
        writer.release()

        n = len(frames)
        rows.append({
            "imgsz": sz, "decode": decode_ms,
            "pre": pre / n, "inf": inf / n, "post": post / n,
            "draw": draw / n, "enc": enc / n,
            "box": nbox / n, "helmet": nhelmet,
        })
        r = rows[-1]
        tot = decode_ms + r["pre"] + r["inf"] + r["post"] + r["draw"] + r["enc"]
        print(f"  imgsz={sz:<5} 완료  총 {tot:.1f} ms/frame  ({1000 / tot:.1f} fps)  helmet {nhelmet}")

    # ── 표 출력 ──
    print("\n■ 구간별 소요 (ms/frame)\n")
    hdr = f"{'imgsz':>6}{'디코딩':>9}{'전처리':>9}{'추론':>9}{'후처리':>9}{'그리기':>9}{'인코딩':>9}{'합계':>10}{'fps':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tot = r["decode"] + r["pre"] + r["inf"] + r["post"] + r["draw"] + r["enc"]
        print(f"{r['imgsz']:>6}{r['decode']:>9.1f}{r['pre']:>9.1f}{r['inf']:>9.1f}"
              f"{r['post']:>9.1f}{r['draw']:>9.1f}{r['enc']:>9.1f}{tot:>10.1f}{1000 / tot:>8.1f}")

    print("\n■ 영상 길이별 처리 시간 · 검출량\n")
    hdr2 = (f"{'imgsz':>6}{'추론만':>10}{'전체':>10}{'실시간비':>10}"
            f"{'1분영상':>10}{'5분영상':>10}{'장당박스':>10}{'helmet':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    base = rows[0]["helmet"] or 1
    for r in rows:
        tot = r["decode"] + r["pre"] + r["inf"] + r["post"] + r["draw"] + r["enc"]
        fps = 1000 / tot
        ratio = args.video_fps / fps
        print(f"{r['imgsz']:>6}{1000 / r['inf']:>9.1f}f{fps:>9.1f}f{ratio:>9.2f}x"
              f"{60 * ratio / 60:>9.1f}분{5 * ratio:>9.1f}분{r['box']:>10.1f}{r['helmet']:>9}")

    print(f"\n  실시간비 = 영상 1초를 처리하는 데 걸리는 시간 (1.0 미만이면 {args.video_fps}fps 실시간 처리 가능)")
    print(f"  helmet   = 측정 프레임 {len(frames)}장에서 검출된 헬멧 총 개수")


if __name__ == "__main__":
    main()
