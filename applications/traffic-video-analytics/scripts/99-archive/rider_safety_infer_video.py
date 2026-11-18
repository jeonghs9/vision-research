#!/usr/bin/env python3
"""드론 영상에 rider-safety 5클래스 모델을 추론해 주석 영상을 만든다.

계획: docs/05-오토바이헬멧및다인승/01. 계획.md

**해상도 주의 — 학습 imgsz 를 그대로 쓰면 안 된다.**
  학습은 imgsz=1024 이지만 입력 영상은 2880x1620 이다. 1024 로 줄이면 헬멧이 사라진다.
  같은 15프레임 샘플에서 측정한 helmet 검출 수:

      imgsz  1024 →  0개 | 1600 → 2개 | 1920 → 7개 | 2560 → 37개

  그래서 **추론은 2560**, 출력만 1920 으로 줄여 파일 크기를 줄인다.

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/99-archive/rider_safety_infer_video.py \
      --video /home/hsjeong/workspace/dataset/etc./test_video/drone/recard_visible_162756.ts \
      --out   /home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/rider-safety-infer/raw_162756.mp4 \
      --gpu 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "plate", 4: "HELMET"}
COLORS = {  # BGR
    0: (80, 255, 80),   # person     연두
    1: (255, 200, 0),   # car        하늘
    2: (0, 180, 255),   # motorcycle 주황
    3: (255, 0, 255),   # plate      마젠타
    4: (0, 60, 255),    # helmet     빨강 (굵게 표시)
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--model",
        default="/home/hsjeong/workspace/applications/traffic-video-analytics/"
        "test/output/rider-safety/001-helmet-5cls/weights/best.pt",
    )
    p.add_argument("--imgsz", type=int, default=2560, help="추론 해상도. 낮추면 헬멧을 놓친다")
    p.add_argument("--out_width", type=int, default=1920, help="출력 가로 크기. 0 이면 원본 유지")
    p.add_argument("--conf", type=float, default=0.25, help="배포 설정과 동일하게 0.25 고정")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--classes", type=int, nargs="*", default=None,
        help="표시할 클래스 ID. 예: --classes 0 2 4 (person·motorcycle·helmet). 생략하면 전체",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 영상을 열 수 없다: {args.video}")

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ow = args.out_width or W
    oh = int(round(H * ow / W)) // 2 * 2  # 인코더가 짝수 높이를 요구한다
    resize = (ow, oh) != (W, H)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    shown = args.classes if args.classes is not None else sorted(NAMES)
    print(f"[CONFIG] imgsz={args.imgsz} conf={args.conf} out_width={ow} "
          f"classes={[NAMES[c] for c in shown]}", flush=True)

    counts = {k: 0 for k in NAMES}
    frames_with_helmet = 0
    n = 0
    t0 = time.time()
    thick = max(2, W // 1400)
    scale = W / 2880 * 0.7
    stem = Path(args.video).stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.gpu,
                          classes=args.classes, verbose=False)[0]
        has_helmet = False
        for box, cls, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            c = int(cls)
            counts[c] += 1
            has_helmet |= c == 4
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[c], thick * 2 if c == 4 else thick)
            cv2.putText(
                frame, f"{NAMES[c]} {sc:.2f}", (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS[c], thick,
            )
        frames_with_helmet += has_helmet

        writer.write(cv2.resize(frame, (ow, oh)) if resize else frame)
        if n % 500 == 0:
            el = time.time() - t0
            print(f"  {stem}: {n}프레임  {n / el:.1f} fps  경과 {el / 60:.1f}분", flush=True)

    cap.release()
    writer.release()
    el = time.time() - t0
    print(f"[DONE] {stem}  {n}프레임  {el / 60:.1f}분 ({n / el:.1f} fps)  헬멧 등장 프레임 {frames_with_helmet:,}")
    print("  검출: " + "  ".join(f"{NAMES[k]} {counts[k]:,}" for k in shown), flush=True)


if __name__ == "__main__":
    main()
