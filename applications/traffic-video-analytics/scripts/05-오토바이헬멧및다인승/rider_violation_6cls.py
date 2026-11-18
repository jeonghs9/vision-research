#!/usr/bin/env python3
"""6클래스 모델(D)로 드론 영상에서 헬멧 미착용·다인승 위반만 표시한다.

계획: docs/05-오토바이헬멧및다인승/

**표시 규칙**
  - 위반만 그린다. 정상 주행은 아무 것도 그리지 않는다.
  - `helmet`    : 오토바이 탑승자 중 머리가 head(맨머리)로 잡힌 사람 → person 박스
                  (헬멧 **미착용** 위반이라는 뜻. 라벨 문구는 `--label_nohelmet` 로 바꿀 수 있다)
  - `2up`       : 탑승자가 2명 이상인 오토바이 → motorcycle 박스
  - head/helmet 박스 자체는 그리지 않는다 (판정에만 쓴다)

**판정 대상**
  머리(helmet 또는 head)가 검출된 탑승자만 판정한다.
  둘 다 없으면 착용 여부를 알 수 없으므로 **판정에서 제외**한다.
  부재 추론(helmet 이 없으면 미착용)을 쓰지 않는 이유는, 헬멧 미탐 1건이
  곧 오탐이 되기 때문이다. head 를 직접 탐지하는 6클래스 모델이므로
  "head 가 잡혔다" 는 적극적 근거로만 위반을 판정한다.

실행 예시:
    python /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/rider_violation_6cls.py \
      --video /home/hsjeong/workspace/dataset/etc./test_video/drone/recard_visible_164800.ts \
      --out   ./test/tmp/rider-safety-infer/violation_164800.mp4 \
      --gpu 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON, CAR, MOTO, PLATE, HELMET, HEAD = range(6)
RED = (60, 60, 255)      # no helmet
ORANGE = (0, 165, 255)   # 2up


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--model",
        default="/home/hsjeong/workspace/applications/traffic-video-analytics/"
        "test/output/rider-safety/snapshots/D_best.pt",
    )
    p.add_argument("--imgsz", type=int, default=2048, help="D 는 2048 로 학습됐다")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out_width", type=int, default=1920, help="0 이면 원본 유지")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--rider_ios", type=float, default=0.30,
                   help="person 이 motorcycle 과 이만큼 겹치면 탑승자로 본다 (person 면적 기준)")
    p.add_argument("--frames", type=int, default=0, help="테스트용 — 앞에서 N 프레임만")
    p.add_argument("--label_nohelmet", default="helmet", help="미착용 위반 라벨 문구")
    p.add_argument("--label_2up", default="2up", help="다인승 위반 라벨 문구")
    return p.parse_args()


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return 0.0 if (x2 <= x1 or y2 <= y1) else (x2 - x1) * (y2 - y1)


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def judge(dets, rider_ios):
    """검출 결과에서 위반을 판정한다.

    returns (no_helmet_person_boxes, multi_rider_moto_boxes, stats)
    """
    people = dets.get(PERSON, [])
    motos = dets.get(MOTO, [])
    helmets = dets.get(HELMET, [])
    heads = dets.get(HEAD, [])

    # ── 각 사람의 머리 상태: 박스 안에 중심이 있는 helmet/head 를 찾는다 ──
    # 겹치면 더 가까운 쪽(중심 거리)을 그 사람의 머리로 본다.
    head_state = {}          # person index -> "helmet" | "head"
    for pi, p in enumerate(people):
        best = None
        for kind, boxes in (("helmet", helmets), ("head", heads)):
            for b in boxes:
                cx, cy = center(b)
                if not (p[0] <= cx <= p[2] and p[1] <= cy <= p[3]):
                    continue
                pcx, pcy = center(p)
                d = (cx - pcx) ** 2 + (cy - pcy) ** 2
                if best is None or d < best[0]:
                    best = (d, kind)
        if best:
            head_state[pi] = best[1]

    # ── 오토바이별 탑승자 ──
    no_helmet, multi = [], []
    judged = skipped = 0
    for m in motos:
        riders = [pi for pi, p in enumerate(people)
                  if inter(p, m) / max(area(p), 1) >= rider_ios]
        if not riders:
            continue
        if len(riders) >= 2:
            multi.append(m)
        for pi in riders:
            st = head_state.get(pi)
            if st is None:
                skipped += 1          # 머리를 못 찾음 → 판정 제외
                continue
            judged += 1
            if st == "head":
                no_helmet.append(people[pi])
    return no_helmet, multi, (judged, skipped)


def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] 영상을 열 수 없다: {args.video}")

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ow = args.out_width or W
    oh = int(round(H * ow / W)) // 2 * 2
    resize = (ow, oh) != (W, H)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    lw = max(2, W // 1400)
    fs = W / 2880 * 0.55          # 라벨은 작게
    n = tot_nh = tot_2up = tot_judged = tot_skip = 0
    fr_nh = fr_2up = 0
    t0 = time.time()
    print(f"[CONFIG] imgsz={args.imgsz} conf={args.conf} rider_ios={args.rider_ios} "
          f"model={Path(args.model).name}", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.gpu, verbose=False)[0]
        dets = {}
        for box, cls in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            dets.setdefault(int(cls), []).append(box)

        no_helmet, multi, (judged, skipped) = judge(dets, args.rider_ios)
        tot_nh += len(no_helmet)
        tot_2up += len(multi)
        tot_judged += judged
        tot_skip += skipped
        fr_nh += bool(no_helmet)
        fr_2up += bool(multi)

        for b in multi:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, lw)
            cv2.putText(frame, args.label_2up, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, ORANGE, lw)
        for b in no_helmet:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, lw)
            cv2.putText(frame, args.label_nohelmet, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, RED, lw)

        writer.write(cv2.resize(frame, (ow, oh)) if resize else frame)
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  {n}프레임  {n / el:.1f} fps  no_helmet {tot_nh:,}  2up {tot_2up:,}", flush=True)
        if args.frames and n >= args.frames:
            break

    cap.release()
    writer.release()
    el = time.time() - t0
    print(f"\n[DONE] {n:,}프레임 · {el / 60:.1f}분 ({n / el:.1f} fps)")
    print(f"  미착용     {tot_nh:,}건 · 등장 프레임 {fr_nh:,} ({fr_nh / max(n,1) * 100:.0f}%)")
    print(f"  2up        {tot_2up:,}건 · 등장 프레임 {fr_2up:,} ({fr_2up / max(n,1) * 100:.0f}%)")
    print(f"  판정한 탑승자 {tot_judged:,} · 머리 미검출로 제외 {tot_skip:,} "
          f"({tot_skip / max(tot_judged + tot_skip, 1) * 100:.0f}%)")


if __name__ == "__main__":
    main()
