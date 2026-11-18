#!/usr/bin/env python3
"""프로젝트 타이틀 슬라이드용 Before/After 이미지.

위: 원본 드론 프레임 (실선·오토바이·차량이 보이는 장면)
아래: 같은 프레임에 박스 + 트랙 ID + 위반 표시(오토바이 2인 탑승) 렌더링
두 패널을 세로로 붙여 PPT 지정 크기로 저장.

준비 (`.ts` 는 seek 시 회색 프레임 → 구간을 png 로 먼저 추출):
  ffmpeg -y -i .../drone/recard_visible_164259.ts \
    -vf "select=between(n\\,2900\\,3120)" -vsync 0 -an \
    test/output/title_beforeafter/seg_%04d.png

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
    /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/04-오토바이다인승/title_before_after.py \
    --frames-dir test/output/title_beforeafter --pick 0 --conf 0.30 \
    --crop 400,440,2450,1300 \
    --out test/output/title_beforeafter/before_after.png

--pick 0 = 워밍업 후 위반 가장 많은 프레임 자동선택 · --pick N = N번째 프레임
--crop x0,y0,x1,y1 = 원본(2880x1620) 좌표 크롭 · --no-badge = 배지 생략
출력은 항상 --cm-w x --cm-h (기본 14.43x11.56cm) @ --dpi (기본 300).
"""
import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

FKRB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FKR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

PERSON, CAR, MOTO = 0, 1, 2
COL = {PERSON: (0, 200, 255), CAR: (255, 200, 0), MOTO: (255, 120, 0)}   # BGR
NAME = {PERSON: "person", CAR: "car", MOTO: "moto"}
VIOL = (0, 0, 235)          # 위반 빨강
IOA_TAU = 0.30


def ioa(pb, mb):
    ix0, iy0 = max(pb[0], mb[0]), max(pb[1], mb[1])
    ix1, iy1 = min(pb[2], mb[2]), min(pb[3], mb[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ap = (pb[2] - pb[0]) * (pb[3] - pb[1])
    return inter / ap if ap else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, help="seg_*.png 프레임 폴더")
    ap.add_argument("--model", default="models/det-best.pt")
    ap.add_argument("--pick", type=int, default=150, help="렌더링할 프레임 순번 (1-base)")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--crop", default=None, help="x0,y0,x1,y1 (원본 좌표). 미지정 시 자동")
    ap.add_argument("--cm-w", type=float, default=14.43)
    ap.add_argument("--cm-h", type=float, default=11.56)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-badge", action="store_true", help="원본/추론결과 라벨 배지 생략")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.frames_dir, "seg_*.png")))
    assert files, "seg_*.png 없음"
    m = YOLO(a.model)

    def parse(r):
        xy = r.boxes.xyxy.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None
               else np.full(len(xy), -1))
        return list(zip(xy, cl, cf, ids))

    def viol_set(bs):
        ps = [b for b in bs if b[1] == PERSON]
        vs = set()
        for (mb, mc, mcf, mid) in [b for b in bs if b[1] == MOTO]:
            if sum(1 for pb, *_ in ps if ioa(pb, mb) >= IOA_TAU) >= 2:
                vs.add(tuple(mb.round(1)))
        return vs

    target_frame, boxes = None, None
    for i, fp in enumerate(files, 1):
        fr = cv2.imread(fp)
        r = m.track(fr, persist=True, imgsz=a.imgsz, conf=a.conf, iou=0.7,
                    classes=[PERSON, CAR, MOTO], tracker="bytetrack.yaml", verbose=False)[0]
        if a.pick > 0 and i == a.pick:
            target_frame, boxes = fr, parse(r)
            break
        if a.pick == 0:  # 자동: 워밍업 후 위반 가장 많은 프레임
            if i < 20:
                continue
            bs = parse(r)
            if boxes is None or len(viol_set(bs)) > len(viol_set(boxes)):
                target_frame, boxes = fr.copy(), bs
            if i >= 90:
                break
    assert target_frame is not None

    H, W = target_frame.shape[:2]
    viol_moto = viol_set(boxes)
    # 트랙 ID 를 화면에 보기 좋게 1..N 으로 재부여 (원 ID 순서 유지)
    raw_ids = sorted({int(b[3]) for b in boxes if b[3] >= 0})
    id_map = {rid: k + 1 for k, rid in enumerate(raw_ids)}

    aspect = a.cm_w / (a.cm_h / 2)
    if a.crop:
        cx0, cy0, cx1, cy1 = map(int, a.crop.split(","))
    else:
        pts = np.array([[(b[0][0] + b[0][2]) / 2, (b[0][1] + b[0][3]) / 2] for b in boxes])
        cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
        ch = int(H * 0.52)
        cw = int(ch * aspect)
        if cw > W:
            cw, ch = W, int(W / aspect)
        cx0 = int(np.clip(cx - cw / 2, 0, W - cw))
        cy0 = int(np.clip(cy - ch * 0.42, 0, H - ch))
        cx1, cy1 = cx0 + cw, cy0 + ch

    def crop(img):
        return img[cy0:cy1, cx0:cx1].copy()

    # 크롭 안에 들어오는 박스만, 면적 큰 것만 라벨
    def visible(xy):
        return xy[2] > cx0 and xy[0] < cx1 and xy[3] > cy0 and xy[1] < cy1

    areas = [(b[0][2] - b[0][0]) * (b[0][3] - b[0][1]) for b in boxes if visible(b[0])]
    amin = np.percentile(areas, 52) if areas else 0

    raw = crop(target_frame)
    ren = target_frame.copy()
    L = {PERSON: "P", CAR: "C", MOTO: "M"}

    # 1차: 일반 박스 + 라벨
    for xy, cl, cf, tid in boxes:
        if not visible(xy) or tuple(xy.round(1)) in viol_moto:
            continue
        x0, y0, x1, y1 = xy.astype(int)
        c = COL[cl]
        cv2.rectangle(ren, (x0, y0), (x1, y1), c, 3)
        if (x1 - x0) * (y1 - y0) >= amin:
            tag = f"{L[cl]}{id_map.get(int(tid), '')}" if tid >= 0 else L[cl]
            cv2.rectangle(ren, (x0, y0 - 22), (x0 + 13 * len(tag) + 6, y0), c, -1)
            cv2.putText(ren, tag, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

    # 2차: 위반 박스 (맨 위에)
    for xy, cl, cf, tid in boxes:
        if tuple(xy.round(1)) not in viol_moto or not visible(xy):
            continue
        x0, y0, x1, y1 = xy.astype(int)
        cv2.rectangle(ren, (x0, y0), (x1, y1), VIOL, 6)
        vt = "VIOLATION 2-up"
        vw = 20 * len(vt)
        lx = int(np.clip(x0 - 4, cx0 + 4, cx1 - vw - 4))
        ly = y0 - 40 if y0 - 40 > cy0 + 6 else min(y1 + 6, cy1 - 40)
        cv2.rectangle(ren, (lx, ly), (lx + vw, ly + 34), VIOL, -1)
        cv2.putText(ren, vt, (lx + 8, ly + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
    ren = crop(ren)

    # ── 캔버스 (PPT 크기) ──
    Wpx = int(round(a.cm_w / 2.54 * a.dpi))
    Hpx = int(round(a.cm_h / 2.54 * a.dpi))
    div = max(4, Hpx // 90)
    ph = (Hpx - div) // 2

    def fit(img):
        s = Wpx / img.shape[1]
        img = cv2.resize(img, (Wpx, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if img.shape[0] >= ph:
            o = (img.shape[0] - ph) // 2
            return img[o:o + ph]
        pad = np.zeros((ph, Wpx, 3), np.uint8)
        o = (ph - img.shape[0]) // 2
        pad[o:o + img.shape[0]] = img
        return pad

    canvas = np.full((Hpx, Wpx, 3), 20, np.uint8)
    canvas[:ph] = fit(raw)
    canvas[ph + div:] = fit(ren)

    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    fb = ImageFont.truetype(FKRB, max(20, Hpx // 34))

    def badge(x, y, txt, fg=(255, 255, 255), bg=(0, 0, 0)):
        tw = d.textlength(txt, font=fb)
        pad = 10
        d.rectangle((x, y, x + tw + pad * 2, y + fb.size + pad), fill=bg)
        d.text((x + pad, y + pad // 2), txt, font=fb, fill=fg)

    if not a.no_badge:
        badge(14, 12, "원본 (Before)")
        badge(14, ph + div + 12, "추론 결과 — 박스 · 트랙 ID · 위반 (After)", bg=(0, 0, 150))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    im.save(a.out, dpi=(a.dpi, a.dpi))
    print(f"saved {a.out}  {im.size}px  = {a.cm_w}x{a.cm_h}cm @ {a.dpi}dpi")
    print(f"crop x{cx0}-{cx1} y{cy0}-{cy1}   위반 오토바이 {len(viol_moto)}대 / 박스 {len(boxes)}개")


if __name__ == "__main__":
    main()
