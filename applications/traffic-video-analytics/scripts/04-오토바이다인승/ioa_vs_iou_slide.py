#!/usr/bin/env python3
"""포트폴리오용: 오토바이 다인승 연관에서 IoU vs IoA 비교 이미지.

`models/det-best.pt` 로 드론 프레임을 검출해 person↔motorcycle 한 쌍을 골라
박스·교집합·IoU·IoA(=교집합/사람박스) 를 한 장에 보여준다.

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
    /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/04-오토바이다인승/ioa_vs_iou_slide.py \
    --video /home/hsjeong/workspace/dataset/etc./test_video/drone/clip_164259_f6025_1500.mp4 \
    --frame 165 --pick rear --out test/output/ioa_vs_iou/f165.png
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

FKR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FKRB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BLUE, YEL, INTER = (255, 128, 0), (0, 200, 255), (80, 200, 80)          # BGR
BLUE_R, YEL_R = (0, 128, 255), (235, 170, 0)                            # RGB
RED, GRN, DK = (200, 40, 40), (30, 150, 60), (40, 40, 40)


def pair_stats(pb, mb):
    ix = [max(pb[0], mb[0]), max(pb[1], mb[1]), min(pb[2], mb[2]), min(pb[3], mb[3])]
    inter = max(0, ix[2] - ix[0]) * max(0, ix[3] - ix[1])
    ap = (pb[2] - pb[0]) * (pb[3] - pb[1])
    ab = (mb[2] - mb[0]) * (mb[3] - mb[1])
    iou = inter / (ap + ab - inter) if inter else 0.0
    ioa = inter / ap if inter else 0.0
    return dict(ix=ix, inter=inter, ap=ap, ab=ab, iou=iou, ioa=ioa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--model", default="models/det-best.pt")
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--pick", default="best",
                    help="best: 목표(IoU 0.26/IoA 0.84)에 가장 가까운 쌍 | "
                         "rear: 가장 큰 오토바이의 뒤쪽(작은 x/y) 탑승자 | front")
    ap.add_argument("--title", default="오토바이 다인승 — 사람↔오토바이 연관")
    ap.add_argument("--subtitle", default="왜 IoU 가 아니라 IoA(포함비) 를 쓰는가")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    cap.set(1, a.frame)
    ok, frame = cap.read()
    assert ok, "frame read fail"
    r = m.predict(frame, imgsz=a.imgsz, conf=a.conf, iou=0.7, classes=[0, 2], verbose=False)[0]
    xy = r.boxes.xyxy.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    persons = [xy[i] for i in range(len(xy)) if cl[i] == 0]
    motos = [xy[i] for i in range(len(xy)) if cl[i] == 2]

    chosen = None
    if a.pick in ("rear", "front"):
        mb = max(motos, key=lambda z: (z[2] - z[0]) * (z[3] - z[1]))
        riders = [(pb, pair_stats(pb, mb)) for pb in persons if pair_stats(pb, mb)["ioa"] > 0.3]
        riders.sort(key=lambda z: z[0][0])  # x 오름차순
        pb, st = riders[0] if a.pick == "rear" else riders[-1]
        chosen = (pb, mb, st)
    else:
        best_d = 1e9
        for mb in motos:
            for pb in persons:
                st = pair_stats(pb, mb)
                if st["ioa"] < 0.7 or st["iou"] > 0.4 or st["inter"] == 0:
                    continue
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                if not (mb[0] < pcx < mb[2] and mb[1] < pcy < mb[3]):
                    continue
                dd = abs(st["iou"] - 0.26) + abs(st["ioa"] - 0.84)
                if dd < best_d:
                    best_d, chosen = dd, (pb, mb, st)
    pb, mb, st = chosen
    print(f"IoU {st['iou']:.3f}  IoA {st['ioa']:.3f}  areaP {st['ap']:.0f}  areaM {st['ab']:.0f}  "
          f"ratio {st['ab'] / st['ap']:.2f}")

    ix = st["ix"]
    pad = 70
    x0 = max(0, int(min(pb[0], mb[0])) - pad)
    y0 = max(0, int(min(pb[1], mb[1])) - pad)
    x1 = min(frame.shape[1], int(max(pb[2], mb[2])) + pad)
    y1 = min(frame.shape[0], int(max(pb[3], mb[3])) + pad)
    crop = frame[y0:y1, x0:x1].copy()
    sh = lambda box: np.array([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0], float)
    pbs, mbs, ixs = sh(pb), sh(mb), sh(ix)

    ov = crop.copy()
    cv2.rectangle(ov, (int(ixs[0]), int(ixs[1])), (int(ixs[2]), int(ixs[3])), INTER, -1)
    crop = cv2.addWeighted(ov, 0.4, crop, 0.6, 0)
    cv2.rectangle(crop, (int(mbs[0]), int(mbs[1])), (int(mbs[2]), int(mbs[3])), BLUE, 3)
    cv2.rectangle(crop, (int(pbs[0]), int(pbs[1])), (int(pbs[2]), int(pbs[3])), YEL, 3)

    PANEL_H = 900
    scl = PANEL_H / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * scl), PANEL_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pbs, mbs = pbs * scl, mbs * scl

    PW = 780
    canvas = Image.new("RGB", (rgb.shape[1] + PW, PANEL_H), (250, 250, 250))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(canvas)
    Ff = lambda s, b=False: ImageFont.truetype(FKRB if b else FKR, s)

    def lab(box, txt, rgbc, above):
        tw = d.textlength(txt, font=Ff(22, True))
        ly = box[1] - 30 if above else box[3] + 2
        d.rectangle((box[0], ly, box[0] + tw + 14, ly + 28), fill=rgbc)
        d.text((box[0] + 7, ly + 3), txt, font=Ff(22, True),
               fill=(0, 0, 0) if rgbc == YEL_R else (255, 255, 255))

    lab(pbs, "person", YEL_R, True)
    lab(mbs, "motorcycle", BLUE_R, False)

    px = rgb.shape[1] + 44

    def xmark(x, yy, c):
        d.line((x, yy, x + 22, yy + 22), fill=c, width=5)
        d.line((x + 22, yy, x, yy + 22), fill=c, width=5)

    def chk(x, yy, c):
        d.line((x, yy + 12, x + 9, yy + 22), fill=c, width=5)
        d.line((x + 9, yy + 22, x + 27, yy), fill=c, width=5)

    y = 44
    d.text((px, y), a.title, font=Ff(32, True), fill=DK); y += 48
    d.text((px, y), a.subtitle, font=Ff(24), fill=(90, 90, 90)); y += 66
    for col, tx in [(BLUE_R, f"오토바이 박스   면적 {st['ab']:,.0f}"),
                    (YEL_R, f"사람 박스        면적 {st['ap']:,.0f}")]:
        d.rectangle((px, y + 3, px + 32, y + 29), outline=col, width=4)
        d.text((px + 44, y), tx, font=Ff(23), fill=DK); y += 42
    d.rectangle((px, y + 3, px + 32, y + 29), fill=(150, 210, 150))
    d.text((px + 44, y), f"교집합           면적 {st['inter']:,.0f}", font=Ff(23), fill=DK); y += 58
    d.line((px, y, px + PW - 70, y), fill=(214, 214, 214), width=2); y += 28

    uni = st['ap'] + st['ab'] - st['inter']
    d.text((px, y), "IoU  =  교집합 / 합집합", font=Ff(25, True), fill=DK); y += 40
    d.text((px + 30, y), f"=  {st['inter']:,.0f} / {uni:,.0f}", font=Ff(22), fill=(120, 120, 120)); y += 38
    d.text((px, y), f"=  {st['iou']:.2f}", font=Ff(38, True), fill=RED)
    xmark(px + 168, y + 12, RED)
    d.text((px + 206, y + 9), "같은 임계 0.30 기준 미달 → 탑승자 놓침", font=Ff(22), fill=RED); y += 68
    d.text((px, y), "IoA  =  교집합 / 사람 박스", font=Ff(25, True), fill=DK); y += 40
    d.text((px + 30, y), f"=  {st['inter']:,.0f} / {st['ap']:,.0f}", font=Ff(22), fill=(120, 120, 120)); y += 38
    d.text((px, y), f"=  {st['ioa']:.2f}", font=Ff(38, True), fill=GRN)
    chk(px + 168, y + 8, GRN)
    d.text((px + 206, y + 9), "임계 0.30 초과 → 탑승자로 연관 (채택)", font=Ff(22), fill=GRN); y += 72
    d.line((px, y, px + PW - 70, y), fill=(214, 214, 214), width=2); y += 24
    for ln in ["드론 하향 뷰에서 오토바이 박스가 사람 박스의",
               f"{st['ab'] / st['ap']:.1f} 배. 사람이 오토바이에 확실히 얹혀 있어도",
               "합집합이 커져 IoU 가 낮게 나온다.",
               "→ 사람 박스 기준 포함비(IoA)로 연관해야 놓치지 않는다."]:
        d.text((px, y), ln, font=Ff(22), fill=(70, 70, 70)); y += 33

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    canvas.save(a.out)
    print("saved", a.out, canvas.size)


if __name__ == "__main__":
    main()
