#!/usr/bin/env python3
"""road 마스크를 ROI 로 써서 원본 픽셀에서 차선을 뽑을 수 있는지 검증한다.

**아이디어** — 통합 seg 모델(road/car/plate)이 도로 영역을 주므로, 그 안에서 밝은
픽셀만 고르면 차선 도색이 나올 것이다. 차선 전용 모델 없이 차선을 얻을 수 있다.

**이 방식의 강점** — 흰색인데 차선이 아닌 것을 지울 수 있다.

    road 마스크        도로 밖(건물·하늘·풀숲) 제외
    - car 마스크       흰색 차량 제외          <- 통합 모델이라야 가능
    - plate 마스크     번호판(흰 사각형) 제외
    = 차선 후보 영역

**검증할 것 3가지** (이 스크립트는 답을 주지 않고 볼 수 있게만 한다)

  1. 차선이 끊김 없이 나오는가        -> 안 나오면 선으로 묶기가 어렵다
  2. 오탐(그림자 경계·젖은 노면·반사)이 얼마나 되는가
  3. **점선의 빈칸과 추출 실패가 구분되는가**  <- 구분 안 되면 이 방식은 못 쓴다

3번이 결정적이다. 실선을 점선으로 오판하면 침범 판정이 무너진다.

**임계 방식** — 고정 임계는 조명 변화에 깨지므로 **ROI 안에서의 백분위**를 쓴다.
도로 픽셀 중 상위 N% 밝기를 차선 후보로 본다. 흰 선은 채도가 낮으므로 채도 상한도 둔다.

**참고** — SAM3 라벨링 때 밝기 기반(`--fill_bright`, 임계 25)을 시험했다가
"차선이 완벽하게 메여져있는건 아닌거같아" 로 판단해 `white line` 프롬프트로 바꿨다.
목적이 달랐지만(구멍 메우기 vs 선 검출) 같은 영상에서의 관측이므로 감안할 것.

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/yolo26seg_cu128/bin/python \
    scripts/03-experiment/lane_from_road_mask.py --n 30
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

VIDEO = "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000157_T.mp4"
WEIGHTS = ("/home/hsjeong/workspace/vision-seg/Yolo26-seg/test/output/"
           "yolo26s_seg_1024_b32/weights/best.pt")
OUT = ("/home/hsjeong/workspace/applications/traffic-video-analytics/"
       "test/tmp/01-lane-postprocess-runs/lane_from_road_mask")

CLASS_ROAD, CLASS_CAR, CLASS_PLATE = 0, 1, 2


def decode(video: str, n: int, stride: int, w: int, h: int):
    """ffmpeg 로 디코딩하며 (w,h) 로 축소해 프레임을 읽는다."""
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video,
         "-vf", f"select='not(mod(n\\,{stride}))',scale={w}:{h}:flags=bilinear",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    nb = w * h * 3
    fr = []
    for _ in range(n):
        b = p.stdout.read(nb)
        if len(b) < nb:
            break
        fr.append(np.frombuffer(b, np.uint8).reshape(h, w, 3).copy())
    p.stdout.close(); p.kill()
    return fr


def class_masks(det, hw, dev):
    """클래스별 합집합 마스크 (H,W) bool 3개."""
    import torch.nn.functional as F
    H, W = hw
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


def extract_lane(rgb_t, roi, pct, sat_max):
    """ROI 안에서 밝기 상위 pct% + 채도 낮은 픽셀을 차선 후보로 본다.

    고정 임계 대신 **ROI 내부 백분위**를 쓴다. 조명이 바뀌어도 도로 대비 상대적으로
    밝은 픽셀을 고르므로 프레임마다 자동으로 맞춰진다.
    """
    if roi.sum() < 100:
        return torch.zeros_like(roi)
    mx, _ = rgb_t.max(dim=0)
    mn, _ = rgb_t.min(dim=0)
    v = mx                                   # HSV 의 V
    s = torch.where(mx > 0, (mx - mn) / mx.clamp(min=1e-6), torch.zeros_like(mx))  # S
    thr = torch.quantile(v[roi].float(), 1.0 - pct / 100.0)
    return roi & (v >= thr) & (s <= sat_max)


def morph_open(mask, k=3):
    """작은 잡음 제거 (침식 후 팽창). GPU 에서 max_pool 로 구현."""
    import torch.nn.functional as F
    x = mask[None, None].float()
    er = -F.max_pool2d(-x, k, 1, k // 2)
    di = F.max_pool2d(er, k, 1, k // 2)
    return di[0, 0] > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=1920, help="디코딩·처리 해상도")
    ap.add_argument("--pct", type=float, default=8.0,
                    help="ROI 안에서 밝기 상위 몇 %% 를 차선 후보로 볼지")
    ap.add_argument("--sat_max", type=float, default=0.35,
                    help="채도 상한. 흰 선은 채도가 낮다")
    ap.add_argument("--car_dilate", type=int, default=9,
                    help="차량 마스크를 이만큼 팽창해 뺀다 (마스크 경계 여유)")
    ap.add_argument("--panel_width", type=int, default=960)
    a = ap.parse_args()

    from ultralytics import YOLO
    from PIL import Image
    import torch.nn.functional as F

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    W = a.width
    H = int(round(2160 * W / 3840)) // 2 * 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"영상   : {Path(a.video).name}  처리 {W}x{H}")
    print(f"임계   : 밝기 상위 {a.pct}%, 채도 <= {a.sat_max}, 차량 팽창 {a.car_dilate}px")
    print(f"출력   : {outdir}\n")

    frames = decode(a.video, a.n, a.stride, W, H)
    model = YOLO(a.weights)

    print(f"{'프레임':>6} {'road%':>7} {'차선후보%':>9} {'road대비%':>9} {'덩어리':>7} {'세장비중앙':>10}")
    print("-" * 60)

    stats = []
    for i, f in enumerate(frames):
        det = model.predict(f[:, :, ::-1], imgsz=a.imgsz, conf=a.conf,
                            device=0, verbose=False)[0]
        m = class_masks(det, (H, W), dev)

        rgb_t = torch.from_numpy(f.copy()).to(dev).permute(2, 0, 1).float()

        # ROI = road − (car 팽창) − plate
        car = m[CLASS_CAR]
        if a.car_dilate > 1:
            car = F.max_pool2d(car[None, None].float(), a.car_dilate, 1,
                               a.car_dilate // 2)[0, 0] > 0.5
        roi = m[CLASS_ROAD] & (~car) & (~m[CLASS_PLATE])

        lane = extract_lane(rgb_t, roi, a.pct, a.sat_max)
        lane = morph_open(lane, 3)

        # 연결요소 통계 — 차선은 가늘고 길다. 세장비로 구분되는지 본다.
        # scipy 가 이 환경에 없으므로 cv2(ultralytics 의존)로 처리한다.
        import cv2
        ln = lane.cpu().numpy().astype(np.uint8)
        nlab, _, st, _ = cv2.connectedComponentsWithStats(ln, connectivity=8)
        elong = []
        for k in range(1, nlab):
            ww, hh, area = st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_AREA]
            if area >= 20:
                elong.append(max(hh, ww) / max(min(hh, ww), 1))
        med_el = float(np.median(elong)) if elong else 0.0

        rp = roi.float().mean().item() * 100
        lp = lane.float().mean().item() * 100
        rr = lp / rp * 100 if rp > 0 else 0
        stats.append((rp, lp, rr, len(elong), med_el))
        print(f"{i:>6} {rp:7.2f} {lp:9.3f} {rr:9.2f} {len(elong):7d} {med_el:10.1f}")

        # 4분할 패널
        pw = a.panel_width; ph = int(round(H * pw / W))
        base = f.astype(np.float32)

        seg = base.copy()
        seg[m[CLASS_ROAD].cpu().numpy()] = seg[m[CLASS_ROAD].cpu().numpy()] * .5 + np.array([150, 60, 190]) * .5
        cm = car.cpu().numpy()
        seg[cm] = seg[cm] * .5 + np.array([30, 230, 70]) * .5

        cand = np.zeros_like(base); cand[ln.astype(bool)] = 255

        ov = base.copy()
        ov[ln.astype(bool)] = ov[ln.astype(bool)] * .3 + np.array([255, 40, 40]) * .7

        canvas = Image.new("RGB", (pw * 4, ph))
        for j, p in enumerate([base, seg, cand, ov]):
            canvas.paste(Image.fromarray(p.clip(0, 255).astype(np.uint8))
                         .resize((pw, ph), Image.BILINEAR), (pw * j, 0))
        canvas.save(outdir / f"{i:02d}.jpg", quality=88)

    s = np.array(stats)
    print(f"\n패널 {len(frames)}장: {outdir}")
    print("  [원본 | road(보라)+car(초록) | 차선 후보 | 원본+후보(빨강)]\n")
    print(f"road 면적 평균        {s[:,0].mean():6.2f} %")
    print(f"차선 후보 면적 평균   {s[:,1].mean():6.3f} % (road 대비 {s[:,2].mean():.2f} %)")
    print(f"덩어리 수 평균        {s[:,3].mean():6.1f}")
    print(f"세장비 중앙값 평균    {s[:,4].mean():6.1f}   (차선이면 크게 나와야 한다)")
    print("\n판단할 것:")
    print("  1. 패널 3에서 차선이 **끊김 없이 이어지는가**")
    print("  2. 그림자 경계·젖은 노면·차량 반사가 후보에 섞이는가")
    print("  3. **점선의 빈칸과 추출 실패가 구분되는가**  <- 안 되면 이 방식은 못 쓴다")


if __name__ == "__main__":
    main()
