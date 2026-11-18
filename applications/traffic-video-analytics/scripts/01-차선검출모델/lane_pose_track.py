#!/usr/bin/env python3
"""YOLO-pose 차선 검출 + 프레임 간 트래킹.

차선은 세상에 고정돼 있고 화면 위치 변화는 드론 움직임 탓이다. 그래서:
  1) (선택) 프레임 간 호모그래피로 드론 자체 움직임을 보정해 이전 차선을 현재 프레임에 정렬
  2) 키포인트 집합 거리로 같은 클래스끼리 매칭 (게이트)
  3) min_hits 프레임 연속 매칭돼야 '확정' → 한두 프레임짜리 오검출 억제
  4) 놓친 트랙은 max_misses 프레임 동안 유지(coast) 후 삭제 → 깜빡임 제거
  5) 키포인트를 EMA 평활 → 떨림 제거

확정 트랙만 그린다. 색: 실선(solid)=노랑, 점선(dashed)=초록. 선 스타일은 둘 다 실선.

기본은 **영상 전체**를 처리한다. 구간만 보려면 --start-sec / --dur-sec 를 준다.

실행 (전체):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
    /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
    scripts/01-차선검출모델/lane_pose_track.py \
    --weights test/output/004-train-yolo26s-pose-1024-445f-300ep/weights/best.pt \
    --video /home/hsjeong/workspace/dataset/etc./test_video/19700101_001037_T.mp4 \
    --out test/output/lane_pose_vis_445f/001037_full.mp4 \
    --conf 0.30 --width 1280 --ego --crf 30

실행 (구간만): 위에 --start-sec 90 --dur-sec 15 추가
"""
import argparse
import os
import shutil
import subprocess
import time
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

YELLOW = (0, 255, 255)   # solid
GREEN = (0, 210, 0)      # dashed
NAME = {0: "solid", 1: "dashed"}
COL = {0: YELLOW, 1: GREEN}


def estimate_homography(prev_gray, cur_gray, orb, bf):
    """prev 프레임 좌표 -> cur 프레임 좌표 매핑 (드론 자체 움직임)."""
    k1, d1 = orb.detectAndCompute(prev_gray, None)
    k2, d2 = orb.detectAndCompute(cur_gray, None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    matches = bf.match(d1, d2)
    if len(matches) < 25:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:250]
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    Hm, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
    return Hm


def warp_pts(pts, Hm):
    if Hm is None:
        return pts
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), Hm)
    return p.reshape(-1, 2)


class Track:
    _next = 0

    def __init__(self, kp, vis, cls, conf, ema):
        self.kp = kp.astype(np.float32)          # (K,2) 평활된 좌표
        self.vis = vis.astype(bool)               # (K,) 최근 검출의 점별 visibility
        self.vel = np.zeros_like(self.kp)
        self.votes = Counter({cls: 1})
        self.conf = conf
        self.ema = ema
        self.hits = 1
        self.misses = 0
        self.tid = Track._next
        Track._next += 1

    @property
    def cls(self):
        return self.votes.most_common(1)[0][0]

    def predict(self, Hm):
        self.kp = warp_pts(self.kp + self.vel, Hm) if Hm is not None else self.kp + self.vel

    def update(self, kp, vis, cls, conf):
        new = kp.astype(np.float32)
        self.vel = 0.7 * self.vel + 0.3 * (new - self.kp)
        self.kp = self.ema * new + (1 - self.ema) * self.kp
        self.vis = vis.astype(bool)
        self.votes[cls] += 1
        self.conf = 0.6 * self.conf + 0.4 * conf
        self.hits += 1
        self.misses = 0

    def miss(self):
        self.misses += 1


class LaneTracker:
    def __init__(self, min_hits=3, max_misses=12, gate=90.0, ema=0.4):
        self.min_hits, self.max_misses, self.gate, self.ema = min_hits, max_misses, gate, ema
        self.merge_d = gate * 0.6
        self.tracks = []

    @staticmethod
    def _d(a, b):
        return float(np.mean(np.linalg.norm(a - b, axis=1)))

    def step(self, dets, Hm=None):
        for t in self.tracks:
            t.predict(Hm)

        # 그리디 매칭: (거리, 트랙idx, det idx) 오름차순
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, (kp, vis, cl, cf) in enumerate(dets):
                if cl != t.cls:
                    continue
                d = self._d(kp, t.kp)
                if d < self.gate:
                    pairs.append((d, ti, di))
        pairs.sort()
        ut, ud = set(range(len(self.tracks))), set(range(len(dets)))
        for d, ti, di in pairs:
            if ti in ut and di in ud:
                kp, vis, cl, cf = dets[di]
                self.tracks[ti].update(kp, vis, cl, cf)
                ut.discard(ti)
                ud.discard(di)
        for ti in ut:
            self.tracks[ti].miss()
        for di in ud:
            kp, vis, cl, cf = dets[di]
            self.tracks.append(Track(kp, vis, cl, cf, self.ema))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self._merge()
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses == 0], \
               [t for t in self.tracks if t.hits >= self.min_hits and t.misses > 0]

    def _merge(self):
        """같은 클래스에서 겹치는 트랙 흡수 (오래된 쪽 유지). 중복 차선 방지."""
        self.tracks.sort(key=lambda t: -t.hits)
        keep = []
        for t in self.tracks:
            dup = False
            for k in keep:
                if k.cls == t.cls and self._d(k.kp, t.kp) < self.merge_d:
                    k.conf = max(k.conf, t.conf)
                    dup = True
                    break
            if not dup:
                keep.append(t)
        self.tracks = keep


def draw(img, kp, cls, conf, thick, vis=None, coasting=False):
    pts = kp.astype(np.int32)
    if vis is not None:
        pts = pts[vis.astype(bool)]   # 저신뢰 점은 선 연결에서 제외 (튀는 점 제거)
    if len(pts) < 2:
        return
    col = COL[cls]
    cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)
    m = pts[len(pts) // 3]
    cv2.putText(img, f"{NAME[cls]} {conf:.2f}" + ("~" if coasting else ""),
                (int(m[0]) + 6, int(m[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def draw_debug_kpts(img, kp, sx, sy, thresh):
    """원본(프레임) 좌표의 keypoint 12개를 신뢰도별 점+숫자로 찍는다. 디버깅 전용.

    초록 = thresh 이상 (선에 남는 점), 빨강 = thresh 미만 (그리기 단계에서 걸러지는 점).
    실제 필터링 기준(`--kpt-vis-thresh`)과 항상 같은 값을 쓴다. `--debug-kpts` 로 켠다.
    """
    for x, y, c in kp[:, :3]:
        px, py = int(x * sx), int(y * sy)
        col = (0, 255, 0) if c >= thresh else (0, 0, 255)
        cv2.circle(img, (px, py), 4, col, -1, cv2.LINE_AA)
        cv2.putText(img, f"{c:.2f}", (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-sec", type=float, default=0, help="시작 위치(초). 기본 0")
    ap.add_argument("--dur-sec", type=float, default=0,
                    help="처리 길이(초). 기본 0 = start-sec 부터 영상 끝까지 전체")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--ego", action="store_true", help="드론 자체 움직임 보정 (ORB 호모그래피)")
    ap.add_argument("--no-track", action="store_true", help="트래킹 없이 프레임별 원본 검출만 (비교용)")
    ap.add_argument("--debug-kpts", action="store_true",
                    help="원본 keypoint 12개를 점+신뢰도 숫자로 함께 그린다 (초록=유지, 빨강=필터링됨)")
    ap.add_argument("--kpt-vis-thresh", type=float, default=0.6,
                    help="이 값 미만인 keypoint는 선 연결에서 제외 (기본 0.6, 기존 0.3에서 상향)")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--gate", type=float, default=90.0, help="매칭 허용 평균 키포인트 거리(px, 출력해상도 기준)")
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--crf", type=int, default=26, help="h264 압축률 (클수록 저용량, 23~32 권장)")
    ap.add_argument("--fps-out", type=float, default=0, help="출력 fps (0=원본). 낮추면 용량↓")
    ap.add_argument("--profile", action="store_true", help="구간별(read/resize/ego/infer/postproc/track/draw/write) 처리 시간 프로파일링 출력")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow = a.width
    oh = int(round(H * ow / W / 2) * 2)
    sx, sy = ow / W, oh / H

    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = max(0, min(int(a.start_sec * fps), total_f))
    end_f = (min(int((a.start_sec + a.dur_sec) * fps), total_f)
             if a.dur_sec > 0 else total_f)
    if end_f <= start_f:
        raise SystemExit(f"처리할 프레임이 없음 (start_f={start_f}, end_f={end_f}, total={total_f})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    thick = max(2, ow // 480)

    tracker = LaneTracker(a.min_hits, a.max_misses, a.gate, a.ema)
    orb = cv2.ORB_create(1500) if a.ego else None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if a.ego else None
    prev_gray = None

    f, n = start_f, 0
    counts = []
    prof = dict(read=0.0, resize=0.0, ego=0.0, infer=0.0, postproc=0.0, track=0.0, draw=0.0, write=0.0)
    pbar = tqdm(total=end_f - start_f, unit="frame", ncols=90)
    while f < end_f:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        t1 = time.perf_counter()
        small = cv2.resize(frame, (ow, oh))
        t2 = time.perf_counter()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if a.ego else None
        Hm = estimate_homography(prev_gray, gray, orb, bf) if (a.ego and prev_gray is not None) else None
        prev_gray = gray
        t3 = time.perf_counter()

        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        t4 = time.perf_counter()
        dets = []
        if r.keypoints is not None and len(r.keypoints):
            kxy = r.keypoints.data.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for i in range(len(kxy)):
                kp = kxy[i]
                if a.debug_kpts:
                    draw_debug_kpts(small, kp, sx, sy, a.kpt_vis_thresh)
                v = kp[:, 2] > a.kpt_vis_thresh if kp.shape[1] > 2 else np.ones(len(kp), bool)
                if v.sum() < 2:
                    continue
                # 키포인트 12개 전부 유지 (트랙 매칭은 고정 길이 필요). 저가시 점도 위치는 예측됨
                # -> 매칭엔 12개 다 쓰고, vis는 따로 들고 다니다 그릴 때만 걸러낸다.
                p = np.stack([kp[:, 0] * sx, kp[:, 1] * sy], 1)
                dets.append((p, v, int(cls[i]), float(cf[i])))
        t5 = time.perf_counter()

        if a.no_track:
            t6 = time.perf_counter()
            for kp, vis, cl, cf in dets:
                draw(small, kp, cl, cf, thick, vis=vis)
            shown = len(dets)
            t7 = time.perf_counter()
        else:
            live, coasting = tracker.step(dets, Hm)
            t6 = time.perf_counter()
            for t in coasting:
                draw(small, t.kp, t.cls, t.conf, thick, vis=t.vis, coasting=True)
            for t in live:
                draw(small, t.kp, t.cls, t.conf, thick, vis=t.vis)
            shown = len(live) + len(coasting)
            t7 = time.perf_counter()

        counts.append(shown)
        tt = (f - start_f) / fps
        cv2.putText(small, f"t={tt:5.1f}s  lanes={shown}"
                    f"  (raw det {len(dets)})" + ("  ego" if a.ego and not a.no_track else "")
                    + ("  [no-track]" if a.no_track else ""),
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(small, "solid=yellow  dashed=green   ~ = coasting(미검출 유지)",
                    (14, oh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
        vw.write(small)
        t8 = time.perf_counter()

        prof["read"] += t1 - t0
        prof["resize"] += t2 - t1
        prof["ego"] += t3 - t2
        prof["infer"] += t4 - t3
        prof["postproc"] += t5 - t4
        prof["track"] += t6 - t5
        prof["draw"] += t7 - t6
        prof["write"] += t8 - t7

        n += 1
        f += 1

    pbar.close()
    cap.release()
    vw.release()
    if n == 0:
        raise SystemExit("처리된 프레임 0 — 영상/구간을 확인하세요")

    if a.profile:
        total = sum(prof.values())
        print(f"\n=== 구간별 처리 시간 ({n} 프레임, 합계 {total:.1f}s, {total / n * 1000:.1f}ms/frame) ===")
        for k, v in prof.items():
            print(f"  {k:9s}  {v:7.1f}s  {v / n * 1000:6.2f}ms/frame  {v / total * 100:5.1f}%")

    if shutil.which("ffmpeg"):
        tmp = a.out + ".raw.mp4"
        os.replace(a.out, tmp)
        cmd = ["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", str(a.crf),
               "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        if a.fps_out > 0:
            cmd += ["-r", str(a.fps_out)]
        cmd.append(a.out)
        print("h264 압축 중...")
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(tmp)

    c = np.array(counts)
    mb = os.path.getsize(a.out) / 1e6
    print(f"{a.out}  ({mb:.1f} MB)")
    print(f"  {n} 프레임, {ow}x{oh}, ego={a.ego}, min_hits={a.min_hits}, max_misses={a.max_misses}")
    print(f"  프레임당 표시 차선: 평균 {c.mean():.1f}  범위 {c.min()}~{c.max()}")


if __name__ == "__main__":
    main()
