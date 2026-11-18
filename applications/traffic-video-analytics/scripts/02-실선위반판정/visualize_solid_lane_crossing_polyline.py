#!/usr/bin/env python3
"""Solid-lane crossing pipeline, polyline(YOLO-pose) lane model variant.

`visualize_solid_lane_crossing_v5.3.1.py`를 기준 베이스라인으로 삼아, 차선
세그멘테이션("lane" 단일 클래스) + 연속성 투표/occlusion-bridge 재구성
로직을 **실선/점선 클래스를 이미 가진 YOLO-pose 폴리라인 모델 + 프레임 간
트래커**로 통째로 교체한 버전이다. 차량 검출/추적, 침범 판정 state
machine(접촉 유지시간 + 중앙통과 디바운스), 번호판 OCR, 렌더링/CSV 골격은
v5.3.1과 동일하게 유지한다 — 바뀌는 건 "차선이 어디 있고 실선/점선 중
무엇이냐"를 알아내는 방법뿐이다.

## v5.3.1과의 차이

| v5.3.1 | 이 파일 |
|---|---|
| seg 모델(`lane_model.predict().masks.xy`) | pose 모델(`lane_model.predict().keypoints`) |
| `classify_lane_continuity`(ROI체인+슬라이스 신호 분석) | 모델이 이미 주는 클래스(0=실선/1=점선) |
| `stabilize_lane_decisions`(LANE_TRACKS 투표) | `LaneTracker`의 hit/miss + 클래스 다수결(`Track.votes`) |
| `stable_solid_fragment_bridges`(차량 가림 보정) | `LaneTracker`의 coast(`max_misses`프레임 속도외삽) + 자체움직임보정 |
| `build_solid_id_map`+bridge | `rasterize_tracks_to_solid_map()` — 트랙 폴리라인을 두께 있는 선으로 래스터화 |
| `evaluate_solid_crossing_trigger` + 접촉/중앙통과 state machine | **무수정, 그대로** — 입력(`solid_map`)만 바뀜 |

`LaneTracker`/`Track`/`estimate_homography`/`warp_pts`는
`scripts/01-차선검출모델/lane_pose_track.py`에서 가져왔다(수치 로직 무수정).
단, 그 스크립트는 자체 축소 해상도(기본 1600px)에서 작업하는 반면, 이
파일은 v5.3.1처럼 **원본 해상도**에서 차량 박스/solid_map을 다룬다 —
keypoint는 스케일 안 하고(ultralytics가 이미 원본 좌표로 돌려줌), ego-motion
호모그래피만 속도를 위해 축소 프레임에서 추정한 뒤 원본 해상도로 환산해서
적용한다(아래 `ORB_WORK_WIDTH`).

실행 예:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
    scripts/02-실선위반판정/visualize_solid_lane_crossing_polyline.py \
    --det-model models/det-best.pt \
    --lane-model models/pose-best.pt \
    --input <입력영상> \
    --output output/<결과영상>.mp4 \
    --csv output/<결과>.csv \
    --device 1 --max-frames 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────
# OCR 캐시 경로 (v5.3.1과 동일)
# ──────────────────────────────────────────────────────────────────────────
APPLICATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCR_CACHE_DIR = Path(
    os.environ.get("TRAFFIC_VIDEO_OCR_CACHE_DIR", APPLICATION_ROOT / "models" / "runtime-cache" / "ocr")
)
DEFAULT_PADDLE_CACHE_DIR = DEFAULT_OCR_CACHE_DIR / ".paddleocr-cache"
PADDLE_DET_MODEL = None
PADDLE_REC_MODEL = None


def normalize_paddle_device(device):
    """Convert YOLO-style device values to PaddleOCR/PaddleX device strings."""
    if device is None:
        return None
    value = str(device).strip()
    if not value:
        return None
    lower = value.lower()
    if lower == "cpu":
        return "cpu"
    if lower.startswith("cuda:"):
        return f"gpu:{value.split(':', 1)[1]}"
    if value.isdigit():
        return f"gpu:{value}"
    return value


def _prepare_paddleocr_environment() -> None:
    """Apply application-local OCR cache settings immediately before PaddleOCR import."""
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_OCR_CACHE_DIR / ".paddlex-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OCR_CACHE_DIR / ".matplotlib-cache"))
    os.environ.setdefault("PADDLE_HOME", str(DEFAULT_PADDLE_CACHE_DIR))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.bytes_, np.str_],
        }


# 번호판 OCR에서 숫자와 문자 혼동을 교정 (khoi03 방식, v5.3.1과 동일)
NUM_TO_CHAR = {
    "0": "D", "1": "T", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P", "$": "S",
}
CHAR_TO_NUM = {
    "A": "4", "B": "8", "C": "0", "D": "0", "E": "6", "F": "5", "G": "6",
    "H": "4", "I": "1", "J": "1", "L": "4", "O": "0", "P": "6", "Q": "0",
    "R": "8", "S": "5", "T": "7", "U": "0", "V": "0", "Y": "1", "Z": "2",
}


class PlateTextStabilizer:
    """track_id 별로 여러 프레임의 OCR 결과를 모아 최빈값으로 안정화한다. (v5.3.1과 동일)"""

    def __init__(self, min_votes: int = 3, history: int = 30):
        self.min_votes = min_votes
        self._buffer = defaultdict(lambda: deque(maxlen=history))
        self._confirmed = {}

    def update(self, track_id: int, text: str, conf: float) -> str:
        if not text:
            return self._confirmed.get(track_id, "")
        buf = self._buffer[track_id]
        buf.append((text, conf))
        score = defaultdict(lambda: [0, 0.0])
        for t, c in buf:
            score[t][0] += 1
            score[t][1] += c
        best_text, best_metric = ("", -1.0)
        for t, (cnt, conf_sum) in score.items():
            avg_conf = conf_sum / cnt
            metric = cnt * avg_conf
            if metric > best_metric:
                best_text, best_metric = (t, metric)
        if len(buf) >= self.min_votes:
            self._confirmed[track_id] = best_text
        return self._confirmed.get(track_id, text)


def clean_plate_text(raw: str) -> str:
    return "".join((ch for ch in raw.upper() if ch.isalnum()))


PLATE_FORMAT_RE = re.compile("^[0-9]{2}[A-Z][0-9]{5}$")
DIGIT_CORRECTIONS = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
LETTER_CORRECTIONS = {
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B", "9": "P",
}


def format_plate_text(raw: str) -> str:
    """Normalize OCR text to '##A ###.##'. (v5.3.1과 동일)"""
    token = clean_plate_text(raw)
    if len(token) < 8:
        return ""
    best = ""
    best_exact_matches = -1
    for start in range(len(token) - 7):
        window = token[start : start + 8]
        chars = list(window)
        for idx in (0, 1, 3, 4, 5, 6, 7):
            chars[idx] = DIGIT_CORRECTIONS.get(chars[idx], chars[idx])
        chars[2] = LETTER_CORRECTIONS.get(chars[2], chars[2])
        normalized = "".join(chars)
        if PLATE_FORMAT_RE.match(normalized):
            exact_matches = sum((a == b for a, b in zip(window, normalized)))
            if exact_matches > best_exact_matches:
                best = normalized
                best_exact_matches = exact_matches
    if not best:
        return ""
    return f"{best[:3]} {best[3:6]}.{best[6:]}"


def extract_ocr_texts(result):
    """Extract OCR texts/confidences from PaddleOCR 2.x or 3.x results. (v5.3.1과 동일)"""
    if not result:
        return ([], [])
    first = result[0]
    if first is None:
        return ([], [])
    if hasattr(first, "get") and "rec_texts" in first:
        texts = [str(t) for t in first.get("rec_texts", []) if t]
        confs = [float(c) for c in first.get("rec_scores", [])]
        return (texts, confs)
    lines = first
    texts = [line[1][0] for line in lines if line and line[1]]
    confs = [float(line[1][1]) for line in lines if line and line[1]]
    return (texts, confs)


class KhoiPaddleOCR:
    """PaddleOCR post-processing style from khoi03 ALPR Colab. (v5.3.1과 동일)"""

    def __init__(self, paddle_ocr):
        self.paddle_ocr = paddle_ocr

    def preprocess(self, img: np.ndarray):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def num2char(self, num: str):
        return "".join((NUM_TO_CHAR.get(ch, ch) for ch in num))

    def char2num(self, char: str):
        return "".join((CHAR_TO_NUM.get(ch, ch) for ch in char))

    def correct(self, text: str):
        if len(text) <= 4:
            return text
        if text[3] == "-":
            return self.char2num(text[:2]) + self.num2char(text[2]) + self.char2num(text[3:])
        if text[4] == "-":
            return self.char2num(text[:2]) + self.num2char(text[2]) + text[3] + self.char2num(text[4:])
        return text

    def read(self, crop: np.ndarray):
        if crop is None or crop.size == 0:
            return ("", 0.0)
        crop = self.preprocess(crop)
        result = self.paddle_ocr.predict(crop)
        texts, confs = extract_ocr_texts(result)
        if not texts:
            return ("", 0.0)
        if len(texts) == 2:
            text = texts[0].replace("-", "") + "-" + texts[1].replace("-", "")
        else:
            text = "".join(texts)
        text = re.sub("['\\\",\\\\.\\\\?:\\\\!]", "", text).strip().upper()
        text = "".join((ch for ch in text if ch.isalnum() or ch == "-"))
        if 7 <= len(text) <= 10:
            text = self.correct(text)
        text = format_plate_text(text)
        return (text, float(np.mean(confs)) if confs else 0.0)


def run_ocr(ocr_engine, crop: np.ndarray):
    """번호판 crop 에 대해 OCR 수행. (text, conf) 반환. 실패 시 ('', 0.0). (v5.3.1과 동일)"""
    if crop is None or crop.size == 0:
        return ("", 0.0)
    h, w = crop.shape[:2]
    if h < 32:
        scale = 32.0 / max(h, 1)
        crop = cv2.resize(crop, (int(w * scale), 32), interpolation=cv2.INTER_CUBIC)
    return ocr_engine.read(crop)


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = (max(a[0], b[0]), max(a[1], b[1]))
    x2, y2 = (min(a[2], b[2]), min(a[3], b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_score(car_box: np.ndarray, plate_box: np.ndarray, metric: str) -> float:
    inter = intersection_area(car_box, plate_box)
    car_area = max(0.0, car_box[2] - car_box[0]) * max(0.0, car_box[3] - car_box[1])
    plate_area = max(0.0, plate_box[2] - plate_box[0]) * max(0.0, plate_box[3] - plate_box[1])
    denominator = plate_area if metric == "plate_ioa" else car_area + plate_area - inter
    return inter / denominator if denominator > 0 else 0.0


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(np.clip(np.floor(box[0]), 0, width))
    y1 = int(np.clip(np.floor(box[1]), 0, height))
    x2 = int(np.clip(np.ceil(box[2]), 0, width))
    y2 = int(np.clip(np.ceil(box[3]), 0, height))
    return (x1, y1, x2, y2)


def draw_box(frame, xyxy, color, label=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_plate_text(frame, xyxy, text, scale=1.4, thickness=3):
    if not text:
        return
    x1, y1, x2, y2 = map(int, xyxy)
    _, frame_w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = (8, 8)
    cx = x1 + (x2 - x1) // 2 - tw // 2
    cx = max(pad_x, min(cx, frame_w - tw - pad_x))
    ty = max(y1 - 14, th + pad_y)
    cv2.rectangle(frame, (cx - pad_x, ty - th - pad_y), (cx + tw + pad_x, ty + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (cx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness)


def draw_vehicle_box(frame: np.ndarray, box: np.ndarray, car_id: int, flagged: bool) -> None:
    x1, y1, x2, y2 = map(int, box)
    color = COLOR_FIRE if flagged else COLOR_CAR
    thickness = 4 if flagged else 2
    label = f"Car #{car_id} SOLID LINE CROSSING" if flagged else f"Car #{car_id}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


DET_CONFIDENCE = 0.25
DET_NMS_IOU = 0.7
TRACKER = "bytetrack.yaml"
COLOR_CAR = (255, 128, 0)
COLOR_FIRE = (0, 0, 255)
COLOR_CENTER = (0, 255, 255)
COLOR_SOLID = (0, 255, 0)
COLOR_DOTTED = (0, 165, 255)
COLOR_PLATE = (0, 0, 255)
BORDER_MARGIN_PX = 8.0

ASSOCIATION_THRESHOLD = 0.7
ASSOCIATION_METRIC = "plate_ioa"

CSV_COLUMNS = [
    "frame", "car_id", "cx", "cy", "box_w", "box_h", "vehicle_scale",
    "border_clipped", "crossing_fired", "flagged", "plate_text",
    "solid_contact_now", "solid_contact_pixels", "trigger_reason",
    "solid_contact_hold_frames", "solid_contact_hold_seconds",
    "solid_center_crossed", "solid_center_pixels",
    "lane_track_id", "lane_track_hits", "lane_track_misses",
    "lane_track_conf", "lane_track_coasting",
]


# ──────────────────────────────────────────────────────────────────────────
# 차량 검출 → per_car 변환 (v5.3.1 VehicleTracker, 무수정)
# ──────────────────────────────────────────────────────────────────────────
class VehicleTracker:
    """Convert YOLO track boxes into the per-car structure without frame flow."""

    def __init__(self, width: int, height: int, car_class: int):
        self.width = width
        self.height = height
        self.car_class = car_class

    def process(self, result) -> tuple[dict[int, dict], dict]:
        boxes = result.boxes
        per_car: dict[int, dict] = {}
        if boxes is None or len(boxes) == 0:
            return (per_car, {"zooming": False, "suspended": False})
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(classes), -1, dtype=np.int32)
        )
        frame_area = max(float(self.width * self.height), 1.0)
        for index, class_id in enumerate(classes):
            car_id = int(track_ids[index])
            if class_id != self.car_class or car_id < 0:
                continue
            box = xyxy[index]
            box_width = float(box[2] - box[0])
            box_height = float(box[3] - box[1])
            center_x = float((box[0] + box[2]) * 0.5)
            center_y = float((box[1] + box[3]) * 0.5)
            per_car[car_id] = {
                "box": box, "box_w": box_width, "box_h": box_height,
                "cx": center_x, "cy": center_y,
                "area": box_width * box_height / frame_area,
            }
        return (per_car, {"zooming": False, "suspended": False})


# ──────────────────────────────────────────────────────────────────────────
# 차선 폴리라인 트래커 — scripts/01-차선검출모델/lane_pose_track.py 에서 가져옴
# (LaneTracker/Track/estimate_homography/warp_pts 수치 로직 무수정)
# ──────────────────────────────────────────────────────────────────────────
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
        self.kp = kp.astype(np.float32)
        self.vis = vis.astype(bool)
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


# lane_pose_track.py 는 축소 출력 해상도(기본 1600px)에서 게이트(90px)를 튜닝했다.
# 이 파일은 원본 해상도에서 작업하므로, gate/merge_d 를 원본폭 비례로 환산해서 쓴다
# (해상도가 커지면 같은 물리적 허용오차가 더 많은 px 로 보이므로).
GATE_REFERENCE_WIDTH = 1600.0

# ego-motion(ORB+호모그래피) 추정은 비용이 커서(전체 파이프라인의 상당 비중을
# 차지한다고 이 프로젝트 자체 실측 기록에 있음) 축소 프레임에서 하고, 결과
# 호모그래피만 원본 해상도로 환산해서 트랙에 적용한다.
ORB_WORK_WIDTH = 960


def rasterize_tracks_to_solid_map(tracks, height: int, width: int, line_thickness: int) -> np.ndarray:
    """실선(class=0) 트랙의 평활화된 폴리라인을 두께 있는 선으로 그려
    인스턴스id 맵을 만든다 — v5.3.1의 build_solid_id_map+bridge 를 대체.

    `evaluate_solid_crossing_trigger`가 기대하는 형식(배경=0, 차선=양의 정수
    인스턴스id)과 맞춘다. id 는 트랙의 tid+1(0은 배경이라 트랙id 0과 겹치지
    않게 +1).
    """
    solid_map = np.zeros((height, width), dtype=np.int32)
    for t in tracks:
        if t.cls != 0:
            continue
        pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
        if len(pts) < 2:
            continue
        pts_int = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(solid_map, [pts_int], False, int(t.tid) + 1, line_thickness, cv2.LINE_8)
    return solid_map


def draw_lane_tracks_overlay(frame: np.ndarray, live: list, coasting: list) -> np.ndarray:
    for t in coasting:
        _draw_one_track(frame, t, coasting=True)
    for t in live:
        _draw_one_track(frame, t, coasting=False)
    return frame


def _draw_one_track(frame: np.ndarray, t, coasting: bool) -> None:
    pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
    if len(pts) < 2:
        return
    color = COLOR_SOLID if t.cls == 0 else COLOR_DOTTED
    pts_int = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts_int], False, color, 3, cv2.LINE_AA)
    anchor = pts_int[len(pts_int) // 3][0]
    label = f"{'solid' if t.cls == 0 else 'dashed'} #{t.tid} {t.conf:.2f}" + ("~" if coasting else "")
    cv2.putText(frame, label, (int(anchor[0]) + 6, int(anchor[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────
# 오프라인 차선 캐시 (--lane-source cache) — GANet 등 같은 프로세스에 못 들어가는
# 백엔드의 결과를 JSONL로 미리 뽑아두고 읽기만 함. `lane_tracker.step(dets, Hm)`
# 호출부는 무수정 — model/cache 모드 모두 동일한 dets 포맷으로 수렴시킨다.
# ──────────────────────────────────────────────────────────────────────────
class LaneCacheReader:
    """JSONL 캐시를 프레임 순서대로 순차 읽는다(메모리에 전체를 올리지 않음).

    캐시 한 줄 = {"frame": N, "lanes": [{"cls": 0|1, "points": [[x,y]x12]}, ...]}.
    `get_dets(frame_index)`가 요청한 frame과 캐시의 다음 레코드가 안 맞으면
    (영상이 다르거나 프레임이 빠진 경우) 그 프레임은 빈 검출로 처리하고
    1회만 경고를 출력한다 — 엉뚱한 캐시를 끼우는 실수를 조용히 넘기지 않기 위함.
    """

    def __init__(self, path: str):
        self._f = open(path, "r")
        self._next = None
        self._mismatch_warned = False
        self._advance()

    def _advance(self) -> None:
        line = self._f.readline()
        self._next = json.loads(line) if line else None

    def get_dets(self, frame_index: int):
        if self._next is None:
            return []
        if self._next["frame"] != frame_index:
            if not self._mismatch_warned:
                print(f"[lane-cache] WARNING: expected frame {frame_index}, cache has "
                      f"{self._next['frame']} — cache/video 프레임이 안 맞을 수 있음 (빈 검출로 처리, 이후 동일 경고는 생략)")
                self._mismatch_warned = True
            return []
        record = self._next
        self._advance()
        dets = []
        for lane in record["lanes"]:
            pts = np.array(lane["points"], dtype=np.float32)
            vis = np.ones(len(pts), dtype=bool)
            dets.append((pts, vis, int(lane["cls"]), 1.0))
        return dets

    def close(self) -> None:
        self._f.close()


# ──────────────────────────────────────────────────────────────────────────
# 침범 판정 state machine — v5.3.1 그대로(무수정). 입력(solid_map)만 바뀜.
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SolidContactConfig:
    box_width_ratio: float = 0.8
    box_height_ratio: float = 0.8
    min_pixels: int = 12


@dataclass(frozen=True)
class CenterPassConfig:
    radius: int = 9
    min_pixels: int = 3


@dataclass
class CenterPassState:
    crossed: bool = False
    max_pixels: int = 0
    instance_id: int = -1


@dataclass
class ContactRun:
    last_frame: int
    consecutive_frames: int
    instance_id: int
    contact_pixels: int


@dataclass(frozen=True)
class FinalTriggerDecision:
    fired: bool = False
    instance_id: int = -1
    reason: str = ""
    solid_contact: bool = False
    solid_contact_pixels: int = 0
    solid_contact_hold_frames: int = 0
    solid_contact_hold_seconds: float = 0.0
    solid_center_crossed: bool = False
    solid_center_pixels: int = 0


CONTACT_CONFIG = SolidContactConfig()
CENTER_CONFIG = CenterPassConfig()
CONTACT_HOLD_SECONDS = 1.5
CONTACT_GAP_GRACE_FRAMES = 8
CONTACT_RUNS: dict[int, ContactRun] = {}
CENTER_PASS_STATES: dict[int, CenterPassState] = {}
CENTER_HIT_STREAKS: dict[int, int] = {}


def centered_contact_rect(
    box: np.ndarray, width: int, height: int, config: SolidContactConfig
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(float, box)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    horizontal_margin = box_width * (1.0 - config.box_width_ratio) * 0.5
    vertical_margin = box_height * (1.0 - config.box_height_ratio) * 0.5
    left = max(0, int(round(x1 + horizontal_margin)))
    top = max(0, int(round(y1 + vertical_margin)))
    right = min(width, int(round(x2 - horizontal_margin)))
    bottom = min(height, int(round(y2 - vertical_margin)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def solid_contact_instance(box: np.ndarray, solid_map: np.ndarray, config: SolidContactConfig) -> tuple[int, int]:
    height, width = solid_map.shape
    rect = centered_contact_rect(box, width, height, config)
    if rect is None:
        return (-1, 0)
    x1, y1, x2, y2 = rect
    ids = solid_map[y1:y2, x1:x2]
    ids = ids[ids > 0]
    if ids.size == 0:
        return (-1, 0)
    instance_ids, counts = np.unique(ids, return_counts=True)
    best_index = int(np.argmax(counts))
    pixels = int(counts[best_index])
    if pixels < config.min_pixels:
        return (-1, pixels)
    return (int(instance_ids[best_index]), pixels)


def solid_center_instance(box: np.ndarray, solid_map: np.ndarray, config: CenterPassConfig) -> tuple[int, int]:
    height, width = solid_map.shape
    x1, y1, x2, y2 = map(float, box)
    center_x = int(round((x1 + x2) * 0.5))
    center_y = int(round((y1 + y2) * 0.5))
    radius = config.radius
    left = max(0, center_x - radius)
    right = min(width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(height, center_y + radius + 1)
    if right <= left or bottom <= top:
        return (-1, 0)
    region = solid_map[top:bottom, left:right]
    yy, xx = np.ogrid[top:bottom, left:right]
    circle = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    ids = region[circle & (region > 0)]
    if ids.size == 0:
        return (-1, 0)
    instance_ids, counts = np.unique(ids, return_counts=True)
    best_index = int(np.argmax(counts))
    pixels = int(counts[best_index])
    if pixels < config.min_pixels:
        return (-1, pixels)
    return (int(instance_ids[best_index]), pixels)


def update_contact_run_with_grace(
    car_id: int, frame_index: int, instance_id: int, contact_pixels: int, contact_now: bool, border_clipped: bool
) -> ContactRun | None:
    """Keep a contact run across a short lane-detection-only dropout."""
    previous = CONTACT_RUNS.get(car_id)
    if border_clipped:
        CONTACT_RUNS.pop(car_id, None)
        return None
    if not contact_now:
        if previous is not None and frame_index - previous.last_frame > CONTACT_GAP_GRACE_FRAMES:
            CONTACT_RUNS.pop(car_id, None)
        return None
    frame_gap = 1 if previous is None else frame_index - previous.last_frame
    if previous is None or frame_gap < 1 or frame_gap > CONTACT_GAP_GRACE_FRAMES + 1:
        consecutive_frames = 1
    else:
        consecutive_frames = previous.consecutive_frames + frame_gap
    current = ContactRun(
        last_frame=frame_index, consecutive_frames=consecutive_frames,
        instance_id=instance_id, contact_pixels=contact_pixels,
    )
    CONTACT_RUNS[car_id] = current
    return current


def evaluate_solid_crossing_trigger(
    *, border_clipped: bool, box: np.ndarray, solid_map: np.ndarray,
    car_id: int, frame_index: int, fps: float,
) -> FinalTriggerDecision:
    contact_instance_id, contact_pixels = solid_contact_instance(box, solid_map, CONTACT_CONFIG)
    contact_now = contact_instance_id >= 0
    contact_run = update_contact_run_with_grace(
        car_id, frame_index, contact_instance_id, contact_pixels, contact_now, border_clipped
    )
    center_instance_id, center_pixels = solid_center_instance(box, solid_map, CENTER_CONFIG)
    center_now = center_instance_id >= 0
    if contact_run is None:
        CENTER_PASS_STATES.pop(car_id, None)
        CENTER_HIT_STREAKS.pop(car_id, None)
        center_state = None
    else:
        if contact_run.consecutive_frames == 1:
            CENTER_PASS_STATES[car_id] = CenterPassState()
            CENTER_HIT_STREAKS[car_id] = 0
        center_state = CENTER_PASS_STATES.setdefault(car_id, CenterPassState())
        streak = CENTER_HIT_STREAKS.get(car_id, 0) + 1 if center_now else 0
        CENTER_HIT_STREAKS[car_id] = streak
        if streak >= 2:
            center_state.crossed = True
            if center_pixels >= center_state.max_pixels:
                center_state.max_pixels = center_pixels
                center_state.instance_id = center_instance_id
    center_crossed = bool(center_state is not None and center_state.crossed)
    remembered_center_pixels = 0 if center_state is None else center_state.max_pixels
    hold_frames = 0 if contact_run is None else contact_run.consecutive_frames
    hold_seconds = hold_frames / max(float(fps), 1e-06)
    required_frames = max(1, int(round(CONTACT_HOLD_SECONDS * fps)))
    fired = bool(hold_frames >= required_frames and (center_now or center_crossed) and (not border_clipped))
    if not fired:
        return FinalTriggerDecision(
            solid_contact=contact_now, solid_contact_pixels=contact_pixels,
            solid_contact_hold_frames=hold_frames, solid_contact_hold_seconds=hold_seconds,
            solid_center_crossed=center_crossed, solid_center_pixels=remembered_center_pixels,
        )
    return FinalTriggerDecision(
        fired=True,
        instance_id=center_state.instance_id if center_state is not None and center_state.instance_id >= 0 else center_instance_id,
        reason=f"solid_contact_{CONTACT_HOLD_SECONDS:g}s+center_crossing_confirmed",
        solid_contact=contact_now, solid_contact_pixels=contact_pixels,
        solid_contact_hold_frames=hold_frames, solid_contact_hold_seconds=hold_seconds,
        solid_center_crossed=True, solid_center_pixels=max(remembered_center_pixels, center_pixels),
    )


def reset_runtime_state() -> None:
    """Reset mutable cross-frame state for a clean standalone invocation."""
    CONTACT_RUNS.clear()
    CENTER_PASS_STATES.clear()
    CENTER_HIT_STREAKS.clear()
    Track._next = 0


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone stable-SOLID lane crossing detector (YOLO-pose polyline lane model) with optional plate OCR"
    )

    # ── 필수 입출력과 실행 환경 ──
    parser.add_argument("--det-model", required=True, help="차량·번호판 검출 모델(.pt). 4클래스 person/car/motorcycle/plate_number")
    parser.add_argument("--lane-source", choices=("model", "cache"), default="model",
                         help="차선 백엔드 선택: model=YOLO-pose 실시간 추론(기본), cache=GANet 등 오프라인 캐시(JSONL) 읽기. "
                              "성능 비교 목적으로 같은 영상을 두 모드로 각각 돌려서 침범판정 결과를 대조하는 용도")
    parser.add_argument("--lane-model", default=None, help="차선 폴리라인(YOLO-pose) 모델(.pt). 2클래스 0=실선/1=점선, 12키포인트. --lane-source model일 때 필수")
    parser.add_argument("--lane-cache", default=None,
                         help="차선 검출 캐시(JSONL) 경로. 한 줄당 한 프레임: "
                              '{"frame": N, "lanes": [{"cls": 0|1, "points": [[x,y]x12]}, ...]}. '
                              "--lane-source cache일 때 필수. 캐시를 만든 영상과 --input이 반드시 같은 파일(동일 해상도·프레임수)이어야 함")
    parser.add_argument("--input", required=True, help="입력 영상 경로")
    parser.add_argument("--output", required=True, help="결과 영상 저장 경로")
    parser.add_argument("--csv", default=None, help="프레임·차량별 판정 로그 CSV 경로. 생략하면 기록하지 않음")
    parser.add_argument("--device", default="1", help="YOLO 추론 GPU 번호. cpu 도 가능")
    parser.add_argument("--paddle-device", default=None, help="OCR 전용 GPU(gpu:0 형식). 생략하면 --device 값을 변환해 사용")
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수. 0이면 전체")
    parser.add_argument("--output-scale", type=float, default=0.5, help="결과 영상 저장 배율. 판정은 원본 해상도에서 수행됨")

    # ── 검출·추적 ──
    parser.add_argument("--car-class", type=int, default=1, help="검출 모델에서 차량에 해당하는 클래스 번호")
    parser.add_argument("--plate-class", type=int, default=3, help="검출 모델에서 번호판에 해당하는 클래스 번호")
    parser.add_argument("--det-imgsz", type=int, default=1024, help="차량 검출 입력 해상도")
    parser.add_argument("--lane-imgsz", type=int, default=1024, help="차선 pose 입력 해상도")
    parser.add_argument("--det-conf", type=float, default=DET_CONFIDENCE, help="차량·번호판 검출 confidence 임계값")
    parser.add_argument("--lane-conf", type=float, default=0.25, help="차선 pose confidence 임계값")
    parser.add_argument("--nms-iou", type=float, default=DET_NMS_IOU, help="검출 NMS IoU 임계값")
    parser.add_argument("--tracker", default=TRACKER, help="차량 추적기 설정. botsort.yaml 은 optical flow GMC 때문에 거부됨")

    # ── 차선 폴리라인 트래커 (LaneTracker) ──
    parser.add_argument("--kpt-vis-thresh", type=float, default=0.6, help="이 값 미만인 keypoint는 선 연결/매칭에서 제외")
    parser.add_argument("--min-hits", type=int, default=3, help="트랙이 확정(그려짐)되려면 필요한 연속 매칭 프레임 수")
    parser.add_argument("--max-misses", type=int, default=12, help="놓친 트랙을 유지(coast)하는 최대 프레임 수")
    parser.add_argument("--gate", type=float, default=90.0, help=f"매칭 허용 평균 키포인트 거리(px, 원본폭 {GATE_REFERENCE_WIDTH:g}px 기준 — 실제 적용 시 원본 해상도 비례로 환산)")
    parser.add_argument("--ema", type=float, default=0.4, help="키포인트 평활 계수(클수록 최신 검출을 더 신뢰)")
    parser.add_argument("--line-thickness", type=int, default=14, help="실선 트랙을 solid_map에 래스터화할 때의 선 두께(px, 원본 해상도 기준)")
    parser.add_argument("--disable-ego", action="store_true", help="드론 자체 움직임(ORB 호모그래피) 보정을 끈다. 기본은 켜짐")

    # ── 침범 판정 (v5.3.1과 동일 의미) ──
    parser.add_argument("--solid-contact-box-width-ratio", type=float, default=0.8, help="접촉 판정용 차량 박스 폭 축소 비율")
    parser.add_argument("--solid-contact-box-height-ratio", type=float, default=0.8, help="접촉 판정용 차량 박스 높이 축소 비율")
    parser.add_argument("--solid-contact-min-pixels", type=int, default=12, help="접촉으로 인정할 최소 실선 픽셀 수")
    parser.add_argument("--solid-contact-hold-seconds", type=float, default=1.5, help="발화까지 필요한 누적 접촉 시간(초). 최대 8프레임 공백은 같은 구간으로 이어짐")
    parser.add_argument("--solid-center-radius", type=int, default=15, help="차량 중앙 판정 원의 반경(px). 화면 노란 원과 동일")
    parser.add_argument("--solid-center-min-pixels", type=int, default=3, help="중앙 원 안에서 통과로 인정할 최소 실선 픽셀 수")
    parser.add_argument("--border-margin", type=float, default=BORDER_MARGIN_PX, help="화면 경계에서 이 거리 안에 걸친 차량은 발화 차단. 0이면 차단 해제(px)")

    # ── 번호판 OCR과 화면 표시 ──
    parser.add_argument("--no-ocr", action="store_true", help="번호판 OCR 비활성화. paddle 계열 패키지가 없어도 실행됨")
    parser.add_argument("--plate-text-scale", type=float, default=1.4, help="화면에 그리는 번호판 글자 크기 배율")
    parser.add_argument("--plate-text-thickness", type=int, default=3, help="화면에 그리는 번호판 글자 두께")
    parser.add_argument("--hide-lane-overlay", action="store_true", help="차선 폴리라인 오버레이를 그리지 않음")
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.lane_source == "model" and not args.lane_model:
        raise ValueError("--lane-model is required when --lane-source model")
    if args.lane_source == "cache" and not args.lane_cache:
        raise ValueError("--lane-cache is required when --lane-source cache")
    if Path(args.tracker).name.lower() == "botsort.yaml":
        raise ValueError("botsort.yaml is not allowed because its default GMC uses sparseOptFlow; use bytetrack.yaml or a verified no-flow tracker")
    if args.border_margin < 0:
        raise ValueError("--border-margin cannot be negative")
    if args.min_hits < 1:
        raise ValueError("--min-hits must be at least 1")
    if args.max_misses < 0:
        raise ValueError("--max-misses cannot be negative")
    if args.gate <= 0:
        raise ValueError("--gate must be positive")
    if not 0 < args.ema <= 1:
        raise ValueError("--ema must be in (0, 1]")
    if args.line_thickness < 1:
        raise ValueError("--line-thickness must be at least 1")
    if not 0 < args.solid_contact_box_width_ratio <= 1:
        raise ValueError("--solid-contact-box-width-ratio must be in (0, 1]")
    if not 0 < args.solid_contact_box_height_ratio <= 1:
        raise ValueError("--solid-contact-box-height-ratio must be in (0, 1]")
    if args.solid_contact_min_pixels < 1:
        raise ValueError("--solid-contact-min-pixels must be at least 1")
    if args.solid_contact_hold_seconds <= 0:
        raise ValueError("--solid-contact-hold-seconds must be positive")
    if args.solid_center_radius < 0 or args.solid_center_min_pixels < 1:
        raise ValueError("solid center radius/minimum pixels are invalid")
    if not 0 < args.output_scale <= 1:
        raise ValueError("--output-scale must be in (0, 1]")


def parse_and_configure_runtime() -> argparse.Namespace:
    global CONTACT_CONFIG, CENTER_CONFIG, CONTACT_HOLD_SECONDS
    args = parse_cli_args()
    validate_cli_args(args)
    CONTACT_CONFIG = SolidContactConfig(
        box_width_ratio=args.solid_contact_box_width_ratio,
        box_height_ratio=args.solid_contact_box_height_ratio,
        min_pixels=args.solid_contact_min_pixels,
    )
    CONTACT_HOLD_SECONDS = args.solid_contact_hold_seconds
    CENTER_CONFIG = CenterPassConfig(radius=args.solid_center_radius, min_pixels=args.solid_center_min_pixels)
    args.center_radius = CENTER_CONFIG.radius
    return args


# ──────────────────────────────────────────────────────────────────────────
# 영상 실행
# ──────────────────────────────────────────────────────────────────────────
def process_video(args: argparse.Namespace) -> None:
    validate_cli_args(args)
    det_model_path = Path(args.det_model).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    required_paths = [(det_model_path, "detection model"), (input_path, "input video")]
    lane_model_path = lane_cache_path = None
    if args.lane_source == "model":
        lane_model_path = Path(args.lane_model).expanduser().resolve()
        required_paths.append((lane_model_path, "lane pose model"))
    else:
        lane_cache_path = Path(args.lane_cache).expanduser().resolve()
        required_paths.append((lane_cache_path, "lane cache (jsonl)"))
    for path, label in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    render_total = (
        min(total_frames, args.max_frames) if args.max_frames > 0 and total_frames > 0 else args.max_frames or total_frames
    )
    out_width = max(2, int(round(width * args.output_scale)) // 2 * 2)
    out_height = max(2, int(round(height * args.output_scale)) // 2 * 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")
    csv_file = csv_writer = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_COLUMNS)

    det_model = YOLO(str(det_model_path))
    lane_model = YOLO(str(lane_model_path)) if args.lane_source == "model" else None
    lane_cache = LaneCacheReader(str(lane_cache_path)) if args.lane_source == "cache" else None
    tracker = VehicleTracker(width, height, args.car_class)
    gate = args.gate * (width / GATE_REFERENCE_WIDTH)
    lane_tracker = LaneTracker(args.min_hits, args.max_misses, gate, args.ema)

    use_ego = not args.disable_ego
    orb = cv2.ORB_create(1500) if use_ego else None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if use_ego else None
    prev_gray = None
    ego_scale = ORB_WORK_WIDTH / width
    ego_h = max(2, int(round(height * ego_scale)))

    ocr_engine = stabilizer = None
    if not args.no_ocr:
        _prepare_paddleocr_environment()
        from paddleocr import PaddleOCR

        paddle_engine = PaddleOCR(
            lang="en", text_detection_model_name=PADDLE_DET_MODEL, text_recognition_model_name=PADDLE_REC_MODEL,
            use_textline_orientation=True, use_doc_orientation_classify=False, use_doc_unwarping=False,
            device=normalize_paddle_device(args.paddle_device or args.device),
        )
        ocr_engine = KhoiPaddleOCR(paddle_engine)
        stabilizer = PlateTextStabilizer(min_votes=3, history=30)

    det_results = det_model.track(
        source=str(input_path), conf=args.det_conf, iou=args.nms_iou, imgsz=args.det_imgsz,
        device=args.device, tracker=args.tracker, stream=True, persist=True, verbose=False,
    )

    flagged_ids: set[int] = set()
    fired_frame: dict[int, int] = {}
    fired_lane: dict[int, int] = {}
    plate_text_by_car: dict[int, str] = {}
    try:
        with tqdm(total=render_total or None, desc="Solid lane crossing (polyline) + OCR", unit="frame", dynamic_ncols=True) as progress:
            for frame_index, result in enumerate(det_results):
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                original = result.orig_img
                frame = original.copy()
                per_car, cam = tracker.process(result)

                Hm = None
                if use_ego:
                    small = cv2.resize(original, (ORB_WORK_WIDTH, ego_h))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        Hm_small = estimate_homography(prev_gray, gray, orb, bf)
                        if Hm_small is not None:
                            s = ego_scale
                            scale_mat = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
                            Hm = np.linalg.inv(scale_mat) @ Hm_small @ scale_mat
                    prev_gray = gray

                if args.lane_source == "cache":
                    dets = lane_cache.get_dets(frame_index)
                else:
                    pose_result = lane_model.predict(
                        source=original, imgsz=args.lane_imgsz, conf=args.lane_conf, device=args.device, verbose=False
                    )[0]
                    dets = []
                    if pose_result.keypoints is not None and len(pose_result.keypoints):
                        kxy = pose_result.keypoints.data.cpu().numpy()
                        cls_arr = pose_result.boxes.cls.cpu().numpy().astype(int)
                        conf_arr = pose_result.boxes.conf.cpu().numpy()
                        for i in range(len(kxy)):
                            kp = kxy[i]
                            v = kp[:, 2] > args.kpt_vis_thresh if kp.shape[1] > 2 else np.ones(len(kp), bool)
                            if v.sum() < 2:
                                continue
                            # ultralytics 가 입력으로 넣은 원본(original) 해상도로 이미 역스케일해서 돌려준다
                            p = kp[:, :2].astype(np.float32)
                            dets.append((p, v, int(cls_arr[i]), float(conf_arr[i])))

                live, coasting = lane_tracker.step(dets, Hm)
                all_tracks = live + coasting
                solid_map = rasterize_tracks_to_solid_map(all_tracks, height, width, args.line_thickness)
                tracks_by_id = {t.tid + 1: t for t in all_tracks}

                if not args.hide_lane_overlay:
                    frame = draw_lane_tracks_overlay(frame, live, coasting)

                car_boxes: dict[int, np.ndarray] = {}
                for car_id, info in per_car.items():
                    box = info["box"]
                    car_boxes[car_id] = box
                    center = np.array([info["cx"], info["cy"]], dtype=np.float32)
                    vehicle_scale = math.hypot(float(info["box_w"]), float(info["box_h"]))
                    x1, y1, x2, y2 = map(float, box)
                    margin = args.border_margin
                    border_clipped = bool(
                        margin > 0 and (x1 <= margin or y1 <= margin or x2 >= width - margin or (y2 >= height - margin))
                    )
                    final_trigger = evaluate_solid_crossing_trigger(
                        border_clipped=border_clipped, box=box, solid_map=solid_map,
                        car_id=car_id, frame_index=frame_index, fps=fps,
                    )
                    combined_fired = final_trigger.fired
                    if combined_fired and car_id not in flagged_ids:
                        flagged_ids.add(car_id)
                        fired_frame[car_id] = frame_index
                        fired_lane[car_id] = final_trigger.instance_id
                        print(
                            f"[SOLID-CROSSING] frame={frame_index} car={car_id} lane={final_trigger.instance_id} "
                            f"reason={final_trigger.reason} hold={final_trigger.solid_contact_hold_seconds:.2f}s"
                        )
                    cv2.circle(frame, tuple(np.rint(center).astype(int)), args.center_radius, COLOR_CENTER, -1, cv2.LINE_AA)
                    draw_vehicle_box(frame, box, car_id, car_id in flagged_ids)
                    if csv_writer is not None:
                        lt = tracks_by_id.get(final_trigger.instance_id)
                        csv_row = [
                            frame_index, car_id,
                            f"{info['cx']:.2f}", f"{info['cy']:.2f}", f"{info['box_w']:.2f}", f"{info['box_h']:.2f}",
                            f"{vehicle_scale:.2f}",
                            int(border_clipped), int(combined_fired), int(car_id in flagged_ids),
                            plate_text_by_car.get(car_id, ""),
                            int(final_trigger.solid_contact), final_trigger.solid_contact_pixels, final_trigger.reason,
                            final_trigger.solid_contact_hold_frames, f"{final_trigger.solid_contact_hold_seconds:.3f}",
                            int(final_trigger.solid_center_crossed), final_trigger.solid_center_pixels,
                            lt.tid if lt is not None else "",
                            lt.hits if lt is not None else "",
                            lt.misses if lt is not None else "",
                            f"{lt.conf:.3f}" if lt is not None else "",
                            int(lt.misses > 0) if lt is not None else "",
                        ]
                        csv_writer.writerow(csv_row)

                if ocr_engine is not None and result.boxes is not None and (len(result.boxes) > 0):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    track_ids = (
                        result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.full(len(classes), -1)
                    )
                    car_indices = [
                        index for index, class_id in enumerate(classes)
                        if class_id == args.car_class and track_ids[index] >= 0
                    ]
                    plate_drawn: set[int] = set()
                    for plate_index, class_id in enumerate(classes):
                        if class_id != args.plate_class or not car_indices:
                            continue
                        score, car_index = max(
                            ((overlap_score(boxes[index], boxes[plate_index], ASSOCIATION_METRIC), index) for index in car_indices),
                            key=lambda item: item[0],
                        )
                        if score < ASSOCIATION_THRESHOLD:
                            continue
                        car_id = int(track_ids[car_index])
                        if car_id not in flagged_ids:
                            continue
                        x1, y1, x2, y2 = clip_box(boxes[plate_index], width, height)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        text, confidence = run_ocr(ocr_engine, original[y1:y2, x1:x2])
                        stable = stabilizer.update(car_id, text, confidence)
                        if stable:
                            plate_text_by_car[car_id] = stable
                        shown = plate_text_by_car.get(car_id, stable)
                        draw_box(frame, boxes[plate_index], COLOR_PLATE)
                        draw_plate_text(frame, boxes[plate_index], shown, scale=args.plate_text_scale, thickness=args.plate_text_thickness)
                        plate_drawn.add(car_id)
                    for car_id in flagged_ids:
                        if car_id in plate_drawn or car_id not in car_boxes:
                            continue
                        text = plate_text_by_car.get(car_id)
                        if not text:
                            continue
                        car_box = car_boxes[car_id]
                        anchor = np.array([car_box[0], car_box[1] - 40, car_box[2], car_box[1] - 40], dtype=np.float32)
                        draw_plate_text(frame, anchor, text, scale=args.plate_text_scale, thickness=args.plate_text_thickness)

                n_solid = sum((1 for t in all_tracks if t.cls == 0))
                n_dashed = sum((1 for t in all_tracks if t.cls == 1))
                cv2.putText(
                    frame,
                    f"frame={frame_index} lanes={len(all_tracks)} solid={n_solid} dashed={n_dashed} "
                    f"coasting={len(coasting)} flagged={len(flagged_ids)}",
                    (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
                )
                if (out_width, out_height) != (width, height):
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                progress.update(1)
                progress.set_postfix(cars=len(per_car), solid=n_solid, flagged=len(flagged_ids), refresh=False)
    finally:
        writer.release()
        if csv_file is not None:
            csv_file.close()
        if lane_cache is not None:
            lane_cache.close()
    if flagged_ids:
        details = "  ".join(
            (
                f"#{car_id}@f{fired_frame[car_id]} lane={fired_lane[car_id]} plate={plate_text_by_car.get(car_id, '?')}"
                for car_id in sorted(flagged_ids, key=lambda cid: fired_frame[cid])
            )
        )
        print(f"[DETECTED] {details}")
    else:
        print("[DETECTED] none")
    if args.csv:
        print(f"[CSV] {Path(args.csv).expanduser().resolve()}")
    print(f"[DONE] {output_path} ({out_width}x{out_height})")


def main() -> None:
    reset_runtime_state()
    process_video(parse_and_configure_runtime())


if __name__ == "__main__":
    main()
