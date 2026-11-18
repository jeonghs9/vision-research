#!/usr/bin/env python3
"""밑변 선분 vs 실선 폴리라인의 기하학적 교차로 침범을 판정한다 (`--hold-seconds`
동안 반대편 유지돼야 확정, 기본 1.5초). 검증용 스크립트(production 코드 아님).

핵심 설계 — 3중 안전장치:
  1) 에피소드 **시작**: `bottom_edge_touches_polyline()`(유한 선분 교차)로
     막는다 — 멀리 떨어진 무관한 선과 "부호가 다르다"는 이유만으로 걸쳐있다고
     오판하는 걸 방지(#9328 사례).
  2) **[안전장치 1: 시간]** 처음 반대편으로 확정되는 순간, 마지막 실제 접촉이
     `--hold-seconds`(1.5초) 이내인지 확인 — 범위를 넘으면(선에서 너무 오래
     멀어진 채 확정됐으면) 무효 처리(1037 영상 `#29062` — 선에서 750px+
     떨어진 채 통과 확정됐던 버그 — 수정).
  3) **[안전장치 2: 교차 비율(ratio) 점프 — 2026-10-16, 박스-크기 가드를
     대체함]** 차량이 **진짜로 탐지 자체가 끊겼다**(border-clip으로 판정만
     건너뛴 것과는 다름) 직후, `crossing_ratio()`(아래 참고)가 직전 관측값
     대비 갑자기 거의 끝(1.0 가까이)까지 뛰었으면 "다른 차량으로 재할당된
     것"으로 보고 무효 처리(1539 영상 `#14925`/`#54009`/`#55533` — 혼잡
     구간에서 짧은 탐지 공백 사이 차량·차선 추적 ID가 다른 실물로 재할당됐던
     버그 — 수정). 에피소드 시작 순간(이땐 "직전 ratio"가 없으니 시작
     ratio 자체가 이미 절반 이상이면 아예 시작하지 않음)과 진행 중 둘 다
     체크한다 — 기존 "박스 크기 1.5배 급변" 가드를 대체한 버전이다(교체
     경위와 실측 근거는 아래 2026-10-16 절 참고).

검증 결과(박스-크기 버전, `crossing_trigger_bottomline.py` 원본 기준): 0816·
1037·1539 영상 × YOLO-pose·GANet 6개 조합 전부에서 기존 정탐(`#4`,`#22816`,
`#22150`) 손실 없이 위 오탐 전부 해결 확인. 상세 분석은
`docs/02-실선위반판정/03. ID재할당_방지가드.md` 참고. **이
test.py는 안전장치 2를 ratio 점프 버전으로 교체한 상태 — 재검증 진행
중/완료 여부는 아래 2026-10-16 절 참고.**

**교차 비율(ratio) 관찰 + 분홍 게이지 시각화(2026-10-15 추가)**: 발화 로직에는
전혀 관여하지 않는 순수 관찰/표시용 추가다. `crossing_ratio()`가 밑변이
`before_sign` 기준 몇 %나 반대편으로 넘어갔는지(0~1, 걸쳐있을 땐 두 끝점의
수직거리 비율로 선형보간) 계산해서 CSV `ratio` 컬럼에 기록하고, 영상에서는
박스 밑변부터 그 비율만큼 분홍색을 채워서 보여준다(확정 시 기존 빨간
VIOLATION 박스로 전환). 정탐 3건(`#4`,`#22816`,`#22150`) 모두 border-clip
공백을 사이에 두고도 수십 프레임에 걸쳐 0→1로 완만하게 상승하는 패턴 확인
(GANet, 3영상 전체 재검증 — 발화 결과 동일, 신규 오탐 없음; YOLO-pose는
아직 재확인 안 함, 로직 자체가 순수 추가라 회귀 가능성은 낮다고 판단).

알려진 한계:
  - 1.5배 크기 기준은 수학적으로 유도한 게 아니라 실측 4개 사례로 보정한
    경험적 수치 — 더 극단적인 사례가 나오면 재검토 필요. 0816 영상
    YOLO-pose의 `#64049`/`#68737`(frame 8200~8500대 반복 오탐)는 별개
    미해결 이슈.
  - **에피소드 시작 조건의 사각지대**: "직전 프레임엔 한쪽에 깔끔히 있다가
    이번 프레임에 막 걸치기 시작"하는 1프레임 전환만 시작으로 인정한다.
    실선과 거의 나란히(사선으로) 달리는 차처럼 "걸쳐있음"(sign=None) 상태가
    여러 프레임 연속되면, 직전 프레임도 계속 걸쳐있는 걸로 평가돼
    "애매하니 skip"에 매 프레임 걸려서 **에피소드가 영원히 시작되지
    않는다** — 실선을 명백히 밟고 있어도 발화도 안 되고 ratio 게이지도
    안 뜸(0816 `#11191` 트럭으로 실측 확인, frame 1631~1705+ 75프레임+
    연속 걸쳐있음에도 에피소드 미시작) — **의도적으로 이대로 둔다**(아래
    2026-10-15 시도 참고, 고치려다 오탐이 크게 늘어서 포기·원복).

2026-10-15 수정 시도 — 실패, 원복됨: before_sign을 직전 1프레임이 아니라
"최근 hold_frames개 기록 중 마지막으로 sign이 확정됐던 시점"까지 거슬러
찾도록 완화해봤으나, **0816/1037/1539 전부 발화 건수가 1건→16~20건으로
폭증**했다. 원인: 신규 추적 ID가 선 근처에서 막 생길 때 박스가 아직
안정되기 전 몇 프레임 동안 흔들리는데, 그중 우연히 "한쪽에 깔끔히 있음"
으로 잡힌 프레임을 lookback이 "원래 있던 쪽"으로 오인해서, 트랙이 생기자
마자 바로(ratio가 시작부터 0.9+인 채로) 에피소드가 열려버렸다(`#5174`
실측: 에피소드 시작 첫 프레임부터 ratio=0.962). 기존의 "직전 1프레임만
본다"는 엄격한 규칙이 사실 "막 생긴 불안정한 트랙"을 걸러주는 역할도
겸하고 있었던 것 — 그 규칙을 풀면서 부작용이 의도한 효과보다 훨씬 컸다.
재시도한다면 "최근 아무 프레임"이 아니라 "최근 K프레임 연속으로 같은
쪽에 안정적으로 있었던 경우"만 인정하는 식으로 훨씬 보수적으로 가야 할 것
— 단, 아직 미구현.

2026-10-16 — 안전장치 2를 "박스 크기" → "ratio 점프"로 교체:

**1단계(관측)**: 박스-크기 가드를 일시적으로 꺼서(취소 동작만 끄고 계산은
유지) 공백 전후 `ratio`가 어떻게 움직이는지 3영상(GANet) 전체로 관측.
결과 — 알려진 오탐 3건은 전부 공백을 만나자마자 ratio가 직전 관측값에서
**곧바로 1.0(완전히 끝)으로 "착지"**했다(`#14925`: 0.21→1.0, `#54009`:
0.02→1.0, `#55533`: 0.15→1.0). 반면 알려진 정탐 2건(공백 중 ratio 데이터가
있는 `#4`,`#22816`)은 공백이 있어도 거의 안 뛰거나(`#4`: 0.85→0.87) 뛰더라도
그 뒤로 계속 서서히 더 올라갔다(`#22816`: 0.11→0.34, 이후 수백 프레임에
걸쳐 추가 상승) — **"바로 끝까지 가버리는지"**가 핵심 구분점. 추가로 252개
전체 차량 중 ratio_jump가 큰(>0.4) "미지정" 차량이 15건 더 있었지만,
가드 없이도 전부 다른 안전장치(hold-seconds 등)에 걸려 실제 발화까지는
안 갔다 — 과잉 탐지 위험이 낮다는 뜻.

추가로 "에피소드 시작 순간에 공백을 끼고 바로 높은 ratio로 등장"하는
사각지대도 확인(5건, ratio 0.78~0.96) — 전부 박스 크기는 거의 안 변한
케이스라 기존 박스-크기 가드로도 못 잡았을 것들. 다행히 이번에도 실제
발화까지는 안 갔지만, 설계상 구멍이라 **에피소드 시작 순간에도 대칭적으로
체크**를 추가했다(아래 참고). 정상적인 "서서히 접근" 시작은 전부
ratio<=0.36이었고 이례적 급발진 패턴은 전부 0.78 이상이라 임계값 0.5로
뚜렷하게 분리됨. 세 known 정탐(`#4`,`#22816`,`#22150`)은 전부 공백 없이
깨끗하게 시작해서 이 체크의 영향을 받지 않음(실측 확인).

**2단계(구현)**: `crossing_trigger_bottomline_test.py`에 박스-크기 가드를
제거하고 ratio 점프 가드로 교체—
  - 진행 중: `true_detection_gap and ratio >= 0.95 and ratio_jump >= 0.4`
    면 에피소드 취소.
  - 시작 순간: `true_detection_gap and start_ratio >= 0.5`면 에피소드를
    아예 시작하지 않음.
둘 다 1단계 실측값에 근거한 임계값이지만, 기존 "1.5배 크기" 기준과
마찬가지로 수학적으로 유도한 값은 아니고 3영상 실측으로 보정한 경험적
수치다 — 더 극단적인 사례가 나오면 재검토 필요(기존 한계와 동일한 성격).
박스 크기가 비슷한 차량끼리 ID가 재할당되는 경우(박스-크기 가드의
사각지대)도 이 신호로는 잡을 수 있을 것으로 기대 — 단, 아직 실측된
사례는 없음. 3영상×2백엔드 재검증 결과는 아래(실행 뒤 갱신 예정).

2026-10-16 진행 중 — 실험: 에고모션 ORB 특징점 수 1500→800. 파이프라인
전체에서 에고모션(ORB)이 최대 병목(38~49%)이라 특징점을 줄이면 속도가
오를 걸로 기대되지만, 도로면처럼 특징이 적은 구간에서 homography 추정이
불안정해져 차선 추적 정확도가 떨어질 위험이 있다 — 3영상×2백엔드로
발화 결과(오탐/정탐 동일한지) + 처리속도(fps) 둘 다 재검증.

실측 결과(5개 값 테스트, 매번 3영상×2백엔드 전체 재검증):

| 특징점 수 | 평균 속도 | 정확도 |
|---|---|---|
| 1500(기존) | 기준 | 전부 정상 |
| 800 | +22~31% | 전부 정상(2회 독립 검증) |
| 700 | +24~34% | 전부 정상 — 단, 650에서 바로 실패해서 안전마진 거의 없음 |
| 650 | +25~36% | **1037 GANet `#22816` 미탐** |
| 500 | +31~41% | **1037 GANet `#22816` 미탐**(650과 동일 사례) |

650/500 둘 다 같은 사례(1037 GANet `#22816`)에서 실패 — CSV로 원인 확인:
에피소드가 시작돼 ratio가 0.68까지 올라갔다가(거의 다 넘어간 상태)
다시 0.28로 내려가며 취소됐다. 특징점이 너무 적어지면 homography(에고
모션) 추정이 흔들리고, 그 노이즈가 차선 위치 추정에 들어가 실제
횡단을 "다시 원래 쪽으로 돌아간 것"으로 오판하게 만든다.

**결론: 800으로 확정, 2026-10-16 `crossing_trigger_bottomline.py`
원본에 승격.** 700은 통과했지만 650에서 바로 실패 재현된 걸 감안하면
마진이 너무 적어서 기각(자세한 분석은
`docs/02-실선위반판정/04. 교차비율게이지.md` 15절 참고).

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
    crossing_trigger_bottomline.py \
    --det-model /home/hsjeong/workspace/applications/traffic-video-analytics/models/det-best.pt \
    --lane-model /home/hsjeong/workspace/applications/traffic-video-analytics/models/pose-best.pt \
    --input clip_first60s.mp4 \
    --out-csv trigger_log.csv \
    --out-video trigger_overlay.mp4 \
    --device 0
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/hsjeong/workspace/applications/traffic-video-analytics/scripts/02-실선위반판정")
from visualize_solid_lane_crossing_polyline import (  # noqa: E402
    BORDER_MARGIN_PX,
    GATE_REFERENCE_WIDTH,
    ORB_WORK_WIDTH,
    LaneTracker,
    VehicleTracker,
    estimate_homography,
)
from ultralytics import YOLO  # noqa: E402


def point_to_polyline_signed_distance(point: np.ndarray, polyline: np.ndarray):
    best_dist = float("inf")
    best_sign = 0.0
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            continue
        t = np.clip(np.dot(point - a, ab) / ab_len2, 0.0, 1.0)
        closest = a + t * ab
        diff = point - closest
        dist = float(np.hypot(diff[0], diff[1]))
        if dist < best_dist:
            best_dist = dist
            cross_z = ab[0] * diff[1] - ab[1] * diff[0]
            best_sign = 1.0 if cross_z >= 0 else -1.0
    return best_dist, best_sign


def segment_to_polyline_relation(p_left: np.ndarray, p_right: np.ndarray, polyline: np.ndarray):
    """박스 밑변(왼쪽 끝~오른쪽 끝)과 실선 폴리라인의 관계.
    둘 다 부호가 같다 -> 그 부호. 다르다 -> 걸쳐있음(None)."""
    d_l, s_l = point_to_polyline_signed_distance(p_left, polyline)
    d_r, s_r = point_to_polyline_signed_distance(p_right, polyline)
    dist = min(d_l, d_r)
    sign = s_l if s_l == s_r else None
    return dist, sign


def crossing_ratio(p_left: np.ndarray, p_right: np.ndarray, polyline: np.ndarray, before_sign: float) -> float:
    """밑변 선분이 "원래 쪽"(before_sign) 기준으로 몇 %나 반대편으로 넘어갔는지
    0~1 연속값으로 반환한다(실험용 — 교차 비율의 연속성을 관찰하기 위함,
    아직 발화 로직에는 안 씀). 두 끝점이 같은 쪽이면 0(원래 쪽) 또는 1(완전히
    반대편). 걸쳐있으면, 두 끝점의 수직거리 비율로 교차점 위치를 선형보간
    (닮은꼴 삼각형 — 국소적으로 선이 직선에 가깝다고 가정, 근사치)해서 "원래
    쪽 끝점 ~ 교차점" 대 "교차점 ~ 반대쪽 끝점" 비율로 환산한다."""
    d_l, s_l = point_to_polyline_signed_distance(p_left, polyline)
    d_r, s_r = point_to_polyline_signed_distance(p_right, polyline)
    if s_l == s_r:
        return 0.0 if s_l == before_sign else 1.0
    if s_l == before_sign:
        d_same, d_other = d_l, d_r
    else:
        d_same, d_other = d_r, d_l
    denom = d_same + d_other
    if denom < 1e-6:
        return 0.5
    return float(np.clip(d_other / denom, 0.0, 1.0))


def _segments_cross(p1, p2, p3, p4) -> bool:
    """선분 p1-p2 와 p3-p4 가 (무한 직선이 아니라) **유한한 구간 안에서**
    실제로 교차하는지. 표준 방향(orientation) 기반 판정 — 거리 개념이 아예
    없어도, 두 선분이 진짜로 겹치지 않으면 True 가 나올 수 없다."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def bottom_edge_touches_polyline(p_left: np.ndarray, p_right: np.ndarray, polyline: np.ndarray) -> bool:
    """밑변 선분이 폴리라인의 **어느 한 구간이라도** 실제로(유한 범위 내에서)
    가로지르는지. #9328 사례(1000px+ 떨어진 무관한 선과 "부호가 다르다"는
    이유만으로 걸쳐있다고 오판)를 막기 위한 진짜 기하학적 교차 검사 —
    거리 제한(--max-solid-dist) 없이도, 실제로 안 닿아있으면 True가 될 수 없다."""
    for i in range(len(polyline) - 1):
        if _segments_cross(p_left, p_right, polyline[i], polyline[i + 1]):
            return True
    return False


# ============================================================
# 번호판 탐지+OCR — scripts/02-실선위반판정/visualize_solid_lane_crossing_v5.3.1.py
# 에서 그대로 이식(2026-10-16). "VIOLATION으로 이미 확정된 차량만" OCR을
# 돌린다(v5.3.1과 동일한 설계 — 매 프레임 모든 차에 OCR 돌리면 너무 느림).
# det-best.pt는 4클래스(0=person,1=car,2=motorcycle,3=plate_number) 모델이라
# 이미 쓰고 있던 모델을 그대로 재사용, 추가 모델 불필요.
# ============================================================

APPLICATION_ROOT = Path("/home/hsjeong/workspace/applications/traffic-video-analytics")
DEFAULT_OCR_CACHE_DIR = Path(
    os.environ.get("TRAFFIC_VIDEO_OCR_CACHE_DIR", APPLICATION_ROOT / "models" / "runtime-cache" / "ocr")
)
DEFAULT_PADDLE_CACHE_DIR = DEFAULT_OCR_CACHE_DIR / ".paddleocr-cache"
PADDLE_DET_MODEL = None  # PaddleOCR 기본 영문 검출 모델
PADDLE_REC_MODEL = None  # PaddleOCR 기본 영문 인식 모델

ASSOCIATION_METRIC = "plate_ioa"  # 번호판 박스가 차량 박스에 겹치는 비율 기준
ASSOCIATION_THRESHOLD = 0.7
COLOR_PLATE = (0, 0, 255)


def normalize_paddle_device(device):
    """YOLO 스타일 device 값을 PaddleOCR/PaddleX 형식으로 변환."""
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


# 번호판 OCR에서 숫자/문자 혼동 교정(khoi03 ALPR 방식)
NUM_TO_CHAR = {"0": "D", "1": "T", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B", "9": "P", "$": "S"}
CHAR_TO_NUM = {
    "A": "4", "B": "8", "C": "0", "D": "0", "E": "6", "F": "5", "G": "6", "H": "4", "I": "1", "J": "1",
    "L": "4", "O": "0", "P": "6", "Q": "0", "R": "8", "S": "5", "T": "7", "U": "0", "V": "0", "Y": "1", "Z": "2",
}


class PlateTextStabilizer:
    """track_id 별로 여러 프레임의 OCR 결과를 모아 최빈값(빈도*평균신뢰도)으로
    안정화한다. 최소 min_votes번 누적돼야 확정값을 내보낸다."""

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
    "0": "O", "1": "I", "2": "Z", "3": "B", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B", "9": "P",
}


def format_plate_text(raw: str) -> str:
    """OCR 원문을 '##A ###.##' 형태로 정규화. 위치별로 흔한 OCR 오인식을 보정."""
    token = clean_plate_text(raw)
    if len(token) < 8:
        return ""
    best = ""
    best_exact_matches = -1
    for start in range(len(token) - 7):
        window = token[start:start + 8]
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
    """PaddleOCR 2.x/3.x 결과 포맷 둘 다에서 텍스트/신뢰도를 뽑는다."""
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
    """PaddleOCR 후처리(khoi03 ALPR Colab 방식) — YOLO가 이미 번호판을 크롭해
    주므로, 이 클래스는 OCR 텍스트 정리/보정만 담당한다."""

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
    """번호판 crop에 대해 OCR 수행. (text, conf) 반환. 실패 시 ('', 0.0)."""
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


def draw_plate_text(frame, xyxy, text, scale=1.4, thickness=3):
    """번호판 박스 위에 인식된 텍스트를 표시."""
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


def _prepare_paddleocr_environment() -> None:
    """PaddleOCR import 직전에 이 프로젝트 전용 캐시 경로를 환경변수로 지정."""
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


class FfmpegWriter:
    """cv2.VideoWriter(mp4v)는 압축이 거의 안 돼서 결과 영상이 수백MB~수GB로
    커진다 — ffmpeg(libx264)에 raw 프레임을 바로 파이프로 흘려서 처음부터
    압축된 h264로 저장한다(끝나고 따로 재인코딩 안 해도 됨)."""

    def __init__(self, out_path: str, w: int, h: int, fps: float, crf: int = 23, preset: str = "fast"):
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


class LaneCacheReader:
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
                      f"{self._next['frame']} — 캐시/영상 프레임이 안 맞을 수 있음 (빈 검출로 처리, 이후 동일 경고 생략)")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model", required=True)
    ap.add_argument("--lane-source", choices=("model", "cache"), default="model")
    ap.add_argument("--lane-model", default=None)
    ap.add_argument("--lane-cache", default=None)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--output-scale", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--car-class", type=int, default=1)
    ap.add_argument("--kpt-vis-thresh", type=float, default=0.6)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=12)
    ap.add_argument("--gate", type=float, default=90.0)
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--hold-seconds", type=float, default=1.5,
                     help="완전히 넘어간 뒤 이 시간(초) 동안 그 상태가 유지돼야 발화 (0이면 즉시 발화)")
    ap.add_argument("--no-ocr", action="store_true", help="번호판 OCR 비활성화 (PaddleOCR 로딩 생략)")
    ap.add_argument("--plate-class", type=int, default=3, help="검출 모델에서 번호판에 해당하는 클래스 id")
    ap.add_argument("--paddle-device", default=None, help="PaddleOCR용 device (생략시 --device 재사용)")
    ap.add_argument("--plate-text-scale", type=float, default=1.4)
    ap.add_argument("--plate-text-thickness", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수 (0이면 끝까지) — "
                     "추적/에피소드 상태가 누적식이라 중간부터 시작은 못하지만, 마지막 관심 프레임 "
                     "직후에서 멈추는 용도로는 쓸 수 있다")
    a = ap.parse_args()
    if a.lane_source == "model" and not a.lane_model:
        raise ValueError("--lane-model is required when --lane-source model")
    if a.lane_source == "cache" and not a.lane_cache:
        raise ValueError("--lane-cache is required when --lane-source cache")

    input_path = Path(a.input).expanduser().resolve()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    hold_frames = max(1, int(round(a.hold_seconds * fps)))

    det_model = YOLO(a.det_model)
    lane_model = YOLO(a.lane_model) if a.lane_source == "model" else None
    lane_cache = LaneCacheReader(a.lane_cache) if a.lane_source == "cache" else None
    tracker = VehicleTracker(width, height, a.car_class)
    gate = a.gate * (width / GATE_REFERENCE_WIDTH)
    lane_tracker = LaneTracker(a.min_hits, a.max_misses, gate, a.ema)

    ocr_engine = stabilizer = None
    if not a.no_ocr:
        _prepare_paddleocr_environment()
        from paddleocr import PaddleOCR
        paddle_engine = PaddleOCR(
            lang="en",
            text_detection_model_name=PADDLE_DET_MODEL,
            text_recognition_model_name=PADDLE_REC_MODEL,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device=normalize_paddle_device(a.paddle_device or a.device),
        )
        ocr_engine = KhoiPaddleOCR(paddle_engine)
        stabilizer = PlateTextStabilizer(min_votes=3, history=30)
    plate_text_by_car: dict[int, str] = {}

    orb = cv2.ORB_create(800)  # [2026-10-16] 1500→800으로 확정 승격. 650/500은 미탐(1037 GANet #22816)
    # 재현돼 기각, 700은 통과했지만 안전마진이 거의 없어 보류 — 800이 최종 채택값.
    # 3영상×2백엔드 전체검증 통과(발화결과 100% 동일, 평균 속도 +25~26%).
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = None
    ego_scale = ORB_WORK_WIDTH / width
    ego_h = max(2, int(round(height * ego_scale)))

    det_results = det_model.track(
        source=str(input_path), conf=0.25, iou=0.5, imgsz=1024,
        device=a.device, tracker="bytetrack.yaml", stream=True, persist=True, verbose=False,
    )

    writer = None
    out_w = out_h = None
    if a.out_video:
        out_w = max(2, int(round(width * a.output_scale)) // 2 * 2)
        out_h = max(2, int(round(height * a.output_scale)) // 2 * 2)
        writer = FfmpegWriter(a.out_video, out_w, out_h, fps)

    # 에피소드 방식 상태 — 차량당 최대 1개의 "지금 막 교차 중인 선" 에피소드.
    # {car_id: {"tid": int, "before_sign": float}}
    active_episode_by_car: dict[int, dict] = {}
    prev_box_by_car: dict[int, tuple] = {}
    last_seen_frame_by_car: dict[int, int] = {}  # "진짜 탐지 공백"(border-clip과 구분) 판정용
    flagged_cars: dict[int, dict] = {}
    last_ratio_by_car: dict[int, float] = {}  # PHASE 1: 에피소드 진행 중 ratio 점프 관찰용

    rows = []
    events = []
    from tqdm import tqdm
    for frame_index, result in enumerate(tqdm(det_results, desc="trigger", unit="frame")):
        if a.max_frames and frame_index >= a.max_frames:
            break
        original = result.orig_img
        frame_draw = original.copy() if writer is not None else None
        per_car, _ = tracker.process(result)

        small = cv2.resize(original, (ORB_WORK_WIDTH, ego_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        Hm = None
        if prev_gray is not None:
            Hm_small = estimate_homography(prev_gray, gray, orb, bf)
            if Hm_small is not None:
                s = ego_scale
                scale_mat = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
                Hm = np.linalg.inv(scale_mat) @ Hm_small @ scale_mat
        prev_gray = gray

        if a.lane_source == "cache":
            dets = lane_cache.get_dets(frame_index)
        else:
            pose_result = lane_model.predict(source=original, imgsz=1024, conf=0.25, device=a.device, verbose=False)[0]
            dets = []
            if pose_result.keypoints is not None and len(pose_result.keypoints):
                kxy = pose_result.keypoints.data.cpu().numpy()
                cls_arr = pose_result.boxes.cls.cpu().numpy().astype(int)
                conf_arr = pose_result.boxes.conf.cpu().numpy()
                for i in range(len(kxy)):
                    kp = kxy[i]
                    v = kp[:, 2] > a.kpt_vis_thresh if kp.shape[1] > 2 else np.ones(len(kp), bool)
                    if v.sum() < 2:
                        continue
                    p = kp[:, :2].astype(np.float32)
                    dets.append((p, v, int(cls_arr[i]), float(conf_arr[i])))

        live, coasting = lane_tracker.step(dets, Hm)
        solid_tracks = [t for t in live if t.cls == 0]
        tid_to_track = {int(t.tid): t for t in (live + coasting)}

        if frame_draw is not None:
            # 지금 추적 중인 차선을 전부(실선=초록, 점선=빨강) 매 프레임 그린다 —
            # 예전엔 "교차 중"인 선만 잠깐 보여서, 모델이 실제로 뭘 보고 있는지
            # 평소엔 화면에서 확인할 수 없었다.
            for t in live:
                pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
                if len(pts) < 2:
                    continue
                pts_int = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
                color = (0, 220, 0) if t.cls == 0 else (0, 255, 255)  # 실선=초록, 점선=노랑
                cv2.polylines(frame_draw, [pts_int], False, color, 2, cv2.LINE_AA)

        for car_id, info in per_car.items():
            box = info["box"]
            x1, y1, x2, y2 = map(float, box)
            bottom_left = np.array([x1, y2], dtype=np.float64)
            bottom_right = np.array([x2, y2], dtype=np.float64)
            border_clipped = bool(
                BORDER_MARGIN_PX > 0 and (
                    x1 <= BORDER_MARGIN_PX or y1 <= BORDER_MARGIN_PX
                    or x2 >= width - BORDER_MARGIN_PX or y2 >= height - BORDER_MARGIN_PX
                )
            )
            # border_clipped는 "탐지는 됐지만 박스가 화면 경계에 닿아 판정만 건너뜀"이고,
            # 여기 true_detection_gap은 "그 프레임엔 아예 탐지 자체가 안 됨"이다 — 완전히
            # 다른 상황이다(#4 사례: 드론 바로 밑을 지나가는 차라 30프레임 내내
            # border_clipped였지만 매 프레임 계속 정상 탐지되고 있었고, 박스 크기도
            # 극단적 각도 때문에 부드럽게 연속적으로 변했을 뿐 — 공백이 아니었다).
            prev_seen_frame = last_seen_frame_by_car.get(car_id)
            true_detection_gap = prev_seen_frame is not None and (frame_index - prev_seen_frame) > 1
            last_seen_frame_by_car[car_id] = frame_index

            episode_tid = None
            episode_sign = None
            episode_pts = None
            episode_ratio = None
            just_flagged = False
            # 안전장치 2(ratio 점프) 진단용 — 기본값
            ratio_jump = float("nan")
            ratio_jump_cancel = False

            if not border_clipped and car_id not in flagged_cars:
                ep = active_episode_by_car.get(car_id)

                if ep is not None:
                    # 이미 "교차 중"인 선이 있음 — live+coasting에서 찾는다. coasting은
                    # LaneTracker 자체의 짧은 유실 허용(최대 --max-misses프레임)이라 여기서
                    # 또 막으면 멀쩡한 정탐까지 1프레임 미탐으로 깨진다(실측 확인:
                    # #22816 정탐이 live-only 제한 때문에 사라짐). 실제 버그(#29062)는
                    # "얼마나 오래 안 닿았는가"의 문제였지 live/coasting 구분이 아니었다.
                    track = tid_to_track.get(ep["tid"])
                    if track is None:
                        active_episode_by_car.pop(car_id, None)  # 선이 완전히 사라짐 — 에피소드 취소
                    else:
                        pts = track.kp[track.vis] if track.vis is not None and track.vis.any() else track.kp
                        if len(pts) >= 2:
                            pts64 = pts.astype(np.float64)
                            still_touching = bottom_edge_touches_polyline(bottom_left, bottom_right, pts64)
                            _, sign = segment_to_polyline_relation(bottom_left, bottom_right, pts64)
                            ratio = crossing_ratio(bottom_left, bottom_right, pts64, ep["before_sign"])
                            if true_detection_gap:
                                prev_ratio = last_ratio_by_car.get(car_id)
                                if prev_ratio is not None:
                                    ratio_jump = abs(ratio - prev_ratio)
                            last_ratio_by_car[car_id] = ratio
                            episode_tid, episode_sign, episode_pts, episode_ratio = ep["tid"], sign, pts, ratio
                            # [안전장치 2 — 교차 비율(ratio) 점프, 2026-10-16 박스-크기 가드 대체]
                            # "진짜 탐지 공백" 직후 ratio가 직전 관측값 대비 갑자기 거의 끝(1.0
                            # 가까이)까지 뛰었으면, 그 사이 다른 차량으로 재할당된 것으로 본다.
                            # 실측(Phase 1 관찰, GANet 3영상): 정탐(`#4`,`#22816`)은 공백을
                            # 끼고도 ratio_jump가 0.02~0.24에 그치고 그 뒤로도 계속 서서히
                            # 올라가는 반면, 오탐(`#14925`,`#54009`,`#55533`)은 공백 직후 바로
                            # ratio=1.0(완전히 끝)으로 "착지"해버린다 — 서서히 vs 순간이동의
                            # 차이가 뚜렷해서, 박스 크기가 비슷한 차량으로 바뀌는 경우(크기
                            # 가드가 못 잡는 사각지대)도 이 신호로는 잡을 수 있다.
                            ratio_jump_cancel = bool(
                                true_detection_gap and ratio >= 0.95 and ratio_jump >= 0.4
                            )
                            if ratio_jump_cancel:
                                active_episode_by_car.pop(car_id, None)  # 탐지 공백 사이 다른 차량으로 재할당된 것으로 판단 — 취소
                                last_ratio_by_car.pop(car_id, None)
                            else:
                                if still_touching:
                                    ep["last_touch_frame"] = frame_index  # 여전히 실제로 닿아있음 — 갱신
                                if sign is None:
                                    pass  # 아직 걸쳐있는 중(또는 통과 확정 대기 중 잠깐 흔들림) — 계속 기다림
                                elif sign == ep["before_sign"]:
                                    active_episode_by_car.pop(car_id, None)  # 원래 쪽으로 복귀 — 취소
                                    last_ratio_by_car.pop(car_id, None)
                                elif ep.get("hold_count", 0) == 0 and frame_index - ep["last_touch_frame"] > hold_frames:
                                    # "반대편으로 처음 확정되는 바로 그 순간"에만 최근성을 본다 — 그
                                    # 순간 마지막 실제 접촉이 hold 기간보다 오래전이면, 선에서 멀어진
                                    # 채 무한 직선 부호만으로 확정된 것으로 보고 무효 처리한다
                                    # (#29062 사례: 마지막 접촉 후 202프레임=6.7초 뒤에야 확정됨 —
                                    # 무효). 한번 hold가 시작된 뒤(hold_count>0)엔 다시 안 본다 —
                                    # border-clip처럼 평가가 잠깐 쉬어도 hold_count 자체가 "그 뒤로도
                                    # 꾸준히 반대편 유지"를 이미 보장하므로 이중으로 깎을 필요가 없다
                                    # (#22816 사례: 26프레임 경계걸림 뒤 hold 축적 도중 또 깎여서
                                    # 억울하게 취소됐던 문제).
                                    active_episode_by_car.pop(car_id, None)
                                    last_ratio_by_car.pop(car_id, None)
                                else:
                                    # 완전히 반대편으로 통과 — 즉시 발화 대신, --hold-seconds 동안
                                    # 그 상태가 유지돼야 확정한다(짧은 노이즈 깜빡임 거르기 위함,
                                    # #9328 사례: 3프레임=0.1초 만에 발화했던 문제).
                                    ep["hold_count"] = ep.get("hold_count", 0) + 1
                                    if ep["hold_count"] >= hold_frames:
                                        flagged_cars[car_id] = {"frame": frame_index, "lane_tid": ep["tid"]}
                                        events.append({"frame": frame_index, "car_id": car_id, "lane_tid": ep["tid"]})
                                        just_flagged = True
                                        active_episode_by_car.pop(car_id, None)
                                        last_ratio_by_car.pop(car_id, None)
                else:
                    # 에피소드 없음 — 지금 막 어떤 실선과 걸치기 시작했는지 확인
                    prev_box = prev_box_by_car.get(car_id)
                    for t in solid_tracks:
                        pts = t.kp[t.vis] if t.vis is not None and t.vis.any() else t.kp
                        if len(pts) < 2:
                            continue
                        pts64 = pts.astype(np.float64)
                        if not bottom_edge_touches_polyline(bottom_left, bottom_right, pts64):
                            continue  # 실제로(유한 구간 안에서) 안 닿아있으면 거리 상관없이 무시
                        _, sign = segment_to_polyline_relation(bottom_left, bottom_right, pts64)
                        if sign is not None:
                            continue  # (이론상 안 나와야 하지만) 방어적으로 한 번 더 확인
                        if prev_box is None:
                            continue  # 이 차의 첫 프레임이라 '원래 쪽'을 모름 — 다음 기회에
                        px1, py1, px2, py2 = prev_box
                        prev_bl = np.array([px1, py2], dtype=np.float64)
                        prev_br = np.array([px2, py2], dtype=np.float64)
                        _, before_sign = segment_to_polyline_relation(prev_bl, prev_br, pts.astype(np.float64))
                        if before_sign is None:
                            continue  # 직전 프레임에도 이미 걸쳐있었음(드묾) — 애매하니 건너뜀
                        start_ratio = crossing_ratio(bottom_left, bottom_right, pts64, before_sign)
                        # [안전장치 2 — 시작 순간, ratio 점프판] "직전 박스"(before_sign 계산용)와
                        # "지금 박스"(교차 시작 판정용) 사이에 진짜 탐지 공백이 있었는데, 시작하자마자
                        # ratio가 이미 절반 이상이면(= 걸치기 시작한 게 아니라 이미 거의 다 넘어간
                        # 상태로 등장함) 그 사이 다른 차량으로 재할당된 것으로 보고 에피소드를 아예
                        # 시작하지 않는다(#55533류 — 공백 직후 첫 "닿음" 판정 시점에 이미 다른 차로
                        # 바뀐 채였던 경우를 시작 순간에 바로 거른다. 진행 중 점프 체크와 대칭).
                        # 실측(Phase 1): 공백 끼고 시작하는 243개 에피소드 중 "정상적인 서서히
                        # 접근" 패턴은 전부 ratio<=0.36이었고, 0.5 이상은 전부 이례적인 급발진
                        # 패턴(0.78~0.96)이었다 — 둘 사이에 뚜렷한 간격이 있어 0.5를 기준으로 삼음.
                        if true_detection_gap and start_ratio >= 0.5:
                            continue
                        active_episode_by_car[car_id] = {
                            "tid": int(t.tid), "before_sign": before_sign, "last_touch_frame": frame_index,
                        }
                        episode_tid, episode_sign, episode_pts = int(t.tid), None, pts
                        episode_ratio = start_ratio
                        last_ratio_by_car[car_id] = episode_ratio
                        break  # 한 프레임에 하나의 에피소드만 시작

            prev_box_by_car[car_id] = (x1, y1, x2, y2)

            _ep_now = active_episode_by_car.get(car_id)
            rows.append({
                "frame": frame_index, "car_id": car_id,
                "episode_tid": episode_tid if episode_tid is not None else -1,
                "episode_sign": episode_sign if episode_sign is not None else float("nan"),
                "episode_active": car_id in active_episode_by_car,
                "hold_count": _ep_now.get("hold_count", 0) if _ep_now is not None else 0,
                "border_clipped": border_clipped,
                "flagged": car_id in flagged_cars,
                "ratio": episode_ratio if episode_ratio is not None else float("nan"),
                # 안전장치 2(ratio 점프 가드) 진단용 컬럼
                "true_detection_gap": true_detection_gap,
                "ratio_jump": ratio_jump,
                "ratio_jump_cancel": ratio_jump_cancel,
            })

            if frame_draw is not None:
                x1i, y1i, x2i, y2i = map(int, box)
                flagged_now = car_id in flagged_cars
                in_episode = car_id in active_episode_by_car
                box_color = (0, 0, 255) if flagged_now else ((0, 165, 255) if in_episode else (0, 255, 255))
                # 교차중(아직 확정 전)이면 밑변에서부터 ratio 비율만큼 분홍색을 "채워서"
                # 보여준다(색 농도가 아니라 채워지는 높이로 진행도를 표현— 사용자 요청).
                if in_episode and not flagged_now and episode_ratio is not None:
                    fill_h = int(round(episode_ratio * (y2i - y1i)))
                    if fill_h > 0:
                        fill_top = y2i - fill_h
                        overlay = frame_draw.copy()
                        cv2.rectangle(overlay, (x1i, fill_top), (x2i, y2i), (180, 105, 255), -1)
                        cv2.addWeighted(overlay, 0.35, frame_draw, 0.65, 0, dst=frame_draw)
                cv2.rectangle(frame_draw, (x1i, y1i), (x2i, y2i), box_color, 3 if flagged_now else 2)
                cv2.line(frame_draw, (x1i, y2i), (x2i, y2i), (255, 220, 0), 3, cv2.LINE_AA)
                label = f"#{car_id}" + (" (교차중)" if in_episode else "")
                if flagged_now:
                    label = f"VIOLATION {label}"
                cv2.putText(frame_draw, label, (x1i, max(0, y1i - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            box_color, 2, cv2.LINE_AA)

        # 번호판 OCR — "이미 VIOLATION으로 확정된 차량"에 대해서만 수행한다
        # (v5.3.1과 동일한 설계: 매 프레임 모든 차에 OCR 돌리면 너무 느림).
        if ocr_engine is not None and result.boxes is not None and len(result.boxes) > 0:
            p_boxes = result.boxes.xyxy.cpu().numpy()
            p_classes = result.boxes.cls.cpu().numpy().astype(int)
            p_track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None else np.full(len(p_classes), -1)
            )
            car_indices = [
                idx for idx, cid in enumerate(p_classes)
                if cid == a.car_class and p_track_ids[idx] >= 0
            ]
            plate_drawn: set[int] = set()
            for plate_index, cid in enumerate(p_classes):
                if cid != a.plate_class or not car_indices:
                    continue
                score, car_index = max(
                    ((overlap_score(p_boxes[idx], p_boxes[plate_index], ASSOCIATION_METRIC), idx)
                     for idx in car_indices),
                    key=lambda item: item[0],
                )
                if score < ASSOCIATION_THRESHOLD:
                    continue
                pcar_id = int(p_track_ids[car_index])
                if pcar_id not in flagged_cars:
                    continue
                px1, py1, px2, py2 = clip_box(p_boxes[plate_index], width, height)
                if px2 <= px1 or py2 <= py1:
                    continue
                text, confidence = run_ocr(ocr_engine, original[py1:py2, px1:px2])
                stable = stabilizer.update(pcar_id, text, confidence)
                if stable:
                    plate_text_by_car[pcar_id] = stable
                shown = plate_text_by_car.get(pcar_id, stable)
                if frame_draw is not None:
                    cv2.rectangle(frame_draw, (px1, py1), (px2, py2), COLOR_PLATE, 2)
                    draw_plate_text(frame_draw, p_boxes[plate_index], shown,
                                     scale=a.plate_text_scale, thickness=a.plate_text_thickness)
                plate_drawn.add(pcar_id)
            if frame_draw is not None:
                for pcar_id in flagged_cars:
                    if pcar_id in plate_drawn or pcar_id not in per_car:
                        continue
                    text = plate_text_by_car.get(pcar_id)
                    if not text:
                        continue
                    pbox = per_car[pcar_id]["box"]
                    anchor = np.array([pbox[0], pbox[1] - 40, pbox[2], pbox[1] - 40], dtype=np.float32)
                    draw_plate_text(frame_draw, anchor, text,
                                     scale=a.plate_text_scale, thickness=a.plate_text_thickness)

        if writer is not None:
            if (out_w, out_h) != (width, height):
                frame_draw = cv2.resize(frame_draw, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame_draw)

    if writer is not None:
        writer.release()
        print(f"[VIDEO] {a.out_video}")
    if lane_cache is not None:
        lane_cache.close()

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"[CSV] {a.out_csv} rows={len(df)}")

    if events:
        details = "  ".join(
            f"#{e['car_id']}@f{e['frame']} lane={e['lane_tid']} plate={plate_text_by_car.get(e['car_id'], '')}"
            for e in events
        )
        print(f"[FLAGGED] {len(events)} cars: {details}")
    else:
        print("[FLAGGED] none")


if __name__ == "__main__":
    main()
