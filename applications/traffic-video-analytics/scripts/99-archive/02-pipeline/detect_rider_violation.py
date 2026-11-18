#!/usr/bin/env python3
"""오토바이 위반 판정 — 다인승 · 헬멧 미착용

드론 교통 영상에서 오토바이 관련 위반 두 가지를 판정한다.

    1. 다인승      오토바이 1대에 탑승자 2명 이상   -> 오토바이+탑승자 통합 빨간 박스
    2. 헬멧 미착용  탑승자의 머리 구간에 헬멧 없음   -> 그 사람에게 노란 박스

탐지 모델은 "여기 사람이 있다", "여기 오토바이가 있다" 만 알려줄 뿐 둘의 관계는 주지
않는다. 따라서 박스의 기하 관계로 "이 사람이 저 오토바이를 탔다" 를 잇는 연관(association)
단계가 판정의 전제가 된다.

    검출 -> 정리 -> 연관 -> 판정 -> 시간 누적 -> 렌더링

헬멧 판정 대상은 오토바이 탑승자로 한정한다. 보행자·자전거 이용자는 대상이 아니며,
연관에 실패한 사람은 판정 자체를 하지 않는다.

미착용은 "헬멧 박스의 부재" 로 판정하는 부정 근거다. 헬멧을 놓치면 그 즉시 미착용 오탐이
되므로, 판정 전에 보류 조건(너무 작음·화면 경계)을 걸고 시간 누적으로 한 프레임 미검출을
흡수한다.

튜닝 값의 기본은 같은 이름의 YAML 파일이 정한다. 명령줄 인자가 항상 우선한다.

사용 예:

    python detect_rider_violation.py \
        --det-model <5클래스.pt> --input <영상.mp4> \
        --output <결과.mp4> --csv <결과.csv> --device 0
"""

from __future__ import annotations
import argparse
import csv
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO


# ── 설정 파일이 없을 때 쓰는 값 ──
# 낮게 잡는다. 앞사람에 가린 뒷사람은 작고 흐릿해 기본값 0.25 로는 놓친다.
# 사람·오토바이 양쪽에 공통 적용되므로, 밀집 구간에서 옆줄 사람이 탑승자로 오연관될 수 있다.
DET_CONFIDENCE = 0.10
DET_NMS_IOU = 0.7
TRACKER = "bytetrack.yaml"

PERSON_CLASS = 0
MOTORCYCLE_CLASS = 2
HELMET_CLASS = 4

# 화면 규칙은 두 가지뿐이다. 오토바이는 파랑, 사람은 노랑, 2인 이상이면 오토바이가 빨강.
# 출력 영상에는 위반 두 가지만 나타난다. 그 외에는 아무것도 그리지 않는다.
COLOR_MOTO = (255, 128, 0)          # 오토바이 (파랑). --draw-all 일 때만
COLOR_VIOLATION = (0, 0, 255)       # 다인승 확정 오토바이+탑승자 통합 박스 (빨강)
COLOR_NO_HELMET = (0, 255, 255)     # 헬멧 미착용 확정 탑승자 (노랑)
COLOR_PERSON = (150, 150, 150)      # 그 외 사람 (회색). --draw-all 일 때만
COLOR_HELMET = (0, 140, 255)        # 검출된 헬멧 박스 (주황)

# 박스 선 두께(px)
THICKNESS_VIOLATION = 3    # 빨강 — 다인승 통합 박스
THICKNESS_NO_HELMET = 2    # 노랑 — 헬멧 미착용 탑승자
THICKNESS_HELMET = 2       # 주황 — 검출된 헬멧
THICKNESS_MOTO = 2         # 파랑 — 정상 오토바이
THICKNESS_PERSON = 2       # 회색 — 그 외 사람
THICKNESS_GHOST = 1        # 회색 — 명단으로만 유지된 탑승자

# 박스 위 글자를 그릴지. 기본은 그리지 않는다. --label 로 켠다.
SHOW_LABELS = False

CSV_COLUMNS = [
    "frame_index",
    "motorcycle_id",
    "motorcycle_conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "rider_count",
    "rider_count_geom",
    "ghost_ids",
    "rider_ids",
    "rider_overlaps",
    "candidate",
    "window_samples",
    "window_ratio",
    "confirmed",
    "confirmed_frame",
    "helmet_scores",
    "bare_riders",
    "undecidable",
    "no_helmet_candidate",
    "no_helmet_confirmed",
]


# ── 기하 계산 ──
def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = (max(a[0], b[0]), max(a[1], b[1]))
    x2, y2 = (min(a[2], b[2]), min(a[3], b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


# 사람 박스가 오토바이 박스에 얼마나 들어가 있는지.
def overlap_score(person_box: np.ndarray, moto_box: np.ndarray, metric: str) -> float:
    inter = intersection_area(person_box, moto_box)
    person_area = box_area(person_box)
    moto_area = box_area(moto_box)
    if metric == "person_ioa":
        denominator = person_area
    elif metric == "min_ioa":
        denominator = min(person_area, moto_area)
    else:  # iou
        denominator = person_area + moto_area - inter
    return inter / denominator if denominator > 0 else 0.0


# 한 물체에 박스가 두 개 이상 잡힌 경우를 걸러낸다.
def deduplicate_boxes(boxes: list[np.ndarray], confs: list[float], threshold: float) -> list[int]:
    """작은 박스가 큰 박스에 거의 들어가 있으면 같은 물체로 보고 conf 높은 쪽만 남긴다.

    NMS 는 IoU 로 판단하므로 큰 박스 안에 작은 박스가 들어앉은 형태(상반신만 잡힌 사람)는
    IoU 가 낮아 살아남는다. 그대로 두면 1인 탑승이 2인으로 세어지므로, min 면적 기준
    포함비로 다시 한 번 거른다.
    """
    order = sorted(range(len(boxes)), key=lambda index: -confs[index])
    kept: list[int] = []
    for index in order:
        duplicate = False
        for kept_index in kept:
            inter = intersection_area(boxes[index], boxes[kept_index])
            smaller = min(box_area(boxes[index]), box_area(boxes[kept_index]))
            if smaller > 0 and inter / smaller >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    return sorted(kept)


def box_center(box: np.ndarray) -> tuple[int, int]:
    return (int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2)))


# 사람 박스 중심이 오토바이 박스(여유분 포함) 안에 들어오는지.
def center_inside(person_box: np.ndarray, moto_box: np.ndarray, margin_ratio: float) -> bool:
    """오토바이 옆에 서 있는 보행자를 걸러내는 조건이다.

    포함비만 쓰면 오토바이 옆에 서서 몸이 걸친 보행자까지 탑승자로 잡힌다.
    탑승자는 오토바이 위에 앉아 있으므로 몸통 중심이 오토바이 박스 안에 떨어진다.
    """
    margin_x = (moto_box[2] - moto_box[0]) * margin_ratio
    margin_y = (moto_box[3] - moto_box[1]) * margin_ratio
    cx = (person_box[0] + person_box[2]) / 2
    cy = (person_box[1] + person_box[3]) / 2
    return bool(
        moto_box[0] - margin_x <= cx <= moto_box[2] + margin_x
        and moto_box[1] - margin_y <= cy <= moto_box[3] + margin_y
    )


# ── 헬멧 판정 ──
def head_zone(person_box: np.ndarray, h_ratio: float) -> np.ndarray:
    """사람 박스의 상단 h_ratio 구간. 드론 하향 뷰에서 정수리가 보이는 영역이다."""
    height = person_box[3] - person_box[1]
    return np.array(
        [person_box[0], person_box[1], person_box[2], person_box[1] + height * h_ratio],
        dtype=np.float32,
    )


def helmet_score(helmet_box: np.ndarray, person_box: np.ndarray, h_ratio: float) -> float:
    """헬멧의 몇 %가 이 사람의 머리 구간에 들어가 있는지.

    분모는 헬멧 면적이다. 사람 면적을 분모로 쓰면 헬멧이 작아 값이 항상 낮게 나온다.
    """
    zone = head_zone(person_box, h_ratio)
    area = box_area(helmet_box)
    return intersection_area(helmet_box, zone) / area if area > 0 else 0.0


def assign_helmets(
    riders: list["RiderLink"], helmet_boxes: list[np.ndarray], h_ratio: float, tau: float
) -> dict[int, float]:
    """탑승자마다 헬멧 착용 점수를 매긴다. 헬멧 1개는 최고 점수 1명에게만 배정한다.

    앞뒤로 탄 두 사람의 머리가 가까워 헬멧 하나가 둘 다에 걸치는 일이 생긴다. 배정하지
    않으면 미착용자까지 착용으로 처리된다.
    """
    scores: dict[int, float] = {}
    for helmet_index, helmet_box in enumerate(helmet_boxes):
        best_rider, best_score = -1, tau
        for rider_index, rider in enumerate(riders):
            score = helmet_score(helmet_box, rider.box, h_ratio)
            if score >= best_score and score > scores.get(rider_index, 0.0):
                best_score, best_rider = score, rider_index
        if best_rider >= 0:
            scores[best_rider] = max(scores.get(best_rider, 0.0), best_score)
    return scores


def helmet_undecidable(
    person_box: np.ndarray, frame_width: int, frame_height: int,
    min_height: float, border_margin: float,
) -> str:
    """헬멧 판정을 보류해야 하는지. 보류 사유 문자열, 아니면 빈 문자열.

    미착용은 부재로 판정하므로, 애초에 헬멧을 볼 수 없는 조건이면 '미착용' 이 아니라
    '판정 불가' 로 빼야 한다.
    """
    if person_box[3] - person_box[1] < min_height:
        return "small"
    if (
        person_box[0] <= border_margin
        or person_box[1] <= border_margin
        or person_box[2] >= frame_width - border_margin
        or person_box[3] >= frame_height - border_margin
    ):
        return "border"
    return ""


# ── 자료 구조 ──
@dataclass
class RiderLink:
    """오토바이 1대에 연관된 사람 1명."""

    person_index: int
    person_id: int          # 트랙 ID. 없으면 -1
    box: np.ndarray
    conf: float
    overlap: float
    ghost: bool = False       # 이번 프레임에 검출되지 않고 명단으로 유지된 탑승자


@dataclass
class MotorcycleState:
    """오토바이 트랙 1개의 시간 누적 상태."""

    history: deque = field(default_factory=deque)
    confirmed: bool = False
    confirmed_frame: int = -1
    max_riders: int = 0
    last_seen: int = -1


# ── 탑승자 연관 ──
def associate_riders(
    person_boxes: list[np.ndarray],
    person_ids: list[int],
    person_confs: list[float],
    moto_boxes: list[np.ndarray],
    tau: float,
    metric: str,
    min_area_ratio: float,
    max_area_ratio: float,
    dedup_ioa: float,
    require_center: bool,
    center_margin: float,
) -> tuple[dict[int, list[RiderLink]], set[int]]:
    """사람을 오토바이에 잇는다. 사람 1명은 가장 많이 겹치는 오토바이 1대에만 배정한다.

    한 사람이 여러 오토바이 박스에 걸쳐 있을 때 양쪽 모두에 세면 두 대가 동시에
    다인승으로 오탐된다. 그래서 최댓값 1대에만 배정한다.
    """
    links: dict[int, list[RiderLink]] = {index: [] for index in range(len(moto_boxes))}
    assigned: set[int] = set()
    for person_index, person_box in enumerate(person_boxes):
        person_area = box_area(person_box)
        best_moto = -1
        best_score = 0.0
        for moto_index, moto_box in enumerate(moto_boxes):
            moto_area = box_area(moto_box)
            if moto_area <= 0:
                continue
            # 오토바이에 비해 너무 작거나 너무 큰 사람 박스는 탑승자가 아니다.
            area_ratio = person_area / moto_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            if require_center and not center_inside(person_box, moto_box, center_margin):
                continue
            score = overlap_score(person_box, moto_box, metric)
            if score > best_score:
                best_score = score
                best_moto = moto_index
        if best_moto >= 0 and best_score >= tau:
            links[best_moto].append(
                RiderLink(
                    person_index=person_index,
                    person_id=person_ids[person_index],
                    box=person_box,
                    conf=person_confs[person_index],
                    overlap=best_score,
                )
            )
            assigned.add(person_index)

    # 같은 탑승자가 두 번 잡힌 경우를 제거한다. 여기서 걸러야 1인 탑승이 2인으로 세어지지 않는다.
    if dedup_ioa < 1.0:
        for moto_index, riders in links.items():
            if len(riders) < 2:
                continue
            keep = deduplicate_boxes([r.box for r in riders], [r.conf for r in riders], dedup_ioa)
            dropped = [riders[i].person_index for i in range(len(riders)) if i not in set(keep)]
            links[moto_index] = [riders[i] for i in keep]
            assigned.difference_update(dropped)
    return links, assigned


# ── 연관 유지 (탑승자 명단) ──
@dataclass
class RiderSlot:
    """오토바이 1대의 탑승자 명단 한 칸."""

    person_id: int
    box: np.ndarray
    conf: float
    overlap: float
    missing: int = 0          # 연속으로 검출되지 않은 프레임 수
    last_seen: int = -1


class RiderRoster:
    """오토바이 트랙별로 탑승자 명단을 들고 다니며 짧은 검출 끊김을 메운다.

    기하 연관만 쓰면 한 프레임 미검출이 곧 탑승자 수 감소로 이어진다. 짧은 끊김을
    명단으로 메워 판정이 흔들리는 것을 줄인다.

    사라진 탑승자를 그대로 유지하면, 같은 사람이 새 트랙 ID 로 다시 잡힐 때 유령과 실물이
    각각 세어진다. 그래서 매 프레임 명단과 현재 탑승자를 ID 우선·위치 차선으로 매칭한 뒤
    남은 것만 유령으로 남긴다.
    """

    def __init__(self, persist: int, match_iou: float) -> None:
        self.persist = persist
        self.match_iou = match_iou
        self.rosters: dict[int, list[RiderSlot]] = {}
        self.last_seen: dict[int, int] = {}

    def update(
        self, track_id: int, riders: list[RiderLink], frame_index: int,
        moto_box: np.ndarray | None = None, center_margin: float = 0.0,
    ) -> list[RiderLink]:
        """이번 프레임의 탑승자 목록에 유령을 더해 돌려준다.

        유령은 현재 오토바이 박스로 다시 검증한다. 트래커가 같은 ID 를 다른 오토바이에
        재할당하면 이전 오토바이의 탑승자가 유령으로 딸려와 통합 박스가 비정상적으로 커진다.
        """
        if self.persist <= 0 or track_id < 0:
            return riders
        slots = self.rosters.setdefault(track_id, [])
        self.last_seen[track_id] = frame_index

        # ① 현재 탑승자를 기존 명단에 매칭한다. ID 가 같으면 우선, 아니면 위치로.
        unmatched = list(range(len(slots)))
        for rider in riders:
            hit = None
            for si in unmatched:
                if slots[si].person_id == rider.person_id:
                    hit = si
                    break
            if hit is None:
                best_score = self.match_iou
                for si in unmatched:
                    score = overlap_score(rider.box, slots[si].box, "iou")
                    if score >= best_score:
                        best_score, hit = score, si
            if hit is not None:
                unmatched.remove(hit)
                slots[hit] = RiderSlot(
                    person_id=rider.person_id, box=rider.box, conf=rider.conf,
                    overlap=rider.overlap, missing=0, last_seen=frame_index,
                )
            else:
                slots.append(RiderSlot(
                    person_id=rider.person_id, box=rider.box, conf=rider.conf,
                    overlap=rider.overlap, missing=0, last_seen=frame_index,
                ))

        # ② 매칭되지 않은 칸은 유예 카운트를 올리고, 한도를 넘으면 버린다.
        for si in sorted(unmatched, reverse=True):
            slots[si].missing += 1
            if slots[si].missing > self.persist:
                slots.pop(si)

        # ③ 위치가 어긋난 유령은 버린다. 오토바이가 바뀌었거나 크게 움직인 경우다.
        if moto_box is not None:
            alive = []
            for slot in slots:
                if slot.missing > 0 and not center_inside(slot.box, moto_box, center_margin):
                    continue
                alive.append(slot)
            slots[:] = alive

        # ④ 실물 + 유예 중인 유령을 합쳐 돌려준다.
        out = list(riders)
        for slot in slots:
            if slot.missing > 0:
                out.append(RiderLink(
                    person_index=-1, person_id=slot.person_id, box=slot.box,
                    conf=slot.conf, overlap=slot.overlap, ghost=True,
                ))
        return out

    def prune(self, frame_index: int, max_age: int = 300) -> None:
        stale = [t for t, f in self.last_seen.items() if frame_index - f > max_age]
        for t in stale:
            self.rosters.pop(t, None)
            self.last_seen.pop(t, None)


# ── 시간 누적 판정 ──
class MultiRiderJudge:
    """트랙 단위로 다인승 후보를 누적해 확정 여부를 결정한다."""

    def __init__(self, window: int, ratio: float, min_samples: int, min_riders: int) -> None:
        self.window = window
        self.ratio = ratio
        self.min_samples = min_samples
        self.min_riders = min_riders
        self.states: dict[int, MotorcycleState] = {}

    def update(self, track_id: int, rider_count: int, frame_index: int) -> MotorcycleState | None:
        """트랙 ID 가 있는 오토바이만 누적한다. ID 가 없으면 None 을 돌려준다."""
        if track_id < 0:
            return None
        state = self.states.get(track_id)
        if state is None:
            state = MotorcycleState(history=deque(maxlen=self.window))
            self.states[track_id] = state
        state.history.append(rider_count >= self.min_riders)
        state.max_riders = max(state.max_riders, rider_count)
        state.last_seen = frame_index
        if not state.confirmed and len(state.history) >= self.min_samples:
            if sum(state.history) / len(state.history) >= self.ratio:
                state.confirmed = True
                state.confirmed_frame = frame_index
        return state

    def window_stats(self, track_id: int) -> tuple[int, float]:
        state = self.states.get(track_id)
        if state is None or not state.history:
            return (0, 0.0)
        return (len(state.history), sum(state.history) / len(state.history))


class NoHelmetJudge:
    """오토바이 트랙 단위로 '미착용 탑승자가 있는 프레임' 을 누적한다.

    사람 트랙으로 누적하지 않는 이유는 탑승자 ID 가 프레임마다 갈려 윈도우가 계속
    초기화되기 때문이다. 확정은 오토바이 단위, 표시는 사람 단위다.
    """

    def __init__(self, window: int, ratio: float, min_samples: int) -> None:
        self.window = window
        self.ratio = ratio
        self.min_samples = min_samples
        self.states: dict[int, MotorcycleState] = {}

    def update(self, track_id: int, has_bare_rider: bool, frame_index: int):
        if track_id < 0:
            return None
        state = self.states.get(track_id)
        if state is None:
            state = MotorcycleState(history=deque(maxlen=self.window))
            self.states[track_id] = state
        state.history.append(has_bare_rider)
        state.last_seen = frame_index
        if not state.confirmed and len(state.history) >= self.min_samples:
            if sum(state.history) / len(state.history) >= self.ratio:
                state.confirmed = True
                state.confirmed_frame = frame_index
        return state

    def prune(self, frame_index: int, max_age: int = 300) -> None:
        stale = [t for t, s in self.states.items() if frame_index - s.last_seen > max_age]
        for t in stale:
            self.states.pop(t, None)


# ── 렌더링 ──
def union_box(boxes: list[np.ndarray]) -> np.ndarray:
    """여러 박스를 모두 감싸는 하나의 박스."""
    stacked = np.stack(boxes)
    return np.array(
        [stacked[:, 0].min(), stacked[:, 1].min(), stacked[:, 2].max(), stacked[:, 3].max()],
        dtype=np.float32,
    )


def draw_violation(
    frame: np.ndarray, moto_box: np.ndarray, rider_boxes: list[np.ndarray],
    track_id: int, rider_count: int, confirmed: bool, ghost_count: int = 0,
) -> None:
    """위반은 오토바이와 탑승자를 하나로 묶은 박스로 그린다.

    개별 박스를 따로 그리면 밀집 구간에서 어느 사람이 어느 오토바이 것인지 읽히지 않는다.
    """
    box = union_box([moto_box] + rider_boxes)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VIOLATION, THICKNESS_VIOLATION)
    if not SHOW_LABELS:
        return
    name = f"Moto #{track_id}" if track_id >= 0 else "Moto #?"
    label = f"{name} riders={rider_count}"
    if ghost_count:
        # 이번 프레임에 검출이 없어 명단으로 채운 탑승자가 있다는 표식.
        label += f"({rider_count - ghost_count}+{ghost_count}?)"
    if confirmed:
        label += " MULTI-RIDER"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    ty = max(th + 6, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), COLOR_VIOLATION, -1)
    cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)


def draw_helmet(frame: np.ndarray, box: np.ndarray, conf: float) -> None:
    """검출된 헬멧 박스를 그대로 그린다. 헬멧이 잡히는지 확인하는 디버깅용이다."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HELMET, THICKNESS_HELMET)


def draw_no_helmet(frame: np.ndarray, box: np.ndarray, person_id: int, confirmed: bool) -> None:
    """헬멧 미착용 탑승자를 노란 박스로 그린다."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_NO_HELMET, THICKNESS_NO_HELMET)
    if not SHOW_LABELS:
        return
    name = f"P#{person_id}" if person_id >= 0 else "P#?"
    label = f"{name} NO-HELMET" if confirmed else name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(frame, (x1, ty - th - 5), (x1 + tw + 6, ty + 3), COLOR_NO_HELMET, -1)
    cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_motorcycle(
    frame: np.ndarray, box: np.ndarray, track_id: int, rider_count: int, violation: bool, confirmed: bool
) -> None:
    """색은 프레임 단위 탑승자 수로, 문구는 트랙 단위 확정 여부로 정한다.

    빨강은 "지금 2명 이상 보인다"는 뜻이고, MULTI-RIDER 문구는 9.4 의 시간 누적까지
    통과했다는 뜻이다. 색만으로는 한 프레임 깜빡임과 확정을 구분할 수 없어 문구를 남긴다.
    """
    color = COLOR_VIOLATION if violation else COLOR_MOTO
    thickness = THICKNESS_VIOLATION if violation else THICKNESS_MOTO
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    name = f"Moto #{track_id}" if track_id >= 0 else "Moto #?"
    label = f"{name} riders={rider_count}"
    if confirmed:
        label += " MULTI-RIDER"
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def draw_person(frame: np.ndarray, box: np.ndarray, person_id: int, ghost: bool = False) -> None:
    """사람은 탑승자든 보행자든 같은 노랑으로 그린다. 유령은 얇게 그리고 ? 를 붙인다."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, THICKNESS_GHOST if ghost else THICKNESS_PERSON)
    name = f"P#{person_id}" if person_id >= 0 else "P#?"
    if ghost:
        name += "?"
    cv2.putText(
        frame, name, (x1, min(frame.shape[0] - 6, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_PERSON, 1 if ghost else 2, cv2.LINE_AA,
    )


def draw_banner(frame: np.ndarray, frame_index: int, moto_count: int, violation_count: int) -> None:
    if not SHOW_LABELS:
        return
    text = f"frame={frame_index}  motorcycles={moto_count}  multi-rider={violation_count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (16 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ── 저장 후 압축 ──
def compress_video(path: Path, crf: int) -> None:
    """OpenCV 가 쓴 mpeg4 파일을 H.264 로 다시 인코딩한다.

    OpenCV 의 mp4v 는 26~32 Mbps 로 나와 5분짜리가 800 MB 를 넘는다. CRF 23 H.264 로
    바꾸면 박스 선과 글자가 유지되면서 1/3 로 줄어든다 (CRF 29 부터는 글자가 뭉개진다).
    ffmpeg 이 없거나 실패하면 원본을 그대로 두고 경고만 남긴다.
    """
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg 이 없어 압축을 건너뛴다")
        return
    temp = path.with_suffix(".h264.mp4")
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(temp),
    ]
    before = path.stat().st_size
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temp.is_file():
        print(f"[WARN] 압축 실패, 원본 유지: {result.stderr.strip()[:200]}")
        temp.unlink(missing_ok=True)
        return
    after = temp.stat().st_size
    temp.replace(path)
    print(f"[COMPRESS] {before/1e6:.0f} MB -> {after/1e6:.0f} MB ({after/before:.0%}) crf={crf}")


# ── 설정 파일 ──
# 이 스크립트와 같은 이름의 .yaml 을 기본 설정으로 읽는다.
CONFIG_PATH = Path(__file__).with_suffix(".yaml")

# YAML 키 -> argparse dest 이름. 여기 없는 인자는 명령줄로만 받는다.
CONFIG_KEYS: dict[str, dict[str, str]] = {
    "detection": {
        "imgsz": "imgsz",
        "det_conf": "det_conf",
        "nms_iou": "nms_iou",
        "tracker": "tracker",
        "person_class": "person_class",
        "motorcycle_class": "motorcycle_class",
        "helmet_class": "helmet_class",
        "person_max_frame_ratio": "person_max_frame_ratio",
        "moto_dedup_ioa": "moto_dedup_ioa",
    },
    "association": {
        "metric": "assoc_metric",
        "tau": "assoc_tau",
        "center_check": "center_check",
        "center_margin": "center_margin",
        "rider_area_min": "rider_area_min",
        "rider_area_max": "rider_area_max",
        "rider_dedup_ioa": "rider_dedup_ioa",
        "rider_persist": "rider_persist",
        "rider_persist_iou": "rider_persist_iou",
    },
    "multi_rider": {
        "min_riders": "min_riders",
        "window": "window",
        "ratio": "ratio",
        "min_samples": "min_samples",
    },
    "helmet": {
        "enabled": "helmet_enabled",
        "conf": "helmet_conf",
        "head_ratio": "head_ratio",
        "tau": "helmet_tau",
        "dedup_ioa": "helmet_dedup_ioa",
        "min_person_height": "helmet_min_person_height",
        "border_margin": "helmet_border_margin",
        "window": "helmet_window",
        "ratio": "helmet_ratio",
        "min_samples": "helmet_min_samples",
    },
    "output": {
        "scale": "output_scale",
        "compress": "compress",
        "video_crf": "video_crf",
    },
    "render": {
        "label": "label",
        "draw_all": "draw_all",
        "draw_helmet": "draw_helmet",
        "debug_helmet": "debug_helmet",
    },
}


def load_config(path: Path) -> dict:
    """YAML 설정을 argparse dest 이름의 평평한 사전으로 읽는다.

    파일이 없으면 빈 사전을 돌려주고 코드 내장 기본값을 쓴다.
    모르는 키는 오타일 가능성이 높으므로 경고만 내고 무시한다.
    """
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    flat: dict = {}
    for section, values in raw.items():
        mapping = CONFIG_KEYS.get(section)
        if mapping is None:
            print(f"[WARN] 설정 파일에 모르는 섹션: {section}")
            continue
        for key, value in (values or {}).items():
            dest = mapping.get(key)
            if dest is None:
                print(f"[WARN] 설정 파일에 모르는 항목: {section}.{key}")
                continue
            flat[dest] = value
    return flat


# ── CLI ──
def parse_cli_args() -> argparse.Namespace:
    """명령줄 인자를 읽는다.

    튜닝 값의 기본은 같은 이름의 YAML 파일이 정한다. 명령줄에 준 값이 항상 이긴다.
    argparse 의 default 를 비워두고, 파싱 뒤에 YAML → 코드 기본값 순으로 채운다.
    """
    parser = argparse.ArgumentParser(
        description="오토바이 위반 판정 — 다인승 + 헬멧 미착용",
        epilog=f"튜닝 기본값: {CONFIG_PATH.name} (명령줄 인자가 우선한다)",
    )

    # ── 입출력 · 실행 환경 (실행마다 달라지므로 YAML 에 두지 않는다) ──
    parser.add_argument("--det-model", required=True, help="검출 모델(.pt). 5클래스면 헬멧 판정 가능")
    parser.add_argument("--input", required=True, help="입력 영상 경로")
    parser.add_argument("--output", default=None, help="결과 영상 경로. --no-video 면 생략 가능")
    parser.add_argument("--csv", default=None, help="판정 로그 CSV 경로. 생략하면 기록하지 않음")
    parser.add_argument("--no-video", action="store_true", help="영상을 쓰지 않고 판정·CSV 만 수행")
    parser.add_argument("--device", default="0", help="추론 GPU 번호. cpu 도 가능")
    parser.add_argument("--config", default=None, help=f"설정 YAML 경로. 생략하면 {CONFIG_PATH.name}")
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수. 0이면 전체")
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="이 프레임부터 처리한다. 결과 영상의 시간 x 30 으로 계산하면 그 지점이 나온다",
    )

    # ── 아래는 전부 YAML 로 관리한다. 명령줄로 주면 그 값이 이긴다 ──
    detection = parser.add_argument_group("검출")
    detection.add_argument("--imgsz", type=int, help="추론 입력 해상도(32의 배수)")
    detection.add_argument("--det-conf", type=float, help="검출 confidence")
    detection.add_argument("--nms-iou", type=float, help="NMS IoU 임계값")
    detection.add_argument("--tracker", help="추적기 설정 파일")
    detection.add_argument("--person-class", type=int, help="사람 클래스 번호")
    detection.add_argument("--motorcycle-class", type=int, help="오토바이 클래스 번호")
    detection.add_argument("--helmet-class", type=int, help="헬멧 클래스 번호")
    detection.add_argument(
        "--person-max-frame-ratio", type=float,
        help="사람 박스가 화면의 이 비율을 넘으면 버린다(군중 오검출 제거)",
    )
    detection.add_argument("--moto-dedup-ioa", type=float, help="오토바이 중복 박스 제거 포함비")

    association = parser.add_argument_group("탑승자 연관")
    association.add_argument(
        "--assoc-metric", choices=("person_ioa", "min_ioa", "iou"), help="연관 척도"
    )
    association.add_argument("--assoc-tau", type=float, help="탑승자로 인정할 겹침 임계값")
    association.add_argument(
        "--no-center-check", dest="center_check", action="store_false", default=None,
        help="사람 중심이 오토바이 박스 안이어야 한다는 조건을 끈다",
    )
    association.add_argument("--center-margin", type=float, help="중심 판정 여유분 비율")
    association.add_argument("--rider-area-min", type=float, help="탑승자 인정 최소 면적비")
    association.add_argument("--rider-area-max", type=float, help="탑승자 인정 최대 면적비")
    association.add_argument("--rider-dedup-ioa", type=float, help="같은 탑승자로 볼 포함비")
    association.add_argument("--rider-persist", type=int, help="검출이 끊겨도 명단에 유지할 프레임 수")
    association.add_argument("--rider-persist-iou", type=float, help="명단 매칭 IoU")

    multi = parser.add_argument_group("다인승 판정")
    multi.add_argument("--min-riders", type=int, help="위반이 되는 최소 탑승자 수")
    multi.add_argument("--window", type=int, help="누적 윈도우(프레임)")
    multi.add_argument("--ratio", type=float, help="윈도우 안 후보 비율")
    multi.add_argument("--min-samples", type=int, help="판정 시작 최소 관측 프레임")

    helmet = parser.add_argument_group("헬멧 미착용 판정")
    helmet.add_argument(
        "--no-helmet-check", dest="helmet_enabled", action="store_false", default=None,
        help="헬멧 판정을 끄고 다인승만 수행",
    )
    helmet.add_argument("--helmet-conf", type=float, help="헬멧 전용 confidence")
    helmet.add_argument("--head-ratio", type=float, help="머리 구간 = 사람 박스 상단 이 비율")
    helmet.add_argument("--helmet-tau", type=float, help="착용으로 볼 IoA")
    helmet.add_argument("--helmet-dedup-ioa", type=float, help="헬멧 중복 박스 제거 포함비")
    helmet.add_argument("--helmet-min-person-height", type=float, help="이보다 낮으면 판정 불가")
    helmet.add_argument("--helmet-border-margin", type=float, help="화면 경계 이 거리 안이면 판정 불가")
    helmet.add_argument("--helmet-window", type=int, help="미착용 누적 윈도우")
    helmet.add_argument("--helmet-ratio", type=float, help="윈도우 안 미착용 비율")
    helmet.add_argument("--helmet-min-samples", type=int, help="미착용 판정 최소 관측")

    out = parser.add_argument_group("출력")
    out.add_argument("--output-scale", type=float, help="결과 영상 저장 배율")
    out.add_argument(
        "--compress", dest="compress", action="store_true", default=None,
        help="저장 후 H.264 로 재인코딩해 용량을 줄인다",
    )
    out.add_argument("--video-crf", type=int, help="재인코딩 품질. 낮을수록 고화질·큰 용량")

    render = parser.add_argument_group("화면 표시")
    render.add_argument(
        "--label", action="store_true", default=None, help="박스 위 글자와 상단 배너를 그린다"
    )
    render.add_argument(
        "--draw-all", action="store_true", default=None,
        help="정상 오토바이(파랑)·사람(회색)도 그린다",
    )
    render.add_argument(
        "--draw-helmet", action="store_true", default=None, help="검출된 헬멧 박스(주황)를 함께 그린다"
    )
    render.add_argument(
        "--debug-helmet", action="store_true", default=None,
        help="헬멧 박스만 그린다. 위반 표시는 모두 끔",
    )

    args = parser.parse_args()
    apply_config(args)
    return args


# 설정 파일에도 명령줄에도 없을 때 쓰는 최종 기본값.
FALLBACK_DEFAULTS = {
    "imgsz": 1536, "det_conf": 0.10, "nms_iou": 0.70, "tracker": "bytetrack.yaml",
    "person_class": 0, "motorcycle_class": 2, "helmet_class": 4,
    "person_max_frame_ratio": 0.02, "moto_dedup_ioa": 0.80,
    "assoc_metric": "person_ioa", "assoc_tau": 0.30, "center_check": True,
    "center_margin": 0.0, "rider_area_min": 0.08, "rider_area_max": 6.0,
    "rider_dedup_ioa": 0.75, "rider_persist": 1, "rider_persist_iou": 0.30,
    "min_riders": 2, "window": 15, "ratio": 0.60, "min_samples": 8,
    "helmet_enabled": True, "helmet_conf": 0.10, "head_ratio": 0.45, "helmet_tau": 0.50,
    "helmet_dedup_ioa": 0.80, "helmet_min_person_height": 40.0, "helmet_border_margin": 20.0,
    "helmet_window": 15, "helmet_ratio": 0.60, "helmet_min_samples": 8,
    "output_scale": 0.5, "compress": False, "video_crf": 23,
    "label": False, "draw_all": False, "draw_helmet": False, "debug_helmet": False,
}


def apply_config(args: argparse.Namespace) -> None:
    """명령줄에서 비어 있는 값을 YAML -> 코드 기본값 순으로 채운다."""
    path = Path(args.config).expanduser() if args.config else CONFIG_PATH
    config = load_config(path)
    args.config_path = path if path.is_file() else None
    for name, fallback in FALLBACK_DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, config.get(name, fallback))


def validate_cli_args(args: argparse.Namespace) -> None:
    if not args.no_video and not args.output:
        raise ValueError("--output 이 필요하다. 영상 없이 판정만 하려면 --no-video 를 준다")
    if args.person_class == args.motorcycle_class:
        raise ValueError("--person-class 와 --motorcycle-class 가 같다")
    if not 0.0 < args.assoc_tau <= 1.0:
        raise ValueError("--assoc-tau 는 0 초과 1 이하여야 한다")
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio 는 0 초과 1 이하여야 한다")
    if args.window < 1:
        raise ValueError("--window 는 1 이상이어야 한다")
    if args.min_samples < 1 or args.min_samples > args.window:
        raise ValueError("--min-samples 는 1 이상 --window 이하여야 한다")
    for name, value in (("--helmet-tau", args.helmet_tau), ("--head-ratio", args.head_ratio),
                        ("--helmet-dedup-ioa", args.helmet_dedup_ioa), ("--helmet-ratio", args.helmet_ratio)):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 는 0 초과 1 이하여야 한다")
    if args.helmet_min_samples < 1 or args.helmet_min_samples > args.helmet_window:
        raise ValueError("--helmet-min-samples 는 1 이상 --helmet-window 이하여야 한다")
    if args.start_frame < 0:
        raise ValueError("--start-frame 은 0 이상이어야 한다")
    if not 0.0 < args.person_max_frame_ratio <= 1.0:
        raise ValueError("--person-max-frame-ratio 는 0 초과 1 이하여야 한다")
    if args.rider_persist < 0:
        raise ValueError("--rider-persist 는 0 이상이어야 한다")
    if not 0.0 < args.rider_persist_iou <= 1.0:
        raise ValueError("--rider-persist-iou 는 0 초과 1 이하여야 한다")
    if args.min_riders < 2:
        raise ValueError("--min-riders 는 2 이상이어야 한다")
    for name, value in (("--rider-dedup-ioa", args.rider_dedup_ioa), ("--moto-dedup-ioa", args.moto_dedup_ioa)):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} 는 0 초과 1 이하여야 한다")
    if args.rider_area_min >= args.rider_area_max:
        raise ValueError("--rider-area-min 은 --rider-area-max 보다 작아야 한다")
    if args.output_scale <= 0:
        raise ValueError("--output-scale 은 0보다 커야 한다")


# ── 프레임 파싱 ──
def split_detections(result, person_class: int, motorcycle_class: int, person_max_frame_ratio: float = 1.0,
                     helmet_class: int = -1, helmet_conf: float = 0.0):
    """YOLO 결과에서 사람/오토바이 박스와 트랙 ID 를 분리한다.

    군중 구간에서 사람 여러 명이 한 박스로 묶여 화면을 거의 덮는 오검출이 나온다.
    그런 박스는 어떤 오토바이와도 크게 겹쳐 탑승자로 오인되므로 여기서 걸러낸다.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ([], [], [], [], [], [], [], [])
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    # 트랙 ID 는 매칭에 실패한 프레임에서 None 이 된다. 그 경우 -1 로 둔다.
    if boxes.id is None:
        ids = np.full(len(classes), -1, dtype=int)
    else:
        ids = boxes.id.cpu().numpy().astype(int)

    frame_height, frame_width = result.orig_img.shape[:2]
    frame_area = float(frame_width * frame_height)
    person_boxes, person_ids, person_confs = ([], [], [])
    moto_boxes, moto_ids, moto_confs = ([], [], [])
    helmet_boxes, helmet_confs = ([], [])
    for index, class_id in enumerate(classes):
        if class_id == person_class:
            if frame_area > 0 and box_area(xyxy[index]) / frame_area > person_max_frame_ratio:
                continue    # 화면을 덮는 사람 박스는 군중 오검출이다
            person_boxes.append(xyxy[index])
            person_ids.append(int(ids[index]))
            person_confs.append(float(confs[index]))
        elif class_id == motorcycle_class:
            moto_boxes.append(xyxy[index])
            moto_ids.append(int(ids[index]))
            moto_confs.append(float(confs[index]))
        elif class_id == helmet_class and confs[index] >= helmet_conf:
            helmet_boxes.append(xyxy[index])
            helmet_confs.append(float(confs[index]))
    return (person_boxes, person_ids, person_confs, moto_boxes, moto_ids, moto_confs,
            helmet_boxes, helmet_confs)


# ── 영상 실행 ──
def process_video(args: argparse.Namespace) -> None:
    validate_cli_args(args)
    det_model_path = Path(args.det_model).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    for path, label in ((det_model_path, "detection model"), (input_path, "input video")):
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
        min(max(0, total_frames - args.start_frame), args.max_frames)
        if args.max_frames > 0 and total_frames > 0
        else args.max_frames or max(0, total_frames - args.start_frame)
    )

    writer = None
    if not args.no_video:
        output_path = Path(args.output).expanduser().resolve()
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

    # 실행 설정을 로그 첫머리에 남긴다. 없으면 나중에 "이거 몇으로 돌렸지" 를
    # 재실행해야만 확인할 수 있다.
    print("[CONFIG] " + "  ".join([
        f"imgsz={args.imgsz}",
        f"conf={args.det_conf}",
        f"nms_iou={args.nms_iou}",
        f"tracker={args.tracker}",
        f"device={args.device}",
        f"assoc={args.assoc_metric}",
        f"tau={args.assoc_tau}",
        f"rider_dedup={args.rider_dedup_ioa}",
        f"moto_dedup={args.moto_dedup_ioa}",
        f"center_check={args.center_check}",
        f"center_margin={args.center_margin}",
        f"person_max_frame={args.person_max_frame_ratio}",
        f"start_frame={args.start_frame}",
        f"rider_persist={args.rider_persist}",
        f"persist_iou={args.rider_persist_iou}",
        f"helmet={'off' if not args.helmet_enabled else 'on'}",
        f"debug_helmet={args.debug_helmet}",
        f"draw_helmet={args.draw_helmet}",
        f"labels={args.label}",
        f"helmet_conf={args.helmet_conf}",
        f"head_ratio={args.head_ratio}",
        f"helmet_tau={args.helmet_tau}",
        f"helmet_window={args.helmet_window}/{args.helmet_ratio}/{args.helmet_min_samples}",
        f"min_riders={args.min_riders}",
        f"window={args.window}",
        f"ratio={args.ratio}",
        f"min_samples={args.min_samples}",
    ]))
    print(f"[CONFIG] model={det_model_path}  input={input_path}")

    global SHOW_LABELS
    SHOW_LABELS = args.label
    det_model = YOLO(str(det_model_path))
    judge = MultiRiderJudge(
        window=args.window, ratio=args.ratio, min_samples=args.min_samples, min_riders=args.min_riders
    )
    roster = RiderRoster(persist=args.rider_persist, match_iou=args.rider_persist_iou)
    helmet_judge = NoHelmetJudge(
        window=args.helmet_window, ratio=args.helmet_ratio, min_samples=args.helmet_min_samples
    )
    no_helmet_violations: dict[int, dict] = {}
    det_results = det_model.track(
        source=str(input_path),
        classes=(
            [args.person_class, args.motorcycle_class]
            if (not args.helmet_enabled)
            else [args.person_class, args.motorcycle_class, args.helmet_class]
        ),
        conf=args.det_conf,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    # 확정된 위반은 트랙 상태가 정리되어도 남겨야 하므로 따로 보관한다.
    violations: dict[int, dict] = {}
    candidate_frames = 0
    untracked_candidates = 0
    # 구간별 소요 시간. 추론은 제너레이터를 기다린 시간으로 잰다.
    timing = {"infer": 0.0, "postprocess": 0.0, "render": 0.0, "write": 0.0, "compress": 0.0}
    wall_start = time.perf_counter()
    infer_mark = time.perf_counter()

    try:
        with tqdm(total=render_total or None, desc="Motorcycle multi-rider", unit="frame", dynamic_ncols=True) as bar:
            for frame_index, result in enumerate(det_results):
                timing["infer"] += time.perf_counter() - infer_mark
                if frame_index < args.start_frame:
                    # 추적 상태를 이어가야 하므로 추론은 하되 처리·저장만 건너뛴다.
                    infer_mark = time.perf_counter()
                    continue
                if args.max_frames > 0 and frame_index - args.start_frame >= args.max_frames:
                    break
                stage_mark = time.perf_counter()
                frame = result.orig_img.copy() if writer is not None else None
                (person_boxes, person_ids, person_confs, moto_boxes, moto_ids, moto_confs,
                 helmet_boxes, helmet_confs) = split_detections(
                    result, args.person_class, args.motorcycle_class, args.person_max_frame_ratio,
                    -1 if not args.helmet_enabled else args.helmet_class, args.helmet_conf,
                )
                # 헬멧도 한 사람에게 두 박스가 잡힐 수 있다.
                if helmet_boxes and args.helmet_dedup_ioa < 1.0 and len(helmet_boxes) > 1:
                    keep = deduplicate_boxes(helmet_boxes, helmet_confs, args.helmet_dedup_ioa)
                    helmet_boxes = [helmet_boxes[i] for i in keep]
                    helmet_confs = [helmet_confs[i] for i in keep]
                # 오토바이 1대가 박스 2개로 잡히면 위반 트랙이 중복 생성된다.
                if args.moto_dedup_ioa < 1.0 and len(moto_boxes) > 1:
                    keep = deduplicate_boxes(moto_boxes, moto_confs, args.moto_dedup_ioa)
                    moto_boxes = [moto_boxes[i] for i in keep]
                    moto_ids = [moto_ids[i] for i in keep]
                    moto_confs = [moto_confs[i] for i in keep]
                links, assigned = associate_riders(
                    person_boxes=person_boxes,
                    person_ids=person_ids,
                    person_confs=person_confs,
                    moto_boxes=moto_boxes,
                    tau=args.assoc_tau,
                    metric=args.assoc_metric,
                    min_area_ratio=args.rider_area_min,
                    max_area_ratio=args.rider_area_max,
                    dedup_ioa=args.rider_dedup_ioa,
                    require_center=args.center_check,
                    center_margin=args.center_margin,
                )

                # 그리기는 판정이 끝난 뒤로 미룬다. 위반 오토바이의 탑승자는 통합 박스에
                # 흡수되므로 개별 노란 박스를 그리면 안 되기 때문이다.
                pending_normal: list[tuple] = []      # (moto_box, track_id, rider_count)
                pending_violation: list[tuple] = []   # (moto_box, rider_boxes, track_id, count, confirmed)
                violation_person_indices: set[int] = set()
                pending_no_helmet: list[tuple] = []    # (person_box, person_id, confirmed)

                frame_violations = 0
                for moto_index, moto_box in enumerate(moto_boxes):
                    track_id = moto_ids[moto_index]
                    geom_riders = links[moto_index]
                    # 짧은 검출 끊김을 명단으로 메운다. --rider-persist 0 이면 그대로 통과한다.
                    riders = roster.update(
                        track_id, geom_riders, frame_index, moto_box, args.center_margin
                    )
                    rider_count = len(riders)
                    geom_count = len(geom_riders)
                    candidate = rider_count >= args.min_riders
                    if candidate:
                        candidate_frames += 1
                    state = judge.update(track_id, rider_count, frame_index)
                    samples, window_ratio = judge.window_stats(track_id)
                    confirmed = bool(state is not None and state.confirmed)
                    if state is None and candidate:
                        # 트랙 ID 가 없으면 시간 누적을 못 한다. 확정하지 않고 세어만 둔다.
                        untracked_candidates += 1
                    # ── 헬멧 미착용 판정 ──
                    helmet_scores: dict[int, float] = {}
                    bare: list[int] = []
                    undecidable: list[str] = []
                    if args.helmet_enabled and riders:
                        helmet_scores = assign_helmets(
                            riders, helmet_boxes, args.head_ratio, args.helmet_tau
                        )
                        for rider_index, rider in enumerate(riders):
                            if rider.ghost:
                                continue    # 검출이 없는 탑승자는 헬멧도 볼 수 없다
                            reason = helmet_undecidable(
                                rider.box, width, height,
                                args.helmet_min_person_height, args.helmet_border_margin,
                            )
                            if reason:
                                undecidable.append(f"{rider.person_id}:{reason}")
                                continue
                            if helmet_scores.get(rider_index, 0.0) < args.helmet_tau:
                                bare.append(rider_index)
                    helmet_state = helmet_judge.update(track_id, bool(bare), frame_index)
                    no_helmet_confirmed = bool(helmet_state is not None and helmet_state.confirmed)
                    if no_helmet_confirmed and track_id not in no_helmet_violations:
                        no_helmet_violations[track_id] = helmet_state.confirmed_frame
                        print(
                            f"[NO-HELMET] frame={helmet_state.confirmed_frame} moto={track_id} "
                            f"bare={len(bare)}"
                        )
                    if frame is not None and bare and no_helmet_confirmed:
                        for rider_index in bare:
                            pending_no_helmet.append(
                                (riders[rider_index].box, riders[rider_index].person_id, True)
                            )

                    if confirmed:
                        frame_violations += 1
                        if track_id not in violations:
                            violations[track_id] = {
                                "frame": state.confirmed_frame,
                                "max_riders": state.max_riders,
                            }
                            print(
                                f"[MULTI-RIDER] frame={state.confirmed_frame} moto={track_id} "
                                f"riders={state.max_riders} window={samples} ratio={window_ratio:.2f}"
                            )
                        else:
                            violations[track_id]["max_riders"] = max(
                                violations[track_id]["max_riders"], state.max_riders
                            )

                    if frame is not None:
                        # 색은 이번 프레임의 탑승자 수로 정한다 (2인 이상 = 빨강).
                        # 확정 트랙은 순간적으로 1명만 잡혀도 빨강을 유지한다.
                        violation = candidate or confirmed
                        if violation:
                            pending_violation.append(
                                (moto_box, [r.box for r in riders], track_id, rider_count,
                                 confirmed, sum(1 for r in riders if r.ghost))
                            )
                            violation_person_indices.update(
                                r.person_index for r in riders if r.person_index >= 0
                            )
                        elif args.draw_all:
                            pending_normal.append((moto_box, track_id, rider_count))

                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                frame_index,
                                track_id,
                                f"{moto_confs[moto_index]:.3f}",
                                f"{moto_box[0]:.1f}",
                                f"{moto_box[1]:.1f}",
                                f"{moto_box[2]:.1f}",
                                f"{moto_box[3]:.1f}",
                                rider_count,
                                geom_count,
                                " ".join(str(r.person_id) for r in riders if r.ghost),
                                " ".join(str(rider.person_id) for rider in riders),
                                " ".join(f"{rider.overlap:.3f}" for rider in riders),
                                int(candidate),
                                samples,
                                f"{window_ratio:.3f}",
                                int(confirmed),
                                state.confirmed_frame if confirmed else "",
                                " ".join(f"{helmet_scores.get(i, 0.0):.2f}" for i in range(len(riders))),
                                " ".join(str(riders[i].person_id) for i in bare),
                                " ".join(undecidable),
                                int(bool(bare)),
                                int(no_helmet_confirmed),
                            ]
                        )

                # ── 실제 그리기 ──
                # 여기까지가 후처리(연관·판정·CSV), 이 아래가 렌더링이다.
                timing["postprocess"] += time.perf_counter() - stage_mark
                stage_mark = time.perf_counter()
                if frame is not None and args.debug_helmet:
                    # 디버깅: 검출된 헬멧 박스만 그린다.
                    for helmet_index, helmet_box in enumerate(helmet_boxes):
                        draw_helmet(frame, helmet_box, helmet_confs[helmet_index])
                elif frame is not None:
                    # ① 참고용 회색 사람·파란 오토바이 — --draw-all 일 때만
                    if args.draw_all:
                        for person_index, person_box in enumerate(person_boxes):
                            if person_index not in violation_person_indices:
                                draw_person(frame, person_box, person_ids[person_index])
                        for moto_box_n, track_id_n, count_n in pending_normal:
                            draw_motorcycle(frame, moto_box_n, track_id_n, count_n, False, False)
                    # ③ 위반은 오토바이+탑승자를 하나로 묶어 빨강. 맨 위에 그린다
                    for moto_box_v, rider_boxes_v, track_id_v, count_v, confirmed_v, ghost_v in pending_violation:
                        draw_violation(
                            frame, moto_box_v, rider_boxes_v, track_id_v, count_v, confirmed_v, ghost_v
                        )
                    # ③ 헬멧 미착용은 맨 위에. 빨간 통합 박스 안에 들어갈 수 있다.
                    for bare_box, bare_id, bare_confirmed in pending_no_helmet:
                        draw_no_helmet(frame, bare_box, bare_id, bare_confirmed)
                    # ④ 검출된 헬멧 박스 (주황)
                    if args.draw_helmet:
                        for helmet_index, helmet_box in enumerate(helmet_boxes):
                            draw_helmet(frame, helmet_box, helmet_confs[helmet_index])

                if frame_index % 300 == 0:
                    roster.prune(frame_index)
                    helmet_judge.prune(frame_index)
                if frame is not None and not args.debug_helmet:
                    draw_banner(frame, frame_index, len(moto_boxes), frame_violations)
                timing["render"] += time.perf_counter() - stage_mark
                stage_mark = time.perf_counter()
                if frame is not None:
                    writer.write(cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA))
                timing["write"] += time.perf_counter() - stage_mark
                bar.update(1)
                infer_mark = time.perf_counter()
    finally:
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    final_path = None
    if not args.no_video:
        final_path = Path(args.output).expanduser().resolve()
        if args.compress:
            compress_mark = time.perf_counter()
            compress_video(final_path, args.video_crf)
            timing["compress"] = time.perf_counter() - compress_mark

    total = time.perf_counter() - wall_start
    processed = max(1, min(frame_index + 1, render_total or frame_index + 1))
    print("")
    print(f"[TIMING] 총 {total:.1f}s / {processed}프레임 = {processed / total:.1f} frame/s")
    for name, label in (("infer", "추론+추적"), ("postprocess", "후처리(연관·판정)"),
                        ("render", "렌더링(박스 그리기)"), ("write", "영상 저장"),
                        ("compress", "H.264 압축")):
        value = timing[name]
        if value <= 0:
            continue
        print(f"[TIMING]   {label:<20} {value:7.1f}s  {value / total:5.1%}  "
              f"({value / processed * 1000:.1f} ms/frame)")
    print("")
    print(f"다인승 확정 트랙 수: {len(violations)}")
    for track_id, info in sorted(violations.items()):
        print(f"  moto #{track_id}: 확정 frame={info['frame']}, 관측된 최대 탑승자 수={info['max_riders']}")
    print(f"프레임 단위 다인승 후보 건수: {candidate_frames}")
    if args.helmet_enabled:
        print(f"헬멧 미착용 확정 트랙 수: {len(no_helmet_violations)}")
    if untracked_candidates:
        print(f"트랙 ID 없어 확정 보류한 후보 건수: {untracked_candidates}")
    if not args.no_video:
        print(f"결과 영상: {final_path}")
    if args.csv:
        print(f"판정 로그: {Path(args.csv).expanduser().resolve()}")


def main() -> None:
    process_video(parse_cli_args())


if __name__ == "__main__":
    main()
