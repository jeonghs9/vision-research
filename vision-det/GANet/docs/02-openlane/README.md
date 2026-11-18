# 02-openlane — OpenLane 데이터셋으로 GANet 학습

OpenLane(Waymo 기반 차량 전방 시점 차선 데이터셋)의 **2D 차선 라벨(`uv`)** 로 GANet을 학습하는
작업의 문서 모음. 기존 드론(si_vietnam_drone) 문서는 `../01-model-review/`에 있고,
**이 폴더와 섞지 않는다.** 드론 문서에서 OpenLane을 언급할 때는 링크만 건다.

| 문서 | 내용 | 상태 |
|---|---|---|
| [1. OpenLane 라벨 분석과 클래스 결정](1.%20OpenLane%20라벨%20분석과%20클래스%20결정.md) | 라벨 구조, category 분포, 실선/점선 2클래스 합치기 선택지 | 분석 완료 · **클래스 확정(2클래스)** |
| [2. 데이터 변환 (CULane 형식)](2.%20데이터%20변환%20%28CULane%20형식%29.md) | uv → GANet 입력 변환 계획 | **완료** (차선 0개 프레임 제외: train 6,988 · val 1,961 · test 1,036) |
| [3. 분할과 vis 검증](3.%20분할과%20vis%20검증.md) | segment 단위 분할, SPLIT.md·vis 30장 | **완료** — 698 / 202 / 100 segment (≈ 7:2:1), train 9,732 · val 2,821 · test 1,393장 |
| [4. 학습·평가 결과](4.%20학습·평가%20결과.md) | 실험(`test/014-…`~) 결과 | 구조 5종(test/014~018 `*-noempty-150ep`) 학습 중 (2026-11-18 14:37 재시작) |
| [5. 데이터 준비 이력과 결정 로그](5.%20데이터%20준비%20이력과%20결정%20로그.md) | 시간순 이력, 결정과 근거, 아직 안 한 것 | 작성 완료 |

## 데이터 위치 (절대경로)

| 항목 | 경로 |
|---|---|
| 원본(수정 금지) | `/home/hsjeong/workspace/dataset/det/OpenLane/raw/` (전체 다운로드 완료: training 798 + validation 202 segment, `images/`·`lane3d_1000/`) |
| 변환 결과 | `/home/hsjeong/workspace/dataset/det/OpenLane/processed/culane_lane_ganet/` |
| 분할·SPLIT.md·vis | `/home/hsjeong/workspace/dataset/det/OpenLane/splits/` (생성 완료) |
| 노션 | 데이터셋 DB "OpenLane" 줄 (https://app.notion.com/p/41bfeb434afc81fcb6c8e55833f532b1) |

## 드론 작업과 분리하는 규칙

1. OpenLane 문서는 `docs/02-openlane/`에만 둔다.
2. 실험 폴더명에 `openlane`을 넣는다 (`test/014-openlane-…`). 하이퍼파라미터는 폴더 안 config에.
3. 모델 파일은 `openlane-ganet-<구성>-v<버전>.pth` — 드론 `drone-ganet-*`는 건드리지 않는다.
4. 스크립트는 파일명으로 구분 (`scripts/01-data/openlane_to_culane.py`). 기존 `polyline_to_culane.py`는 그대로.
5. **드론 val 점수(F1 0.774 등)와 OpenLane 점수는 직접 비교하지 않는다** — 시점(항공 vs 전방)·라벨 정의·평가 셋이 다르다.

## 결정 대기 항목 (사용자 확인 필요)

- [x] 클래스 확정: 혼합선→실선, 연석 제외, 점선(1,3,7,9)/실선(2,4,8,10,5,6,11,12) → 문서 1 §5 (unknown 0은 재확인 후 확정)
- [x] 입력 해상도 1280×864 (너비 1280 + 32배수 높이), backbone ResNet34, 드론 가중치 미사용 → 문서 4 §1
- [x] 분할: segment 단위, 15프레임 간격, train 698 / val 202 / test 100 (문서 3, `splits/SPLIT.md`)
- [ ] 정지 위주 segment·차선 0개 프레임을 학습에서 줄일지 (현재는 그대로 포함)
- [x] epoch 150, 구조 5종(P3·SA·DA 조합) 동시 학습 (문서 4 §1-1)
