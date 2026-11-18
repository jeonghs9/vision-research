# GANet

si_vietnam_drone 드론 항공뷰 도로 영상에 [GANet](https://github.com/Wolfwjs/GANet)
(A Keypoint-based Global Association Network for Lane Detection, CVPR 2022)을
적용해보는 프로젝트. YOLinO(`../YOLinO/`)와 같은 데이터셋을 다른 방법론으로
시도해보는 비교 실험.

상세 검토: [docs/01-model-review/1. GANet 적용 가능성 검토.md](docs/01-model-review/1.%20GANet%20적용%20가능성%20검토.md)

## 데이터셋별 문서 위치

| 데이터셋 | 문서 | 실험(`test/`) | 모델(`models/`) |
|---|---|---|---|
| si_vietnam_drone (드론 항공뷰) | [docs/01-model-review/](docs/01-model-review/) | 001~013 | `drone-ganet-*` |
| OpenLane (차량 전방 시점) | [docs/02-openlane/](docs/02-openlane/README.md) | 014~ (예정, 이름에 `openlane`) | `openlane-ganet-*` (예정) |

두 데이터셋의 점수는 직접 비교하지 않는다.

## 구조

- `source/` — GANet 원 저장소 clone (`mmdet2_25` 브랜치, `.git` 포함 원형 유지)
- `docs/` — 논문 검토, 설정 결정 기록
- `scripts/01-data` — si_vietnam_drone 라벨 → CULane 형식 변환
- `scripts/02-train`, `scripts/04-eval` — 학습·평가·시각화 스크립트
- `test/` — 실험별 임시 결과 (`test/001-...`)
- `models/` — 채택한 최종 모델만
- `output/` — 비교용 시각화 결과 (`01~04` 모델 학습·평가 결과, `05` 점선 미탐 원인분석, `06` 특징맵·BEV 무학습 실험, `07` P3 제거 특징맵 분석)

## 환경

`ganet_cu101` (python 3.8 / torch 1.8 / cudatoolkit 10.1, mmcv-full 1.6.2 +
mmdet 2.25.3) — `yolino_cu126`과 별도 격리된 conda 환경. 구축 완료, GPU에서
커스텀 CUDA 연산(deform_conv) 동작까지 확인. 구축 과정·주의사항:
[docs/01-model-review/2. 환경 구축 (ganet_cu101).md](docs/01-model-review/2.%20환경%20구축%20%28ganet_cu101%29.md)

## 진행 상황 (2026-11-13 기준)

> 전체 실험(003~013) 성능 한 표: [16. 전체 실험 성능 정리](docs/01-model-review/16.%20전체%20실험%20성능%20정리.md) — 1위 P3del−SA(008) ep100~150 평균 F1 .774

- 데이터 변환(`culane_lane_ganet`, 431/163 train/val, YOLinO와 동일 분할):
  [3. 데이터 변환과 학습 스모크 테스트](docs/01-model-review/3.%20데이터%20변환과%20학습%20스모크%20테스트.md)
- 본학습(res18/res34 1클래스, 1280x736, batch=16, 150epoch) 완료:
  [4. 본학습 파라미터 결정과 실행](docs/01-model-review/4.%20본학습%20파라미터%20결정과%20실행.md)
- epoch별 자동 시각화(`output/01-cls1-에폭별val확인/`) + 진행 관찰:
  [5. 자동 시각화 체계와 epoch별 진행 관찰](docs/01-model-review/5.%20자동%20시각화%20체계와%20epoch별%20진행%20관찰.md)
- **최종 체크포인트 선정 완료(1클래스)** — res18 epoch140, res34 epoch150,
  둘 다 F1 0.758(163장 val 전체 기준)로 동률. `models/`에 승격:
  [7. 최종 체크포인트 선정 (1클래스)](docs/01-model-review/7.%20최종%20체크포인트%20선정%20%281클래스%29.md)
- 2클래스(실선/점선) 확장 계획: [6. 2클래스(실선·점선) 확장 계획](docs/01-model-review/6.%202클래스%28실선·점선%29%20확장%20계획.md)
- **2클래스 본학습·선정 완료** — res18 epoch120(F1 0.738), res34 epoch130(F1 0.749),
  `models/drone-ganet-res{18,34}-2cls-v1.pth`:
  [8. 2클래스 구현과 결과](docs/01-model-review/8.%202클래스%20구현과%20결과.md)
  (영상 적용·후처리는 문서 9~11)
- **P3 제거 구조(P3del) 2클래스 학습 완료, 채택 미정** — LFA·head를 P3(1/8) 대신 P4(1/16)에 걸어 다시 학습.
  val F1: res18 0.736 → 0.770(ep140), res34 0.747 → 0.766(ep150). 다만 원거리 장면(부산 다리)에서 가는 실선을 놓침.
  이후 비교는 **res34로 고정**. BEV 이미지 입력 비교와 히트맵 읽는 법(Grad-CAM 기여도 vs conf map)은 문서 12의 8·9절.
  `test/005-res18-2cls-1280-P3del-150ep`, `test/006-res34-2cls-1280-P3del-150ep`:
  [12. P3 제거 구조 학습 (P3del)](docs/01-model-review/12.%20P3%20제거%20구조%20학습%20%28P3del%29.md)
- **SA(C5 attention) 제거 구조(noSA) 학습 완료, 채택 미정** — C5 뒤 attention만 빼고 다시 학습.
  val F1: 기존 0.747 → 기존−SA 0.763(ep140), P3del 0.766 → P3del−SA 0.780(ep110). 차이가 epoch 간 흔들림과 비슷해서 1회 학습으로는 "빼도 안 나빠짐"까지만 확인.
  `test/007-res34-2cls-1280-noSA-150ep`, `test/008-res34-2cls-1280-P3del-noSA-150ep`:
  [13. SA 제거 구조 학습 (noSA)](docs/01-model-review/13.%20SA%20제거%20구조%20학습%20%28noSA%29.md)
- **SA 대신 Deformable Attention(DA) 구조 학습 완료, 채택 안 함 권장** — P3del에서 C5 전역 SA만 DA(head 8)로 교체, 위치 인코딩 방식 2가지(`query`/`add`) 비교.
  val F1: DA `query` 0.768(ep150), `add` 0.768(ep90) — 둘 다 P3del−SA 0.780 미만(ep100~150 범위가 겹치지 않음).
  `source/projects/plugin/models/dcn_fpn.py`에 `DeformAttentionLayer` 추가(미커밋).
  `test/009-res34-2cls-1280-P3del-DAq-150ep`, `test/010-res34-2cls-1280-P3del-DAadd-150ep`:
  [14. SA 대신 Deformable Attention 구조 학습 (DA)](docs/01-model-review/14.%20SA%20대신%20Deformable%20Attention%20구조%20학습%20%28DA%29.md)
- **C5에 APT(Adaptive Patch) 토큰 병합 적용 완료, 채택 안 함 권장** — 입력 64×64 엔트로피 < 6.5 인 2×2 칸을 큰 패치 토큰으로 합침.
  ep100~150 평균 val F1: SA .758→.769, DA .763→.762, SA제거 .774→.765 — 셋 다 P3del−SA(.774) 이하, 속도 +1.3~2.9ms.
  `dcn_fpn.py` APT 부분·`ganet.py` `extract_feat` 추가(미커밋).
  `test/011-…-P3del-APT-150ep`, `012-…-P3del-DAq-APT-150ep`, `013-…-P3del-noSA-APT-150ep`:
  [15. C5에 APT(Adaptive Patch) 토큰 병합 적용 (APT)](docs/01-model-review/15.%20C5에%20APT%28Adaptive%20Patch%29%20토큰%20병합%20적용%20%28APT%29.md)
