# GANet Feature 분석 2단계

GANet(`drone-ganet-res34-2cls-v1`)에서 FPN 단계 일부를 **학습 없이 추론에서만** 빼거나 바꿨을 때 특징맵이 어떻게 달라지는지 본 그림입니다.

- 프레임: `1` = AIhub 부산 다리, `2` = 베트남 드론 001392, `3` = 0816 영상 f990(33초)
- 파일명: `{프레임}-{그림번호}. 이름.png`
- 스크립트는 `GANet/scripts/03-vis/`에 있습니다. GANet 소스는 수정하지 않았고, 스크립트 안에서 `neck.forward`만 바꿔 끼웠습니다.
- 비교 기준: `../06-cls2-특징맵BEV무학습분석/figures/{n}-1. FPN단계별_히트맵.png` (기존 경로)

## 경로

```text
기존:     C3 ──────────────┐
          C4 ───────┐      │
          C5 → SA → AC5 → +C4 = P4 → +C3 = P3 → DCN → fpn_conv → head   (출력 1/8)

P3 제거:  C4 ───────┐
          C5 → SA → AC5 → +C4 = P4 → DCN → fpn_conv → head                (출력 1/16, C3 안 씀)
```

## 그림별 테스트 과정

| 번호 | 내용 | 경로 | 스크립트 |
|---|---|---|---|
| 1 | P3 제거 시 FPN 단계별 히트맵(활성 크기, 실선/점선 conf 기여도). 기존 1-1 그림에서 P3 칸을 뺀 6단계: C3·C4·C5·C5+SA·P4(DCN 전)·P4(최종). C3는 백본에서 계산되어 C4의 입력이 되지만 neck에서는 쓰지 않습니다 | P4를 기존 P3 자리에 그대로 넣습니다. DCN(`def_convs[0]`, 1×7) → `fpn_convs[0]` → head는 기존 가중치를 그대로 씁니다. offset 헤드 입력은 DCN 전 P4입니다 | `ganet_fpn_stage_heatmap_noP3.py` |
| 2 | P3 제거 시 최종 검출 결과 vs 기존. 줄: 기존(P3, 1/8) / P3 제거(offset 값 그대로) / P3 제거 + offset ½. 칸: 실선 conf, 점선 conf, 최종 검출 | 1과 같은 경로입니다. head는 offset을 1/8 칸 단위로 학습했으므로, 1/16 칸으로 환산한 ½ 버전도 함께 봤습니다. 후처리 문턱값(root_thr=1, cluster_by_center_thr=5, points_thr=10)은 칸 단위 그대로 둡니다 | `ganet_noP3_detection.py` |
| 3 | P3del 구조로 **다시 학습한** 모델의 최종 검출 vs 기존 (**res34 고정**). 줄: res34 기존 / res34 P3del. 칸: 실선 conf, 점선 conf, 최종 검출 | 기존은 `models/` 채택본(res34 ep130). P3del은 `GANet/test/006` epoch_150 | `ganet_p3del_trained_compare.py` |
| 4 | P3del로 다시 학습한 res34의 FPN 단계별 히트맵 (1번 그림과 같은 방식) | C3 · C4 · C5 · C5+SA · P4(DCN 전) · P4(최종). C3는 neck에서 안 쓰지만 C4의 입력이라 함께 그립니다 | `ganet_p3del_trained_compare.py` (3과 같이 생성) |
| 5 | 4번 그림(P3del로 **다시 학습한** res34의 FPN 단계별 히트맵)에 **BEV로 편 이미지**를 입력한 버전. 배경도 BEV 이미지 | 이미지 → BEV(고정 비율 사다리꼴: 윗변 y=0.45H 가운데 40% 폭 → 화면 전체) → backbone → C4·C5 → attention → P4 → head. 모델은 `GANet/test/006` epoch_150 | `ganet_p3del_trained_compare_bev.py` |
| 6 | 3번 그림(P3del 학습 모델 최종 검출 vs 기존)에 **BEV로 편 이미지**를 입력한 버전. 줄: res34 기존 / res34 P3del, 결과도 BEV 이미지 위에 그림 | 5와 같은 사다리꼴. 모델은 3과 같습니다 | `ganet_p3del_trained_compare_bev.py` (5와 같이 생성) |
| 7 | 3번(원본 입력)과 6번(BEV 입력) 최종 검출을 한 장에 모은 그림. 줄: 원본·res34 기존 / 원본·res34 P3del / BEV·res34 기존 / BEV·res34 P3del. 칸: 실선 conf, 점선 conf, 최종 검출 | 모델은 3, BEV 사다리꼴은 5·6과 같습니다. 수치는 3·6번 표와 같습니다 | `ganet_p3del_orig_vs_bev_detection.py` |
| 8 | res34 P3del 학습 모델의 **실선·점선 구분 근거** 히트맵. 2줄 = '실선 점수 − 점선 점수'를 키운 곳, 3줄 = 줄인 곳(점선 쪽). 2·3줄은 같은 기준으로 칠함 | 목표값 = 차선 칸(실선 또는 점선 conf > 0.4)에서 Σ(실선 conf − 점선 conf). Grad-CAM을 ReLU 없이 계산해 부호로 나눔. 원본 입력 | `ganet_solid_vs_dashed_cam.py` |
| 9 | 8과 같은 그림, res34 기존 모델 | 단계: C3 · C4 · C5 · C5+SA · P4 · P3(DCN 전) · P3(최종) | `ganet_solid_vs_dashed_cam.py` (8과 같이 생성) |
| 10 | **원본 입력** FPN 단계별 히트맵: res34 기존 vs P3del (블록 2개). SA 적용/미적용 비교 전 기준 그림 | 칸은 두 구조를 맞춘 7칸: C3 · C4 · C5 · C5+SA · P4 · P3(DCN 전, 기존만) · 최종. 블록마다 3줄(활성 크기, 실선·점선 conf 기여도) | `ganet_fpn_heatmap_4way.py` |
| 11 | 10과 같은 그림, **BEV 입력** | 10과 같음. BEV 사다리꼴은 5·6·7과 같음 | `ganet_fpn_heatmap_4way.py` (10과 같이 생성) |
| 12 | **SA(C5 attention) 적용 vs 미적용** FPN 단계별 히트맵, **원본 입력**. 블록 4개: res34 기존 / 기존−SA / P3del / P3del−SA | 10번과 같은 7칸. SA 없는 모델은 C5 다음에 바로 lateral conv로 가므로 C5+SA 칸이 "SA 없음"으로 빔. 기존−SA = `GANet/test/007` epoch_140, P3del−SA = `GANet/test/008` epoch_110 (각각 val F1 최고 epoch) | `ganet_fpn_heatmap_sa.py` |
| 13 | 12와 같은 그림, **BEV 입력** | 12와 같음. BEV 사다리꼴은 5·6·7과 같음 | `ganet_fpn_heatmap_sa.py` (12와 같이 생성) |
| 14 | SA 적용 vs 미적용 **최종 검출**, **원본 입력**. 줄: 기존 / 기존−SA / P3del / P3del−SA. 칸: 실선 conf, 점선 conf, 최종 검출 | 모델은 12와 같음. 검출 함수는 3번 그림과 같음 | `ganet_sa_detection.py` |
| 15 | 14와 같은 그림, **BEV 입력** | 14와 같음 | `ganet_sa_detection.py` (14와 같이 생성) |
| 16 | **C5 뒤 SA / 없음 / DA(query)** FPN 단계별 히트맵, **원본 입력**. 블록 3개: res34 P3del+SA / P3del−SA / P3del+DA query | 12번에서 P3 칸을 뺀 6칸: C3 · C4 · C5 · C5+attention · P4(DCN 전) · 최종. P3del = `GANet/test/006` epoch_150, P3del−SA = `test/008` epoch_110, P3del+DA query = `test/009` epoch_150 (Deformable-DETR 방식, 문서 14) | `ganet_fpn_heatmap_da.py` |
| 17 | 16과 같은 그림, **BEV 입력** | 16과 같음. BEV 사다리꼴은 5·6·7과 같음 | `ganet_fpn_heatmap_da.py` (16과 같이 생성) |
| 18 | P3del+SA / P3del−SA / P3del+DA query **최종 검출**, **원본 입력**. 칸: 실선 conf, 점선 conf, 최종 검출 | 모델은 16과 같음. 검출 함수는 3·14번 그림과 같음 | `ganet_da_detection.py` |
| 19 | 18과 같은 그림, **BEV 입력** | 18과 같음 | `ganet_da_detection.py` (18과 같이 생성) |
| 20 | **APT 3종** FPN 단계별 히트맵, **원본 입력**. 블록 3개: res34 P3del+SA+APT / P3del+DA+APT / P3del−SA+APT | 16과 같은 6칸. C5+attention 칸은 SA·DA+APT면 attention 뒤(64ch), −SA+APT면 병합 뒤 C5(512ch). `test/011` epoch_110, `test/012` epoch_110, `test/013` epoch_150 (각 최고 F1, APT τ=6.5, 문서 15) | `ganet_fpn_heatmap_apt.py` |
| 21 | 20과 같은 그림, **BEV 입력** | 20과 같음 | `ganet_fpn_heatmap_apt.py` (20과 같이 생성) |
| 22 | APT 3종 **최종 검출**, **원본 입력**. 칸: 실선 conf, 점선 conf, 최종 검출 | 모델은 20과 같음. 검출 함수는 18번과 같음 | `ganet_apt_detection.py` |
| 23 | 22와 같은 그림, **BEV 입력** | 22와 같음 | `ganet_apt_detection.py` (22와 같이 생성) |
| 24 | **실험별 경로 과정도** (그림 아닌 구조도). 004 기존, 006~013 res34 2클래스 9종의 단계 흐름, 뺀 단계는 점선. 줄마다 ep100~150 평균 F1·시간 | 경로는 `dcn_fpn.py` DeformFPN.forward 기준, 수치는 문서 16 | `ganet_experiment_path_diagram.py` |
| 25 | 24와 같은 과정도, **BEV 입력**. 원본을 사다리꼴로 펴 같은 모델에 넣음(21·23번과 같은 입력), 결과는 BEV 좌표 그대로. APT 엔트로피도 BEV 이미지에서 계산. F1은 BEV 평가가 없어 뺌 | 24와 같음 | `ganet_experiment_path_diagram.py --bev` |

`24. 경로별_과정도_실험별구조.pptx`: 24번 그림을 편집 가능한 도형으로 옮긴 슬라이드 2장(① 004·007·006·008 덜어내기, ② 009~013 바꾸기·더하기). 생성: `cd /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/pptx && npm install && NODE_PATH=$PWD/node_modules node build.js` (`node_modules`는 git 제외, 처음 한 번 `npm install`)

실행:

```bash
cd /home/hsjeong/workspace/vision-det/GANet/source
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap_noP3.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_noP3_detection.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare_bev.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_orig_vs_bev_detection.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_solid_vs_dashed_cam.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_4way.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_sa.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_sa_detection.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_da.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_da_detection.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_heatmap_apt.py
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_apt_detection.py
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_experiment_path_diagram.py
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_experiment_path_diagram.py --bev
```

18·19 P3del+SA / P3del−SA / P3del+DA(query) 최종 검출 (선 수: 실선+점선):

| 프레임 | 입력 | P3del+SA | P3del−SA | P3del+DA query |
|---|---|---|---|---|
| 1 | 원본 | 2+1 | 2+1 | 3+2 |
| 1 | BEV | 2+1 | 1+1 | 2+1 |
| 2 | 원본 | 1+2 | 1+2 | 1+2 |
| 2 | BEV | 3+1 | 2+0 | 3+1 |
| 3 | 원본 | 0+3 | 0+3 | 0+3 |
| 3 | BEV | 2+1 | 2+1 | 2+1 |

P3del+SA·P3del−SA 값은 14·15번 표와 같습니다(같은 체크포인트). 설정·해석은 `vision-det/GANet/docs/01-model-review/14. SA 대신 Deformable Attention 구조 학습 (DA).md` 7-3절.

14·15 SA 적용 vs 미적용 최종 검출 (선 수: 실선+점선). 선별 점 수는 문서 13의 3절에 있습니다:

| 프레임 | 입력 | 기존 | 기존−SA | P3del | P3del−SA |
|---|---|---|---|---|---|
| 1 | 원본 | 4+1 | 2+1 | 2+1 | 2+1 |
| 1 | BEV | 3+1 | 2+1 | 2+1 | 1+1 |
| 2 | 원본 | 1+2 | 1+2 | 1+2 | 1+2 |
| 2 | BEV | 3+1 | 1+2 | 3+1 | 2+0 |
| 3 | 원본 | 0+3 | 0+3 | 0+3 | 0+3 |
| 3 | BEV | 3+1 | 2+1 | 2+1 | 2+1 |

val 163장 F1(위치만): 기존 0.747 → 기존−SA 0.763, P3del 0.766 → P3del−SA 0.780.
설정·전체 결과는 `vision-det/GANet/docs/01-model-review/13. SA 제거 구조 학습 (noSA).md`에 있습니다.

3 다시 학습한 res34 모델의 최종 검출 (선별 점 수 / 버려진 점):

| 프레임 | 클래스 | res34 기존 | res34 P3del |
|---|---|---|---|
| 1 | 실선 | (38, 46, 39, 23) / 14 | (22, 23) / 0 |
| 1 | 점선 | (46) / 23 | (23) / 11 |
| 2 | 실선 | (92) / 0 | (44) / 0 |
| 2 | 점선 | (65, 92) / 81 | (39, 45) / 27 |
| 3 | 점선 | (36, 78, 59) / 14 | (21, 41, 26) / 8 |

6 BEV 입력 최종 검출, P3del 학습 res34 모델 (선별 점 수 / 버려진 점):

| 프레임 | 클래스 | res34 기존 | res34 P3del |
|---|---|---|---|
| 1 | 실선 | (11, 54, 59) / 36 | (23, 29) / 0 |
| 1 | 점선 | (66) / 60 | (38) / 21 |
| 2 | 실선 | (41, 26, 92) / 5 | (22, 11, 44) / 0 |
| 2 | 점선 | (16) / 1 | (11) / 2 |
| 3 | 실선 | (62, 43, 14) / 5 | (31, 19) / 0 |
| 3 | 점선 | (38) / 23 | (20) / 4 |

val 163장 F1(위치만 평가): res18 0.736 → 0.770, res34 0.747 → 0.766.

결론: 학습 없이 P3만 빼면 검출이 크게 나빠집니다. head가 P3 모양의 입력과 1/8 칸 단위 offset만 학습했기 때문입니다.
그래서 P3 제거 구조로 다시 학습했습니다(그림 3·4). 설정과 전체 결과는
`vision-det/GANet/docs/01-model-review/12. P3 제거 구조 학습 (P3del).md`에 있습니다.
BEV 입력 결과(그림 5~7)는 문서 12의 8절에 있습니다. 히트맵 줄별 의미(활성 크기 / 기여도 / conf map 차이)와 검증 수치는 9절에 있습니다.

2 최종 검출 (선별 점 수 / 버려진 점):

| 프레임 | 클래스 | 기존 (P3) | P3 제거, offset 그대로 | P3 제거, offset ½ |
|---|---|---|---|---|
| 1 | 실선 | (38, 46, 39, 23) / 14 | 없음 / 16 | 없음 / 16 |
| 1 | 점선 | (46) / 23 | 없음 / 14 | (14) / 0 |
| 2 | 실선 | (92) / 0 | 없음 / 40 | (40) / 0 |
| 2 | 점선 | (65, 92) / 81 | 없음 / 95 | (41) / 54 |
| 3 | 점선 | (36, 78, 59) / 14 | (11) / 60 | (28) / 43 |


conf 0.4를 넘는 칸 수: 1/16 칸 하나는 1/8 칸 4개 넓이이므로 기존 값을 4로 나눠 함께 적었습니다.

| 프레임 | 클래스 | 기존 (1/8) | 기존 ÷4 | P3 제거 (1/16) |
|---|---|---|---|---|
| 1 | 실선 | 294 | 73.5 | 20 |
| 1 | 점선 | 134 | 33.5 | 14 |
| 2 | 실선 | 145 | 36.3 | 46 |
| 2 | 점선 | 455 | 113.8 | 114 |
| 3 | 실선 | 0 | 0 | 0 |
| 3 | 점선 | 366 | 91.5 | 77 |
