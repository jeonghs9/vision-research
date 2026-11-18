# res34 4구조 비교 (기존 / 기존−SA / P3del / P3del−SA) — val 이미지별

> 2026-11-11. 관련: [docs/01-model-review/13. SA 제거 구조 학습 (noSA).md](../../docs/01-model-review/13.%20SA%20제거%20구조%20학습%20%28noSA%29.md) · [12. P3 제거 구조 학습 (P3del).md](../../docs/01-model-review/12.%20P3%20제거%20구조%20학습%20%28P3del%29.md)

- 이미지: `culane_lane_ganet/list/val.txt`(= test.txt) 163장. epoch 선택도 이 163장으로 했으므로 낙관적인 수치입니다.
- 모델(모두 res34, 각 구조의 val F1 최고 epoch): 기존 = `models/drone-ganet-res34-2cls-v1.pth`(ep130), 기존−SA = `test/007` ep140, P3del = `test/006` ep150, P3del−SA = `test/008` ep110
- 매칭: `scripts/04-eval/score_checkpoint.py`와 같음(평균 x 오차 40px 이하, 원본 3840 기준). 클래스 일치 = 맞춘 선 중 실선/점선까지 맞은 수
- 그림: `images/{원본 stem}.png` — 왼쪽 위 정답, 오른쪽 2x2 = 윗줄 P3 있음(기존 · 기존−SA), 아랫줄 P3 없음(P3del · P3del−SA). 왼쪽 칸 SA 있음, 오른쪽 칸 SA 없음
- 이미지별 전체 수치: `비교표.csv`

실행:

```bash
cd /home/hsjeong/workspace/vision-det/GANet/source
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3sa_val_compare.py
```

## 합계

정답 선 1030개.

| 모델 | P3 | SA | 맞춘 선 | 예측 선 | 틀린 예측 | 놓친 정답 | 클래스 일치(맞춘 선 중) |
|---|:-:|:-:|---|---|---|---|---|
| 기존 | O | O | 665 | 751 | 86 | 365 | 647 (97.3%) |
| 기존−SA | O | X | 675 | 740 | 65 | 355 | 657 (97.3%) |
| P3del | X | O | 674 | 731 | 57 | 356 | 662 (98.2%) |
| P3del−SA | X | X | 689 | 737 | 48 | 341 | 671 (97.4%) |

## 쌍별 비교 (이미지 수)

이미지별 점수 = 맞춘 선 − 틀린 예측 선. 차이 = 뒤 모델 점수 − 앞 모델 점수(+면 뒤 모델이 나음).

| 비교 | 앞 → 뒤 | 좋아짐 | 같음 | 나빠짐 |
|---|---|---|---|---|
| SA 제거 (P3 있음) | 기존 → 기존−SA | 36 | 111 | 16 |
| SA 제거 (P3 없음) | P3del → P3del−SA | 30 | 123 | 10 |
| P3 제거 (SA 있음) | 기존 → P3del | 43 | 102 | 18 |
| P3 제거 (SA 없음) | 기존−SA → P3del−SA | 41 | 102 | 20 |
| 기존 → 둘 다 제거 | 기존 → P3del−SA | 54 | 99 | 10 |

## 차이 큰 이미지

### SA 제거 (P3 있음): 기존 → 기존−SA

**기존−SA이(가) 나아진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | 기존−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000310` | +3 | 6 | 4/6 | 5/5 |
| `19700101_001037_T__000420` | +3 | 6 | 4/5 | 6/6 |
| `19700101_001037_T__000425` | +3 | 6 | 4/5 | 6/6 |
| `19700101_001037_T__000265` | +2 | 5 | 2/3 | 3/3 |
| `19700101_001037_T__000294` | +2 | 6 | 5/7 | 6/7 |
| `19700101_001037_T__000300` | +2 | 6 | 6/8 | 6/6 |
| `19700101_001037_T__000360` | +2 | 6 | 5/7 | 5/5 |
| `19700101_001037_T__000365` | +2 | 6 | 4/7 | 5/7 |
| `19700101_001037_T__000419` | +2 | 6 | 3/5 | 4/5 |
| `19700101_001037_T__000490` | +2 | 8 | 6/8 | 6/6 |

**기존−SA이(가) 나빠진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | 기존−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000590` | -3 | 7 | 6/6 | 5/7 |
| `19700101_001037_T__000330` | -2 | 6 | 6/8 | 5/8 |
| `19700101_001037_T__000570` | -2 | 9 | 8/8 | 6/6 |
| `19700101_001037_T__000270` | -1 | 6 | 5/5 | 4/4 |
| `19700101_001037_T__000275` | -1 | 6 | 5/5 | 5/6 |
| `19700101_001037_T__000290` | -1 | 6 | 5/7 | 4/6 |
| `19700101_001037_T__000410` | -1 | 6 | 5/5 | 4/4 |
| `19700101_001037_T__000415` | -1 | 7 | 5/5 | 4/4 |
| `19700101_001037_T__000460` | -1 | 9 | 7/8 | 6/7 |
| `19700101_001037_T__000465` | -1 | 9 | 7/8 | 6/7 |

### SA 제거 (P3 없음): P3del → P3del−SA

**P3del−SA이(가) 나아진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | P3del 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000290` | +2 | 6 | 4/5 | 6/7 |
| `19700101_001037_T__000300` | +2 | 6 | 5/6 | 6/6 |
| `19700101_001037_T__000335` | +2 | 6 | 4/6 | 5/6 |
| `19700101_001037_T__000650` | +2 | 8 | 7/8 | 8/8 |
| `19700101_001037_T__000210` | +1 | 6 | 0/0 | 1/1 |
| `19700101_001037_T__000256` | +1 | 4 | 0/0 | 1/1 |
| `19700101_001037_T__000265` | +1 | 5 | 2/2 | 3/3 |
| `19700101_001037_T__000305` | +1 | 6 | 4/5 | 5/6 |
| `19700101_001037_T__000330` | +1 | 6 | 5/7 | 5/6 |
| `19700101_001037_T__000340` | +1 | 6 | 5/7 | 5/6 |

**P3del−SA이(가) 나빠진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | P3del 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000280` | -1 | 6 | 5/5 | 4/4 |
| `19700101_001037_T__000310` | -1 | 6 | 5/5 | 5/6 |
| `19700101_001037_T__000420` | -1 | 6 | 6/6 | 6/7 |
| `19700101_001037_T__000490` | -1 | 8 | 6/6 | 6/7 |
| `19700101_001037_T__000495` | -1 | 8 | 6/6 | 5/5 |
| `19700101_001037_T__000505` | -1 | 8 | 6/6 | 6/7 |
| `19700101_001037_T__000510` | -1 | 8 | 6/6 | 6/7 |
| `19700101_001037_T__000515` | -1 | 8 | 6/6 | 6/7 |
| `19700101_001037_T__000565` | -1 | 8 | 7/8 | 6/7 |
| `19700101_001037_T__000705` | -1 | 8 | 8/8 | 7/7 |

### P3 제거 (SA 있음): 기존 → P3del

**P3del이(가) 나아진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | P3del 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000310` | +3 | 6 | 4/6 | 5/5 |
| `19700101_001037_T__000419` | +3 | 6 | 3/5 | 5/6 |
| `19700101_001037_T__000420` | +3 | 6 | 4/5 | 6/6 |
| `19700101_001037_T__000425` | +3 | 6 | 4/5 | 6/6 |
| `19700101_001037_T__000585` | +3 | 8 | 6/7 | 8/8 |
| `19700101_001037_T__000365` | +2 | 6 | 4/7 | 5/7 |
| `19700101_001037_T__000490` | +2 | 8 | 6/8 | 6/6 |
| `19700101_001037_T__000500` | +2 | 8 | 5/6 | 6/6 |
| `19700101_001037_T__000535` | +2 | 8 | 6/7 | 7/7 |
| `19700101_001037_T__000260` | +1 | 4 | 0/0 | 1/1 |

**P3del이(가) 나빠진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | P3del 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000355` | -2 | 6 | 6/8 | 5/8 |
| `19700101_001037_T__000285` | -1 | 6 | 6/7 | 4/4 |
| `19700101_001037_T__000295` | -1 | 6 | 6/7 | 5/6 |
| `19700101_001037_T__000305` | -1 | 6 | 6/8 | 4/5 |
| `19700101_001037_T__000315` | -1 | 6 | 5/5 | 4/4 |
| `19700101_001037_T__000320` | -1 | 6 | 6/6 | 5/5 |
| `19700101_001037_T__000325` | -1 | 6 | 6/7 | 4/4 |
| `19700101_001037_T__000330` | -1 | 6 | 6/8 | 5/7 |
| `19700101_001037_T__000345` | -1 | 6 | 6/8 | 5/7 |
| `19700101_001037_T__000360` | -1 | 6 | 5/7 | 4/6 |

### P3 제거 (SA 없음): 기존−SA → P3del−SA

**P3del−SA이(가) 나아진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존−SA 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000290` | +3 | 6 | 4/6 | 6/7 |
| `19700101_001037_T__000590` | +3 | 7 | 5/7 | 7/8 |
| `19700101_001037_T__000330` | +2 | 6 | 5/8 | 5/6 |
| `19700101_001037_T__000415` | +2 | 7 | 4/4 | 6/6 |
| `19700101_001037_T__000419` | +2 | 6 | 4/5 | 6/7 |
| `19700101_001037_T__000465` | +2 | 9 | 6/7 | 7/7 |
| `19700101_001037_T__000550` | +2 | 7 | 6/7 | 7/7 |
| `19700101_001037_T__000570` | +2 | 9 | 6/6 | 8/8 |
| `19700101_001037_T__000580` | +2 | 9 | 6/6 | 8/8 |
| `19700101_001037_T__000591` | +2 | 8 | 6/7 | 7/7 |

**P3del−SA이(가) 나빠진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존−SA 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000285` | -2 | 6 | 6/6 | 4/4 |
| `19700101_001037_T__000355` | -2 | 6 | 6/8 | 5/8 |
| `19700101_001037_T__000360` | -2 | 6 | 5/5 | 4/5 |
| `19700101_001037_T__000215` | -1 | 6 | 1/1 | 0/0 |
| `19700101_001037_T__000255` | -1 | 4 | 1/1 | 0/0 |
| `19700101_001037_T__000280` | -1 | 6 | 5/5 | 4/4 |
| `19700101_001037_T__000294` | -1 | 6 | 6/7 | 5/6 |
| `19700101_001037_T__000295` | -1 | 6 | 6/7 | 5/6 |
| `19700101_001037_T__000310` | -1 | 6 | 5/5 | 5/6 |
| `19700101_001037_T__000315` | -1 | 6 | 5/5 | 5/6 |

### 기존 → 둘 다 제거: 기존 → P3del−SA

**P3del−SA이(가) 나아진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000419` | +4 | 6 | 3/5 | 6/7 |
| `19700101_001037_T__000425` | +3 | 6 | 4/5 | 6/6 |
| `19700101_001037_T__000585` | +3 | 8 | 6/7 | 8/8 |
| `19700101_001037_T__000265` | +2 | 5 | 2/3 | 3/3 |
| `19700101_001037_T__000290` | +2 | 6 | 5/7 | 6/7 |
| `19700101_001037_T__000300` | +2 | 6 | 6/8 | 6/6 |
| `19700101_001037_T__000310` | +2 | 6 | 4/6 | 5/6 |
| `19700101_001037_T__000335` | +2 | 6 | 4/6 | 5/6 |
| `19700101_001037_T__000365` | +2 | 6 | 4/7 | 5/7 |
| `19700101_001037_T__000420` | +2 | 6 | 4/5 | 6/7 |

**P3del−SA이(가) 나빠진 이미지 (상위 10)**

| 이미지 | 차이 | 정답 | 기존 맞춤/예측 | P3del−SA 맞춤/예측 |
|---|---|---|---|---|
| `19700101_001037_T__000355` | -2 | 6 | 6/8 | 5/8 |
| `19700101_001037_T__000280` | -1 | 6 | 6/7 | 4/4 |
| `19700101_001037_T__000285` | -1 | 6 | 6/7 | 4/4 |
| `19700101_001037_T__000295` | -1 | 6 | 6/7 | 5/6 |
| `19700101_001037_T__000315` | -1 | 6 | 5/5 | 5/6 |
| `19700101_001037_T__000320` | -1 | 6 | 6/6 | 6/7 |
| `19700101_001037_T__000325` | -1 | 6 | 6/7 | 5/6 |
| `19700101_001037_T__000373` | -1 | 7 | 6/7 | 5/6 |
| `19700101_001037_T__000375` | -1 | 7 | 6/7 | 5/6 |
| `19700101_001037_T__000685` | -1 | 8 | 8/8 | 7/7 |

## 영상: 기존 vs P3del−SA 나란히 (0816 · AIhub1)

- `videos/0816영상_기존vsP3del-SA.mp4` — 0816 영상 전체(8983프레임, 4분 59초), 353MB
- `videos/AIhub1영상_기존vsP3del-SA.mp4` — `dataset/etc./test_image/aihub_CCTV고속도로/1.mp4` 전체(96프레임, 6.4초, 15fps), 5.9MB
- 3840x1080. 왼쪽 = 기존(ep130), 오른쪽 = P3del−SA(test/008 ep110). 초록 실선 · 빨강 점선
- 각 모델 결과를 원본 해상도에 그린 뒤 1920x1080으로 맞췄음. 평활화 등 추가 후처리 없음(모델 기본 후처리만)
- AIhub 원본은 1920x1080인데 config의 `crop_bbox`가 3840x2160으로 고정이라, 좌표를 원본 크기로 환산해서 그림
- 용량: NVENC(cq23)로 만든 뒤 `ffmpeg libx264 -crf 28 -preset medium`으로 다시 압축(0816 1.4GB → 353MB). 압축 전 파일은 지움

| 영상 | 프레임당 평균 차선 수: 기존 | P3del−SA |
|---|---|---|
| 0816 | 3.95 | 4.20 |
| AIhub1 | 4.24 | 2.91 |

(이 절은 손으로 덧붙임 — `ganet_p3sa_val_compare.py`를 다시 돌리면 README가 새로 써지므로 다시 붙여야 함)

```bash
cd /home/hsjeong/workspace/vision-det/GANet/source
O=/home/hsjeong/workspace/vision-det/GANet/output/08-cls2-P3SA4구조비교-val결과/videos
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_sa_p3_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000816_T.mp4" "$O/_raw_0816영상_기존vsP3del-SA.mp4"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_sa_p3_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1.mp4" "$O/_raw_AIhub1영상_기존vsP3del-SA.mp4"
cd "$O"; for f in _raw_*.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart -an "${f#_raw_}" && rm "$f"
done
```

## 영상: 기존 vs P3del+DA(query) 나란히 (0816 · AIhub1)

- `videos/0816영상_기존vsP3del+DA.mp4` — 0816 영상 전체(8983프레임, 4분 59초, 30fps), 353MB (압축 전 1.4GB)
- `videos/AIhub1영상_기존vsP3del+DA.mp4` — `dataset/etc./test_image/aihub_CCTV고속도로/1.mp4` 전체(96프레임, 6.4초, 15fps), 5.9MB (압축 전 17MB)
- 3840x1080. 왼쪽 = 기존(ep130), 오른쪽 = P3del+DA(P3 제거 + SA→Deformable Attention 교체, query 방식, `test/009` epoch_150). 초록 실선 · 빨강 점선
- 만드는 방법·압축은 위 P3del−SA 영상과 같음 (`ganet_da_video_compare.py` = `ganet_sa_p3_video_compare.py`에서 오른쪽 모델만 바꾼 것, 두 체크포인트 strict 로드 확인)
| 영상 | 프레임당 평균 차선 수: 기존 | P3del+DA | (참고) P3del−SA |
|---|---|---|---|
| 0816 | 3.95 | 4.02 | 4.20 |
| AIhub1 | 4.24 | 4.21 | 2.91 |

- DA 구조 설명·val F1은 `vision-det/GANet/docs/01-model-review/14. SA 대신 Deformable Attention 구조 학습 (DA).md`

```bash
cd /home/hsjeong/workspace/vision-det/GANet/source
O=/home/hsjeong/workspace/vision-det/GANet/output/08-cls2-P3SA4구조비교-val결과/videos
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_da_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000816_T.mp4" "$O/_raw_0816영상_기존vsP3del+DA.mp4"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_da_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1.mp4" "$O/_raw_AIhub1영상_기존vsP3del+DA.mp4"
cd "$O"; for f in _raw_*+DA.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart -an "${f#_raw_}" && rm "$f"
done
```

## 영상: P3del−SA vs P3del+DA(query) 나란히 (0816 · AIhub1)

- `videos/0816영상_P3del-SAvsP3del+DA.mp4` — 0816 영상 전체(8983프레임, 4분 59초, 30fps), 354MB (압축 전 1.4GB)
- `videos/AIhub1영상_P3del-SAvsP3del+DA.mp4` — AIhub `1.mp4` 전체(96프레임, 6.4초, 15fps), 5.9MB (압축 전 17MB)
- 3840x1080. 왼쪽 = P3del−SA(P3·SA 제거, `test/008` epoch_110), 오른쪽 = P3del+DA(P3 제거 + SA→DA 교체, query, `test/009` epoch_150)
- `ganet_nosa_vs_da_video_compare.py` = `ganet_da_video_compare.py`에서 왼쪽 모델만 바꾼 것. 두 config가 같은 기준 config(P3del) 모듈을 고쳐 쓰므로
  모델마다 `projects.cfgs` 캐시를 지우고 읽음(문서 14 6절). 두 체크포인트 strict 로드 확인
- 프레임당 평균 차선 수: 0816 P3del−SA 4.20 / P3del+DA 4.02, AIhub1 2.91 / 4.21 (모델별 단독 영상의 값과 같음)

```bash
cd /home/hsjeong/workspace/vision-det/GANet/source
O=/home/hsjeong/workspace/vision-det/GANet/output/08-cls2-P3SA4구조비교-val결과/videos
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_nosa_vs_da_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000816_T.mp4" "$O/_raw_0816영상_P3del-SAvsP3del+DA.mp4"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
  /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_nosa_vs_da_video_compare.py "/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1.mp4" "$O/_raw_AIhub1영상_P3del-SAvsP3del+DA.mp4"
cd "$O"; for f in _raw_*SAvsP3del+DA.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p -movflags +faststart -an "${f#_raw_}" && rm "$f"
done
```

