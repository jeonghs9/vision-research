# scripts/ 실행 가이드

**이 파일에 없는 스크립트는 지금 쓸 일 없는 코드입니다**
(실험용·구버전은 `99-archive/`에 있거나 각 폴더 안에 남아있어도 무시하세요).
새 스크립트를 쓰기 시작하거나 역할이 바뀌면 이 파일부터 갱신합니다
(규칙: 프로젝트 루트 `CLAUDE.md` 참고).

**2026-11-03 주제별 재구성**: 기존 `01-data/02-pipeline/03-train/04-inference/05-eval`
(스테이지 축)을 "어떤 위반·어떤 작업 대상인지"(주제 축)로 다시 묶었다. `docs/`, `output/`도
같은 축으로 번호를 맞췄다(03, 06은 `docs/03-migration`, `docs/06-portfolio`용으로 비워둠).

---

## 01-차선검출모델 — 차선 검출 자체 (GANet·YOLO-pose, 라벨링·학습)

### 차선 라벨링 → YOLO-pose 변환
1. `labeling_polyline.py` — 차선 폴리라인 라벨링 GUI (PyQt6)
   ```bash
   python scripts/01-차선검출모델/labeling_polyline.py
   ```
2. `polyline_to_yolo_pose.py` — 라벨을 YOLO-pose 형식으로 변환
   ```bash
   PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
     scripts/01-차선검출모델/polyline_to_yolo_pose.py \
     --src <폴리라인 라벨 폴더> --kpt 12
   ```

### 차선만 보고 싶을 때 (침범판정 없이 차선 추적 품질만 확인)
`lane_pose_track.py` — YOLO-pose 차선 검출+추적 영상 생성
```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5 \
  /home/hsjeong/miniconda3/envs/Yolov26_env/bin/python \
  scripts/01-차선검출모델/lane_pose_track.py \
  --weights <pose모델> --video <영상> --out <결과영상> --conf 0.30 --ego
```
`lane_pose_track_roadmask.py` — 위에 도로 세그 마스크 필터 추가(도로 밖 오검출 제거)판
`lane_pose_vis.py` — pose val셋 시각화(점이 아니라 선으로 이어 그림)

GANet 아키텍처 단계별(FPN/confidence/offset) 분석과 미탐 원인 추적은
`vision-det/GANet/output/05-cls2-점선미탐원인분석/`과 관련 분석 문서 참고.

---

## 02-실선위반판정 — 검출된 차선으로 실선 침범 판정

### 실선 침범 판정 — 배포용
`visualize_solid_lane_crossing_v5.3.1.py` — seg 모델 기반, **검증된 배포 버전**
```bash
python scripts/02-실선위반판정/visualize_solid_lane_crossing_v5.3.1.py \
  --det-model models/det-best.pt --lane-model models/seg-best.pt \
  --input <영상> --output <결과영상> --csv <결과csv> --device 0
```

### 실선 침범 판정 — 실험 중 (YOLO-pose/GANet 비교용, 아직 배포 아님)
`visualize_solid_lane_crossing_polyline.py` — 위와 로직 동일, 차선 백엔드만 교체 가능
```bash
python scripts/02-실선위반판정/visualize_solid_lane_crossing_polyline.py \
  --det-model models/det-best.pt --lane-model models/pose-best.pt \
  --input <영상> --output <결과영상> --csv <결과csv> --device 0
```

### 실선 침범 판정 — 밑변 선분 vs 실선 기하 교차 (GANet/YOLO-pose 검증용, production 아님)
`crossing_trigger_bottomline.py` — 차량 박스 밑변 선분과 실선 폴리라인의 기하학적
교차로 침범을 판정(`--hold-seconds` 유지돼야 확정). ratio-jump 안전장치·번호판 OCR
포함. 상세 설계는 `docs/02-실선위반판정/02~04. (선분교차_침범트리거~교차비율게이지)` 참고.
```bash
PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/paddleocr_gpu_cu126/bin/python \
  scripts/02-실선위반판정/crossing_trigger_bottomline.py \
  --det-model models/det-best.pt --lane-model models/pose-best.pt \
  --input <영상> --out-csv <결과csv> --out-video <결과영상> --device 0
# GANet 백엔드로 바꾸려면 --lane-model 대신 --lane-source cache --lane-cache <jsonl캐시>
```
`crossing_trigger_bottomline_test.py` — 위 스크립트의 다음 실험용 스테이징 사본
(아직 원본에 승격 전인 변경을 먼저 검증하는 용도, 완성본 아님). 실행 방법 동일.

---

## 04-오토바이다인승 — 다인승만 판정 (헬멧 없음)

`detect_motorcycle_multi_rider.py` — 4클래스 모델, **2인 이상 탑승만** 봄
```bash
python scripts/04-오토바이다인승/detect_motorcycle_multi_rider.py \
  --det-model models/det-best.pt --input <영상> --output <결과영상> --device 1
```

`audit_rider_merge.py` — 다인승 중복제거가 실수로 합친 박스 쌍 감사(비정기적 사용)
`ioa_vs_iou_slide.py`, `title_before_after.py` — 포트폴리오용 다인승 비교 그림 재생성
(`docs/06-portfolio/01. 포트폴리오_그림.md` 참고, 비정기적 사용)

---

## 05-오토바이헬멧및다인승 — 헬멧 미착용 + 다인승 동시 판정 (6클래스 통합)

`rider_violation_6cls.py` — 6클래스(D모델), 둘 다 잡음. `detect_motorcycle_multi_rider.py`보다
상위호환(다인승도 잡고 헬멧도 잡음) — 둘 다 필요하면 이거 하나만 쓰면 됨
```bash
python scripts/05-오토바이헬멧및다인승/rider_violation_6cls.py \
  --video <영상> --out <결과영상> --gpu 0
```

### 데이터 파이프라인 — 01~05 순서대로 실행
`rider_safety_01_build_helmet_pool.py` → `02_merge_labels.py` → `03_split.py`
→ `04_vis_labels.py` → `05_drone_split.py`
```bash
python scripts/05-오토바이헬멧및다인승/rider_safety_01_build_helmet_pool.py \
  --raw_dir <roboflow원본> --out_dir <01단계출력>
python scripts/05-오토바이헬멧및다인승/rider_safety_02_merge_labels.py \
  --helmet_dir <01출력> --sam3_dir <SAM3자동라벨> --out_dir <02단계출력>
python scripts/05-오토바이헬멧및다인승/rider_safety_03_split.py \
  --merged_dir <02출력> --eval_source <드론유사뷰 데이터셋>
python scripts/05-오토바이헬멧및다인승/rider_safety_04_vis_labels.py \
  --images <이미지> --labels <라벨> --out <검수용시각화> --n 100
python scripts/05-오토바이헬멧및다인승/rider_safety_05_drone_split.py \
  --images <드론프레임> --labels <6클래스라벨> --out <최종split>
```
(새로 라벨링할 일 없으면 안 돌림 — 이미 만들어진 `splits/002-drone-6cls/` 사용 중)

### 학습/벤치마크
`run_rider_ABC.sh` — rider-safety 6클래스 A/B/C 실험(병합vs파인튜닝vs스크래치) 자동 실행
```bash
nohup bash scripts/05-오토바이헬멧및다인승/run_rider_ABC.sh > test/output/rider-safety/run_ABC.log 2>&1 &
```
`rider_safety_bench_imgsz.py` — 해상도별 추론 속도(디코딩~인코딩 구간별 분해)
```bash
python scripts/05-오토바이헬멧및다인승/rider_safety_bench_imgsz.py \
  --video <영상> --gpu 0 --frames 300
```

### 학습 중 체크포인트 자동 추론
`watch_epoch_ckpt_infer.sh` — 새 epoch 체크포인트 나올 때마다 자동 추론+기록
```bash
nohup bash scripts/05-오토바이헬멧및다인승/watch_epoch_ckpt_infer.sh \
  > test/tmp/rider-safety-infer/watch.log 2>&1 &
```

---

## 99-archive — 더 이상 쓰지 않는 코드

하위에 기존 `02-pipeline/`, `03-experiment/`, `05-bench/` 서브폴더가 있다(스테이지
재구성 이전부터 쓰던 archive 전용 번호라 위 주제 번호와는 무관).
`rider_safety_infer_video.py`(구버전 5클래스 헬멧풀만 추론, D모델 6클래스로 대체됨),
`dataset_prepare_vehicle_cls_split.py`(차량분류 데이터 분할 — 차선·라이더 어느
주제에도 속하지 않아 전용 폴더를 만들지 않고 여기 둠, 다시 쓸 일이 생기면 새 주제
폴더로 뺄 것)가 최상위에 있다.
