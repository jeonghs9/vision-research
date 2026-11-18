# GANet scripts — 데이터셋별 구분

파일명은 바꾸지 않고, 아래 표와 각 스크립트 맨 위 설명(docstring)의 `[드론용]` / `[OpenLane용]` 표기로 구분한다.
config는 `source/projects/cfgs/drone/` (드론) · `source/projects/cfgs/openlane/` (OpenLane)로 폴더가 나뉘어 있다.

| 스크립트 | 대상 | 설명 |
|---|---|---|
| `01-data/polyline_to_culane.py` | **드론** | si_vietnam_drone 폴리라인 → CULane 형식 |
| `01-data/openlane_make_split.py` | **OpenLane** | segment 단위 분할 목록(15프레임 간격) |
| `01-data/openlane_vis_split.py` | **OpenLane** | test 30장 vis (category별 / 2클래스) |
| `01-data/openlane_to_culane.py` | **OpenLane** | uv 라벨 → CULane 형식(2클래스) |
| `01-data/openlane_vis_converted.py` | **OpenLane** | 변환 라벨 확인 vis |
| `01-data/openlane_export_splits.py` | **OpenLane** | split별 이미지 복사 + 변환 라벨 저장 |
| `02-train/train_openlane_res34_2cls.sh` | **OpenLane** | OpenLane res34 2클래스 학습 |
| `04-eval/openlane_score_checkpoint.py` | **OpenLane** | 체크포인트 1개 val/test 채점 (위치 F1 + 실선·점선별 F1, 20px@1920) |
| `04-eval/openlane_score_epochs.sh` | **OpenLane** | 여러 epoch 일괄 채점 → `test/<실험>/eval/` |
| `04-eval/openlane_watch_eval.sh` | **OpenLane** | 학습 중 새 체크포인트를 자동 val 채점 |
| `02-train/fix_worker_affinity.sh` | 공통 | DataLoader worker CPU 묶임(0,40) 우회 — 학습 중 백그라운드 실행 |
| `04-eval/score_checkpoint.py` | **드론** | 드론 val 163장 P/R (3840px 기준 임계값) |
| `04-eval/visualize_checkpoint.py` | **드론** | 드론 val 10장 추론 vis |
| `04-eval/export_lane_cache.py` | **드론** | 영상 추론 결과 JSONL 캐시 |
| `04-eval/infer_video_2cls.py` | **드론** | 영상 전체 차선 검출 |
| `04-eval/offset_xy_heatmap.py` | **드론** | offset head 히트맵 |


## 분석·그림 스크립트 (2026-11-18 `/home/hsjeong/tmp`에서 옮김, 파일명 그대로)

| 위치 | 대상 | 설명 |
|---|---|---|
| `03-vis/ganet_*.py` (53개) | **드론** | 문서 1~16과 `output/04~08` 그림·영상 생성. 각 파일 맨 위 docstring에 실행 명령 |
| `03-vis/pptx/build.js` | **드론** | 그림 24 → 편집 가능한 pptx 2장. 처음 한 번 `npm install` |
| `04-eval/all_metrics_table.py` | **드론** | 003~013 성능표(문서 16). `test/<실험>/eval/` 로그를 읽음 |
| `04-eval/{da_f1,apt_f1}/run_follow.sh`, `04-eval/nosa_f1/run.sh` | **드론** | 학습을 따라가며 epoch별 val F1 → `test/<실험>/eval/<번호>.log` |
| `04-eval/apt_analysis/` | **드론** | APT τ 표(entropy_stats·split), 동치 검증(smoke_apt), 속도(bench) — 문서 15 |
| `04-eval/da_analysis/` | **드론** | DA 변형·위치 인코딩 비율 분석 — 문서 14 |
| `04-eval/check_c4_same.py`, `check_heatmap_rows_same.py` | **드론** | 특징맵·히트맵 동일성 확인 |
| `05-bench/all_speed_bench.py` | **드론** | 11개 모델 params·속도(문서 16) |
| `05-bench/params_speed.py` | **드론** | SA 제거 실험 params·속도(문서 13) |
