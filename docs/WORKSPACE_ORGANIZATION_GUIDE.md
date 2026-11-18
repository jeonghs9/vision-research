# 연구 워크스페이스 정리 가이드

이 문서는 `/home/hsjeong/workspace`의 연구 코드, 데이터셋, 실험 결과와 문서를 일관된 방식으로 관리하기 위한 기준이다.

핵심 원칙은 **연구 분야 아래에 프로젝트를 두고, 프로젝트 하나가 원본 코드부터 최종 결과까지 완결된 구조를 갖게 하는 것**이다.

> 이 문서는 정리 기준만 정의한다. 기존 폴더를 자동으로 이동하거나 이름을 변경하지 않는다.

## 1. 권장 최상위 구조

```text
workspace/
├── vision-det/             # 객체 검출, 소형 객체 검출
├── vision-seg/             # 세그멘테이션, SAM, 차선 분할
├── vision-ocr/             # 번호판 및 문자 인식
├── model-opt/              # 양자화, 경량화, 모델 압축
├── datasets/               # 여러 프로젝트가 공유하는 데이터셋
├── docs/                   # 워크스페이스 공용 문서
└── PROJECTS.md             # 전체 연구 프로젝트 인덱스
```

### 디렉터리 역할

| 디렉터리 | 역할 | 프로젝트 예시 |
|---|---|---|
| `vision-det/` | detection 계열 연구 | `small-object-detection`, `yolo11`, `yolo26` |
| `vision-seg/` | segmentation 계열 연구 | `sam3`, `road-lane-segmentation` |
| `vision-ocr/` | OCR 계열 연구 | `vehicle-plate-ocr` |
| `model-opt/` | 모델 최적화 연구 | `mqbench-qat`, `ptq`, `lora` |
| `datasets/` | 공유 데이터셋 원본과 전처리본 | `aitod`, `visdrone`, `aihub` |
| `docs/` | 프로젝트에 속하지 않는 공용 문서 | 운영 가이드, 연구 인덱스 |

최상위 분야명은 필요할 때만 추가한다. 프로젝트 하나를 위해 분야 폴더를 계속 늘리지 않는다.

## 2. 프로젝트 이름

프로젝트 폴더는 시간이 지나도 연구 목적을 기억할 수 있도록 설명적으로 작성한다. 프로젝트는 가장 가까운 연구 분야 아래에 둔다.

```text
vision-det/
└── small-object-detection/

vision-seg/
├── sam3/
└── road-lane-segmentation/

vision-ocr/
└── vehicle-plate-ocr/

model-opt/
└── yolo-quantization/
```

### 이름 규칙

- 영문 소문자와 하이픈(`-`)을 사용한다.
- 기반 기술 자체를 관리하는 프로젝트는 `sam3`, `yolo26`처럼 기술명을 사용해도 된다.
- 특정 연구 문제는 `small-object-detection`, `vehicle-plate-ocr`처럼 목적을 이름에 담는다.
- `test`, `new`, `final`, `project1`처럼 맥락이 없는 이름은 사용하지 않는다.
- 하나의 폴더는 하나의 연구 질문 또는 명확한 목표를 나타낸다.

좋은 예:

```text
small-object-detection
vehicle-detection-and-plate-ocr
road-lane-segmentation
```

피해야 할 예:

```text
YOLO_PROJECT
final_test
new_model_2
3차모델
```

### 2.1 Conda 가상환경 이름

가상환경 이름은 프로젝트와 CUDA 버전을 바로 알 수 있게 다음 형식을 사용한다.

```text
<프로젝트명>_cu<CUDA 버전>
```

- 프로젝트명은 영문 소문자와 숫자를 사용한다.
- CUDA 버전에서 점(`.`)은 생략한다. 예: CUDA 12.6 → `cu126`
- 같은 프로젝트에 환경이 하나라면 Python 버전은 이름에 넣지 않고 프로젝트 `README.md`에 기록한다.
- CUDA를 사용하지 않는 환경은 `<프로젝트명>_cpu`를 사용한다.

예:

```text
videomt_cu126
segformer_cu111
videomt_cpu
```

VidEoMT의 표준 Conda 가상환경 이름은 `videomt_cu126`이다.

이 규칙은 새로 만드는 환경에 우선 적용한다. 기존 환경은 editable 설치, 실행 스크립트와 절대경로 의존성을 확인하기 전에 이름을 바꾸지 않는다.

## 3. 프로젝트 내부 구조

**2026-09-16부터 새로 만드는 프로젝트는 아래 골격을 쓴다.** 이 문서 맨 위 원칙대로
기존 프로젝트(SAM3 등)를 자동으로 옮기지는 않는다 — 옛 골격은 3.4절에 남겨둔다.

```text
vision-det/YOLinO/
├── README.md
├── source/             # clone한 원본 Git 저장소, 원형 유지
├── docs/                # 프로젝트 문서 — 폴더별 정리, 파일명 규칙은 3.0
├── scripts/             # 직접 작성한 실행·변환·평가 스크립트
├── test/                # 스크립트를 돌렸을 때 임시로 결과를 보기 위한 공간
│   ├── models/          # 실험별 학습 체크포인트 (실험 폴더명과 대응, 원본은 복사)
│   ├── tmp/             # 정식 실험이 아닌 디버그·스모크 테스트
│   └── log/             # 체크포인트 없는 평가 전용 실행의 로그
├── models/              # test 로 비교한 것 중 현재 채택한 모델만 (한 개~소수)
└── output/              # 비교 영상 등 보고서에 쓸 만한 결과. 용도별 하위 폴더 + 설명
```

| 경로 | 내용 |
|---|---|
| `README.md` | 프로젝트 목적, 설치법, 주요 스크립트, 대표 결과 |
| `source/` | clone한 원본 Git 저장소. `.git`도 이 안에 둔다 |
| `docs/` | 실험 기록, 검토 문서, 논문/모델 요약 등 — 3.0 |
| `scripts/` | 학습, 평가, 변환, 시각화 실행 스크립트 — 3.0 |
| `test/` | 실험 단위 임시 결과. 자유롭게 지우거나 다시 만들 수 있다 |
| `test/models/` | 실험별 학습 체크포인트. `test/<실험명>/` 과 이름을 맞춘다. **복사본** — 이동시키지 않는다 |
| `test/tmp/` | 정식 번호 실험이 아닌 디버그·스모크 테스트·실패한 시도. 이름 규칙 없음, 언제든 삭제 |
| `test/log/` | 체크포인트를 만들지 않는 평가 전용 실행의 로그·수치 결과 |
| `models/` | test 로 비교한 것 중 **현재 채택한 모델만**. 더 나은 모델이 나오면 교체한다 |
| `output/` | 비교 영상·표·그림처럼 보고서·발표에 바로 쓸 결과물 |

`config/`, `assets/`, `archive/`, `results/` 는 새 골격에 없다.

- 실행 설정(`params.yaml` 등)은 별도 폴더 없이 **해당 스크립트나 `test/<실험>/` 안에 같이 둔다.**
  `--config` 처럼 경로를 직접 지정할 수 있는 도구라면 이 방식이 자연스럽다.
- README용 작은 이미지·표는 `docs/` 하위 폴더 안에 함께 둔다.
- 옛 버전을 따로 모아두는 `archive/`는 두지 않는다 — `test/`에 실험 단위로 이미
  남아 있으므로, 지우지 않는 한 그 자체가 이력이 된다.
- 채택 결과를 한데 모으던 `results/`는 성격이 다른 `models/`(모델 자체)와
  `output/`(보고서용 산출물)로 나뉘었다.

### 3.0 `docs/`와 `scripts/`는 파일이 4개를 넘으면 역할별 폴더로 나눈다

평평하게 쌓으면 개수가 늘수록 무엇이 어디 있는지 파일명만으로 추측해야 한다.
**파일이 4개를 넘으면** 아래 형식의 하위 폴더로 나눈다.

```text
NN-역할이름/          # 두 자리 번호 + 하이픈 + 영문 소문자 역할명
```

**`docs/` 예시**

```text
docs/
├── README.md                  # 인덱스. 폴더별 담는 것과 문서 목록
├── 01-lane-postprocess/       # 후처리 알고리즘 사양·검증
├── 02-model-selection/        # 모델 비교와 선정 근거
└── 03-migration/              # 이관·정리 기록
```

**`docs/` 파일명 규칙 (신규 골격, 2026-09-16~)** — 각 `NN-역할이름/` 폴더 **안에서
번호를 새로 매기고, 번호 뒤는 한글로** 쓴다.

```text
docs/
├── README.md
├── 01-model-review/
│   ├── 1. YOLinO 핵심 아이디어 요약.md
│   └── 2. 데이터셋 적용 가능성 검토.md
└── 02-integration/
    └── 1. 커스텀 데이터 로더 설계.md
```

번호는 그 폴더 안에서 읽는 순서다. 폴더를 옮기더라도 번호를 포함한 파일명은 그대로 둔다.

**`scripts/` 예시** — 학습 프로젝트는 대체로 이 흐름을 따른다.

```text
scripts/
├── 01-data/          # 데이터 준비·변환·검증·시각화
├── 02-train/         # 학습
├── 03-inference/     # 추론 — 결과물을 만드는 것
├── 04-eval/          # 정확도 평가
└── 05-bench/         # 속도 측정 — 실험이지 결과물이 아니다
```

프로젝트 성격에 따라 이름은 달라진다. 예를 들어 라벨링 파이프라인이라면
`01-data / 02-labeling / 03-vis` 가 맞다. **필요한 단계만 두고 번호는 연속으로** 매긴다
(쓰지 않는 단계를 위해 번호를 비워두지 않는다).

**`04-eval` 과 `05-bench` 를 나누는 이유** — 성격이 다르다. 평가는 "이 모델이 얼마나
정확한가", 벤치마크는 "어디가 느린가" 를 본다. 한쪽만 있으면 한쪽만 만든다.

#### 공통 규칙

- **번호는 작업 흐름 순서**로 매긴다. 알파벳순이 아니라 "먼저 읽어야 할 것"이 앞이다.
- **역할명은 담기는 파일의 성격**이지 주제가 아니다. `01-lane` 이 아니라
  `01-lane-postprocess`, `02-model-selection` 처럼 **무엇을 하는지**가 드러나야 한다.
- **`docs/`에는 `README.md`가 필수다.** 폴더별로 무엇을 담는지 한 줄씩 적어, 다음에
  문서를 만들 때 어디 넣을지 고민하지 않게 한다.
- **파일명은 바꾸지 않는다.** 폴더로 옮기기만 한다. 파일명을 함께 바꾸면 다른 문서·
  스크립트에 박힌 참조가 조용히 깨진다.

#### 옮기기 전에 반드시 참조를 검색한다

```bash
grep -rl "<파일명>" --include="*.md" --include="*.py" --include="*.yaml" --include="*.sh" .
```

경로가 박힌 곳을 찾아 함께 고친다. 이 과정을 건너뛰면 링크가 깨진 것을 한참 뒤에 알게 된다.

**실행 중이거나 예약된 작업이 있으면 먼저 끝내고 옮긴다.** 백그라운드로 대기 중인
스크립트가 옛 경로를 참조하고 있으면, 나중에 깨어날 때 조용히 실패한다.

옮긴 뒤에는 문서에 적힌 경로가 실제로 존재하는지 확인한다.

```bash
grep -rhoE "[A-Za-z0-9_-]+/scripts/[0-9A-Za-z_/.-]+\.(py|sh)" --include="*.md" . \
  | sort -u | while read ref; do [ -f "$ref" ] || echo "❌ $ref"; done
```

### 3.1 `source` 원본과 사용자 코드를 분리한다

`source/`는 외부 Git 저장소의 원형을 최대한 유지한다. 직접 사용하는 파이썬 파일과 실행 셸은 프로젝트 루트의 `scripts/`에 둔다.

```text
vision-seg/sam3/
├── source/
│   ├── .git/
│   ├── sam3/
│   ├── pyproject.toml
│   └── README.md
└── scripts/
    ├── batch-multiclass-yolo.py
    └── batch-solid-white-lane.py
```

원본 코드 자체를 수정해야 한다면 수정 내역을 Git commit으로 남긴다. 원본 저장소 안에 임시 스크립트나 결과 이미지를 추가하지 않는다.

### 3.2 `test` / `models` / `output` 을 성격으로 구분한다 (신규 골격)

```text
test/                        # 실험 단위 임시 결과 — 자유롭게 지우거나 다시 만든다
├── 001-<핵심변경점>/         # 정식 실험 하나 (config, metrics, logs, figures)
├── 002-<핵심변경점>/
├── models/                  # 실험별 체크포인트. test/<실험명>/ 과 이름을 맞춤
│   ├── 001-<핵심변경점>/best.pt
│   └── 002-<핵심변경점>/best.pt
├── tmp/                     # 정식 실험이 아닌 디버그·스모크 테스트·실패한 시도
└── log/                     # 체크포인트 없이 로그·수치만 남기는 평가 전용 실행

models/                      # test 로 비교한 것 중 현재 채택한 모델만
└── <목적>-<방법>-v<버전>.pt

output/                      # 보고서·발표에 바로 쓸 결과물
├── README.md                # 하위 폴더 목록과 각각 무엇인지 한 줄 설명
├── <용도-설명>/
└── <용도-설명>/
```

- `test/`는 통째로 `.gitignore`에 등록한다. epoch 마다의 중간 산출물도 여기 머문다.
- `test/models/`는 **비교를 위해** 여러 실험의 체크포인트를 모아두는 곳이다.
  최종 채택 전까지는 여기서만 관리한다. **체크포인트(.pt/.pth) 파일은 반드시 여기로
  복사(copy)해서 남긴다 — 이동(move)해서 원본 `test/<실험명>/`쪽 사본을 없애지 않는다.**
  실수로 `test/models/`를 지워도 `test/<실험명>/`에 원본이 남아있어야 한다.
- `test/tmp/`는 정식 번호 실험이 아닌 디버그·스모크 테스트·중간에 실패한 시도를 모은다.
  이름 규칙이 없고 언제든 지워도 된다 — `test/` 바로 아래에 흩어놓지 않고 여기로 모은다.
- `test/log/`는 체크포인트를 만들지 않는 실행(기존 체크포인트로 다시 돌린 평가 등)의
  로그·수치 결과를 남긴다. 학습(체크포인트 생성)은 `test/<실험명>/`에, 평가 전용
  실행은 `test/log/`에 둔다.
- `models/`는 **지금 쓰는 모델만** 남긴다 — 여러 버전을 쌓아두는 곳이 아니다.
  더 나은 모델이 나오면 이전 것을 교체한다(옛 버전은 `test/models/`에 이미 남아 있다).
  모델 카드 또는 간단한 설명을 함께 남긴다.
- `output/`는 사람이 읽을 결과물(비교 영상, 표, 그림)만 담는다. 원시 로그나 중간
  체크포인트는 넣지 않는다 — 그런 건 `test/`다. 하위 폴더는 `docs/`처럼 **용도가
  드러나는 이름 + README 한 줄 설명**을 갖춘다.
- 대용량 원본 데이터셋은 `test/`나 `output/`에 복사하지 않는다.

### 3.3 YOLinO 적용 예시 (신규 골격)

```text
vision-det/YOLinO/
├── README.md
├── source/                  # KIT-MRT/YOLinO Git 저장소, 원형 유지
├── docs/
│   ├── README.md
│   └── 01-model-review/
│       ├── 1. YOLinO 핵심 아이디어 요약.md
│       └── 2. si_vietnam_drone 적용 가능성 검토.md
├── scripts/
│   ├── 01-data/              # 원본 폴리라인 → YOLinO 라벨 변환
│   ├── 02-train/
│   └── 03-inference/
├── test/
│   ├── 001-train-60ep/          # 정식 실험 (raw output)
│   ├── models/                  # 001-train-60ep/best_model.pth 등 (복사본)
│   ├── tmp/                     # 디버그·스모크 테스트·실패한 시도
│   └── log/                     # 체크포인트 없는 평가 전용 실행 로그
├── models/
└── output/
    └── README.md
```

### 3.4 이전 골격 (기존 프로젝트만 해당, 새 프로젝트에는 쓰지 않음)

SAM3 등 2026-09-16 이전에 만든 프로젝트는 `config/` · `assets/` · `archive/` 와
`test/{cases,output}` + `results/{models,exports,metrics,figures,samples}` 골격을 그대로 쓴다.
자동으로 옮기지 않는다(문서 맨 위 원칙).

```text
vision-seg/sam3/                     # 예시 — 이전 골격
├── README.md
├── source/                         # 현재 SAM3/sam3 Git 저장소
├── scripts/
│   ├── batch-multiclass-yolo.py   # 현재 sam3_batch_multiclass_yolo.py
│   └── batch-solid-white-lane.py  # 현재 sam3_batch_solid_white_lane.py
├── config/
├── docs/
├── assets/
├── test/
│   ├── cases/                      # 보존, 테스트 코드·fixture
│   └── output/                     # 재생성 가능, .gitignore
│       ├── images/
│       └── videos/
├── results/                        # 검토 후 채택한 결과
│   ├── models/
│   ├── figures/
│   └── samples/
└── archive/
```

`source/` 내부에 이미 존재하는 upstream `assets/`, `examples/`, `scripts/`는 원본 저장소 구성으로 유지한다. 프로젝트 루트의 `scripts/`와 `assets/`는 사용자가 직접 관리하는 파일만 담는다.

## 4. 실험 결과 이름

> 신규 골격(3.2)에서는 모든 실험을 `test/` 안에서 수행한다 — 보존용으로 옮기는 별도
> 폴더가 없다. 그중 채택한 모델 하나만 `models/`로 승격한다(9절). 이전 골격은
> `results/experiments/`를 썼다(3.4).

이름은 `순번-핵심변경점` 형식을 사용한다.

```text
test/                     # 신규 골격. 이전 골격은 results/experiments/
├── 001-yolo11-baseline/
├── 002-piou/
├── 003-piou-c3k2/
├── 004-star-backbone/
└── 005-star-lora/
```

날짜, GPU 번호, seed, 전체 데이터셋 이름과 긴 하이퍼파라미터는 폴더명에 넣지 않는다. 해당 정보는 `config.yaml`에 기록한다.

### 실험 내부 구조

신규 골격에서는 체크포인트(`weights/`)를 실험 폴더가 아니라 같은 이름의
`test/models/003-piou-c3k2/`에 둔다(3.2) — 실험 폴더는 가볍게 유지된다.

```text
test/003-piou-c3k2/
├── config.yaml
├── metrics.csv
├── figures/
└── logs/

test/models/003-piou-c3k2/
├── best.pt
└── last.pt
```

학습과 검증은 별도 실험으로 나누지 않는다.

```text
# 피해야 할 형태
003-piou-train/
004-piou-val/

# 권장 형태
003-piou/
├── train/
└── eval/
```

## 5. `config.yaml` 최소 기록 항목

```yaml
project: vision-det/small-object-detection
experiment: 003-piou-c3k2
status: complete

model: yolo11s
task: detection
dataset: visdrone
image_size: 1024
seed: 42

method:
  loss: piou
  backbone: c3k2

environment: SOD-PAPER
code_revision: "<git commit>"
started_at: 2026-08-06
notes: "PIoU와 C3k2 조합 비교"
```

## 6. 프로젝트 `README.md` 템플릿

```markdown
# Small Object Detection

AI-TOD, VisDrone, DOTA에서 소형 객체 검출 성능을 개선하는 연구.

## 목표

- APs 개선
- 작은 객체의 localization 오류 감소

## 구성

- Base model: YOLO11s
- Methods: PIoU, C3k2, STAR backbone
- Datasets: AI-TOD, VisDrone, DOTA
- Metrics: APs, mAP50-95
- Environment: SOD-PAPER

## 현재 상태

- Status: active
- Best run: `005-star-lora`
- Next: DOTA 교차 데이터셋 평가
```

## 7. 루트 `PROJECTS.md` 템플릿

`PROJECTS.md`는 전체 연구의 목차 역할을 한다.

```markdown
# Research Projects

| 경로 | 목적 | 상태 | 대표 결과 | 환경 |
|---|---|---|---|---|
| vision-det/small-object-detection | 소형 객체 AP 개선 | active | 005-star-lora | SOD-PAPER |
| vision-ocr/vehicle-plate-ocr | 차량 검출 및 번호판 OCR | active | 018-flow-association | paddleocr_gpu_cu126 |
| vision-seg/road-lane-segmentation | 도로 차선 분할 | paused | 007-segformer | segformer_cu111 |
| model-opt/yolo-quantization | PTQ/QAT 및 경량화 | active | 012-mqbench-qat | mqbench_env |
```

상태값은 아래 네 가지로 통일한다.

```text
idea       아직 시작하지 않음
active     진행 중
paused     일시 중단
complete   종료
```

## 8. 데이터셋 구조

공유 데이터셋은 프로젝트 내부에 복사하지 않고 `datasets/`에서 한 번만 관리한다.

```text
datasets/
├── aitod/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── visdrone/
│   ├── raw/
│   ├── processed/
│   └── splits/
└── aihub/
    ├── vehicle/
    ├── plate/
    └── lane/
```

- `raw/`: 수정하지 않는 원본
- `processed/`: 포맷 변환, 정제, 타일링 결과
- `splits/`: train/val/test 목록과 YAML

프로젝트에서는 데이터셋을 복사하지 않고 설정 파일의 경로 또는 심볼릭 링크로 참조한다.

### 8.1 분할 결과에는 반드시 `vis/`를 만든다

**train/val/test 분할을 만들 때마다 `vis/`를 함께 생성한다. 선택이 아니라 필수다.**

```text
splits/<분할이름>/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── vis/                        # test 무작위 30장, 라벨을 원본 위에 시각화
└── SPLIT.md                    # 분할 근거와 통계
```

| 항목 | 규칙 |
|---|---|
| 대상 | **test** split |
| 장수 | **30장** |
| 선택 | 무작위 (**seed 고정** — 재실행 시 같은 장이 나와야 비교가 된다) |
| 파일명 | 원본 stem 그대로 (원본과 대조 가능하게) |

**패널 구성**은 전처리 유무로 갈린다.

```text
전처리 없음 : 이미지 | 라벨 오버레이                         (2분할)
전처리 있음 : 원본(변환 전) | 변환 후 | 라벨 오버레이         (3분할)
```

BEV(IPM), 타일링, 크롭처럼 **모습이 크게 바뀌는 전처리**를 했다면 3분할이라야 변환이
맞았는지 판단할 수 있다. 변환 후 이미지만 보면 "원본에 있던 차선이 제대로 넘어왔는지"를
알 수 없다. 각 패널 위에 제목과 클래스별 픽셀 수를 적는다.

원본과 변환본의 **종횡비가 다르면 폭이 아니라 높이를 맞춰** 나란히 놓는다
(예: 원본 1920x1200 종횡비 1.6 vs BEV 512x768 종횡비 0.67 — 폭을 맞추면 한쪽이 뭉개진다).

**왜 필수인가.** 라벨 변환은 조용히 틀린다. 좌표계가 뒤집히거나(예: `image_size`가
`[height, width]` 순서), 클래스 ID가 밀리거나, 폴리라인이 잘못 래스터화돼도 파일은 정상적으로
만들어지고 학습도 돌아간다. 몇 시간을 태운 뒤 성능이 이상해서야 알게 된다.
**30장만 눈으로 보면 30초 만에 잡힌다.**

라벨이 폴리라인·폴리곤처럼 **원본과 다른 형태**라면 변환 결과를 보여줘야 한다.
GT 패널에 보이는 것이 원본 라벨이 아니라 **변환된 마스크**임을 문서에 명시한다.

**선이 얇은 데이터(차선, 균열, 전선 등)는 축소 시 사라진다.** 시각화용으로 팽창시켰다면
"라벨 N px 굵게 표시"처럼 이미지 안에 반드시 적는다. 안 그러면 실제 라벨 두께로 오해한다.

### 8.2 `SPLIT.md` 최소 기록 항목

```markdown
# split: subset-holdout

- 생성: 2026-08-07  /  스크립트: scripts/make_split.py
- 원본: datasets/aihub/lane/raw
- 분할 단위: **클립**(연속 프레임 묶음) — 프레임 단위 분할은 누수

| split | 클립 | 장수 | 전경% |
|---|---:|---:|---:|
| train | 1,419 | 18,094 | 3.73 |
| val | 352 | 4,461 | 3.65 |
| test | 594 | 6,423 | 3.58 |

## 분할 근거
- test = 특정 서브셋 통째로 홀드아웃 (다른 주행이라 누수 없음)
- 서브셋 내부 클립 간 ID 거리 118~453 vs 서브셋 간 7,768~43,260

## 검증
- split 간 공유 클립 0
- 이미지/라벨 stem 불일치 0
- 라벨 고유값 {0,1,2}, dtype uint8
- vis/ 30장 육안 확인 완료
```

### 8.3 영상에서 뽑은 프레임은 반드시 클립 단위로 나눈다

주행 영상·CCTV처럼 **연속 프레임**을 담은 데이터셋은 인접 프레임이 사실상 같은 장면이다
(실측: 인접 프레임의 화소 평균 절대차가 무작위 쌍의 1/3). 프레임 단위로 무작위 분할하면
같은 장면이 train과 test에 동시에 들어가 **성능이 부풀려진다.**

- 파일명이 연속 정수면 그 구간을 하나의 클립으로 보고 **클립째 배정**한다
- 가능하면 한 단계 더 나아가 **촬영 세션(폴더) 단위로 test를 홀드아웃**한다
- 분할 후 **split 간 공유 클립이 0인지 assert**로 확인한다

## 9. 모델 보관 규칙

모든 학습 체크포인트는 실험 중엔 `test/models/<실험명>/`에 둔다(3.2). 여러 실험을
비교한 뒤 **현재 채택한 모델 하나만** 프로젝트의 `models/`로 승격한다. 새 모델이
더 나으면 `models/`의 내용을 교체한다 — 옛 버전은 `test/models/`에 이미 남아 있으므로
따로 보관할 필요가 없다.

```text
vision-det/small-object-detection/models/
└── sod-piou-v1.pt          # 현재 채택 모델. 다음 승격 때 교체됨
```

(이전 골격은 `results/models/`에 여러 버전을 함께 두었다. 3.4)

최종 모델 파일명은 다음 형식을 권장한다.

```text
<목적>-<방법 또는 모델>-v<버전>.<확장자>
```

## 10. 현재 워크스페이스의 배치와 예상 매핑

아래 표에서 `이전 완료` 항목은 실제 경로 이동과 실행 검증이 끝난 상태다. `검토` 항목은 정리 방향을 설명하기 위한 제안이며 실제 이동 전 경로 의존성을 점검해야 한다.

| 상태 | 기존 또는 현재 경로 | 권장 또는 적용 위치 |
|---|---|---|
| 이전 완료 | `SAM3` | `vision-seg/sam3` |
| 이전 완료 | `SAM3/sam3` Git 저장소 | `vision-seg/sam3/source` |
| 이전 완료 | `YOLO-MASTER` | `vision-det/YOLO-MASTER` |
| 이전 완료 | `YOLO26-MASTER` | `vision-det/YOLO26-MASTER` |
| 이전 완료 | `Yolo11` | `vision-det/Yolo11` |
| 검토 | `SOD-PAPER` | `vision-det/small-object-detection` |
| 검토 | `Yolo26/ultralytics/PROJECT/OCR-CAR-CATEGORY` | `vision-ocr/vehicle-plate-ocr` |
| 검토 | `Yolo26/ultralytics/SWEEP` | 관련 프로젝트의 `test/sweeps` |
| 검토 | `Yolo26/ultralytics/UTIL` | 관련 프로젝트의 `scripts` |
| 검토 | `vision-det/YOLO26-MASTER`, `Yolo26` | `vision-det/yolo26` 구조로 통합 |
| 검토 | `vision-det/YOLO-MASTER`, `vision-det/Yolo11` | 역할을 확인한 뒤 `vision-det/yolo11` 구조로 통합 |
| 검토 | `dataset` | `datasets` |
| 이전 완료 | `md` | `docs` |

### 10.1 Ultralytics 공용 실행 결과 경로

현재 사용자 공용 Ultralytics 설정의 기본 실행 결과 경로는 다음과 같다.

```text
/tmp/Ultralytics/runs
```

`project`를 생략하면 task별로 다음과 같은 경로가 사용된다.

```text
/tmp/Ultralytics/runs/detect/<experiment-name>
/tmp/Ultralytics/runs/segment/<experiment-name>
/tmp/Ultralytics/runs/classify/<experiment-name>
```

이 경로는 자유롭게 지울 수 있는 임시 실행 결과용이다. `/tmp`는 재부팅이나 시스템 정리로 삭제될 수 있으므로 채택한 체크포인트, 지표, 표와 그림은 프로젝트의 `results/`로 승격한다. 보존이 필요한 학습에서는 코드나 CLI의 `project`를 영구 경로로 명시한다.

## 11. 적용 순서

기존 경로를 한 번에 변경하면 데이터 YAML, 학습 스크립트, 체크포인트와 로그에 저장된 절대경로가 깨질 수 있다. 다음 순서로 점진적으로 적용한다.

1. `PROJECTS.md`를 만들고 현재 연구를 분야별로 등록한다.
2. 새 프로젝트부터 표준 골격을 적용한다.
3. `SAM3`처럼 작고 구조가 분명한 프로젝트 하나로 먼저 시험한다.
4. Git 원본을 `source/`, 사용자 코드를 `scripts/`로 분리한다.
5. 기존 경로에는 임시 심볼릭 링크를 두어 호환성을 유지한다.
6. 데이터셋의 원본과 복사본을 구분하고 마지막에 이동한다.
7. 이동 후 학습·평가 스크립트의 smoke test를 실행한다.
8. 문제가 없을 때 기존 호환 링크를 제거한다.

## 12. 최종 체크리스트

- 프로젝트 이름만 보고 연구 목적을 이해할 수 있는가?
- 각 프로젝트에 최신 `README.md`가 있는가?
- Git 원본 저장소가 프로젝트의 `source/`에 분리되어 있는가?
- 직접 작성한 실행 코드가 `scripts/`에 모여 있는가?
- (신규 골격) 실험은 `test/`, 채택한 모델은 `models/`, 보고서용 산출물은 `output/`에 있는가?
  (이전 골격 프로젝트는 `test/output/` / `results/` 기준을 그대로 쓴다 — 3.4)
- 실험이 `001-...`, `002-...` 순서로 정리되어 있는가?
- `final`, `new`, `test2`, `train10` 같은 이름이 남아 있지 않은가?
- 데이터셋 원본이 여러 프로젝트에 복사되어 있지 않은가?
- `models/`에 **지금 채택한 모델만** 있고, 비교용 나머지는 `test/models/`에 있는가?
- (신규 골격) `docs/` 파일명이 `NN-역할이름/` 폴더 안에서 `1. 한글설명.md` 형식인가?
- `PROJECTS.md`에서 현재 연구 상태를 한눈에 볼 수 있는가?
- **모든 분할에 `vis/` 30장과 `SPLIT.md`가 있는가? (8.1~8.2절)**
- **영상 프레임 데이터를 클립 단위로 나눴는가? (8.3절)**
