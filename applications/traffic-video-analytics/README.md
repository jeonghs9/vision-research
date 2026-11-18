# Traffic Video Analytics

Traffic-video pipeline for vehicle and plate detection, lane instance segmentation, tracking, solid-lane crossing detection, and optional plate OCR.

## Current release

The verified deployment entry point is:

```text
scripts/02-실선위반판정/visualize_solid_lane_crossing_v5.3.1.py
```

This standalone release was migrated from the legacy project and revalidated after its OCR cache was moved into the application model area.

A second entry point covers motorcycle multi-rider (2+ occupants) detection:

```text
scripts/04-오토바이다인승/detect_motorcycle_multi_rider.py
```

It reuses the same 4-class detection model, needs no PaddleOCR, and runs in either
`Yolov26_env` or `paddleocr_gpu_cu126`. See `docs/04-오토바이다인승/` (multi-rider only)
and `docs/05-오토바이헬멧및다인승/` (combined helmet + multi-rider).

## Runtime dependencies

- YOLO runtime: `/home/hsjeong/workspace/Yolo26/ultralytics` (`ultralytics 8.4.6` editable install)
- Integrated runtime environment: `paddleocr_gpu_cu126`
- Detection and lane models: configured in `configs/runtime.yaml`
- PaddleOCR is optional when `--no-ocr` is supplied

`vision-det/YOLO26-MASTER` is not a dependency of this application.

## Layout

```text
src/       reusable modules for future extraction from the standalone release
scripts/   executable entry points and application-specific data tools
configs/   runtime paths and application defaults
docs/      design, experiment, and release records
tests/     unit, integration, and small test fixtures (pytest code; currently empty, see docs/03-migration)
test/      experiment scratch space — training runs, sweeps, raw logs; excluded from Git
deploy/    deployment environment and service files
models/    local weights and engines; excluded from Git
output/    curated, report-ready results (videos, figures, metrics); excluded from Git
```

`tests/` (pytest 코드)와 `test/` (실험 스크래치)는 이름이 비슷하지만 용도가 다르다 —
자세한 정리 배경은 `docs/03-migration/` 참고.

The v5.3.1 implementation remains standalone during the first migration stage. Splitting it into `src/` modules is deferred until behavior-preserving integration tests are in place.

PaddleOCR runtime caches are stored under `models/runtime-cache/ocr/` and are excluded from Git. Set `TRAFFIC_VIDEO_OCR_CACHE_DIR` only when an alternate cache location is required.
