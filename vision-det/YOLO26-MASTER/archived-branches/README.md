# 정리한 YOLO26 실험 브랜치 보관 (2026-11-18)

`vision-research` 저장소에서 아래 브랜치 3개와 stash 1개를 지우기 전에 내용을 이 폴더에 남겼다.
세 브랜치 모두 2026-05-27 `c06157b`(원본 복구)에서 갈라졌고 `main`에 합쳐진 적이 없다.
당시 경로는 워크스페이스 맨 위 `YOLO26-MASTER/`였고, 2026-08-10에 `vision-det/YOLO26-MASTER/`로 옮겨졌다.

| 브랜치 | 마지막 commit | 내용 | 어디로 옮겼나 |
|---|---|---|---|
| `0-moe_loss` | `4ab4253` 2026-06-09 | MoE aux loss 별도 슬롯 + cosine decay(0.3→0.05), batch_size 버그 수정, `compute_ap` 표준(pre-8.4.50) 식으로 복원, `yolo26-master-s-noE2E-reg16.yaml`, 결과 문서 | 코드: `0-moe_loss-code.patch` (적용 안 함) / 문서: `../YOLO26-MASTER_result.md` |
| `1-residual` | `6ea63c2` 2026-06-04 | MoE 에 shared expert + `_residual_aggregation` (0-moe_loss 의 loss 변경 일부 포함) | 코드: `1-residual-code.patch` (적용 안 함) |
| `YOLO26-QUANTIZATION-DOCS` | `e3e6b7e` 2026-07-06 | 0-moe_loss 전체 + YOLO26 StarGate CPU INT8 양자화 문서 9개 | 문서: `Yolo26/docs/quantization/` |
| stash `fcf3d37` | 2026-07-14 | 결과 문서 +101줄, 양자화 문서 09 +87줄 | 두 문서에 반영함 |

- `YOLO26-MASTER_result.md` 는 stash 판(가장 최신)이다.
- 코드 변경은 실험용 변형이라 `main` 코드에는 넣지 않았다. 다시 쓰려면 `main` 기준으로:
  ```bash
  cd /home/hsjeong/workspace
  git apply vision-det/YOLO26-MASTER/archived-branches/0-moe_loss-code.patch
  ```
- `runs/YOLO26/260526_*` · `260601_*_Weight-residual` 가중치는 shared expert 가 들어 있어
  `1-residual-code.patch` 를 적용한 코드로만 올바르게 동작한다 (main 코드로 불러도 에러 없이 틀린 결과가 날 수 있음).
- `260604_*_noE2E-reg16` 의 모델 설정 `yolo26-master-s-noE2E-reg16.yaml` 은 `0-moe_loss-code.patch` 안에 있다.
- 결과 문서의 mAP 는 `0-moe_loss` 의 표준 `compute_ap` 로 잰 값이다. main(8.4.50 strict)로 재면 conf=0.25 에서 약 0.1 낮다.
- 브랜치를 이력째 되살리려면 (`yolo26-branches.bundle`, 필요한 기준 commit 은 `main` 에 있음).
  되살린 브랜치는 2026-08-10 폴더 이동 이전 구조라 **checkout 하지 말고** 위 patch 로 쓸 것
  (checkout 하면 맨 위 `YOLO26-MASTER/` 가 생기고 옛 `.gitignore` 가 적용됨):
  ```bash
  cd /home/hsjeong/workspace
  git fetch vision-det/YOLO26-MASTER/archived-branches/yolo26-branches.bundle \
    'refs/heads/*:refs/heads/*' 'refs/archive/*:refs/archive/*'
  ```
