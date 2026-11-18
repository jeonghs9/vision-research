#!/usr/bin/env bash
# [OpenLane용] OpenLane 2클래스(실선/점선) GANet res34 학습 — 드론 가중치 미사용, 입력 1280x864.
# 문서: docs/02-openlane/ (config: source/projects/cfgs/openlane/*.py)
#
# 사용: bash train_openlane_res34_2cls.sh <GPU> <실험폴더명> <config 이름(.py 제외)>
#   config 이름: res34_s8_full_2cls_noSA | res34_s16_full_2cls_P3del_noSA | res34_s16_full_2cls_P3del
#                res34_s8_full_2cls_DAq  | res34_s16_full_2cls_P3del_DAq | res34_s8_full_2cls(원본 P3+SA)
# 결과(체크포인트·로그): /home/hsjeong/workspace/vision-det/GANet/test/<실험폴더명>/
# 예:
#   bash /home/hsjeong/workspace/vision-det/GANet/scripts/02-train/train_openlane_res34_2cls.sh \
#     0 014-openlane-res34-2cls-1280-noSA-150ep res34_s8_full_2cls_noSA
set -euo pipefail
GPU="${1:?GPU 번호}"; EXP="${2:?실험 폴더명}"; CFG="${3:?config 이름}"; shift 3
cd /home/hsjeong/workspace/vision-det/GANet/source
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python tools/train.py \
  "projects/cfgs/openlane/${CFG}.py" \
  --work-dir "/home/hsjeong/workspace/vision-det/GANet/test/${EXP}" \
  --no-validate --gpu-id 0 "$@"
