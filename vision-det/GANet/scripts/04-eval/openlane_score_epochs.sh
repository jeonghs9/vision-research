#!/usr/bin/env bash
# [OpenLane용] 실험 폴더의 체크포인트들을 openlane_score_checkpoint.py로 채점한다.
#
# 사용: bash openlane_score_epochs.sh <GPU> <실험폴더명> <config 이름(.py 제외)> <val|test> [epoch 목록]
#   - val: epoch 목록을 안 주면 폴더에 있는 epoch_*.pth 전부 → best epoch 고르기용
#   - test: 반드시 val로 고른 epoch 하나만 준다 (test로 모델을 고르지 않는다)
# 결과: /home/hsjeong/workspace/vision-det/GANet/test/<실험>/eval/<split>_epoch<N>.json, <split>_scores.txt (RESULT 줄 모음)
# 예:
#   bash /home/hsjeong/workspace/vision-det/GANet/scripts/04-eval/openlane_score_epochs.sh \
#     5 015-openlane-res34-2cls-1280-P3del-noSA-150ep res34_s16_full_2cls_P3del_noSA val
#   bash /home/hsjeong/workspace/vision-det/GANet/scripts/04-eval/openlane_score_epochs.sh \
#     5 015-openlane-res34-2cls-1280-P3del-noSA-150ep res34_s16_full_2cls_P3del_noSA test 140
set -euo pipefail
GPU="${1:?GPU}"; EXP="${2:?실험 폴더}"; CFG="${3:?config 이름}"; SPLIT="${4:?val|test}"; shift 4
ROOT=/home/hsjeong/workspace/vision-det/GANet
D="$ROOT/test/$EXP"; OUT="$D/eval"; mkdir -p "$OUT"
if [ $# -gt 0 ]; then EPS="$*"; else
  [ "$SPLIT" = test ] && { echo "test는 epoch을 지정해야 합니다 (val로 고른 것 하나)"; exit 1; }
  EPS=$(ls "$D"/epoch_*.pth | sed 's/.*epoch_\([0-9]*\)\.pth/\1/' | sort -n | tr '\n' ' ')
fi
cd "$ROOT/source"
for ep in $EPS; do
  line=$(PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$GPU" /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    "$ROOT/scripts/04-eval/openlane_score_checkpoint.py" "projects/cfgs/openlane/${CFG}.py" "$D/epoch_${ep}.pth" \
    --split "$SPLIT" --out-json "$OUT/${SPLIT}_epoch${ep}.json" 2>/dev/null | grep '^RESULT')
  echo "epoch=${ep} ${line}" | tee -a "$OUT/${SPLIT}_scores.txt"
done
