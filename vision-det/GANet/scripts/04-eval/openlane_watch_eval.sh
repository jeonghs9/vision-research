#!/usr/bin/env bash
# [OpenLane용] 학습 중인 실험들의 새 체크포인트(epoch_N.pth)를 감시하다가 나오는 대로 val 채점한다.
# 한 GPU에서 맡은 실험들을 순서대로 처리한다 (채점끼리 GPU를 다투지 않게).
#
# 사용: bash openlane_watch_eval.sh <채점 GPU> <실험폴더명>:<config 이름> [<실험폴더명>:<config 이름> ...]
# 출력: 각 실험 test/<실험>/eval/val_scores.txt (+ JSON), 진행 로그는 표준출력 (EVAL / DEAD / DONE 줄)
# 종료: 맡은 실험이 모두 epoch_150 채점까지 끝나거나, 학습 프로세스가 없어지고 남은 체크포인트도 다 채점했을 때
set -uo pipefail
GPU="${1:?GPU}"; shift
ROOT=/home/hsjeong/workspace/vision-det/GANet
declare -A DONE_ALL
while true; do
  all_done=1
  for pair in "$@"; do
    EXP="${pair%%:*}"; CFG="${pair##*:}"; D="$ROOT/test/$EXP"
    [ -n "${DONE_ALL[$EXP]:-}" ] && continue
    all_done=0
    for ck in $(ls "$D"/epoch_*.pth 2>/dev/null | sort -V); do
      ep=$(basename "$ck" .pth); ep=${ep#epoch_}
      [ -f "$D/eval/val_epoch${ep}.json" ] && continue
      # 저장 중인 파일 피하기: 60초 이상 변경 없을 때만
      [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -lt 60 ] && continue
      echo "$(date '+%m-%d %H:%M') EVAL start $EXP epoch $ep (GPU $GPU)"
      bash "$ROOT/scripts/04-eval/openlane_score_epochs.sh" "$GPU" "$EXP" "$CFG" val "$ep" | sed "s/^/$(date '+%m-%d %H:%M') EVAL $EXP /"
      [ -f "$D/eval/val_epoch${ep}.json" ] || echo "$(date '+%m-%d %H:%M') EVAL FAIL $EXP epoch $ep"
    done
    if [ -f "$D/eval/val_epoch150.json" ]; then DONE_ALL[$EXP]=1; echo "$(date '+%m-%d %H:%M') DONE $EXP (val 채점 15개 끝)"; continue; fi
    if ! pgrep -f "work-dir $D" > /dev/null; then
      left=0; for ck in "$D"/epoch_*.pth; do [ -e "$ck" ] || continue; e=$(basename "$ck" .pth); [ -f "$D/eval/val_epoch${e#epoch_}.json" ] || left=$((left+1)); done
      if [ "$left" -eq 0 ]; then DONE_ALL[$EXP]=1; echo "$(date '+%m-%d %H:%M') DEAD $EXP 학습 프로세스 없음 (epoch_150 전), 남은 체크포인트 채점 완료"; fi
    fi
  done
  [ $all_done -eq 1 ] && break
  sleep 60
done
echo "$(date '+%m-%d %H:%M') WATCH END GPU $GPU"
