#!/usr/bin/env bash
# [공통] GANet 학습·평가의 DataLoader worker 프로세스가 CPU 0,40 두 개에만 묶이는 문제(2026-11-18 발견) 우회.
# 원인 코드는 못 찾음(파이썬 코드엔 affinity 설정 없음, 메인 프로세스는 0-79 정상, worker만 0,40).
# tools/train.py / openlane_score_checkpoint.py 프로세스(모든 스레드)의 affinity를 30초마다 전체 CPU로 되돌린다.
# epoch마다 worker가 새로 생겨도 다시 풀어 준다. 학습이 다 끝나면 수동 종료(pkill -f fix_worker_affinity).
# 사용: setsid nohup bash fix_worker_affinity.sh > /dev/null 2>&1 &
ALL="0-$(( $(nproc --all) - 1 ))"
while true; do
  for p in $(pgrep -f "tools/train.py|openlane_score_checkpoint.py|score_checkpoint.py"); do
    cur=$(taskset -cp "$p" 2>/dev/null | awk -F': ' '{print $2}')
    [ -n "$cur" ] && [ "$cur" != "$ALL" ] && taskset -a -cp "$ALL" "$p" > /dev/null 2>&1
  done
  sleep 30
done
