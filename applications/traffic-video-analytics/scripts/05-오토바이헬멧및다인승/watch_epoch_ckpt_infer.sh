#!/usr/bin/env bash
# 학습 중 새로 저장되는 epochNN.pt 를 감지해 자동으로 영상 추론·압축한다.
#
# 계획: docs/05-오토바이헬멧및다인승/03. 학습및영상추론.md §7 (1번 항목)
#
# 왜 필요한가
#   Ultralytics 는 best.pt / last.pt 를 계속 덮어쓴다. save_period=10 으로 남는
#   epochNN.pt 도 학습이 끝나면 어느 시점이 좋았는지 되짚기 어렵다.
#   새 체크포인트가 나올 때마다 같은 클립·같은 설정으로 추론해 두면
#   에폭별 헬멧 검출 추이를 나중에 표 하나로 비교할 수 있다.
#
# 동작
#   1. WEIGHTS 폴더를 30초마다 확인
#   2. 새 epochNN.pt 발견 → snapshots/ 로 복사 (학습이 지워도 남기기 위함)
#   3. 추론 (imgsz 2560, person·motorcycle·helmet)
#   4. H.264 압축 후 raw 삭제
#   5. summary.tsv 에 검출 수 누적 기록
#
# 실행:
#   nohup bash /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/watch_epoch_ckpt_infer.sh \
#     > /home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/rider-safety-infer/watch.log 2>&1 &
#
# 중단:
#   pkill -f watch_epoch_ckpt_infer.sh

set -u

RUN=002-helmet-5cls-ep50repro
BASE=/home/hsjeong/workspace/applications/traffic-video-analytics
WEIGHTS="$BASE/test/output/rider-safety/$RUN/weights"
SNAP="$BASE/test/output/rider-safety/snapshots"
OUT="$BASE/output/rider-safety-infer"
# 클립 2개를 모두 돌린다 — 한 구간에만 우연히 잘/못 나오는 경우를 걸러내기 위함
DRONE=/home/hsjeong/workspace/dataset/etc./test_video/drone
VIDEOS=(
    "$DRONE/clip_164259_f3990_1800.mp4:f3990"   # 1,800프레임 30fps · ep189 기준점 있음
    "$DRONE/clip_164259_f6025_1500.mp4:f6025"   # 1,500프레임 25fps · 원본 마지막 1분
)
SCRIPT="$BASE/scripts/99-archive/rider_safety_infer_video.py"
PY=/home/hsjeong/miniconda3/envs/Yolov26_env/bin/python
GPU=${GPU:-4}          # 학습이 0,1,2 와 3~7 을 쓰므로 여유 있는 쪽을 고른다
IMGSZ=${IMGSZ:-2560}
INTERVAL=${INTERVAL:-30}

mkdir -p "$SNAP" "$OUT"
SUMMARY="$OUT/epoch_summary.tsv"
[ -f "$SUMMARY" ] || printf 'epoch\tclip\tframes\tperson\tmotorcycle\thelmet\thelmet_frames\tinfer_min\tfile\n' > "$SUMMARY"

echo "[WATCH] 시작 $(date '+%F %T')"
echo "  대상   : $WEIGHTS/epoch*.pt"
for v in "${VIDEOS[@]}"; do echo "  영상   : $(basename "${v%%:*}")  [tag ${v##*:}]"; done
echo "  설정   : imgsz=$IMGSZ gpu=$GPU classes=person,motorcycle,helmet"
echo "  요약   : $SUMMARY"

while true; do
    # 학습이 끝나고 새 체크포인트도 없으면 종료
    if ! pgrep -f "_temp_.*\.py" > /dev/null 2>&1; then
        shopt -s nullglob
        remaining=0
        for w in "$WEIGHTS"/epoch*.pt; do
            ep=$(basename "$w" .pt); ep=${ep#epoch}
            for v in "${VIDEOS[@]}"; do
                [ -f "$OUT/infer_ep${ep}_${v##*:}.mp4" ] || remaining=1
            done
        done
        shopt -u nullglob
        if [ "$remaining" -eq 0 ]; then
            echo "[WATCH] 학습 종료 · 미처리 체크포인트 없음 → 감시 종료 $(date '+%F %T')"
            exit 0
        fi
    fi

    shopt -s nullglob
    for w in "$WEIGHTS"/epoch*.pt; do
        ep=$(basename "$w" .pt); ep=${ep#epoch}
        # 두 클립이 모두 끝났으면 건너뛴다
        all_done=1
        for v in "${VIDEOS[@]}"; do
            [ -f "$OUT/infer_ep${ep}_${v##*:}.mp4" ] || all_done=0
        done
        [ "$all_done" -eq 1 ] && continue

        # 쓰기 완료 확인 — 크기가 두 번 연속 같아야 안전하다
        s1=$(stat -c%s "$w" 2>/dev/null) || continue
        sleep 5
        s2=$(stat -c%s "$w" 2>/dev/null) || continue
        [ "$s1" != "$s2" ] && continue

        echo "[WATCH] epoch $ep 감지 $(date '+%F %T')"
        snap="$SNAP/${RUN}_epoch${ep}.pt"
        cp "$w" "$snap"          # 학습이 지워도 남긴다

        for v in "${VIDEOS[@]}"; do
            video="${v%%:*}"; tag="${v##*:}"
            done_mp4="$OUT/infer_ep${ep}_${tag}.mp4"
            [ -f "$done_mp4" ] && continue

            log="$OUT/log_ep${ep}_${tag}.txt"
            raw="$OUT/raw_ep${ep}_${tag}.mp4"
            PYTHONNOUSERSITE=1 "$PY" "$SCRIPT" \
                --video "$video" --model "$snap" --out "$raw" \
                --imgsz "$IMGSZ" --classes 0 2 4 --gpu "$GPU" > "$log" 2>&1

            if [ ! -f "$raw" ]; then
                echo "[WATCH] epoch $ep [$tag] 추론 실패 — $log 확인"
                continue
            fi

            ffmpeg -y -i "$raw" -c:v libx264 -preset medium -crf 26 \
                -pix_fmt yuv420p -movflags +faststart -an "$done_mp4" \
                > "$OUT/ff_ep${ep}_${tag}.log" 2>&1 && rm -f "$raw"

            # 요약 기록 — [DONE] 줄과 그 다음 검출 줄에서 뽑는다
            awk -v ep="$ep" -v tag="$tag" -v f="$(basename "$done_mp4")" '
                /^\[DONE\]/ { fr=$3; sub(/프레임/,"",fr); mins=$4; sub(/분/,"",mins); hf=$NF }
                /^  검출:/  { p=$3; m=$5; h=$7 }
                END { gsub(/,/,"",p); gsub(/,/,"",m); gsub(/,/,"",h); gsub(/,/,"",hf)
                      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", ep, tag, fr, p, m, h, hf, mins, f }
            ' "$log" >> "$SUMMARY"

            echo "[WATCH] epoch $ep [$tag] 완료 → $(basename "$done_mp4")"
            tail -1 "$SUMMARY"
        done
    done
    shopt -u nullglob
    sleep "$INTERVAL"
done
