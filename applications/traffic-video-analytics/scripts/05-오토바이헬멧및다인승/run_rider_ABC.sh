#!/usr/bin/env bash
# 실험 A·B·C 자동 실행. OOM 이면 batch 를 줄여 재시도한다.
#
# 계획: docs/05-오토바이헬멧및다인승/06. 드론영상_자체라벨링.md
#
# 전제: SAM3 라벨링과 영상 단위 분할이 이미 끝나 있어야 한다.
#   splits/002-drone-6cls/  train 4,505 / val 1,504 / test 342
#
# 실험 설계 — 같은 val/test(영상 단위 홀드아웃)로 세 전략을 비교한다
#   A  COCO + 드론 단순 병합, 스크래치        GPU 0-3, 60 epochs
#   B  det-best.pt 파인튜닝, 드론 단독        GPU 4-7, 150 epochs, lr 낮춤
#   C  드론 단독 스크래치                     GPU 4-7 (B 완료 후), 150 epochs
#
#   A vs B : 병합이 나은가, 파인튜닝이 나은가
#   B vs C : COCO 사전학습이 실제로 도움이 되는가
#   A vs C : 데이터 구성만의 효과
#
# 실행:
#   nohup bash /home/hsjeong/workspace/applications/traffic-video-analytics/scripts/05-오토바이헬멧및다인승/run_rider_ABC.sh \
#     > /home/hsjeong/workspace/applications/traffic-video-analytics/test/output/rider-safety/run_ABC.log 2>&1 &
#
# 중단: pkill -f run_rider_ABC && pkill -f "yolo detect train" && pkill -f _temp_

set -u

BASE=/home/hsjeong/workspace/applications/traffic-video-analytics
OUT="$BASE/test/output/rider-safety"
DATA=/home/hsjeong/workspace/Yolo26/DATA-YAML
SPLIT=/home/hsjeong/workspace/dataset/det/rider-safety/splits/002-drone-6cls
MODEL_YAML=/home/hsjeong/workspace/Yolo26/ultralytics/ultralytics/cfg/models/26/yolo26s_STAR-GATE_ver1.0.1.yaml
PRETRAIN="$BASE/models/det-best.pt"
YOLO=/home/hsjeong/miniconda3/envs/Yolov26_env/bin/yolo
BATCHES="40 32 24 16 8"     # OOM 시 이 순서로 줄여가며 재시도

mkdir -p "$OUT"
cd /home/hsjeong/workspace/Yolo26/ultralytics || exit 1

# 전제 확인 — 분할이 없으면 학습할 수 없다
for sp in train val test; do
    n=$(find "$SPLIT/images/$sp" -name '*.jpg' 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && { echo "[ERROR] $SPLIT/images/$sp 가 비어 있다"; exit 1; }
    echo "[INFO] $sp $n 장"
done

# ── OOM 재시도 학습 ───────────────────────────────────────────────────
# DDP 에서 batch 는 GPU 수로 나누어떨어져야 한다 (trainer.py:323 이 // 로 나눈다).
# 나누어떨어지지 않으면 조용히 내림되어 의도와 다른 배치로 학습된다.
train_retry() {
    local name=$1 model=$2 data=$3 epochs=$4 device=$5 lr0=$6 lrf=$7 warm=$8 pat=$9
    local log="$OUT/train_${name%%-*}.log"
    local ngpu; ngpu=$(echo "$device" | tr ',' '\n' | wc -l)

    for b in $BATCHES; do
        [ $((b % ngpu)) -ne 0 ] && continue
        echo "[$(date '+%F %T')] $name 시작 (batch=$b, GPU $device)"
        PYTHONNOUSERSITE=1 $YOLO detect train \
            model="$model" data="$data" \
            epochs="$epochs" imgsz=1024 batch="$b" device="$device" \
            lr0="$lr0" lrf="$lrf" momentum=0.9267 weight_decay=0.001315 \
            warmup_epochs="$warm" warmup_momentum=0.6599 warmup_bias_lr=0.2229 \
            workers=8 seed=0 patience="$pat" save_period=10 \
            project="$OUT" name="$name" exist_ok=True \
            > "$log" 2>&1
        local rc=$?

        if grep -qiE "out of memory|OutOfMemoryError|CUDA out of memory" "$log"; then
            echo "[$(date '+%F %T')] $name OOM (batch=$b) → 축소 재시도"
            pkill -f "_temp_" 2>/dev/null
            sleep 30                                  # GPU 메모리 반환 대기
            rm -rf "${OUT:?}/$name"                   # 실패한 실행 디렉터리 정리
            continue
        fi
        if [ $rc -eq 0 ]; then
            echo "[$(date '+%F %T')] $name 완료 (batch=$b)"
            echo "$b" > "$OUT/$name.batch"
            return 0
        fi
        echo "[$(date '+%F %T')] $name 실패 (exit $rc, OOM 아님) — $log 확인"
        return $rc
    done
    echo "[ERROR] $name — batch 8 까지 시도했으나 OOM 지속"
    return 1
}

# ── A · B 동시 시작 ───────────────────────────────────────────────────
train_retry "A-merged-6cls"    "$MODEL_YAML" "$DATA/data_rider_A_merged_6cls.yaml" \
            60  "0,1,2,3" 0.009906 0.1101 0.4234 20 &
A_PID=$!

train_retry "B-finetune-drone" "$PRETRAIN"   "$DATA/data_rider_BC_drone_6cls.yaml" \
            150 "4,5,6,7" 0.001    0.01   1.0    30 &
B_PID=$!

# ── B 완료 후 C 시작 (같은 GPU 재사용) ────────────────────────────────
wait $B_PID
echo "[$(date '+%F %T')] B 종료 → C 시작"
train_retry "C-scratch-drone"  "$MODEL_YAML" "$DATA/data_rider_BC_drone_6cls.yaml" \
            150 "4,5,6,7" 0.009906 0.1101 0.4234 30 &
C_PID=$!

wait $A_PID $C_PID
echo "[$(date '+%F %T')] A·B·C 전부 종료"

# ── 공통 test 로 세 모델 평가 ─────────────────────────────────────────
echo "[$(date '+%F %T')] test 평가 (164800, 342장, conf 0.25)"
for e in A-merged-6cls B-finetune-drone C-scratch-drone; do
    w="$OUT/$e/weights/best.pt"
    [ -f "$w" ] || { echo "  $e: best.pt 없음 — 건너뜀"; continue; }
    echo "===== $e  (batch $(cat "$OUT/$e.batch" 2>/dev/null || echo '?'))"
    PYTHONNOUSERSITE=1 GITHUB_ACTIONS=1 $YOLO detect val model="$w" \
        data="$DATA/data_rider_BC_drone_6cls.yaml" split=test \
        imgsz=1024 batch=8 device=0 conf=0.25 workers=4 \
        project="$OUT/eval" name="$e" exist_ok=True plots=False 2>&1 \
        | grep -E "^ *(all|person|car|motorcycle|plate_number|helmet|head) "
done
echo "[$(date '+%F %T')] 완료"
