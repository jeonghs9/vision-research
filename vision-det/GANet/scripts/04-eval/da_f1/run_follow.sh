#!/bin/bash
# 학습을 따라가며 epoch_N.pth 가 생기면 바로 val F1 측정. $1=gpu $2=config $3=expdir $4=log
cd /home/hsjeong/workspace/vision-det/GANet/source
for e in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150; do
  until [ -f $3/epoch_$e.pth ]; do sleep 60; done
  sleep 30  # 저장이 끝날 때까지
  echo "=== epoch $e" >> $4
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$1 /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    ../scripts/04-eval/score_checkpoint.py $2 $3/epoch_$e.pth 2>&1 | grep -v "^\[" | tail -4 >> $4
done
echo DONE >> $4
