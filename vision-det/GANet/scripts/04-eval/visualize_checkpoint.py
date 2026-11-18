#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] 체크포인트 하나로 고정된 val 이미지 10장에 대해 추론하고 결과를 그려서 저장한다.

GANet의 GANet.show_result()/vis_one()이 이미 예측 차선을 원본 이미지에 그려주는
기능을 갖고 있어서(PIL로 선을 그리고, img_metas의 gt_points_for_show가 있으면
정답도 같이 그림) 그걸 그대로 쓴다 — mmdet 표준 tools/test.py --show-dir 와
같은 경로, 다만 이미지 10장 고정 리스트 + 임의의 출력 폴더를 지정할 수 있게
얇게 감쌌다.

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/04-eval/visualize_checkpoint.py \
    projects/cfgs/drone/res18_s8_full.py \
    test/001-res18-1cls-1280-150ep/epoch_10.pth \
    --list /home/hsjeong/workspace/dataset/etc./si_vietnam_drone/culane_lane_ganet/list/demo10.txt \
    --out-dir output/02-cls1-에폭별val이미지확인/res18/ep10
"""
import argparse
import os
import sys

sys.path.insert(0, "./")

import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.utils import build_dp, compat_cfg, get_device
from projects.plugin import *  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--list", required=True, help="추론할 이미지 목록(list/*.txt)")
    ap.add_argument("--out-dir", required=True, help="결과 이미지를 저장할 폴더")
    ap.add_argument("--gpu-id", type=int, default=0)
    a = ap.parse_args()

    cfg = mmcv.Config.fromfile(a.config)
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [a.gpu_id]
    cfg.device = get_device()

    # 10장 고정 리스트로 교체 (그 외 test 설정은 config 그대로)
    cfg.data.test.data_list = [a.list]
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    dataset.open_check()
    dataset.set_all_scenes()
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None  # 체크포인트로 덮어쓸 거라 사전학습 재다운로드 불필요
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model.CLASSES = getattr(dataset, "CLASSES", None)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    os.makedirs(a.out_dir, exist_ok=True)
    single_gpu_test(model, data_loader, show=False, out_dir=a.out_dir)
    print(f"저장 완료: {a.out_dir}")


if __name__ == "__main__":
    main()
