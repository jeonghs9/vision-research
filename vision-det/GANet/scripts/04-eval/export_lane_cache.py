#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] GANet 추론 결과를 JSONL 캐시로 저장한다 (영상당 1회, `ganet_cu101` 환경).

목적: GANet(`ganet_cu101`: torch1.8/CUDA10.1/Python3.8)과
traffic-video-analytics 침범판정 파이프라인(`paddleocr_gpu_cu126`:
torch2.6/CUDA12.6/Python3.12)은 같은 프로세스에 못 들어간다(바이너리
비호환, 실측 확인). 두 프로세스를 연결하는 유일한 방법은 GANet 결과를
먼저 파일로 뽑아두고 다른 프로세스가 그 파일을 읽는 것 — 이 스크립트가
그 파일(캐시)을 만든다. `infer_video_2cls.py`의 모델 로딩(`build_model`)·
전처리(`preprocess`)·추론 3단계 패턴을 그대로 재사용하되, 그리기/NVENC
인코딩은 전부 뺐다(캐시 생성은 실시간성이 필요 없어 단순 동기 루프로 충분).

좌표계: `result["final_dict_list"]`의 `lanes`는 `PostProcessor`가 이미
원본 영상 해상도로 역변환해서 반환한다(`infer_video_2cls.py`가 그대로
증명) — 캐시도 원본 해상도 좌표 그대로 저장한다. 캐시를 만든 영상과
나중에 캐시를 읽는 영상은 반드시 같은 파일(같은 해상도·프레임수)이어야
한다.

점 개수: GANet 한 차선의 점 개수는 가변인데 소비 측(`LaneTracker`)의
매칭 거리 계산은 두 폴리라인의 점 개수가 같아야 하므로, 호 길이 기준
균등 재샘플링으로 12개(YOLO-pose와 동일 관례)로 통일해서 저장한다.

클래스 변환: GANet은 1=실선/2=점선(1-indexed) -> 캐시는 0=실선/1=점선으로
저장(소비 쪽이 GANet 관례를 몰라도 되게).

신뢰도: GANet `final_dict_list`는 차선 단위 스칼라 신뢰도를 직접 주지
않으므로 캐시의 `conf`는 1.0 고정.

캐시 포맷 (JSONL, 한 줄 = 한 프레임):
  {"frame": 0, "lanes": [{"cls": 0, "points": [[x1,y1],...,[x12,y12]]}, ...]}

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/04-eval/export_lane_cache.py \
    projects/cfgs/drone/res34_s8_full_2cls.py \
    /home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth \
    "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000157_T.mp4" \
    --out-cache /home/hsjeong/workspace/applications/traffic-video-analytics/test/tmp/polyline-crossing-tuning/19700101_000157_T_res34-2cls.jsonl \
    --start-sec 105 --end-sec 165 \
    --gpu-id 0
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "./")

import cv2
import numpy as np
import torch
import mmcv

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa


def build_model(cfg_path, ckpt_path, gpu_id):
    cfg = mmcv.Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    if "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = model.cuda(gpu_id)
    model.eval()
    return model, cfg


def preprocess(frame_bgr, img_scale_wh, mean, std):
    resized = cv2.resize(frame_bgr, img_scale_wh, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normed = (rgb - mean) / std
    chw = normed.transpose(2, 0, 1)
    return torch.from_numpy(chw).float()


def resample_polyline(points, n=12):
    """호 길이 기준 균등 재샘플링으로 점 개수를 n개로 통일한다.

    points: GANet이 반환한 가변 길이 [[x,y], ...] (대체로 y 오름차순/내림차순
    한 방향으로 정렬돼 있음). 점이 1개뿐이면 n개로 복제, 2개 이상이면
    각 구간 거리를 누적해서 np.interp로 등간격 보간한다.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("video")
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=12)
    ap.add_argument("--start-sec", type=float, default=None,
                     help="이 시각(초)부터 캐시 생성 (생략시 영상 처음부터)")
    ap.add_argument("--end-sec", type=float, default=None,
                     help="이 시각(초)까지 캐시 생성 (생략시 영상 끝까지)")
    a = ap.parse_args()

    model, cfg = build_model(a.config, a.checkpoint, a.gpu_id)
    img_scale_wh = tuple(cfg.img_scale)  # (W, H)
    mean = np.array(cfg.img_norm_cfg["mean"], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg["std"], dtype=np.float32)

    cap = cv2.VideoCapture(a.video)
    assert cap.isOpened(), f"cannot open {a.video}"
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(round(a.start_sec * src_fps)) if a.start_sec is not None else 0
    end_frame = int(round(a.end_sec * src_fps)) if a.end_sec is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    meta_base = dict(
        img_shape=(img_scale_wh[1], img_scale_wh[0], 3),
        ori_shape=(h, w, 3),
        offset_x=0,
        offset_y=0,
    )

    os.makedirs(os.path.dirname(a.out_cache) or ".", exist_ok=True)

    idx = start_frame
    n_lanes_total = 0
    gpu = a.gpu_id
    t0 = time.time()

    with open(a.out_cache, "w") as f_out, torch.no_grad():
        while end_frame is None or idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame, img_scale_wh, mean, std).unsqueeze(0).cuda(gpu)

            meta = dict(meta_base)
            meta["filename"] = meta["ori_filename"] = f"frame_{idx:06d}.jpg"

            feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
            pred = model.bbox_head.forward_test(feats)
            result = model.post_processing(pred, meta)

            lanes, _, _, classes = result["final_dict_list"]
            n_lanes_total += len(lanes)

            record_lanes = []
            for lane, cls in zip(lanes, classes):
                resampled = resample_polyline(lane, n=a.n_points)
                record_lanes.append({
                    "cls": int(cls) - 1,  # GANet 1/2(실선/점선) -> 캐시 0/1
                    "points": np.round(resampled, 1).tolist(),
                })

            f_out.write(json.dumps({
                "frame": idx - start_frame,
                "lanes": record_lanes,
            }) + "\n")

            idx += 1
            if (idx - start_frame) % 100 == 0:
                elapsed = time.time() - t0
                print(f"frame {idx - start_frame} done, {elapsed:.1f}s elapsed "
                      f"({(idx - start_frame) / elapsed:.2f} fps)", flush=True)

    cap.release()
    n_done = idx - start_frame
    elapsed = time.time() - t0
    print(f"done cache={a.out_cache} frames={n_done} lanes_total={n_lanes_total} "
          f"elapsed_sec={elapsed:.1f} fps={n_done / elapsed:.2f}")


if __name__ == "__main__":
    main()
