#!/usr/bin/env python3
"""[드론용 · si_vietnam_drone] 2클래스(실선/점선) 체크포인트로 영상 전체에 차선 검출을 적용하고,
파이프라인 단계별(전처리/모델 추론/후처리/그리기/인코딩) 처리 속도를
측정한다.

입력 프레임(원본 해상도, 예: 3840x2160)은 학습 때와 동일하게 config의
img_scale(1280x736)로 리사이즈해서 모델에 넣고, 예측 좌표는 원본 해상도로
역변환해서 영상 위에 그린다(`PostProcessor.crop_bbox`가 원본 전체 프레임을
가리키므로 비율 변환만 하면 됨 — val 평가/시각화와 동일한 경로).

색상은 val 시각화(`ganet.py`의 `CLASS_COLORS`)와 동일: 실선=초록, 점선=빨강
(cv2가 프레임을 BGR로 다루므로 같은 BGR 튜플을 그대로 재사용).

`GANet.forward_test`(backbone+neck -> head forward+디코딩 -> 후처리) 내부를
그대로 재현하되 각 구간을 직접 호출해서 시간을 잰다 — 동작은 원본과 동일,
타이밍만 세분화. GPU 비동기 실행 때문에 구간 경계마다 `torch.cuda.synchronize()`로
동기화한 뒤 시간을 끊는다(안 하면 이전 구간 작업이 다음 구간 시간에 섞여
들어간다).

**인코딩은 GPU 하드웨어 인코더(NVENC, `ffmpeg -c:v h264_nvenc`)로 별도
스레드/프로세스에서 비동기로 수행한다** — 메인 스레드(읽기/전처리/추론/
후처리/그리기)가 다음 프레임을 처리하는 동안 이전 프레임의 인코딩이 동시에
진행된다(큐로 연결, producer-consumer 구조). 이전 버전(`cv2.VideoWriter`
mp4v, 완전 순차 실행)과 비교해서 실측했을 때 전체 처리시간이 크게
줄어든다(분석: docs 참고). 인코더 스레드가 못 따라오면 큐가 가득 차서
`writer.write()`가 블로킹되는데, 그 대기시간을 `encode_backpressure`
단계로 따로 집계한다 — 0에 가까울수록 인코딩이 병목에서 완전히 빠졌다는
뜻.

실행:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python \
    scripts/04-eval/infer_video_2cls.py \
    projects/cfgs/drone/res34_s8_full_2cls.py \
    /home/hsjeong/workspace/vision-det/GANet/models/drone-ganet-res34-2cls-v1.pth \
    "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000157_T.mp4" \
    --out-video output/03-cls2-영상결과확인/res34/19700101_000157_T_res34-2cls.mp4 \
    --out-report output/03-cls2-영상결과확인/res34/속도측정.md \
    --gpu-id 0
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict

sys.path.insert(0, "./")

import cv2
import numpy as np
import torch
import mmcv

from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa

CLASS_COLORS = {
    1: (0, 220, 0),   # 실선 - 초록 (BGR, val 시각화와 동일)
    2: (0, 0, 220),   # 점선 - 빨강 (BGR, val 시각화와 동일)
}

STAGE_NAMES = OrderedDict([
    ("read", "영상 디코딩(프레임 읽기)"),
    ("preprocess", "전처리(리사이즈+정규화+GPU 업로드)"),
    ("backbone_neck", "모델 추론 - backbone+neck"),
    ("head_decode", "모델 추론 - head forward+히트맵 디코딩"),
    ("postprocess", "후처리(그룹핑/거리필터/좌표 역변환)"),
    ("draw", "결과 그리기(polylines)"),
    ("encode_backpressure", "인코딩 큐 대기(백프레셔, 인코더가 못 따라올 때만 발생)"),
])


class NvencWriter:
    """ffmpeg h264_nvenc로 별도 프로세스에 raw 프레임을 비동기로 흘려보낸다.

    쓰기(`write`)는 큐에 넣기만 하고 바로 반환 — 실제 인코딩은 별도 스레드가
    ffmpeg 자식 프로세스의 stdin에 파이프로 흘려보내면서 진행되므로, 메인
    스레드(GPU 추론)와 동시에 돌아간다. 큐가 가득 차면(인코더가 못 따라오면)
    `write`가 블로킹되고, 그 시간을 `wait_sec`에 누적한다.
    """

    def __init__(self, out_path, w, h, fps, gpu_id, preset="fast", maxsize=64,
                 cq=23, maxrate="80M"):
        # 비트레이트/화질을 안 정하면 nvenc가 기본값(~3Mbps)으로 압축한다 --
        # 3840x2160에는 턱없이 부족해서 화질이 심하게 뭉개졌다(실측으로 확인,
        # 버스 글자가 뭉개질 정도). -rc vbr -cq로 "이 화질을 유지해라"는
        # 목표치를 직접 지정(libx264의 -crf와 같은 개념) -- cq 숫자가
        # 작을수록 고화질(0~51). cq=23이 이전에 품질 확인했던
        # libx264-crf20 결과(~58Mbps)와 비슷한 비트레이트가 나오고 육안
        # 손실도 거의 없어 기본값으로 채택(cq=19는 83Mbps까지 치솟아
        # 파일이 불필요하게 커짐). maxrate/bufsize는 복잡한 장면에서
        # 비트레이트가 과도하게 치솟지 않게 상한만 둔 안전장치.
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "h264_nvenc", "-preset", preset, "-gpu", str(gpu_id),
            "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
            "-maxrate", maxrate, "-bufsize", maxrate,
            "-pix_fmt", "yuv420p", out_path,
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.q = queue.Queue(maxsize=maxsize)
        self.wait_sec = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self.proc.stdin.write(item.tobytes())
        self.proc.stdin.close()

    def write(self, frame):
        t0 = time.time()
        self.q.put(frame)
        self.wait_sec += time.time() - t0

    def release(self):
        t0 = time.time()
        self.q.put(None)
        self._thread.join()
        self.proc.wait()
        return time.time() - t0  # 드레인(꼬리) 대기 시간


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


def mmss(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("video")
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--line-width", type=int, default=6)
    ap.add_argument("--nvenc-preset", default="fast")
    ap.add_argument("--queue-size", type=int, default=64)
    ap.add_argument("--cq", type=int, default=23)
    ap.add_argument("--maxrate", default="80M")
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
    n_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(a.out_video), exist_ok=True)
    os.makedirs(os.path.dirname(a.out_report), exist_ok=True)
    writer = NvencWriter(a.out_video, w, h, src_fps, a.gpu_id,
                          preset=a.nvenc_preset, maxsize=a.queue_size,
                          cq=a.cq, maxrate=a.maxrate)

    meta_base = dict(
        img_shape=(img_scale_wh[1], img_scale_wh[0], 3),
        ori_shape=(h, w, 3),
        offset_x=0,
        offset_y=0,
    )

    stage_sec = OrderedDict((k, 0.0) for k in STAGE_NAMES)
    idx = 0
    n_lanes_total = 0
    gpu = a.gpu_id

    def sync():
        torch.cuda.synchronize(gpu)

    t_total0 = time.time()
    with torch.no_grad():
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            t1 = time.time()
            stage_sec["read"] += t1 - t0

            tensor = preprocess(frame, img_scale_wh, mean, std).unsqueeze(0).cuda(gpu)
            sync()
            t2 = time.time()
            stage_sec["preprocess"] += t2 - t1

            meta = dict(meta_base)
            meta["filename"] = meta["ori_filename"] = f"frame_{idx:06d}.jpg"

            feats = model.extract_feat(tensor.type(torch.cuda.FloatTensor))
            sync()
            t3 = time.time()
            stage_sec["backbone_neck"] += t3 - t2

            pred = model.bbox_head.forward_test(feats)
            sync()
            t4 = time.time()
            stage_sec["head_decode"] += t4 - t3

            result = model.post_processing(pred, meta)
            t5 = time.time()
            stage_sec["postprocess"] += t5 - t4

            lanes, _, _, classes = result["final_dict_list"]
            n_lanes_total += len(lanes)

            for lane, cls in zip(lanes, classes):
                pts = np.array(lane, dtype=np.int32).reshape(-1, 1, 2)
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=a.line_width)
            t6 = time.time()
            stage_sec["draw"] += t6 - t5

            writer.write(frame)  # 큐에 넣기만 함 -- 실제 인코딩은 별도 스레드가 동시 진행
            t7 = time.time()
            stage_sec["encode_backpressure"] += t7 - t6

            idx += 1

    t_loop_end = time.time()
    cap.release()
    drain_sec = writer.release()  # 남은 큐 비우기 + ffmpeg 종료 대기 (여기서부터는 중첩 안 됨)
    t_total1 = time.time()

    main_loop_sec = t_loop_end - t_total0
    total_elapsed = t_total1 - t_total0
    total_video_sec = idx / src_fps
    stage_sum = sum(stage_sec.values())

    with open(a.out_report, "w") as f:
        f.write("# 속도 측정 결과 (파이프라인 단계별, 인코딩 중첩+NVENC 적용)\n\n")
        f.write(f"- 모델: `{a.config}` + `{os.path.basename(a.checkpoint)}`\n")
        f.write(f"- 입력 영상: `{a.video}`\n")
        f.write(f"- 원본 해상도: {w}x{h} -> 모델 입력 해상도: {img_scale_wh[0]}x{img_scale_wh[1]}\n")
        f.write(f"- 총 프레임: {idx} (영상 길이 {total_video_sec:.1f}초 = "
                f"{total_video_sec / 60:.2f}분, {src_fps:.1f}fps)\n")
        f.write(f"- 검출된 차선 수(전체 프레임 합): {n_lanes_total} "
                f"(프레임당 평균 {n_lanes_total / idx:.2f}개)\n")
        f.write(f"- 인코더: GPU 하드웨어(`h264_nvenc`, preset={a.nvenc_preset}), "
                f"별도 스레드/프로세스에서 비동기 실행(메인 루프와 중첩)\n")
        f.write(f"- 메인 루프(읽기~그리기, 인코딩 제외) 시간: {main_loop_sec:.1f}초 "
                f"({main_loop_sec / 60:.2f}분)\n")
        f.write(f"- 인코더 드레인(메인 루프 끝난 뒤 남은 프레임 인코딩 대기) 시간: "
                f"{drain_sec:.1f}초 ({drain_sec / 60:.2f}분) -- 여기부터는 중첩이 안 됨(꼬리)\n")
        f.write(f"- **총 처리 시간(wall-clock): {total_elapsed:.1f}초 ({total_elapsed / 60:.2f}분)** "
                f"= 메인 루프 + 드레인\n")
        f.write(f"- 평균 처리 속도: {idx / total_elapsed:.2f} fps "
                f"(프레임당 {total_elapsed / idx * 1000:.1f}ms)\n")
        rt_factor = total_video_sec / total_elapsed if total_elapsed > 0 else float("inf")
        f.write(f"- 실시간 대비 배속: {rt_factor:.2f}x "
                f"({'실시간보다 빠름(처리시간 < 영상길이)' if rt_factor >= 1 else '실시간보다 느림(처리시간 > 영상길이)'})\n\n")

        f.write("## 메인 루프 단계별 속도 (읽기/전처리/모델 추론/후처리/그리기)\n\n")
        f.write("GPU는 비동기 실행이라 각 단계 경계마다 `torch.cuda.synchronize()`로 "
                "동기화한 뒤 시간을 측정 — 단계 간 시간이 섞이지 않는다. "
                "`인코딩 큐 대기`는 인코더 스레드가 못 따라와서 메인 루프가 블로킹된 시간만 "
                "잡힌다(0에 가까우면 인코딩이 더 이상 병목이 아니라는 뜻) -- 실제 인코딩 작업 "
                "자체는 이 표에 안 잡히고 메인 루프와 동시에 진행된다.\n\n")
        f.write("| 단계 | 총 시간(초) | 총 시간(분:초) | 비율(%) | 평균(ms/frame) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for key, name in STAGE_NAMES.items():
            sec = stage_sec[key]
            pct = sec / stage_sum * 100 if stage_sum > 0 else 0.0
            f.write(f"| {name} | {sec:.1f} | {mmss(sec)} | {pct:.1f}% | {sec / idx * 1000:.2f} |\n")
        f.write(f"| **합계(=메인 루프 시간)** | **{stage_sum:.1f}** | **{mmss(stage_sum)}** | 100.0% | "
                f"**{stage_sum / idx * 1000:.2f}** |\n")

        f.write("\n## 모델 추론만 따로 보면(backbone+neck + head+디코딩)\n\n")
        infer_sec = stage_sec["backbone_neck"] + stage_sec["head_decode"]
        infer_fps = idx / infer_sec if infer_sec > 0 else float("inf")
        f.write(f"- 순수 GPU 추론 시간: {infer_sec:.1f}초({infer_sec / 60:.2f}분), "
                f"프레임당 {infer_sec / idx * 1000:.2f}ms, {infer_fps:.2f} fps\n")

    print(f"done video={a.out_video} report={a.out_report} "
          f"frames={idx} total_sec={total_elapsed:.1f} fps={idx / total_elapsed:.2f}")


if __name__ == "__main__":
    main()
