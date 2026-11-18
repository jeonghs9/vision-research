"""영상 전체(기본 0816 영상, 인자로 다른 영상 가능)에 res34 기존 vs P3del−SA(P3·SA 둘 다 제거)를 돌려 좌우로 나란히 붙인 비교 영상.

왼쪽 = 기존(models/drone-ganet-res34-2cls-v1.pth, ep130), 오른쪽 = P3del−SA(test/008 epoch_110).
각 모델의 검출·그리기는 scripts/04-eval/infer_video_2cls.py 와 같음(원본 해상도에 그린 뒤, 각 절반을 1920x1080으로 줄여 3840x1080으로 붙임).
후처리 문턱값은 config 값 그대로(P3del 계열도 칸 단위 숫자를 그대로 둠). 인코딩은 같은 NvencWriter.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_sa_p3_video_compare.py [입력 영상] [출력 mp4]
나온 영상은 용량이 커서(NVENC cq23) ffmpeg libx264 -crf 28 로 다시 압축해 둠(README 참고).
"""
import importlib.util
import os
import sys
import time
sys.path.insert(0, "./")
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

G = "/home/hsjeong/workspace/vision-det/GANet"
spec = importlib.util.spec_from_file_location("iv", f"{G}/scripts/04-eval/infer_video_2cls.py")
iv = importlib.util.module_from_spec(spec); spec.loader.exec_module(iv)

# 인자: [입력 영상] [출력 mp4]. 없으면 0816 영상
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/home/hsjeong/workspace/dataset/etc./test_video/19700101_000816_T.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{G}/output/08-cls2-P3SA4구조비교-val결과/videos/0816영상_기존vsP3del-SA.mp4"
MODELS = [
    ("기존 (P3 O · SA O)", "projects/cfgs/drone/res34_s8_full_2cls.py", f"{G}/models/drone-ganet-res34-2cls-v1.pth"),
    ("P3del−SA (P3 X · SA X)", "projects/cfgs/drone/res34_s16_full_2cls_P3del_noSA.py",
     f"{G}/test/008-res34-2cls-1280-P3del-noSA-150ep/epoch_110.pth"),
]
HALF = (1920, 1080)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def make_label(text):
    """라벨 상자 범위와 글씨 마스크를 한 번만 만든다(한글이라 PIL 사용). 프레임마다 상자는 어둡게, 글씨는 흰색으로 덮음."""
    f = ImageFont.truetype(FONT, 44)
    pil = Image.new("L", HALF, 0)
    d = ImageDraw.Draw(pil)
    x0, y0, x1, y1 = d.textbbox((20, 14), text, font=f)
    d.text((20, 14), text, font=f, fill=255)
    box = (max(x0 - 12, 0), max(y0 - 10, 0), x1 + 12, y1 + 10)
    alpha = (np.asarray(pil, np.float32) / 255)[box[1]:box[3], box[0]:box[2], None]
    return box, alpha


def put_label(img, lab):
    (x0, y0, x1, y1), alpha = lab
    roi = img[y0:y1, x0:x1].astype(np.float32) * 0.37
    img[y0:y1, x0:x1] = (roi * (1 - alpha) + 255 * alpha).astype(np.uint8)

models = [(tag, *iv.build_model(c, k, 0)) for tag, c, k in MODELS]
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
writer = iv.NvencWriter(OUT, HALF[0] * 2, HALF[1], fps, 0)

labels = [make_label(t) for t, _, _ in models]

n_lanes = [0, 0]
idx, t0 = 0, time.time()
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        halves = []
        for m, (tag, model, cfg) in enumerate(models):
            img_scale = tuple(cfg.img_scale)
            mean = np.array(cfg.img_norm_cfg["mean"], np.float32)
            std = np.array(cfg.img_norm_cfg["std"], np.float32)
            t = iv.preprocess(frame, img_scale, mean, std).unsqueeze(0).cuda(0)
            meta = dict(img_shape=(img_scale[1], img_scale[0], 3), ori_shape=(h, w, 3), offset_x=0, offset_y=0,
                        filename=f"frame_{idx:06d}.jpg", ori_filename=f"frame_{idx:06d}.jpg")
            pred = model.bbox_head.forward_test(model.extract_feat(t))
            lanes, _, _, classes = model.post_processing(pred, meta)["final_dict_list"]
            n_lanes[m] += len(lanes)
            canvas = frame.copy()
            # post_process 의 crop_bbox 가 config 에 3840x2160 으로 고정이라, 원본이 다른 해상도(AIhub 1920x1080)면 좌표를 원본 크기로 환산
            cb = cfg.crop_bbox
            sx, sy = w / (cb[2] - cb[0]), h / (cb[3] - cb[1])
            for lane, cls in zip(lanes, classes):
                pts = (np.array(lane, dtype=np.float32) * [sx, sy]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], isClosed=False, color=iv.CLASS_COLORS.get(cls, (255, 255, 255)), thickness=6)
            small = cv2.resize(canvas, HALF, interpolation=cv2.INTER_AREA)
            put_label(small, labels[m])
            halves.append(small)
        writer.write(np.ascontiguousarray(np.hstack(halves)))
        idx += 1
        if idx % 1000 == 0:
            print(f"{idx}/{n_total} 프레임, {time.time() - t0:.0f}초", flush=True)
cap.release()
writer.release()
print(f"done {OUT} frames={idx} 걸린시간={time.time() - t0:.0f}초 "
      f"프레임당 평균 차선 수: 기존 {n_lanes[0] / idx:.2f}, P3del−SA {n_lanes[1] / idx:.2f}")
