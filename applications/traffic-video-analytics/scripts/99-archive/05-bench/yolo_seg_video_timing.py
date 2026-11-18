#!/usr/bin/env python3
"""YOLO segmentation timing over every frame of a video."""
import argparse
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(args.video)
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    timing_ms = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    decode_seconds = 0.0
    frames = 0

    started = time.perf_counter()
    while True:
        tick = time.perf_counter()
        ok, frame = capture.read()
        decode_seconds += time.perf_counter() - tick
        if not ok:
            break
        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        for section in timing_ms:
            timing_ms[section] += float(result.speed[section])
        frames += 1
        if frames % 500 == 0:
            print(f"{frames}/{expected}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    capture.release()

    print(f"frames={frames} expected={expected}")
    print(f"decode={decode_seconds:.6f}s")
    for section, milliseconds in timing_ms.items():
        print(f"{section}={milliseconds / 1000.0:.6f}s")
    print(f"model_sections={sum(timing_ms.values()) / 1000.0:.6f}s")
    print(f"wall_total={elapsed:.6f}s")
    print(f"fps={frames / elapsed:.6f}")


if __name__ == "__main__":
    main()
