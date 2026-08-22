#!/usr/bin/env python3
"""YOLO person detection worker for local-camera-vision.

Reads JSON requests on stdin and returns JSON responses on stdout.

Protocol (one JSON object per line):
  REQUEST:  {"image_path": "/path/to/frame.jpg", "confidence": 0.5}
  RESPONSE: {"person_detected": true, "detections": [...], "elapsed_ms": 4.2}
"""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

MODEL_NAME = "yolo11n"

PERSON_CLASS = 0  # COCO class index for "person"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()
    # Pre-warm message so caller knows model is loading
    print(json.dumps({"status": "loading", "model": MODEL_NAME}), flush=True)

    t0 = time.perf_counter()
    model = YOLO(args.model)
    load_ms = (time.perf_counter() - t0) * 1000

    # Single warm-up inference (GPU kernels JIT-compiled on first call)
    warm = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model(warm, verbose=False, conf=0.5, classes=[PERSON_CLASS])

    print(json.dumps({
        "status": "ready",
        "model": MODEL_NAME,
        "load_ms": round(load_ms, 1),
        "device": str(model.device),
    }), flush=True)

    # Main loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"invalid json: {exc}"}), flush=True)
            continue

        image_path = req.get("image_path", "")
        confidence = float(req.get("confidence", 0.5))

        if not image_path or not Path(image_path).is_file():
            print(json.dumps({
                "error": f"image not found: {image_path}",
                "person_detected": False,
                "detections": [],
            }), flush=True)
            continue

        t0 = time.perf_counter()
        try:
            results = model(image_path, verbose=False, conf=confidence, classes=[PERSON_CLASS])
        except Exception as exc:
            print(json.dumps({
                "error": str(exc),
                "person_detected": False,
                "detections": [],
            }), flush=True)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    detections.append({
                        "bbox": [round(v, 1) for v in bbox],
                        "confidence": round(conf, 4),
                        "class": "person",
                    })

        print(json.dumps({
            "person_detected": len(detections) > 0,
            "detections": detections,
            "elapsed_ms": round(elapsed_ms, 1),
        }), flush=True)


if __name__ == "__main__":
    main()
