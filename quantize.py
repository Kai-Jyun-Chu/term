"""
Quantize a YOLO model to INT8 TFLite.

Run:
    python3 quantize.py --weights yolov8n.pt --data coco128.yaml

Then evaluate the output with:
    python3 evaluate.py --weights yolov8n_saved_model/yolov8n_int8.tflite --data coco128.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize a YOLO model to INT8 TFLite.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO checkpoint to quantize.")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset YAML for INT8 calibration.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export and calibration.")
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for Ultralytics default.")
    parser.add_argument("--batch", type=int, default=1, help="Calibration batch size.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of calibration data to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    output = model.export(
        format="tflite",
        int8=True,
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        fraction=args.fraction,
    )

    print(f"Quantized TFLite model: {output}")
    if Path(output).exists():
        print(f"Evaluate with: python3 evaluate.py --weights {output} --data {args.data}")


if __name__ == "__main__":
    main()
