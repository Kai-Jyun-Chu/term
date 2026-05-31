"""
Export entry point for the pruned/finetuned YOLO model.
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model.")
    parser.add_argument("--weights", required=True, help="Model checkpoint to export.")
    parser.add_argument("--format", default="onnx", help="Export format, e.g. onnx, engine, openvino.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    output = model.export(format=args.format, imgsz=args.imgsz, device=args.device)
    print(f"Exported model to: {output}")


if __name__ == "__main__":
    main()
