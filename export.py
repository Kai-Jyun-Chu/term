"""
Export entry point for the pruned/finetuned YOLO model.

This is a placeholder for deployment formats such as ONNX, TensorRT, or OpenVINO.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model.")
    parser.add_argument("--weights", required=True, help="Model checkpoint to export.")
    parser.add_argument("--format", default="onnx", help="Export format, e.g. onnx, engine, openvino.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Export will be implemented after pruning and finetuning are stable. "
        f"Requested weights={args.weights}, format={args.format}"
    )


if __name__ == "__main__":
    main()
