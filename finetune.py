"""
Finetuning entry point for the pruned YOLO model.

This is a placeholder for the next step. After pruning works, we can wire this
to your dataset YAML and decide how to load the pruned architecture cleanly.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a pruned YOLO model.")
    parser.add_argument("--weights", default="yolov8n_pruned_sample.pt", help="Pruned checkpoint path.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML, e.g. data.yaml.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Finetuning will be implemented after the pruning smoke test works. "
        f"Requested weights={args.weights}, data={args.data}"
    )


if __name__ == "__main__":
    main()
