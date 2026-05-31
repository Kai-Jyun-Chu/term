"""
Validate one designated YOLOv8 model on a dataset.

Run:
    python3 evaluate.py --weights yolov8n_pruned_10.pt --data coco128.yaml

Pruned checkpoints saved from prune_yolo.py use a custom C2fV2 module, so this
file imports C2fV2 before loading the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

# Needed so torch can unpickle yolov8n_pruned_sample.pt.
from prune_yolo import C2fV2  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one YOLO model.")
    parser.add_argument("--data", default="coco128.yaml", help="Dataset YAML used by model.val().")
    parser.add_argument("--weights", required=True, help="YOLO checkpoint to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for Ultralytics default.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    return parser.parse_args()


def validate_model(weights: str, data: str, imgsz: int, device: str, batch: int):
    print(f"\nValidating model: {weights}")
    model = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, device=device, batch=batch)
    box = metrics.box
    return {
        "weights": weights,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def print_result(result: dict[str, float | str]) -> None:
    print(f"\nResult ({Path(str(result['weights'])).name})")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"mAP50:     {result['map50']:.4f}")
    print(f"mAP50-95:  {result['map50_95']:.4f}")


def main() -> None:
    args = parse_args()

    result = validate_model(args.weights, args.data, args.imgsz, args.device, args.batch)
    print_result(result)


if __name__ == "__main__":
    main()
