"""
Try experimental low-bit/INT4 TFLite conversion for a YOLO model.

TensorFlow Lite does not expose normal post-training INT4 quantization for this
YOLO export path. This script tries TensorFlow's hidden low-bit QAT flag and
then inspects the TFLite flatbuffer to report whether INT4 tensors were actually
created.

Run:
    python3 quantize_int4.py --weights yolov8n.pt

Evaluate the output with:
    python3 evaluate.py --weights yolov8n_saved_model/yolov8n_experimental_int4.tflite --data coco128.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try experimental INT4 TFLite conversion.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO checkpoint used if SavedModel is missing.")
    parser.add_argument("--saved-model", default="", help="Existing SavedModel directory. Defaults to <weights>_saved_model.")
    parser.add_argument("--output", default="", help="Output .tflite path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for SavedModel export and calibration.")
    parser.add_argument("--calibration-samples", type=int, default=4, help="Random calibration samples.")
    return parser.parse_args()


def ensure_saved_model(weights: str, saved_model: Path, imgsz: int) -> None:
    if saved_model.exists():
        return
    print(f"SavedModel not found at {saved_model}, exporting first...")
    model = YOLO(weights)
    model.export(format="saved_model", imgsz=imgsz)


def representative_dataset(imgsz: int, samples: int):
    for _ in range(samples):
        yield [np.random.random((1, imgsz, imgsz, 3)).astype(np.float32)]


def tensor_type_counts(tflite_path: Path) -> dict[str, int]:
    type_names = {getattr(schema.TensorType, name): name for name in dir(schema.TensorType) if name.isupper()}
    model = schema.Model.GetRootAsModel(tflite_path.read_bytes(), 0)
    counts: dict[str, int] = {}
    for subgraph_idx in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_idx)
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            tensor_type = type_names.get(tensor.Type(), str(tensor.Type()))
            counts[tensor_type] = counts.get(tensor_type, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    saved_model = Path(args.saved_model) if args.saved_model else weights_path.with_suffix("").with_name(f"{weights_path.stem}_saved_model")
    output = Path(args.output) if args.output else saved_model / f"{weights_path.stem}_experimental_int4.tflite"

    ensure_saved_model(args.weights, saved_model, args.imgsz)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(args.imgsz, args.calibration_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Hidden TensorFlow flag. This only creates true low-bit output for models
    # prepared with low-bit quantization-aware training.
    converter._experimental_low_bit_qat = True

    print("Converting with TensorFlow experimental low-bit QAT flag...")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(converter.convert())

    counts = tensor_type_counts(output)
    print(f"Saved: {output}")
    print(f"Size: {output.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Tensor types: {counts}")
    if counts.get("INT4", 0) == 0:
        print("Warning: no INT4 tensors were produced; this output is an experimental INT8-style TFLite model.")
    print(f"Evaluate with: python3 evaluate.py --weights {output} --data coco128.yaml")


if __name__ == "__main__":
    main()
