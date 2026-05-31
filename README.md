# YOLOv8 Pruning Term Project

This project is a small experiment for pruning a YOLOv8 model with Torch-Pruning.
The current main goal is to test whether YOLOv8 can be pruned and still run a
valid forward pass.

## Files

- `TP.ipynb`  
  Notebook version of the pruning experiment. It explains the basic pruning
  flow, loads YOLOv8, prunes channels, checks the forward pass, and saves a
  pruned checkpoint.

- `prune_yolo.py`  
  Main Python script for structured pruning. It loads `yolov8n.pt`, replaces
  YOLOv8 `C2f` blocks with a pruning-friendly version, prunes part of the model,
  tests a forward pass, and saves the result.

- `evaluate.py`  
  Validation script for one selected model. It uses
  `YOLO(...).val(data="coco128.yaml")` and prints precision, recall, mAP50, and
  mAP50-95.

- `finetune.py`  
  Placeholder script for future finetuning after pruning. It already has command
  line arguments for weights, dataset YAML, epochs, image size, and device, but
  the training logic is not implemented yet.

- `export.py`  
  Export script for YOLO models. It can export formats supported by Ultralytics,
  including TFLite, ONNX, TensorRT, and OpenVINO.

- `quantize.py`  
  Quantization script for exporting a YOLO model to INT8 TFLite. It uses a
  dataset YAML such as `coco128.yaml` for calibration.

- `quantize_int4.py`  
  Experimental low-bit TFLite conversion script. TensorFlow may still output
  INT8 tensors, so the script also reports the tensor types inside the file.

- `yolov8n.pt`  
  Original YOLOv8 nano model weights used as the pruning input.

- `yolov8n_pruned_sample.pt`  
  Example output checkpoint saved after running the pruning experiment.

## How To Run

Install the needed packages first:

```bash
pip install ultralytics torch-pruning
```

Run the pruning script:

```bash
python3 prune_yolo.py --weights yolov8n.pt --ratio 0.10
```

Optional arguments:

```bash
python3 prune_yolo.py --weights yolov8n.pt --ratio 0.10 --imgsz 640 --device cpu --output yolov8n_pruned_sample.pt
```

Evaluate one selected model:

```bash
python3 evaluate.py --weights yolov8n_pruned_sample.pt --data coco128.yaml
```

Export YOLOv8 to TFLite:

```bash
python3 export.py --weights yolov8n.pt --format tflite
```

The exported TFLite file is saved under `yolov8n_saved_model/`.

Quantize YOLOv8 to INT8 TFLite:

```bash
python3 quantize.py --weights yolov8n.pt --data coco128.yaml
```

For a smaller calibration run, use:

```bash
python3 quantize.py --weights yolov8n.pt --data coco128.yaml --fraction 0.25
```

Evaluate the quantized model:

```bash
python3 evaluate.py --weights yolov8n_saved_model/yolov8n_int8.tflite --data coco128.yaml
```

Try experimental INT4/low-bit conversion:

```bash
python3 quantize_int4.py --weights yolov8n.pt
```

## Current Status

Pruning works as a smoke test. Finetuning and export are planned next steps, but
they are not implemented yet.
