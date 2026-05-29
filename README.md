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

- `finetune.py`  
  Placeholder script for future finetuning after pruning. It already has command
  line arguments for weights, dataset YAML, epochs, image size, and device, but
  the training logic is not implemented yet.

- `export.py`  
  Placeholder script for future model export. It is planned for formats such as
  ONNX, TensorRT, or OpenVINO, but the export logic is not implemented yet.

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

## Current Status

Pruning works as a smoke test. Finetuning and export are planned next steps, but
they are not implemented yet.
