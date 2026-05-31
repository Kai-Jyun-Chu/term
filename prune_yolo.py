"""
Simple YOLOv8 structured pruning experiment with Torch-Pruning.

Run:
    python prune_yolo.py --weights yolov8n.pt --ratio 0.10

This script is intentionally small. It only checks that pruning can run and that
the pruned model still has a valid forward pass. Finetuning comes after this.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv, Detect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune YOLOv8 with Torch-Pruning.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 checkpoint path or model name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size used for graph tracing.")
    parser.add_argument("--ratio", type=float, default=0.10, help="Channel pruning ratio, e.g. 0.10 for 10%.")
    parser.add_argument("--round-to", type=int, default=1, help="Round pruned channel counts to this multiple.")
    parser.add_argument("--local-pruning", action="store_true", help="Prune the ratio from each layer instead of globally.")
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--output", default="yolov8n_pruned_sample.pt", help="Output checkpoint path.")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_model(model: torch.nn.Module, example_inputs: torch.Tensor) -> tuple[float, float]:
    """Return MACs and parameter count for a model and dummy input."""
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return float(macs), float(params)


def infer_shortcut(block: Bottleneck) -> bool:
    """Return whether an Ultralytics Bottleneck block uses a residual shortcut."""
    c1 = block.cv1.conv.in_channels
    c2 = block.cv2.conv.out_channels
    return c1 == c2 and hasattr(block, "add") and block.add


class C2fV2(nn.Module):
    """A pruning-friendly version of YOLOv8 C2f.

    Ultralytics C2f splits channels with torch.chunk(). Torch-Pruning can miss
    that dependency, so this version uses two explicit 1x1 convolutions instead.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv0(x), self.cv1(x)]
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, dim=1))


def transfer_c2f_weights(old: C2f, new: C2fV2) -> None:
    """Copy weights from the original C2f block into C2fV2."""
    new.cv2 = old.cv2
    new.m = old.m
    for attr in ["i", "f", "type", "np"]:
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))

    old_state = old.state_dict()
    new_state = new.state_dict()

    old_weight = old_state["cv1.conv.weight"]
    half_channels = old_weight.shape[0] // 2
    new_state["cv0.conv.weight"] = old_weight[:half_channels]
    new_state["cv1.conv.weight"] = old_weight[half_channels:]

    for key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = old_state[f"cv1.bn.{key}"]
        new_state[f"cv0.bn.{key}"] = old_bn[:half_channels]
        new_state[f"cv1.bn.{key}"] = old_bn[half_channels:]

    for key, value in old_state.items():
        if not key.startswith("cv1."):
            new_state[key] = value

    new.load_state_dict(new_state)


def replace_c2f_with_c2fv2(module: nn.Module) -> None:
    """Recursively replace YOLOv8 C2f modules with C2fV2 modules."""
    for name, child in module.named_children():
        if isinstance(child, C2f):
            first_block = child.m[0]
            replacement = C2fV2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=infer_shortcut(first_block),
                g=first_block.cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            transfer_c2f_weights(child, replacement)
            setattr(module, name, replacement)
        else:
            replace_c2f_with_c2fv2(child)


def unique_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """Return modules in original order without duplicates."""
    unique = []
    seen = set()
    for module in modules:
        module_id = id(module)
        if module_id not in seen:
            unique.append(module)
            seen.add(module_id)
    return unique


def collect_sensitive_layers(model: nn.Module) -> tuple[list[nn.Module], list[str]]:
    """Collect layers that should be protected during pruning.

    For YOLO, the first stem layer and the final neck outputs feeding Detect are
    especially sensitive. Protecting them keeps the detector interface more
    stable while global pruning chooses less important channels elsewhere.
    """
    layers = getattr(model, "model", None)
    has_indexed_layers = isinstance(layers, (nn.ModuleList, nn.Sequential))
    protected: list[nn.Module] = []
    descriptions: list[str] = []

    if has_indexed_layers and len(layers) > 0:
        protected.append(layers[0])
        descriptions.append("layer 0 first stem")

    for module in model.modules():
        if isinstance(module, Detect):
            protected.append(module)
            descriptions.append("Detect head")

            if has_indexed_layers:
                for source_idx in module.f:
                    source = layers[source_idx]
                    protected.append(source)
                    descriptions.append(f"layer {source_idx} Detect input/final neck output")

    return unique_modules(protected), descriptions


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_path = Path(args.output)

    torch.manual_seed(0)
    print(f"Loading {args.weights} on {device}...")

    # Ultralytics YOLO is a wrapper. Torch-Pruning needs the real nn.Module inside it.
    yolo = YOLO(args.weights)
    model = yolo.model.to(device)

    # YOLOv8 C2f needs this conversion for Torch-Pruning to discover prunable
    # channel groups correctly.
    replace_c2f_with_c2fv2(model)
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Torch-Pruning traces the dependency graph with a dummy input.
    # The shape should match the image size you plan to train/export with.
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    base_macs, base_params = count_model(model, example_inputs)
    print(f"Before pruning: {base_params / 1e6:.2f}M params, {base_macs / 1e9:.2f}G MACs")

    # Keep sensitive layers untouched: Detect, Detect input layers/final neck
    # outputs, and the first stem layer.
    sensitive_layers, sensitive_descriptions = collect_sensitive_layers(model)
    ignored_layers = sensitive_layers
    pruning_ratio_dict = {module: 0.0 for module in sensitive_layers}
    print("Protected layers:")
    for description in sensitive_descriptions:
        print(f"  - {description}")

    # Magnitude importance removes channels with smaller weight norms first.
    importance = tp.importance.MagnitudeImportance(p=2)

    # MetaPruner removes channels and also updates dependent layers. Global
    # pruning is safer for very small ratios: local pruning can remove one
    # channel from almost every layer, which often collapses pretrained
    # detection confidence before finetuning.
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=args.ratio,
        pruning_ratio_dict=pruning_ratio_dict,
        ignored_layers=ignored_layers,
        global_pruning=not args.local_pruning,
        round_to=args.round_to,
    )

    print(f"Pruning {args.ratio:.0%} of prunable channels...")
    pruner.step()

    pruned_macs, pruned_params = count_model(model, example_inputs)

    # Smoke test: if this fails, the architecture was broken by pruning.
    with torch.no_grad():
        _ = model(example_inputs)

    print(f"After pruning:  {pruned_params / 1e6:.2f}M params, {pruned_macs / 1e9:.2f}G MACs")
    print(f"Params smaller: {(1 - pruned_params / base_params) * 100:.1f}%")
    print(f"MACs fewer:     {(1 - pruned_macs / base_macs) * 100:.1f}%")
    print("Forward pass OK.")

    # Structural pruning changes layer shapes. Saving the full module is simpler
    # for experiments than saving only state_dict.
    checkpoint = {
        "model": copy.deepcopy(model).cpu(),
        "source_weights": args.weights,
        "imgsz": args.imgsz,
        "pruning_ratio": args.ratio,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved pruned checkpoint to {output_path.resolve()}")


if __name__ == "__main__":
    main()
