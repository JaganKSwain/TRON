"""ONNX Exporter for AgroBot models (Phase 8).

Exports Model A (ConvNeXt-Tiny + GeM) and Model B (MobileNetV2-0.5) to ONNX format
with verified input/output signatures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.dataset import NUM_ALL_CLASSES, NUM_PLANTVILLAGE_CLASSES
from agrobot.models.classifier import PlantClassifier
from agrobot.models.mcu import MCU_INPUT_SIZE, McuClassifier, build_mcu_model
from agrobot.paths import EXPORT_DIR, RUNS_DIR, ensure_dirs


def export_model_a_onnx(
    checkpoint_path: Path,
    out_path: Path,
    image_size: int = 224,
    device: str = "cpu",
) -> Path:
    """Export Model A (ConvNeXt-Tiny + GeM) to ONNX."""
    print(f"\n--- Exporting Model A to ONNX: {out_path} ---", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("args", {})
    num_classes = NUM_ALL_CLASSES if model_args.get("include_not_a_leaf", False) else NUM_PLANTVILLAGE_CLASSES

    model = PlantClassifier(
        backbone_name=model_args.get("backbone", "convnext_tiny"),
        pretrained=False,
        num_classes=num_classes,
        pool_type=model_args.get("pool_type", "gem"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=14,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    # Verify with onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Verify with onnxruntime
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"input": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    np_diff = float(abs(ort_out - torch_out).max())
    print(f"Model A ONNX verified successfully! (Max diff vs PyTorch: {np_diff:.6f})", flush=True)
    return out_path


def export_model_b_onnx(
    checkpoint_path: Path,
    out_path: Path,
    image_size: int = MCU_INPUT_SIZE,
    device: str = "cpu",
) -> Path:
    """Export Model B (MobileNetV2-0.5) to ONNX."""
    print(f"\n--- Exporting Model B to ONNX: {out_path} ---", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = 15

    model = build_mcu_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["student_state"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=14,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    # Verify with onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    # Verify with onnxruntime
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"input": dummy_input.numpy()})[0]
    torch_out = model(dummy_input).detach().numpy()
    np_diff = float(abs(ort_out - torch_out).max())
    print(f"Model B ONNX verified successfully! (Max diff vs PyTorch: {np_diff:.6f})", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AgroBot models to ONNX")
    parser.add_argument("--model-a-ckpt", type=Path, default=RUNS_DIR / "model_a_convnext" / "best_model.pt")
    parser.add_argument("--model-b-ckpt", type=Path, default=RUNS_DIR / "model_b_mcu" / "best_model.pt")
    parser.add_argument("--out-dir", type=Path, default=EXPORT_DIR / "onnx")
    args = parser.parse_args()

    ensure_dirs()
    if args.model_a_ckpt.exists():
        export_model_a_onnx(args.model_a_ckpt, args.out_dir / "model_a_convnext.onnx", image_size=224)
    else:
        print(f"Warning: Model A checkpoint not found at {args.model_a_ckpt}")

    if args.model_b_ckpt.exists():
        export_model_b_onnx(args.model_b_ckpt, args.out_dir / "model_b_mcu.onnx", image_size=MCU_INPUT_SIZE)
    else:
        print(f"Warning: Model B checkpoint not found at {args.model_b_ckpt}")


if __name__ == "__main__":
    main()
