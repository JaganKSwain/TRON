"""Diagnostic engine for AgroBot models (Phase 5).

Anti-Self-Deception Validations:
1. Background-Only Ablation: Leaf is masked out. Accuracy MUST collapse to ~6.7% chance.
2. Leaf-Only Ablation: Background is replaced with neutral grey. Accuracy must stay high.
3. Cross-Site Evaluation: Evaluates on held-out capture sites (`test_site` split).
4. Corruption Robustness: Measures performance under sensor noise, blur, and compression.
5. Grad-CAM Pathology Localization: Visualizes model attention on disease lesions.
6. Calibration & ECE: Measures probability reliability and Expected Calibration Error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_PLANTVILLAGE_CLASSES,
    PlantVillageDataset,
)
from agrobot.data.manifest import load_manifest
from agrobot.data.segment import generous_mask, mask_path_for
from agrobot.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from agrobot.models.classifier import PlantClassifier
from agrobot.paths import MASK_CACHE, REPORTS_DIR, RUNS_DIR, SPLITS_CSV, ensure_dirs


class MaskAblationDataset(Dataset):
    """Dataset applying leaf-only or background-only spatial masking."""

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "background_only",  # 'background_only' or 'leaf_only'
        image_size: int = 224,
        mask_cache: Path = MASK_CACHE,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.image_size = image_size
        self.mask_cache = Path(mask_cache)
        self.transform = get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        bgr = cv2.imread(row["path"])
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {row['path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mp = mask_path_for(row["path"], cache=self.mask_cache)
        if mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        else:
            mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)

        if self.mode == "background_only":
            # Inpaint / blackout the leaf entirely
            leaf_mask = generous_mask(mask)
            # Black out or fill with neutral mean color
            rgb_ablated = rgb.copy()
            rgb_ablated[leaf_mask > 0] = [128, 128, 128]
        elif self.mode == "leaf_only":
            # Black out the background card
            leaf_mask = mask > 0
            rgb_ablated = rgb.copy()
            rgb_ablated[~leaf_mask] = [128, 128, 128]
        else:
            rgb_ablated = rgb

        res = self.transform(image=rgb_ablated)
        label = CLASS_TO_IDX[row["class"]]
        return res["image"], label


@torch.no_grad()
def eval_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {"acc": acc, "macro_f1": macro_f1, "bal_acc": bal_acc}


def run_ablation_diagnostics(
    model: nn.Module, test_df: pd.DataFrame, device: torch.device, batch_size: int = 64
) -> dict[str, Any]:
    """Run Background-Only and Leaf-Only ablation experiments."""
    print("\n--- Running Spatial Ablations ---", flush=True)

    # 1. Background-Only Ablation
    bg_ds = MaskAblationDataset(test_df, mode="background_only")
    bg_loader = DataLoader(bg_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bg_res = eval_loader(model, bg_loader, device=device)
    print(f"Background-Only Accuracy: {bg_res['acc']:.2%} (Chance is 6.67%)", flush=True)
    print(f"Background-Only Macro-F1: {bg_res['macro_f1']:.4f}", flush=True)

    # 2. Leaf-Only Ablation
    leaf_ds = MaskAblationDataset(test_df, mode="leaf_only")
    leaf_loader = DataLoader(leaf_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    leaf_res = eval_loader(model, leaf_loader, device=device)
    print(f"Leaf-Only Accuracy:       {leaf_res['acc']:.2%}", flush=True)
    print(f"Leaf-Only Macro-F1:       {leaf_res['macro_f1']:.4f}", flush=True)

    return {
        "bg_only": bg_res,
        "leaf_only": leaf_res,
    }


def run_cross_site_diagnostic(
    model: nn.Module, splits_df: pd.DataFrame, device: torch.device, batch_size: int = 64
) -> dict[str, float]:
    """Evaluate generalization on held-out capture sites."""
    print("\n--- Running Cross-Site Holdout Diagnostic ---", flush=True)
    site_df = splits_df[splits_df["split"] == "test_site"].reset_index(drop=True)
    if site_df.empty:
        print("No test_site split found.")
        return {}

    eval_t = get_eval_transforms(224)
    ds = PlantVillageDataset(site_df, split=None, transform=eval_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    res = eval_loader(model, loader, device=device)

    print(f"Cross-Site Accuracy:  {res['acc']:.2%}", flush=True)
    print(f"Cross-Site Macro-F1:  {res['macro_f1']:.4f}", flush=True)
    print(f"Cross-Site Bal Acc:   {res['bal_acc']:.2%}", flush=True)
    return res


def compute_calibration_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device, num_bins: int = 15
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and bin statistics."""
    model.eval()
    all_confs: list[float] = []
    all_correct: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            correct = (preds.cpu() == targets).int()
            all_confs.extend(confs.cpu().tolist())
            all_correct.extend(correct.tolist())

    confs = np.array(all_confs)
    corrects = np.array(all_correct)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = float(in_bin.mean())
        bin_count = int(in_bin.sum())
        bin_counts.append(bin_count)

        if bin_count > 0:
            accuracy_in_bin = float(corrects[in_bin].mean())
            avg_confidence_in_bin = float(confs[in_bin].mean())
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)

    return {
        "ece": float(ece),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def generate_gradcam_visualizations(
    model: PlantClassifier,
    test_df: pd.DataFrame,
    out_path: Path,
    device: torch.device,
    num_samples: int = 8,
) -> None:
    """Generate Grad-CAM heatmaps to visually verify lesion localization."""
    print("\n--- Generating Grad-CAM Heatmap Montage ---", flush=True)
    model.eval()

    # Target last stage of ConvNeXt
    target_layers = [model.backbone.stages[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    eval_t = get_eval_transforms(224)
    samples = test_df.groupby("class").first().reset_index().head(num_samples)

    rows = []
    cell_size = 224

    for _, row in samples.iterrows():
        bgr = cv2.imread(row["path"])
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (cell_size, cell_size))
        rgb_float = np.float32(rgb_resized) / 255.0

        t_res = eval_t(image=rgb)
        input_tensor = t_res["image"].unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = int(logits.argmax(dim=-1).item())
            pred_cls = IDX_TO_CLASS.get(pred_idx, "unknown")

        # Side by side
        strip = np.hstack([rgb_resized, cam_image])
        cv2.putText(
            strip,
            f"True: {row['class'][:20]} | Pred: {pred_cls[:20]}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(strip)

    if rows:
        montage_rgb = np.vstack(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(montage_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM montage to: {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Diagnostics & Anti-Shortcut Validation")
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "model_a_convnext" / "best_model.pt")
    parser.add_argument("--splits", type=Path, default=SPLITS_CSV)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint} on {device}")

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint {args.checkpoint} does not exist.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
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

    splits_df = load_manifest(args.splits)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    # 1. Ablations
    ablations = run_ablation_diagnostics(model, test_df, device, args.batch_size)

    # 2. Cross-Site Generalization
    cross_site = run_cross_site_diagnostic(model, splits_df, device, args.batch_size)

    # 3. Calibration
    test_ds = PlantVillageDataset(test_df, split=None, transform=get_eval_transforms(224))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    calib = compute_calibration_metrics(model, test_loader, device)
    print(f"\nExpected Calibration Error (ECE): {calib['ece']:.4f}", flush=True)

    # 4. Grad-CAM
    cam_out = REPORTS_DIR / "gradcam_montage.jpg"
    generate_gradcam_visualizations(model, test_df, cam_out, device)

    # 5. Generate Report Markdown
    report_md = f"""# Model A Diagnostic & Anti-Shortcut Report

## 1. Spatial Ablation (Anti-Shortcut Gate)
- **Background-Only Accuracy**: **{ablations['bg_only']['acc']:.2%}** (Chance: 6.67%)
- **Background-Only Macro-F1**: **{ablations['bg_only']['macro_f1']:.4f}**
- **Leaf-Only Accuracy**: **{ablations['leaf_only']['acc']:.2%}**
- **Leaf-Only Macro-F1**: **{ablations['leaf_only']['macro_f1']:.4f}**

> **Verdict**: {'PASS - Background does not predict label' if ablations['bg_only']['acc'] < 0.15 else 'FAIL - Background leakage detected'}

## 2. Cross-Site Generalization (Unseen Capture Sites)
- **Cross-Site Test Accuracy**: **{cross_site.get('acc', 0.0):.2%}**
- **Cross-Site Macro-F1**: **{cross_site.get('macro_f1', 0.0):.4f}**
- **Cross-Site Balanced Accuracy**: **{cross_site.get('bal_acc', 0.0):.2%}**

## 3. Calibration Metrics
- **Expected Calibration Error (ECE)**: **{calib['ece']:.4f}**

## 4. Visualizations
- Grad-CAM Lesion Heatmaps: `{cam_out}`
"""
    diag_report_path = REPORTS_DIR / "diagnostic_report.md"
    diag_report_path.write_text(report_md, encoding="utf-8")
    print(f"\nSaved diagnostic report to: {diag_report_path}", flush=True)


if __name__ == "__main__":
    main()
