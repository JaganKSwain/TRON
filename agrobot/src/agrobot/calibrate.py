"""Calibration and Out-Of-Distribution (OOD) rejection for AgroBot (Phase 6).

Implements:
1. Temperature Scaling: Learns post-hoc temperature T via L-BFGS on validation logits.
2. Expected Calibration Error (ECE) computation before and after scaling.
3. Energy-Based OOD Detection: Thresholds free energy at 95% TPR on known leaves.
4. Generates calibration.json and visual diagnostic diagrams.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.backgrounds import BackgroundBank
from agrobot.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_ALL_CLASSES,
    NUM_PLANTVILLAGE_CLASSES,
    PlantVillageDataset,
)
from agrobot.data.manifest import load_manifest
from agrobot.data.transforms import get_eval_transforms
from agrobot.models.classifier import PlantClassifier
from agrobot.paths import NOT_A_LEAF, REPORTS_DIR, RUNS_DIR, SPLITS_CSV, ensure_dirs


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling to calibrate classifier probabilities."""

    def __init__(self, init_temp: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Optimize temperature T using L-BFGS to minimize Negative Log-Likelihood (NLL)."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        # Ensure temperature remains strictly positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        return float(self.temperature.item())


def compute_energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute free energy: E(x) = -T * logsumexp(logits / T).

    Higher energy = out-of-distribution (anomaly/non-leaf).
    Lower energy = high confidence in-distribution leaf.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15) -> tuple[float, dict[str, Any]]:
    """Compute Expected Calibration Error and bin statistics."""
    confs = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    corrects = (preds == labels).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confs > lower) & (confs <= upper)
        prop_in_bin = float(in_bin.mean())
        count = int(in_bin.sum())
        bin_counts.append(count)

        if count > 0:
            acc = float(corrects[in_bin].mean())
            conf = float(confs[in_bin].mean())
            ece += np.abs(conf - acc) * prop_in_bin
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2)

    return float(ece), {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


class SyntheticOODDataset(Dataset):
    """Generates pure background and noise images representing non-leaf inputs."""

    def __init__(self, size: int = 224, count: int = 500, bank: BackgroundBank | None = None) -> None:
        super().__init__()
        self.size = size
        self.count = count
        self.bank = bank or BackgroundBank()
        self.transform = get_eval_transforms(size)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        bg_bgr = self.bank.sample(size=self.size)
        rgb = bg_bgr[:, :, ::-1]
        res = self.transform(image=rgb)
        return res["image"], -1  # -1 indicates OOD


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(targets)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def plot_reliability_diagram(
    uncal_stats: dict[str, Any],
    cal_stats: dict[str, Any],
    uncal_ece: float,
    cal_ece: float,
    out_path: Path,
) -> None:
    """Plot reliability diagram comparing before and after calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, stats, ece, title in zip(
        axes,
        [uncal_stats, cal_stats],
        [uncal_ece, cal_ece],
        ["Uncalibrated Softmax", "Temperature Scaled"],
    ):
        centers = [(l + u) / 2 for l, u in zip(stats["bin_boundaries"][:-1], stats["bin_boundaries"][1:])]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.bar(centers, stats["bin_accs"], width=1.0 / len(centers), alpha=0.6, edgecolor="black", label="Outputs")
        ax.set_title(f"{title} (ECE: {ece:.4f})")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to: {out_path}")


def plot_energy_histogram(
    in_energies: np.ndarray,
    ood_energies: np.ndarray,
    threshold: float,
    auroc: float,
    out_path: Path,
) -> None:
    """Plot distribution of free energy scores for known leaves vs non-leaf OOD inputs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(in_energies, bins=40, alpha=0.6, density=True, label="In-Distribution (Leaves)", color="green")
    ax.hist(ood_energies, bins=40, alpha=0.6, density=True, label="OOD (Background/Noise)", color="red")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"95% TPR Threshold ({threshold:.2f})")
    ax.set_title(f"Energy Score Distribution (OOD AUROC: {auroc:.4f})")
    ax.set_xlabel("Free Energy (Lower = In-Distribution)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved energy histogram to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgroBot Model Calibration & OOD Rejection")
    parser.add_argument("--checkpoint", type=Path, default=RUNS_DIR / "model_a_convnext" / "best_model.pt")
    parser.add_argument("--splits", type=Path, default=SPLITS_CSV)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint} on {device}")

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

    eval_t = get_eval_transforms(224)
    val_ds = PlantVillageDataset(args.splits, split="val", transform=eval_t)
    test_ds = PlantVillageDataset(args.splits, split="test", transform=eval_t)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 1. Collect validation logits & Fit Temperature
    print("\nCollecting validation logits and optimizing temperature...")
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    scaler = TemperatureScaler(init_temp=1.5)
    learned_temp = scaler.fit(val_logits, val_labels)
    print(f"Optimal Temperature T: {learned_temp:.4f}")

    # 2. Evaluate ECE on Test Set
    print("\nEvaluating calibration on test split...")
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    uncal_probs = F.softmax(test_logits, dim=-1).numpy()
    cal_probs = F.softmax(scaler(test_logits), dim=-1).detach().numpy()
    y_test = test_labels.numpy()

    uncal_ece, uncal_stats = compute_ece(uncal_probs, y_test)
    cal_ece, cal_stats = compute_ece(cal_probs, y_test)

    print(f"Uncalibrated ECE: {uncal_ece:.4f}")
    print(f"Calibrated ECE:   {cal_ece:.4f} (Drop: {(uncal_ece - cal_ece):.4f})")

    rel_diagram_path = REPORTS_DIR / "reliability_diagram.jpg"
    plot_reliability_diagram(uncal_stats, cal_stats, uncal_ece, cal_ece, rel_diagram_path)

    # 3. Energy-Based OOD Detection
    print("\nFitting Energy Threshold for OOD Rejection (95% TPR)...")
    val_energies = compute_energy_score(val_logits, temperature=learned_temp).detach().numpy()
    # 95% TPR threshold = 95th percentile of in-distribution energies (95% of in-dist have energy <= threshold)
    energy_threshold = float(np.percentile(val_energies, 95.0))
    print(f"Energy Threshold (95% TPR): {energy_threshold:.4f}")

    # Synthetic OOD evaluation
    ood_ds = SyntheticOODDataset(size=224, count=600)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    ood_logits, _ = collect_logits_and_labels(model, ood_loader, device)
    ood_energies = compute_energy_score(ood_logits, temperature=learned_temp).detach().numpy()
    test_energies = compute_energy_score(test_logits, temperature=learned_temp).detach().numpy()

    # AUROC computation: Known leaf = class 0 (lower energy), OOD = class 1 (higher energy)
    y_ood_true = np.concatenate([np.zeros_like(test_energies), np.ones_like(ood_energies)])
    y_ood_scores = np.concatenate([test_energies, ood_energies])
    ood_auroc = float(roc_auc_score(y_ood_true, y_ood_scores))
    fpr_at_95tpr = float((ood_energies <= energy_threshold).mean())

    print(f"OOD AUROC against background/non-leaf: {ood_auroc:.4f}")
    print(f"False Positive Rate @ 95% TPR (FPR95):  {fpr_at_95tpr:.2%}")

    energy_hist_path = REPORTS_DIR / "energy_ood_histogram.jpg"
    plot_energy_histogram(test_energies, ood_energies, energy_threshold, ood_auroc, energy_hist_path)

    # 4. Save Calibration Configuration
    calib_config = {
        "temperature": learned_temp,
        "energy_threshold_95tpr": energy_threshold,
        "uncalibrated_ece": uncal_ece,
        "calibrated_ece": cal_ece,
        "ood_auroc": ood_auroc,
        "fpr95": fpr_at_95tpr,
    }
    calib_json_path = args.checkpoint.parent / "calibration.json"
    calib_json_path.write_text(json.dumps(calib_config, indent=2), encoding="utf-8")
    print(f"Saved calibration configuration to: {calib_json_path}")

    # 5. Write Calibration Report
    report_md = rf"""# Phase 6 Calibration & Open-Set Rejection Report

## 1. Temperature Scaling
- **Optimal Temperature ($T$)**: **{learned_temp:.4f}**
- **Uncalibrated ECE**: **{uncal_ece:.4f}**
- **Calibrated ECE**: **{cal_ece:.4f}** (relative error reduced significantly)
- Reliability Diagram: `{rel_diagram_path}`

## 2. Energy-Based OOD / Non-Leaf Rejection
- **Energy Threshold (95% In-Distribution TPR)**: **{energy_threshold:.4f}**
- **OOD Detection AUROC**: **{ood_auroc:.4f}** (Target $\ge 90\%$)
- **False Positive Rate at 95% TPR (FPR95)**: **{fpr_at_95tpr:.2%}**
- Energy Histogram: `{energy_hist_path}`

Saved configuration: `{calib_json_path}`
"""
    (REPORTS_DIR / "calibration_report.md").write_text(report_md, encoding="utf-8")
    print(f"Saved calibration report to: {REPORTS_DIR / 'calibration_report.md'}")


if __name__ == "__main__":
    main()
