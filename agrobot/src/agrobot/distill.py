"""Knowledge Distillation and QAT training engine for Model B (MCU) (Phase 7).

Distills Model A (ConvNeXt-Tiny + GeM teacher) into Model B (MobileNetV2-0.5 student)
for deployment on Renesas RA8P1 / Ethos-U55 NPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_ALL_CLASSES,
    NUM_PLANTVILLAGE_CLASSES,
    MixupCutmixCollator,
    PlantVillageDataset,
    build_weighted_sampler,
    compute_class_weights,
)
from agrobot.data.manifest import load_manifest
from agrobot.data.transforms import get_eval_transforms, get_train_transforms
from agrobot.models.classifier import PlantClassifier
from agrobot.models.mcu import MCU_INPUT_SIZE, McuClassifier, build_mcu_model
from agrobot.paths import REPORTS_DIR, RUNS_DIR, SPLITS_CSV, ensure_dirs


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss combining Cross-Entropy and Temperature-Scaled KL-Divergence."""

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.6,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.class_weights = class_weights
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Hard Cross-Entropy Loss (against targets / smoothed labels)
        if targets.ndim == 2:
            # Soft targets from mixup/label smoothing
            log_probs = F.log_softmax(student_logits, dim=-1)
            if self.class_weights is not None:
                weights = self.class_weights.to(student_logits.device)
                ce_loss = -(targets * log_probs * weights).sum(dim=-1).mean()
            else:
                ce_loss = -(targets * log_probs).sum(dim=-1).mean()
        else:
            ce_loss = F.cross_entropy(
                student_logits, targets, weight=self.class_weights.to(student_logits.device) if self.class_weights is not None else None
            )

        # 2. Soft KL Divergence Loss
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = self.kl_div(student_soft, teacher_soft) * (T * T)

        # Combined Loss
        total_loss = (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss
        return total_loss, ce_loss, kl_loss


@torch.no_grad()
def evaluate_student(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets_t = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets_t)

        total_loss += float(loss.item())
        num_batches += 1
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    avg_loss = total_loss / max(1, num_batches)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1, "bal_acc": bal_acc}


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 3.0,
    alpha: float = 0.6,
    device: torch.device = torch.device("cuda"),
    exp_dir: Path = RUNS_DIR / "model_b_mcu",
) -> dict[str, Any]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    distill_criterion = DistillationLoss(temperature=temperature, alpha=alpha)

    best_val_f1 = 0.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    print(f"\nStarting MCU Distillation Training ({epochs} epochs)...", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        student_model.train()
        train_loss_accum = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # Forward teacher (frozen) at full resolution
                with torch.no_grad():
                    # If student input is 160, resize to 224 for teacher if needed
                    if images.shape[-1] != 224:
                        teacher_images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
                    else:
                        teacher_images = images
                    teacher_logits = teacher_model(teacher_images)

                # Forward student
                student_logits = student_model(images)
                loss, ce_l, kl_l = distill_criterion(student_logits, teacher_logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += float(loss.item())
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss_accum / max(1, num_batches)

        # Validate
        val_metrics = evaluate_student(student_model, val_loader, device)
        is_best = val_metrics["macro_f1"] > best_val_f1

        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "student_state": student_model.state_dict(),
                    "val_metrics": val_metrics,
                    "arch": "mobilenetv2_050",
                    "input_size": MCU_INPUT_SIZE,
                },
                exp_dir / "best_model.pt",
            )

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
            f"{' [* BEST]' if is_best else ''}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["macro_f1"],
            "val_acc": val_metrics["acc"],
        })

    # Evaluate Best Checkpoint on Test Set
    print("\nEvaluating Best Student Checkpoint on Test Split...", flush=True)
    best_ckpt = torch.load(exp_dir / "best_model.pt", map_location=device)
    student_model.load_state_dict(best_ckpt["student_state"])
    test_metrics = evaluate_student(student_model, test_loader, device)

    print(f"Student Test Accuracy:  {test_metrics['acc']:.4f}", flush=True)
    print(f"Student Test Macro-F1:  {test_metrics['macro_f1']:.4f}", flush=True)
    print(f"Student Test Bal Acc:   {test_metrics['bal_acc']:.4f}", flush=True)

    # Save summary
    summary_md = f"""# Model B (MCU MobileNetV2-0.5) Distillation Summary

## Performance
- **Architecture**: MobileNetV2 (`width_mult=0.5`)
- **Input Resolution**: `{MCU_INPUT_SIZE}x{MCU_INPUT_SIZE}`
- **Best Validation Epoch**: {best_epoch}
- **Best Validation Macro-F1**: **{best_val_f1:.4f}**
- **Test Set Accuracy**: **{test_metrics['acc']:.4f}**
- **Test Set Macro-F1**: **{test_metrics['macro_f1']:.4f}**
- **Test Balanced Accuracy**: **{test_metrics['bal_acc']:.4f}**

## Knowledge Distillation Hyperparameters
- Teacher: Model A (`convnext_tiny` + GeM)
- Distillation Temperature: `{temperature}`
- KD Loss Weight ($\alpha$): `{alpha}`
- Optimizer: AdamW, lr={lr}, epochs={epochs}
"""
    (exp_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    (REPORTS_DIR / "distill_report.md").write_text(summary_md, encoding="utf-8")

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model B MCU Knowledge Distillation")
    parser.add_argument("--teacher", type=Path, default=RUNS_DIR / "model_a_convnext" / "best_model.pt")
    parser.add_argument("--splits", type=Path, default=SPLITS_CSV)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--image-size", type=int, default=MCU_INPUT_SIZE)
    parser.add_argument("--exp-name", type=str, default="model_b_mcu")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AgroBot MCU Distillation: {args.exp_name} ===")
    print(f"Device: {device} | Student Image Size: {args.image_size}x{args.image_size}")

    # 1. Load Teacher Model A
    print(f"Loading Teacher Checkpoint: {args.teacher}")
    t_ckpt = torch.load(args.teacher, map_location=device)
    t_args = t_ckpt.get("args", {})
    num_classes = NUM_ALL_CLASSES if t_args.get("include_not_a_leaf", False) else NUM_PLANTVILLAGE_CLASSES

    teacher = PlantClassifier(
        backbone_name=t_args.get("backbone", "convnext_tiny"),
        pretrained=False,
        num_classes=num_classes,
        pool_type=t_args.get("pool_type", "gem"),
    ).to(device)
    teacher.load_state_dict(t_ckpt["model_state"])
    teacher.eval()

    # 2. Build Student Model B
    student = build_mcu_model(num_classes=num_classes, pretrained=True).to(device)

    # 3. Data Loaders (Student resolution 160x160)
    train_t = get_train_transforms(image_size=args.image_size, aug_level="medium", p_bg_swap=0.5)
    eval_t = get_eval_transforms(image_size=args.image_size)

    splits_df = load_manifest(args.splits)
    train_df = splits_df[splits_df["split"] == "train"].reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)

    train_ds = PlantVillageDataset(train_df, split=None, transform=train_t)
    val_ds = PlantVillageDataset(val_df, split=None, transform=eval_t)
    test_ds = PlantVillageDataset(test_df, split=None, transform=eval_t)

    sampler = build_weighted_sampler(train_df)
    collator = MixupCutmixCollator(prob=0.3, num_classes=num_classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_distillation(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        device=device,
        exp_dir=RUNS_DIR / args.exp_name,
    )


if __name__ == "__main__":
    main()
