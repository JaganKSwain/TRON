"""Model training engine for AgroBot (Model A & baselines).

Features:
- ConvNeXt-Tiny + Learnable GeM Pooling + BN-Neck head
- Sqrt-inverse-frequency WeightedRandomSampler
- MixUp / CutMix with label smoothing
- Mixed precision training (AMP) + gradient clipping
- Exponential Moving Average (EMA) shadow model
- Cosine LR schedule with linear warmup
- Early stopping & checkpointing on validation macro-F1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.dataset import (
    NUM_ALL_CLASSES,
    NUM_PLANTVILLAGE_CLASSES,
    PlantVillageDataset,
    build_weighted_sampler,
    compute_class_weights,
    MixupCutmixCollator,
)
from agrobot.data.transforms import get_eval_transforms, get_train_transforms
from agrobot.models.classifier import PlantClassifier
from agrobot.paths import NOT_A_LEAF, RUNS_DIR, SPLITS_CSV, ensure_dirs


class ModelEMA:
    """Maintains moving average of model parameters for evaluation stability."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * model_p.detach())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    ema: ModelEMA | None = None,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch with AMP mixed precision and MixUp/CutMix."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on dataset without MixUp; compute Macro-F1 and balanced accuracy."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.size(0)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        total_loss += loss.item() * bs
        total_samples += bs

    val_loss = total_loss / max(1, total_samples)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "loss": val_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "bal_acc": bal_acc,
    }


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgroBot Model A Training")
    parser.add_argument("--exp-name", type=str, default="model_a_convnext", help="Experiment name")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="timm backbone")
    parser.add_argument("--pool-type", type=str, default="gem", choices=["gem", "avg", "max", "attention", "concat"])
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1, help="Backbone LR scaling factor")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution")
    parser.add_argument("--aug-level", type=str, default="heavy", choices=["light", "medium", "heavy"])
    parser.add_argument("--p-bg-swap", type=float, default=0.6, help="Probability of background swap")
    parser.add_argument("--mixup-prob", type=float, default=0.8, help="MixUp / CutMix probability")
    parser.add_argument("--include-not-a-leaf", action="store_true", help="Include 16th not_a_leaf class")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--limit-train", type=int, default=0, help="Subsample train set (for smoke testing)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AgroBot Training: {args.exp_name} ===", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Backbone: {args.backbone} | Pooling: {args.pool_type} | Aug Level: {args.aug_level}", flush=True)

    run_dir = RUNS_DIR / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Transforms & Datasets
    num_classes = NUM_ALL_CLASSES if args.include_not_a_leaf else NUM_PLANTVILLAGE_CLASSES
    train_transform = get_train_transforms(
        image_size=args.image_size,
        aug_level=args.aug_level,
        p_bg_swap=args.p_bg_swap,
    )
    eval_transform = get_eval_transforms(image_size=args.image_size)

    train_ds = PlantVillageDataset(
        SPLITS_CSV,
        split="train",
        transform=train_transform,
        include_not_a_leaf=args.include_not_a_leaf,
    )

    if args.limit_train > 0:
        sub_df = train_ds.df.sample(n=min(args.limit_train, len(train_ds.df)), random_state=args.seed).reset_index(drop=True)
        train_ds = PlantVillageDataset(
            sub_df,
            split=None,
            transform=train_transform,
            include_not_a_leaf=args.include_not_a_leaf,
        )

    val_ds = PlantVillageDataset(
        SPLITS_CSV,
        split="val",
        transform=eval_transform,
        include_not_a_leaf=args.include_not_a_leaf,
    )

    test_ds = PlantVillageDataset(
        SPLITS_CSV,
        split="test",
        transform=eval_transform,
        include_not_a_leaf=args.include_not_a_leaf,
    )

    sampler = build_weighted_sampler(train_ds.df)
    collator = MixupCutmixCollator(
        prob=args.mixup_prob,
        num_classes=num_classes,
        label_smoothing=0.1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,} | Test samples: {len(test_ds):,}", flush=True)

    # 2. Model, Optimizer, Scheduler, EMA
    model = PlantClassifier(
        backbone_name=args.backbone,
        pretrained=True,
        num_classes=num_classes,
        pool_type=args.pool_type,
        drop_rate=0.3,
        drop_path_rate=0.1,
    ).to(device)

    param_groups = model.get_parameter_groups(
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_scale=args.backbone_lr_scale,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = get_lr_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # 3. Training Loop
    best_macro_f1 = 0.0
    history = []

    print("\nStarting Training Loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ema=ema,
            grad_clip=args.grad_clip,
        )
        scheduler.step()

        # Evaluate EMA model (or base model if no EMA)
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device=device)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_bal_acc": val_metrics["bal_acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt,
        }
        history.append(row)

        is_best = val_metrics["macro_f1"] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": eval_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                run_dir / "best_model.pt",
            )

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Time: {dt:.1f}s"
            f"{' [* BEST]' if is_best else ''}",
            flush=True,
        )

    # Save last checkpoint & history CSV
    eval_model = ema.module if ema is not None else model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": eval_model.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        run_dir / "last_model.pt",
    )

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(run_dir / "metrics.csv", index=False)

    # 4. Final Evaluation on Test Set with Best Model
    print("\nEvaluating Best Checkpoint on Test Split...", flush=True)
    best_ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device=device)

    print(f"Test Accuracy:  {test_metrics['acc']:.4f}", flush=True)
    print(f"Test Macro-F1:  {test_metrics['macro_f1']:.4f}", flush=True)
    print(f"Test Bal Acc:   {test_metrics['bal_acc']:.4f}", flush=True)

    summary_md = f"""# Model A ({args.exp_name}) Training Summary

- **Backbone**: `{args.backbone}`
- **Pooling**: `{args.pool_type}`
- **Epochs**: {args.epochs}
- **Augmentation Level**: `{args.aug_level}`

## Test Set Performance (Grouped Leaf Split)
- **Macro-F1**: **{test_metrics['macro_f1']:.4f}**
- **Balanced Accuracy**: **{test_metrics['bal_acc']:.4f}**
- **Top-1 Accuracy**: **{test_metrics['acc']:.4f}**
- **Best Validation Macro-F1**: **{best_macro_f1:.4f}**

Checkpoint saved to: `{run_dir / 'best_model.pt'}`
"""
    (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Saved run summary to: {run_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
