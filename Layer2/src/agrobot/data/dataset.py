"""PyTorch Dataset, sampling utilities, and batch collation for AgroBot.

Handles:
1. Multi-class indexing (15 PlantVillage classes + optional 16th '__not_a_leaf__' reject class).
2. Grouped and split-aware data loading from manifest/splits.csv.
3. Synchronized mask loading for background swapping.
4. Sqrt-inverse-frequency class weighting to counteract 21x class imbalance.
5. MixUp / CutMix batch collator with label smoothing.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.backgrounds import BackgroundBank
from agrobot.data.manifest import load_manifest
from agrobot.data.segment import mask_path_for
from agrobot.paths import MASK_CACHE, NOT_A_LEAF, SPLITS_CSV

PLANTVILLAGE_CLASSES = (
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
)

ALL_CLASSES = PLANTVILLAGE_CLASSES + (NOT_A_LEAF,)

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(ALL_CLASSES)}
NUM_PLANTVILLAGE_CLASSES = len(PLANTVILLAGE_CLASSES)
NUM_ALL_CLASSES = len(ALL_CLASSES)


def one_hot(
    labels: torch.Tensor,
    num_classes: int = NUM_PLANTVILLAGE_CLASSES,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Convert integer labels to one-hot floats with optional label smoothing."""
    bs = labels.size(0)
    out = torch.full((bs, num_classes), smoothing / num_classes, dtype=torch.float32, device=labels.device)
    out.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + (smoothing / num_classes))
    return out


class PlantVillageDataset(Dataset):
    """Dataset for PlantVillage images and associated leaf masks."""

    def __init__(
        self,
        df_or_path: pd.DataFrame | Path | str = SPLITS_CSV,
        split: str | None = "train",
        transform: Callable | None = None,
        mask_cache: Path = MASK_CACHE,
        include_not_a_leaf: bool = False,
        not_a_leaf_fraction: float = 0.05,
        bank: BackgroundBank | None = None,
        load_mask: bool | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.mask_cache = Path(mask_cache)
        self.include_not_a_leaf = include_not_a_leaf
        self.bank = bank or (BackgroundBank() if include_not_a_leaf else None)

        if isinstance(df_or_path, (str, Path)):
            self.df = load_manifest(df_or_path)
        else:
            self.df = df_or_path.copy()

        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        self.samples = self.df.to_dict("records")
        self.num_real_samples = len(self.samples)

        # Only load masks when training/augmenting (skip for eval to maximize throughput)
        self.load_mask = (split == "train") if load_mask is None else load_mask

        # 16th class: not_a_leaf virtual samples
        if include_not_a_leaf and self.num_real_samples > 0:
            self.num_not_a_leaf = int(math.ceil(self.num_real_samples * not_a_leaf_fraction))
        else:
            self.num_not_a_leaf = 0

    def __len__(self) -> int:
        return self.num_real_samples + self.num_not_a_leaf

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Synthetic not_a_leaf sample
        if idx >= self.num_real_samples:
            size = 256
            bg_bgr = self.bank.sample(size=size)  # type: ignore[union-attr]
            img_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
            mask = np.zeros((size, size), dtype=np.uint8)
            label = CLASS_TO_IDX[NOT_A_LEAF]

            if self.transform is not None:
                res = self.transform(image=img_rgb, mask=mask)
                img_t = res["image"]
            else:
                img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

            return img_t, label

        # Real sample
        item = self.samples[idx]
        img_bgr = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image at: {item['path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cls_name = item["class"]
        label = CLASS_TO_IDX[cls_name]

        if self.load_mask:
            mask_p = mask_path_for(item["path"], cache=self.mask_cache)
            if mask_p.exists():
                mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)
            else:
                mask = np.full(img_rgb.shape[:2], 255, dtype=np.uint8)

            if self.transform is not None:
                res = self.transform(image=img_rgb, mask=mask, exclude_class=cls_name)
                img_t = res["image"]
            else:
                img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        else:
            if self.transform is not None:
                res = self.transform(image=img_rgb)
                img_t = res["image"]
            else:
                img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        return img_t, label


def compute_class_weights(df: pd.DataFrame, beta: float = 0.5) -> torch.Tensor:
    """Compute normalized sqrt-inverse-frequency class weights for cross-entropy loss."""
    counts = df["class"].value_counts().to_dict()
    weights = []
    for cls in PLANTVILLAGE_CLASSES:
        c = counts.get(cls, 1)
        weights.append((1.0 / c) ** beta)
    w_t = torch.tensor(weights, dtype=torch.float32)
    return w_t / w_t.mean()


def build_weighted_sampler(df: pd.DataFrame, beta: float = 0.5) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances class sampling frequencies."""
    counts = df["class"].value_counts().to_dict()
    cls_weight = {cls: (1.0 / max(1, count)) ** beta for cls, count in counts.items()}
    sample_weights = [cls_weight.get(r["class"], 1.0) for _, r in df.iterrows()]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class MixupCutmixCollator:
    """Batch collator applying MixUp and CutMix with label smoothing."""

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
        num_classes: int = NUM_PLANTVILLAGE_CLASSES,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def __call__(
        self, batch: list[tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([img for img, _ in batch], dim=0)
        labels = torch.tensor([lbl for _, lbl in batch], dtype=torch.int64)

        targets = one_hot(labels, num_classes=self.num_classes, smoothing=self.label_smoothing)

        if self.prob <= 0.0 or np.random.rand() > self.prob or len(batch) < 2:
            return images, targets

        bs, _, h, w = images.shape
        perm = torch.randperm(bs)

        use_cutmix = np.random.rand() < self.switch_prob

        if use_cutmix and self.cutmix_alpha > 0.0:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            # Bounding box for CutMix
            cut_rat = math.sqrt(1.0 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)

            cx = int(np.random.randint(w))
            cy = int(np.random.randint(h))

            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
            actual_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            targets = actual_lam * targets + (1.0 - actual_lam) * targets[perm]
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[perm]
            targets = lam * targets + (1.0 - lam) * targets[perm]

        return images, targets
