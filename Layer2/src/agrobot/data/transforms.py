"""Data augmentation and preprocessing transforms for AgroBot.

Features:
1. `RandomBackgroundSwap`: The core defense against PlantVillage capture-site shortcuts.
   Takes an image + binary leaf mask and blends the leaf over a randomly sampled
   cross-class harvested backdrop, procedural soil/foliage/sky texture, or field photo.
2. Multi-tier training transforms: `light`, `medium`, and `heavy`.
3. Standardized eval transforms for validation, testing, and cross-site diagnostic.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.data.backgrounds import BackgroundBank

# Standard ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomBackgroundSwap(DualTransform):
    """Composites leaf foreground onto a randomly chosen backdrop.

    Expects RGB uint8 image and single-channel uint8 mask (0=bg, 255=leaf).
    """

    def __init__(
        self,
        bank: BackgroundBank | None = None,
        blur_edge_ksize: int = 3,
        blur_edge_sigma: float = 1.0,
        always_apply: bool = False,
        p: float = 0.6,
    ) -> None:
        super().__init__(p=p)
        self.bank = bank or BackgroundBank()
        self.blur_edge_ksize = blur_edge_ksize
        self.blur_edge_sigma = blur_edge_sigma

    def apply_with_params(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        if np.random.rand() > self.p:
            return kwargs
        img = kwargs.get("image")
        mask = kwargs.get("mask")
        exclude_class = kwargs.get("exclude_class")
        if img is not None and mask is not None:
            mask_mean = float(mask.mean())
            if 2.0 <= mask_mean <= 253.0:
                h, w = img.shape[:2]
                bg_bgr = self.bank.sample(size=max(h, w), exclude_class=exclude_class)
                bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
                if bg_rgb.shape[:2] != (h, w):
                    bg_rgb = cv2.resize(bg_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                m_float = (mask > 0).astype(np.float32)
                if self.blur_edge_ksize > 1:
                    m_float = cv2.GaussianBlur(
                        m_float,
                        (self.blur_edge_ksize, self.blur_edge_ksize),
                        self.blur_edge_sigma,
                    )
                alpha = np.clip(m_float, 0.0, 1.0)[..., None]
                blended = (img.astype(np.float32) * alpha + bg_rgb.astype(np.float32) * (1.0 - alpha))
                kwargs["image"] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        return kwargs

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return img

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        return mask

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("blur_edge_ksize", "blur_edge_sigma")


def get_train_transforms(
    image_size: int = 224,
    aug_level: str = "heavy",
    p_bg_swap: float = 0.6,
    bank: BackgroundBank | None = None,
) -> A.Compose:
    """Build training transform pipeline for given intensity level."""
    aug_level = aug_level.lower()
    if aug_level not in {"light", "medium", "heavy"}:
        raise ValueError(f"Unknown aug_level: {aug_level!r}. Choose 'light', 'medium', or 'heavy'.")

    transforms: list[Any] = []

    if aug_level == "light":
        transforms = [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.4),
        ]
    elif aug_level == "medium":
        if p_bg_swap > 0.0:
            transforms.append(RandomBackgroundSwap(bank=bank, p=p_bg_swap))
        transforms.extend([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0), ratio=(0.8, 1.2)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-180, 180), border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), fill=0, p=0.4),
        ])
    else:  # heavy
        if p_bg_swap > 0.0:
            transforms.append(RandomBackgroundSwap(bank=bank, p=p_bg_swap))
        transforms.extend([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.4, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), rotate=(-180, 180), shear=(-15, 15), border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.Perspective(scale=(0.05, 0.1), border_mode=cv2.BORDER_REFLECT_101, p=0.4),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=25, p=0.5),
            A.RandomGamma(gamma_limit=(75, 125), p=0.5),
            A.CLAHE(clip_limit=3.0, p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7)),
                A.Defocus(radius=(1, 3)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.4),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            ], p=0.4),
            A.ImageCompression(quality_range=(40, 95), p=0.4),
            A.Downscale(scale_range=(0.6, 0.95), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.5),
        ])

    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Build deterministic evaluation transform pipeline."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
