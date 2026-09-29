"""Unit tests for augmentations and transforms pipeline."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from agrobot.data.backgrounds import BackgroundBank
from agrobot.data.transforms import (
    RandomBackgroundSwap,
    get_eval_transforms,
    get_train_transforms,
)


def test_random_background_swap_preserves_shape_and_dtype():
    bank = BackgroundBank()
    swap = RandomBackgroundSwap(bank=bank, p=1.0)

    # 256x256 RGB image and centred leaf mask
    img = np.full((256, 256, 3), 100, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 60, 255, -1)

    res = swap(image=img, mask=mask)
    out = res["image"]

    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    # Center of leaf should retain original image value
    assert np.allclose(out[128, 128], [100, 100, 100], atol=10)
    # Outside leaf should have changed from original 100
    assert not np.allclose(out[10, 10], [100, 100, 100], atol=1)


def test_random_background_swap_handles_empty_or_full_mask():
    bank = BackgroundBank()
    swap = RandomBackgroundSwap(bank=bank, p=1.0)
    img = np.full((128, 128, 3), 50, dtype=np.uint8)

    # Empty mask
    res_empty = swap(image=img, mask=np.zeros((128, 128), dtype=np.uint8))
    assert np.array_equal(res_empty["image"], img)

    # Full mask
    res_full = swap(image=img, mask=np.full((128, 128), 255, dtype=np.uint8))
    assert np.array_equal(res_full["image"], img)


def test_train_transforms_levels():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 50, 255, -1)

    for level in ("light", "medium", "heavy"):
        t = get_train_transforms(image_size=160, aug_level=level, p_bg_swap=0.5)
        res = t(image=img, mask=mask)
        tensor = res["image"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 160, 160)
        assert tensor.dtype == torch.float32


def test_eval_transforms():
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    t = get_eval_transforms(image_size=224)
    res = t(image=img)
    tensor = res["image"]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
