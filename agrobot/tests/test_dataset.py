"""Unit tests for PlantVillageDataset, samplers, and batch collators."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from agrobot.data.dataset import (
    CLASS_TO_IDX,
    NUM_PLANTVILLAGE_CLASSES,
    PlantVillageDataset,
    build_weighted_sampler,
    compute_class_weights,
    MixupCutmixCollator,
    one_hot,
)
from agrobot.paths import SPLITS_CSV


def test_one_hot_encoding_and_smoothing():
    labels = torch.tensor([0, 2, 4], dtype=torch.int64)
    smooth = 0.1
    oh = one_hot(labels, num_classes=5, smoothing=smooth)
    assert oh.shape == (3, 5)
    assert torch.allclose(oh.sum(dim=1), torch.ones(3))
    # Active class should have 1 - 0.1 + 0.1/5 = 0.92
    assert pytest.approx(float(oh[0, 0]), rel=1e-3) == 0.92
    # Inactive class should have 0.1/5 = 0.02
    assert pytest.approx(float(oh[0, 1]), rel=1e-3) == 0.02


def test_dataset_splits_loading():
    if not SPLITS_CSV.exists():
        pytest.skip("splits.csv does not exist")

    ds_train = PlantVillageDataset(SPLITS_CSV, split="train")
    ds_val = PlantVillageDataset(SPLITS_CSV, split="val")
    ds_test = PlantVillageDataset(SPLITS_CSV, split="test")

    assert len(ds_train) > 10000
    assert len(ds_val) > 2000
    assert len(ds_test) > 2000

    img, label = ds_train[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert 0 <= label < NUM_PLANTVILLAGE_CLASSES


def test_dataset_not_a_leaf_integration():
    if not SPLITS_CSV.exists():
        pytest.skip("splits.csv does not exist")

    ds = PlantVillageDataset(SPLITS_CSV, split="val", include_not_a_leaf=True, not_a_leaf_fraction=0.05)
    assert len(ds) > ds.num_real_samples

    # Fetch synthetic sample from the end of dataset
    img, label = ds[len(ds) - 1]
    assert isinstance(img, torch.Tensor)
    assert label == CLASS_TO_IDX["__not_a_leaf__"]


def test_compute_class_weights_and_sampler():
    data = [
        {"path": "p1", "class": "Potato___healthy", "split": "train"},
        {"path": "p2", "class": "Potato___Early_blight", "split": "train"},
        {"path": "p3", "class": "Potato___Early_blight", "split": "train"},
    ]
    df = pd.DataFrame(data)
    weights = compute_class_weights(df, beta=0.5)
    assert weights.shape == (NUM_PLANTVILLAGE_CLASSES,)
    assert pytest.approx(float(weights.mean()), rel=1e-3) == 1.0

    sampler = build_weighted_sampler(df, beta=0.5)
    assert len(sampler) == len(df)


def test_mixup_cutmix_collator():
    collator = MixupCutmixCollator(prob=1.0, label_smoothing=0.0, num_classes=15)
    batch = [
        (torch.zeros(3, 32, 32), 0),
        (torch.ones(3, 32, 32), 1),
    ]
    imgs, targets = collator(batch)
    assert imgs.shape == (2, 3, 32, 32)
    assert targets.shape == (2, 15)
    assert torch.allclose(targets.sum(dim=1), torch.ones(2), atol=1e-5)
