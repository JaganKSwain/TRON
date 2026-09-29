"""Unit tests for pooling layers and Model A classifier."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agrobot.models.classifier import PlantClassifier
from agrobot.models.pooling import (
    AttentionPool2d,
    ConcatPool2d,
    GeMPool2d,
    build_pooling,
)


def test_gem_pool_equals_gap_at_p1():
    pool = GeMPool2d(p=1.0, learnable=False)
    x = torch.rand(4, 32, 7, 7) + 0.1
    gem_out = pool(x)
    gap_out = F.avg_pool2d(x, (7, 7)).flatten(1)
    assert torch.allclose(gem_out, gap_out, atol=1e-5)


def test_gem_pool_gradients_flow_to_p():
    pool = GeMPool2d(p=3.0, learnable=True)
    x_raw = torch.rand(2, 16, 7, 7, requires_grad=True)
    x = x_raw + 0.1
    out = pool(x)
    loss = out.sum()
    loss.backward()

    assert pool.p.grad is not None
    assert torch.isfinite(pool.p.grad)
    assert x_raw.grad is not None


def test_attention_pool():
    pool = AttentionPool2d(in_features=64)
    x = torch.randn(4, 64, 7, 7)
    out = pool(x)
    assert out.shape == (4, 64)


def test_concat_pool():
    pool = ConcatPool2d(p=3.0)
    x = torch.rand(4, 32, 7, 7) + 0.1
    out = pool(x)
    assert out.shape == (4, 64)


def test_build_pooling_factory():
    for p_type in ("gem", "avg", "max", "attention", "concat"):
        layer, dim = build_pooling(p_type, in_features=128)
        assert isinstance(layer, torch.nn.Module)
        if p_type == "concat":
            assert dim == 256
        else:
            assert dim == 128


def test_plant_classifier_forward():
    # Use lightweight mobilenetv2 backbone for fast unit test
    model = PlantClassifier(
        backbone_name="mobilenetv2_050",
        pretrained=False,
        num_classes=15,
        pool_type="gem",
    )
    x = torch.randn(2, 3, 160, 160)
    logits = model(x)
    assert logits.shape == (2, 15)

    logits, feats = model(x, return_features=True)
    assert logits.shape == (2, 15)
    assert feats.shape[0] == 2

    param_groups = model.get_parameter_groups(base_lr=1e-3)
    assert len(param_groups) == 3
