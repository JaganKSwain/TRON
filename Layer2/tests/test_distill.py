"""Unit tests for Phase 7 knowledge distillation module."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from agrobot.distill import DistillationLoss
from agrobot.models.mcu import build_mcu_model


def test_distillation_loss_hard_and_soft_balance() -> None:
    loss_fn = DistillationLoss(temperature=3.0, alpha=0.5)

    student_logits = torch.randn(4, 15)
    teacher_logits = torch.randn(4, 15)
    targets = torch.tensor([0, 2, 4, 1])

    total_loss, ce_loss, kl_loss = loss_fn(student_logits, teacher_logits, targets)

    assert total_loss.item() > 0.0
    assert ce_loss.item() > 0.0
    assert kl_loss.item() >= 0.0
    assert torch.isclose(total_loss, 0.5 * ce_loss + 0.5 * kl_loss, atol=1e-5)


def test_distillation_loss_identical_logits_minimizes_kl() -> None:
    loss_fn = DistillationLoss(temperature=2.0, alpha=1.0)

    logits = torch.randn(4, 15)
    targets = torch.tensor([0, 1, 2, 3])

    total_loss, _, kl_loss = loss_fn(logits, logits, targets)
    assert kl_loss.item() < 1e-4, "KL divergence between identical logits should be ~0"


def test_student_forward_shape_at_mcu_resolution() -> None:
    model = build_mcu_model(num_classes=15, pretrained=False)
    inputs = torch.randn(2, 3, 160, 160)
    outputs = model(inputs)
    assert outputs.shape == (2, 15)
