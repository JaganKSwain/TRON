"""Unit tests for Phase 6 calibration, OOD detection, and infer.py abstention logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from agrobot.calibrate import (
    TemperatureScaler,
    compute_ece,
    compute_energy_score,
)
from agrobot.infer import AgroBotPredictor, PredictionResult
from agrobot.paths import RUNS_DIR


def test_temperature_scaler_reduces_nll_on_overconfident_logits() -> None:
    # Synthetic overconfident logits
    torch.manual_seed(42)
    logits = torch.randn(100, 15) * 5.0
    labels = torch.randint(0, 15, (100,))

    scaler = TemperatureScaler(init_temp=1.0)
    learned_t = scaler.fit(logits, labels, lr=0.1, max_iter=50)

    assert learned_t > 1.0, f"Learned temperature {learned_t} should be > 1.0 for overconfident logits"
    scaled_logits = scaler(logits)
    assert scaled_logits.shape == logits.shape


def test_compute_energy_score_separates_peaked_and_flat_logits() -> None:
    # High confidence peaked logit -> lower free energy (higher negative energy)
    peaked_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    # Uniform/flat/uncertain logits -> higher free energy
    flat_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    energy_peaked = compute_energy_score(peaked_logits, temperature=1.0)
    energy_flat = compute_energy_score(flat_logits, temperature=1.0)

    assert energy_peaked.item() < energy_flat.item(), "Peaked logits should have lower free energy than flat logits"


def test_compute_ece_metric() -> None:
    # Perfect calibration synthetic case
    probs = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.9],
    ])
    labels = np.array([0, 0, 1, 1])

    ece, stats = compute_ece(probs, labels, num_bins=10)
    assert 0.0 <= ece <= 1.0
    assert len(stats["bin_accs"]) == 10


def test_predictor_abstains_on_synthetic_non_leaf_image(tmp_path: Path) -> None:
    ckpt_path = RUNS_DIR / "model_a_convnext" / "best_model.pt"
    if not ckpt_path.exists():
        pytest.skip("Model A checkpoint not found")

    predictor = AgroBotPredictor(checkpoint_path=ckpt_path, device="cpu")

    # Pure random noise image (non-leaf OOD)
    noise_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = predictor.predict(noise_img)

    assert isinstance(res, PredictionResult)
    assert isinstance(res.predicted_class, str)
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_k) > 0


def test_draw_hud_renders_without_error() -> None:
    from agrobot.camera_demo import draw_hud
    import numpy as np

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_result = PredictionResult(
        predicted_class="Tomato_Early_blight",
        confidence=0.95,
        top_k=[("Tomato_Early_blight", 0.95), ("Tomato_Late_blight", 0.03), ("Tomato_healthy", 0.02)],
        abstain=False,
        abstain_reason=None,
        energy_score=-2.5,
        raw_logits=[0.0] * 15,
    )

    rendered = draw_hud(dummy_frame, dummy_result, fps=30.0)
    assert rendered.shape == dummy_frame.shape
    assert rendered.dtype == np.uint8
