"""Tests for the RA8P1 export chain (P0).

These guard the properties that make the MCU model shippable. They are cheap:
the expensive artifacts (TFLite graph, Vela report) are produced by
`export_tflite_int8.py` / `vela_check.py` and read back here, so a broken
architecture edit fails in seconds rather than after a training run.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from agrobot.export_tflite_int8 import write_calibration_npy
from agrobot.models.mcu import MCU_INPUT_SIZE, McuClassifier, build_mcu_model
from agrobot.paths import EXPORT_DIR
from agrobot.vela_check import (
    FLASH_BUDGET_KB,
    MIN_NPU_OFFLOAD,
    SRAM_BUDGET_KB,
    VELA_SUPPORTED_OPS,
)

TFLITE_SUMMARY = EXPORT_DIR / "mcu" / "tflite_summary.json"
VELA_REPORT = EXPORT_DIR / "vela" / "vela_report.json"


# ---------------------------------------------------------------------------
# Architecture: fast, no artifacts needed
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mcu_model() -> McuClassifier:
    # pretrained=False keeps the test offline; only the graph shape matters here.
    return build_mcu_model(num_classes=16, pretrained=False).eval()


def test_mcu_model_uses_only_vela_safe_activations(mcu_model):
    """SiLU/GELU would silently fall back to the M85 CPU."""
    banned = (torch.nn.SiLU, torch.nn.GELU, torch.nn.Mish, torch.nn.ELU,
              torch.nn.LeakyReLU, torch.nn.PReLU)
    offenders = [n for n, m in mcu_model.named_modules() if isinstance(m, banned)]
    assert not offenders, f"unsupported activations: {offenders}"
    acts = {type(m).__name__ for _, m in mcu_model.named_modules()
            if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Hardswish))}
    assert acts <= {"ReLU", "ReLU6", "Hardswish"}


def test_vela_guard_rejects_an_unsupported_activation(mcu_model):
    """The guard must actually fire -- a silent guard is worse than none."""
    mcu_model.classifier = torch.nn.Sequential(torch.nn.SiLU(), mcu_model.classifier)
    with pytest.raises(ValueError, match="SiLU"):
        mcu_model._assert_vela_safe()


def test_mcu_model_fits_the_weight_budget(mcu_model):
    """INT8 weights must fit flash; ideally the 1 MB MRAM part too."""
    n = sum(p.numel() for p in mcu_model.parameters())
    assert n / 1024 < FLASH_BUDGET_KB, f"{n:,} params exceeds flash budget"
    assert n < 1_000_000, f"{n:,} params is larger than this design intends"


def test_mcu_model_emits_logits_not_probabilities(mcu_model):
    """Softmax stays off-graph so the energy score is available for rejection."""
    with torch.no_grad():
        out = mcu_model(torch.randn(1, 3, MCU_INPUT_SIZE, MCU_INPUT_SIZE))
    assert out.shape == (1, 16)
    assert abs(float(out.exp().sum()) - 1.0) > 1e-3, "output looks like a softmax"


def test_calibration_data_must_be_nhwc(tmp_path):
    """onnx2tf feeds this straight to TFLite without transposing; NCHW is silent poison."""
    with pytest.raises(ValueError, match="NHWC"):
        write_calibration_npy(tmp_path / "bad.npy", data=np.zeros((4, 3, 64, 64), np.float32))
    ok = write_calibration_npy(tmp_path / "ok.npy", n=2, size=32)
    assert np.load(ok).shape == (2, 32, 32, 3)


# ---------------------------------------------------------------------------
# Exported artifacts: skip when the export has not been run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tflite_summary() -> dict:
    if not TFLITE_SUMMARY.exists():
        pytest.skip("run export_tflite_int8.py first")
    return json.loads(TFLITE_SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vela_report() -> dict:
    if not VELA_REPORT.exists():
        pytest.skip("run vela_check.py first")
    return json.loads(VELA_REPORT.read_text(encoding="utf-8"))


def test_exported_graph_contains_only_npu_operators(tflite_summary):
    unsupported = set(tflite_summary["ops"]) - VELA_SUPPORTED_OPS
    assert not unsupported, f"operators Ethos-U55 cannot run: {sorted(unsupported)}"


def test_exported_graph_has_no_transposes(tflite_summary):
    """A per-conv layout shuffle is Vela-legal but would dominate the runtime."""
    assert tflite_summary["ops"].get("TRANSPOSE", 0) == 0


def test_export_is_fully_integer(tflite_summary):
    assert tflite_summary["n_float_tensors"] == 0
    assert tflite_summary["input"]["dtype"] == "int8"
    assert tflite_summary["output"]["dtype"] == "int8"


def test_export_input_is_nhwc_at_the_expected_size(tflite_summary):
    assert tflite_summary["input"]["shape"] == [1, MCU_INPUT_SIZE, MCU_INPUT_SIZE, 3]


def test_conversion_preserved_the_function(tflite_summary):
    """Low correlation here means a layout bug, not quantisation loss."""
    fid = tflite_summary.get("fidelity")
    if fid is None:
        pytest.skip("summary predates the fidelity check")
    assert fid["layout_ok"]
    assert fid["logit_pearson_r_min"] > 0.90


def test_vela_places_everything_on_the_npu(vela_report):
    assert vela_report["n_ops_cpu"] == 0
    assert vela_report["npu_offload"] >= MIN_NPU_OFFLOAD


def test_vela_fits_ra8p1_memory(vela_report):
    assert vela_report["sram_kb"] <= SRAM_BUDGET_KB
    assert vela_report["flash_kb"] <= FLASH_BUDGET_KB


def test_vela_was_compiled_against_the_real_arena_size(vela_report):
    """Vela's default assumes 4 MB; passing that would over-report the fit."""
    assert vela_report["arena_cache_size"] == SRAM_BUDGET_KB * 1024


def test_vela_gate_verdict_is_fully_measured(vela_report):
    assert vela_report["unmeasured"] == []
    assert vela_report["passed"] is True
