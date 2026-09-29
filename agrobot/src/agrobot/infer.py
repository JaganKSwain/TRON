"""Inference and Abstention Prediction Engine for AgroBot (Phase 6).

Provides the high-level `AgroBotPredictor` API with calibrated probabilities
and energy-based Out-Of-Distribution / non-leaf abstention logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from agrobot.calibrate import TemperatureScaler, compute_energy_score
from agrobot.data.dataset import (
    IDX_TO_CLASS,
    NUM_ALL_CLASSES,
    NUM_PLANTVILLAGE_CLASSES,
    PLANTVILLAGE_CLASSES,
)
from agrobot.data.transforms import get_eval_transforms
from agrobot.models.classifier import PlantClassifier
from agrobot.paths import NOT_A_LEAF, RUNS_DIR


@dataclass
class PredictionResult:
    """Diagnostic prediction result with open-set abstention details."""

    predicted_class: str
    confidence: float  # Calibrated probability (0.0 to 1.0)
    top_k: list[tuple[str, float]]
    abstain: bool  # True if model rejects prediction (e.g. non-leaf / OOD / uncertain)
    abstain_reason: str | None  # Reason for abstention, if any
    energy_score: float  # Free energy score (lower = more confident leaf)
    raw_logits: list[float]


class AgroBotPredictor:
    """Predictor with integrated temperature scaling and OOD energy-based abstention."""

    def __init__(
        self,
        checkpoint_path: Path | str = RUNS_DIR / "model_a_convnext" / "best_model.pt",
        calibration_path: Path | str | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: int = 224,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.transform = get_eval_transforms(image_size)

        # 1. Load Model Checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        model_args = ckpt.get("args", {})
        self.include_not_a_leaf = model_args.get("include_not_a_leaf", False)
        num_classes = NUM_ALL_CLASSES if self.include_not_a_leaf else NUM_PLANTVILLAGE_CLASSES

        self.model = PlantClassifier(
            backbone_name=model_args.get("backbone", "convnext_tiny"),
            pretrained=False,
            num_classes=num_classes,
            pool_type=model_args.get("pool_type", "gem"),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # 2. Load Calibration Config
        if calibration_path is None:
            calibration_path = self.checkpoint_path.parent / "calibration.json"

        self.calibration_path = Path(calibration_path)
        if self.calibration_path.exists():
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
            self.temperature = float(cal_data.get("temperature", 1.0))
            self.energy_threshold = float(cal_data.get("energy_threshold_95tpr", 10.0))
        else:
            self.temperature = 1.0
            self.energy_threshold = 100.0  # Open-set disabled by default if uncalibrated

    def _preprocess(self, image_input: str | Path | np.ndarray | Image.Image) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            bgr = cv2.imread(str(image_input))
            if bgr is None:
                raise ValueError(f"Could not load image from: {image_input}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 3:
                # Assume RGB if numpy array
                rgb = image_input
            else:
                raise ValueError(f"Unexpected image array shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        res = self.transform(image=rgb)
        return res["image"].unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        top_k: int = 3,
    ) -> PredictionResult:
        """Predict pathology class with temperature-calibrated confidence and abstention."""
        tensor = self._preprocess(image_input)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            raw_logits = self.model(tensor)

        energy = float(compute_energy_score(raw_logits, temperature=self.temperature).item())
        cal_logits = raw_logits / self.temperature
        probs = F.softmax(cal_logits, dim=-1)[0]

        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(0)))
        top_k_list = [
            (IDX_TO_CLASS.get(int(idx.item()), "unknown"), float(prob.item()))
            for idx, prob in zip(top_indices, top_probs)
        ]

        pred_class, pred_conf = top_k_list[0]

        # Check Abstention Criteria
        abstain = False
        abstain_reason = None

        if energy > self.energy_threshold:
            abstain = True
            abstain_reason = f"energy_score_above_threshold ({energy:.2f} > {self.energy_threshold:.2f}): likely non-leaf / OOD"
        elif pred_class == NOT_A_LEAF:
            abstain = True
            abstain_reason = "classified_as_not_a_leaf"
        elif pred_conf < self.confidence_threshold:
            abstain = True
            abstain_reason = f"low_calibrated_confidence ({pred_conf:.2%} < {self.confidence_threshold:.2%})"

        return PredictionResult(
            predicted_class=pred_class,
            confidence=pred_conf,
            top_k=top_k_list,
            abstain=abstain,
            abstain_reason=abstain_reason,
            energy_score=energy,
            raw_logits=raw_logits[0].cpu().tolist(),
        )
