"""Plant disease classifier model architecture (Model A).

Combines:
- timm feature extraction backbone (default: ConvNeXt-Tiny)
- Pluggable spatial pooling (GeM, Attention, Avg, Concat)
- BN-neck classification head with dropout
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agrobot.models.pooling import build_pooling


class PlantClassifier(nn.Module):
    """Model A: High-capacity plant pathology classifier."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 15,
        pool_type: str = "gem",
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pool_type = pool_type

        # Backbone without default classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            # Fallback probe
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                feat_map = self.backbone.forward_features(dummy)
                in_features = feat_map.shape[1]

        self.in_features = in_features

        # Spatial Pooling
        self.pool, pooled_dim = build_pooling(pool_type, in_features)

        # BN-Neck Classifier Head
        self.bn_neck = nn.BatchNorm1d(pooled_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(pooled_dim, num_classes)

        # Initialize head weights cleanly
        nn.init.constant_(self.bn_neck.weight, 1.0)
        nn.init.constant_(self.bn_neck.bias, 0.0)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [B, C, H, W] spatial feature map from backbone."""
        return self.backbone.forward_features(x)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.extract_feature_map(x)
        # Apply custom pooling
        pooled = self.pool(feat_map)
        if pooled.dim() > 2:
            pooled = pooled.flatten(1)

        norm_feat = self.bn_neck(pooled)
        logits = self.fc(self.dropout(norm_feat))

        if return_features:
            return logits, norm_feat
        return logits

    def get_parameter_groups(
        self,
        base_lr: float,
        weight_decay: float = 1e-2,
        backbone_lr_scale: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Return differentiated optimizer parameter groups with scaled backbone LR."""
        backbone_params = []
        head_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if bias or 1D parameter (e.g. norm weights, GeM p)
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            elif name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": head_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "lr": base_lr,
                "weight_decay": 0.0,
            },
        ]
