"""Custom pooling layers for AgroBot classifiers.

Includes:
1. `GeMPool2d`: Generalized Mean pooling with learnable p parameter.
   Preserves localized lesion activation cues that Global Average Pooling (GAP) washes out.
2. `AttentionPool2d`: 2D spatial attention pooling.
3. `ConcatPool2d`: Concatenates multiple poolings (e.g. GeM + Max).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPool2d(nn.Module):
    """Generalized Mean Pooling 2D: f(x) = (1/HW * sum(x^p))^(1/p).

    When p=1.0, GeM is equivalent to standard Global Average Pooling.
    As p -> inf, GeM approaches Global Max Pooling.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else torch.tensor([p], dtype=torch.float32)
        self.eps = eps
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W]
        # Clamp inputs to eps to avoid NaN in pow when x is 0 or negative
        x_clamped = x.clamp(min=self.eps)
        p = self.p.to(x.device) if not self.learnable else self.p
        # (mean(x^p))^(1/p)
        x_pow = x_clamped.pow(p)
        pooled = F.avg_pool2d(x_pow, (x_pow.size(-2), x_pow.size(-1)))
        return pooled.pow(1.0 / p).flatten(1)

    def extra_repr(self) -> str:
        p_val = self.p.item() if isinstance(self.p, nn.Parameter) else float(self.p)
        return f"p={p_val:.3f}, eps={self.eps}, learnable={self.learnable}"


class AttentionPool2d(nn.Module):
    """Spatial self-attention pooling over 2D feature maps."""

    def __init__(self, in_features: int, hidden_features: int | None = None) -> None:
        super().__init__()
        hidden = hidden_features or max(16, in_features // 4)
        self.attn = nn.Sequential(
            nn.Conv2d(in_features, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        scores = self.attn(x).view(b, 1, h * w)  # [B, 1, HW]
        weights = F.softmax(scores, dim=-1)  # [B, 1, HW]
        x_flat = x.view(b, c, h * w)  # [B, C, HW]
        out = torch.bmm(x_flat, weights.transpose(1, 2)).squeeze(-1)  # [B, C]
        return out


class ConcatPool2d(nn.Module):
    """Concatenates GeM and Max pooled representations."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        self.gem = GeMPool2d(p=p, eps=eps, learnable=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gem_out = self.gem(x)
        max_out = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)
        return torch.cat([gem_out, max_out], dim=1)


def build_pooling(pool_type: str = "gem", in_features: int = 768) -> tuple[nn.Module, int]:
    """Factory creating pooling layer and returning (pooling_module, output_dim)."""
    pool_type = pool_type.lower()
    if pool_type == "gem":
        return GeMPool2d(p=3.0, learnable=True), in_features
    elif pool_type == "avg":
        return nn.AdaptiveAvgPool2d((1, 1)), in_features
    elif pool_type == "max":
        return nn.AdaptiveMaxPool2d((1, 1)), in_features
    elif pool_type == "attention":
        return AttentionPool2d(in_features=in_features), in_features
    elif pool_type == "concat":
        return ConcatPool2d(p=3.0, learnable=True), in_features * 2
    else:
        raise ValueError(f"Unknown pool_type: {pool_type!r}. Choose from: 'gem', 'avg', 'max', 'attention', 'concat'.")
