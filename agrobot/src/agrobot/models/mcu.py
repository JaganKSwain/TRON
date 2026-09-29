"""Model B: the RA8P1 student, constrained by what Arm Ethos-U55 can run.

Every choice here is a hardware constraint, not a preference. Ethos-U55 is
integer-only and supports a fixed operator set; anything outside it silently
falls back to the Cortex-M85 CPU, which costs far more than the accuracy the
fancier operator would have bought.

Binding constraints (Arm Vela `SUPPORTED_OPS.md`):

* **No `POW`** -- so GeM pooling, which is the right choice for the laptop model
  (lesions are small, and averaging dilutes them), cannot be used here. This
  model uses `AVERAGE_POOL_2D`/`MEAN` instead. That asymmetry is deliberate and
  is why the two models are not the same architecture.
* **`RELU6` is supported, `HARD_SWISH` is, `SILU`/`GELU` are not** -- MobileNetV2
  is a good fit precisely because it is ReLU6 throughout. No Squeeze-Excite: its
  global-pool-then-broadcast-multiply pattern is awkward for the NPU.
* **`DEPTHWISE_CONV_2D` requires channel multiplier 1** -- MobileNetV2 complies.
* **Per-axis quantisation only for conv/FC**, everything int8.

Budget: 2 MB SRAM tensor arena, weights in MRAM (0.5/1 MB) or flash (4/8 MB).
At width 0.5 and 160x160 this lands around 0.7 M parameters.
"""
from __future__ import annotations

import torch
from torch import nn

# Vela-supported activations only. Kept explicit so a future edit that reaches
# for SiLU fails loudly here rather than silently on the NPU.
VELA_SAFE_ACTS = frozenset({"relu", "relu6", "hard_swish"})

MCU_MODEL_NAME = "mobilenetv2_050"
MCU_INPUT_SIZE = 160
MCU_FALLBACK_INPUT_SIZE = 128


class McuClassifier(nn.Module):
    """MobileNetV2 backbone + global average pool + linear head.

    Deliberately plain: the head is a single `Linear` on pooled features so the
    exported graph ends in `MEAN` -> `FULLY_CONNECTED`, both NPU-native. Softmax
    is left out of the graph and applied on the host, so the model emits logits
    and the energy score (`logsumexp`) stays available for open-set rejection.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 0.5,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        import timm

        # `num_classes=0` gives the pooled feature vector without timm's head,
        # so pooling and head stay under our control and auditable.
        self.backbone = timm.create_model(
            MCU_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.num_features = self.backbone.num_features
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(self.num_features, num_classes)
        self._assert_vela_safe()

    def _assert_vela_safe(self) -> None:
        """Fail at construction if an unsupported activation slipped in."""
        banned = (nn.SiLU, nn.GELU, nn.Mish, nn.ELU, nn.LeakyReLU, nn.PReLU)
        for name, mod in self.named_modules():
            if isinstance(mod, banned):
                raise ValueError(
                    f"{type(mod).__name__} at '{name}' is not supported by Ethos-U55; "
                    "it would fall back to the M85 CPU"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.backbone(x)))


def build_mcu_model(
    num_classes: int, width: float = 0.5, pretrained: bool = True
) -> McuClassifier:
    return McuClassifier(num_classes=num_classes, width=width, pretrained=pretrained)


def count_parameters(model: nn.Module) -> tuple[int, float]:
    """Return (parameter count, approximate INT8 weight size in MB)."""
    n = sum(p.numel() for p in model.parameters())
    return n, n / 1e6
