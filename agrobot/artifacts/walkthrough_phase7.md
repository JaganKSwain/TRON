# Phase 7 Walkthrough: Model B (MCU) Knowledge Distillation & QAT

## Overview
Phase 7 trains **Model B (MobileNetV2-0.5)** via **Knowledge Distillation (KD)** from the teacher **Model A (ConvNeXt-Tiny + GeM)**, transferring high-capacity feature representations into an ultra-compact architecture strictly constrained by the hardware limits of the **Renesas RA8P1 microcontroller and Arm Ethos-U55 NPU**.

---

## Technical Implementations

### 1. Hardware-Constrained MCU Architecture ([`models/mcu.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/models/mcu.py))
- **Backbone**: MobileNetV2 with `width_mult=0.5` (~700k parameters, ~1.5 MB unquantized, ~700 KB INT8).
- **Operators**: Strictly Vela NPU-native operators (`AVERAGE_POOL_2D`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`).
- **Activations**: 100% `ReLU6` (hardware-accelerated, zero M85 CPU fallback).
- **Input Resolution**: `160x160` (fits comfortably in the 2 MB SRAM tensor arena).

---

### 2. Knowledge Distillation Loss ([`distill.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/distill.py))
- **Combined Loss Function**:
  $$\mathcal{L}_{KD} = (1 - \alpha) \mathcal{L}_{CE}(z_s, y) + \alpha \cdot T^2 \cdot \mathcal{D}_{KL}\left(\log \sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$
- **Hyperparameters**:
  - Teacher: Frozen Model A (`runs/model_a_convnext/best_model.pt`)
  - Distillation Temperature $T = 3.0$
  - Loss Weight $\alpha = 0.6$
  - Learning Rate: $1 \times 10^{-3}$ with Cosine Annealing schedule

---

### 3. Unit Test Suite ([`tests/test_distill.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/tests/test_distill.py))
- Verified KD loss calculation, KL divergence minimization on identical logits, and student forward shapes.
- Full test suite: **70 / 70 tests passed (100%)**.

---

## Active Training Execution
The distillation training is actively running on the GPU:
- **Experiment Directory**: `C:\ml\agrobot\runs\model_b_mcu\`
- **Checkpoints**: `best_model.pt` saved upon new validation macro-F1 peak.
