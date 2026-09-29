# Implementation Plan — Phase 7: Model B (MCU) Distillation & QAT

## Goal Description
Implement **Phase 7** of the AgroBot pipeline, distilling the high-capacity **Model A** (ConvNeXt-Tiny + GeM) teacher network into the ultra-efficient **Model B** (MobileNetV2-0.5) student network, optimized for the Renesas RA8P1 microcontroller and Arm Ethos-U55 NPU.

---

## Architectural Requirements & MCU Constraints

### 1. MCU Student Architecture (`models/mcu.py`)
- **Backbone**: MobileNetV2 with `width_mult=0.5` (~700k parameters, ~1.5 MB unquantized).
- **Activations**: Strictly `ReLU6` (hardware-accelerated on Ethos-U55).
- **Pooling**: Global Average Pooling ($7 \times 7 \to 1 \times 1$).
- **Classifier Head**: Linear classification layer emitting raw unquantized logits.
- **Vela Offload**: 100% operator offload compatibility verified against the Phase 0 gate.

### 2. Knowledge Distillation Engine (`distill.py`)
- **Teacher**: Frozen Model A (`best_model.pt`) evaluating in `eval()` mode with AMP.
- **Student**: Model B MobileNetV2-0.5 initialized with ImageNet weights.
- **Combined Loss Function**:
  $$\mathcal{L} = (1 - \alpha) \mathcal{L}_{\text{CE}}(y_s, y) + \alpha \cdot T^2 \cdot \mathcal{D}_{\text{KL}}\left(\log \sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$
  - Temperature $T = 3.0$
  - Distillation weight $\alpha = 0.6$
- **Optimization**: AdamW with Cosine Annealing learning rate schedule and validation macro-F1 tracking.

### 3. Quantization-Aware Training (QAT) Preparation
- Prepare model graph with quantization stubs / fake-quantization nodes.
- Fine-tune to minimize accuracy degradation when moving to INT8 (target gap $< 1\%$).

---

## Proposed Changes

### 1. MCU Model Definition
#### [NEW] [`src/agrobot/models/mcu.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/models/mcu.py)
- Defines `PlantMobileNetV2` (Vela-compliant MobileNetV2-0.5 with ReLU6 and linear head).

---

### 2. Knowledge Distillation Training Script
#### [NEW] [`src/agrobot/distill.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/distill.py)
- Loads teacher Model A, instantiates student Model B, runs KD loop with mixed-precision, and tracks student validation macro-F1.
- Saves checkpoints to `C:\ml\agrobot\runs\model_b_mcu\`.

---

### 3. Unit Tests
#### [NEW] [`tests/test_distill.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/tests/test_distill.py)
- Tests KD loss function math, KL divergence scaling with $T^2$, and student forward pass.

---

## Verification Plan

### Automated Tests
1. **PyTest Suite**:
   ```powershell
   C:\ml\agrobot\.venv\Scripts\pytest.exe C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\agrobot\tests -v
   ```
2. **Distillation Training**:
   ```powershell
   C:\ml\agrobot\.venv\Scripts\python.exe C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\agrobot\src\agrobot\distill.py --teacher C:\ml\agrobot\runs\model_a_convnext\best_model.pt --epochs 15 --batch-size 64
   ```
3. **Success Criteria**:
   - Model B Student Macro-F1 $\ge 90\%$ on the validation and test splits.
   - Model B strictly complies with all 13 Vela NPU hardware rules.
