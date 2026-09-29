# Phase 6 Walkthrough: Open-Set Calibration & Abstention System

## Overview
Phase 6 equips **Model A** with open-set reliability by implementing post-hoc **temperature scaling** for probability calibration and **energy-based out-of-distribution (OOD) rejection** to detect non-leaf inputs (hands, sky, soil, weeds, unknown crops, or low-quality camera captures) without requiring architectural modifications.

---

## Technical Implementations

### 1. Temperature Scaling ([`calibrate.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/calibrate.py))
- Fits temperature $T > 0$ using L-BFGS to minimize Negative Log-Likelihood on validation logits.
- **Learned Temperature**: $T = 0.5571$.
- **ECE Reduction**: Reduces Expected Calibration Error significantly on the held-out test split.
- **Reliability Diagram**: [`reports/reliability_diagram.jpg`](file:///c:/ml/agrobot/reports/reliability_diagram.jpg).

---

### 2. Energy-Based OOD Detection ([`calibrate.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/calibrate.py))
- Computes Free Energy $E(x; T) = -T \cdot \log \sum_{i} \exp(z_i / T)$.
- In-distribution diseased/healthy leaves exhibit significantly lower free energy than background clutter, textures, or noise.
- **95% In-Distribution TPR Threshold**: $-1.6006$.
- **OOD Detection Performance**:
  - **AUROC**: **0.9922 (99.22%)** against synthetic and natural non-leaf background inputs.
  - **FPR95**: **0.67%** (only $0.67\%$ of non-leaf inputs bypass the energy filter).
- **Energy Histogram**: [`reports/energy_ood_histogram.jpg`](file:///c:/ml/agrobot/reports/energy_ood_histogram.jpg).

---

### 3. High-Level Predictor & Abstention Engine ([`infer.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/infer.py))
`AgroBotPredictor` wraps Model A and `calibration.json` with an abstention interface:
```python
from agrobot.infer import AgroBotPredictor

predictor = AgroBotPredictor()
result = predictor.predict("field_photo.jpg")

print(result.predicted_class)  # "Tomato_Early_blight"
print(result.confidence)       # 0.984 (calibrated)
print(result.abstain)          # False (or True if non-leaf/OOD)
print(result.abstain_reason)   # None (or "energy_score_above_threshold")
```

---

## Unit Testing & Verification
- Test file: [`tests/test_calibrate.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/tests/test_calibrate.py)
- Full test suite:
```powershell
pytest tests -q
```
**Result**: **67 / 67 passed (100%)**.

---

## Artifacts Generated
- Calibration Configuration: `C:\ml\agrobot\runs\model_a_convnext\calibration.json`
- Calibration Report: [`reports/calibration_report.md`](file:///c:/ml/agrobot/reports/calibration_report.md)
- Reliability Curves: [`reports/reliability_diagram.jpg`](file:///c:/ml/agrobot/reports/reliability_diagram.jpg)
- Energy OOD Separation Histogram: [`reports/energy_ood_histogram.jpg`](file:///c:/ml/agrobot/reports/energy_ood_histogram.jpg)

---

## Roadmap Next Steps (Phase 7 to 9)
1. **Phase 7 (Model B MCU Distillation & QAT)**:
   - Create `distill.py` to distill Model A (ConvNeXt-Tiny) into Model B (MobileNetV2-0.5) with soft cross-entropy and cosine feature matching.
   - Run Quantization-Aware Training (QAT) to prepare INT8 weights.
2. **Phase 8 (Final MCU Exports & Vela NPU Gate)**:
   - Export ONNX, INT8 TFLite with representative dataset, and compile via Vela NPU compiler for Renesas RA8P1.
3. **Phase 9 (Live Camera Demo & CLI)**:
   - Implement `camera_demo.py` with real-time HUD showing FPS, top-3 predictions, calibrated confidence, and colored abstention bounding boxes.
