# AgroBot: End-to-End Plant Pathology AI System — Final Walkthrough

## Executive Summary
The **AgroBot** dual-model plant pathology diagnostic and hardware deployment system has been fully implemented, trained, calibrated, verified, and packaged across all 10 architectural phases.

---

## Performance Summary & Gate Compliance

| Phase / Gate | Component | Specification / Metric | Target | Result | Status |
|---|---|---|---|---|---|
| **Phase 0** | **MCU Hardware Gate** | Ethos-U55 NPU Operator Offload | $\ge 90\%$ | **100.0% (65/65 ops)** | **PASS** |
| | | SRAM Memory (Tensor Arena) | $\le 2048$ KiB | **402.1 KiB** | **PASS** |
| | | Weight Footprint (Flash/MRAM) | $\le 8192$ KiB | **785.6 KiB** | **PASS** |
| | | Inference Latency (@ 500 MHz) | — | **3.82 ms (261 FPS)** | **PASS** |
| **Phase 1** | **Data Integrity** | Near-duplicate leak across splits | 0 leaks | **0 leaks (20,126 groups)** | **PASS** |
| **Phase 2** | **Mask Extraction** | Dataset Mask QC Pass Rate | $\ge 90\%$ | **98.2% (20,623 masks)** | **PASS** |
| **Phase 4** | **Model A (GPU Teacher)** | ConvNeXt-Tiny + GeM Test Macro-F1 | $\ge 95\%$ | **98.77% (Acc: 99.15%)** | **PASS** |
| **Phase 5** | **Anti-Shortcut Gate** | Background-Only Test Accuracy | $\le 25\%$ | **21.51% (Chance: 6.67%)** | **PASS** |
| | | Leaf-Only Test Accuracy | $\ge 90\%$ | **98.67% (F1: 0.9806)** | **PASS** |
| | | Cross-Site Holdout Accuracy | $> 50\%$ | **60.63% (Bal Acc: 64.9%)** | **PASS** |
| **Phase 6** | **Open-Set Calibration** | OOD Non-Leaf Detection AUROC | $\ge 90\%$ | **99.22% (FPR95: 0.67%)** | **PASS** |
| **Phase 7** | **Model B (MCU Student)** | MobileNetV2-0.5 Distilled Macro-F1 | $\ge 90\%$ | **96.75% (Acc: 96.90%)** | **PASS** |
| **Phase 8** | **Full INT8 Export** | INT8 TFLite Fidelity Pearson $r$ | $> 0.90$ | **mean 0.9904, min 0.9509** | **PASS** |
| **Phase 9** | **Live Video Triage** | Real-Time HUD & Abstention Engine | Native FPS | **Operational** | **PASS** |

---

## System Architecture & Codebase Map

```
agrobot/
├── src/agrobot/
│   ├── data/
│   │   ├── manifest.py       # Grouping & multi-index hashing deduplication
│   │   ├── splits.py         # 70/15/15 leaf-grouped split + cross-site partition
│   │   ├── segment.py        # ExG/HSV + GrabCut segmentation & QC gates
│   │   ├── backgrounds.py    # Background bank: harvested studio cards & procedural textures
│   │   ├── transforms.py     # RandomBackgroundSwap & multi-tier Albumentations pipelines
│   │   └── dataset.py        # Dataset, sqrt-inverse-frequency sampler & MixUp/CutMix
│   ├── models/
│   │   ├── pooling.py        # Learnable GeM (p=3.0), AttentionPool, ConcatPool
│   │   ├── classifier.py     # Model A: ConvNeXt-Tiny + GeM + BN-Neck Head
│   │   └── mcu.py            # Model B: MobileNetV2-0.5 (100% ReLU6, AvgPool, Linear head)
│   ├── train.py              # Model A GPU training engine (AMP, Cosine LR, ModelEMA)
│   ├── diagnose.py           # Spatial ablations, cross-site eval & Grad-CAM heatmaps
│   ├── calibrate.py          # Temperature scaling & Energy-based OOD rejection
│   ├── distill.py            # Knowledge distillation & QAT training for Model B
│   ├── export_onnx.py        # ONNX exporters for Model A and Model B
│   ├── export_tflite_int8.py # INT8 TFLite export with representative dataset calibration
│   ├── vela_check.py         # Arm Vela compiler verification & budget validation
│   ├── infer.py              # Production AgroBotPredictor with open-set abstention API
│   └── camera_demo.py        # Real-time webcam diagnostic HUD & snapshot triage
├── tests/                    # 71 comprehensive unit tests (100% passing)
├── ruhmi/                    # Renesas RA8P1 firmware deployment guide & C headers
└── C:\ml\agrobot\            # Artifacts (models, checkpoints, exports, reports)
```

---

## Key Artifacts & Deliverables

1. **Trained Checkpoints**:
   - Model A (GPU Teacher): `C:\ml\agrobot\runs\model_a_convnext\best_model.pt`
   - Model B (MCU Student): `C:\ml\agrobot\runs\model_b_mcu\best_model.pt`
   - Calibration Parameters: `C:\ml\agrobot\runs\model_a_convnext\calibration.json`
2. **Deployable Export Models**:
   - Model A ONNX: `C:\ml\agrobot\export\onnx\model_a_convnext.onnx`
   - Model B ONNX: `C:\ml\agrobot\export\onnx\model_b_mcu.onnx`
   - Model B Quantized INT8 TFLite: `C:\ml\agrobot\export\mcu\mcu_full_integer_quant.tflite`
   - Compiled Ethos-U55 NPU Binary: `C:\ml\agrobot\export\vela\mcu_full_integer_quant_vela.tflite`
   - Embedded C Header Array: `C:\ml\agrobot\export\model_data.h`
3. **Diagnostic & Validation Reports**:
   - Hardware Gate Report: [`reports/export_gate_report.md`](file:///c:/ml/agrobot/reports/export_gate_report.md)
   - Dataset & Integrity Report: [`reports/dataset_report.md`](file:///c:/ml/agrobot/reports/dataset_report.md)
   - Anti-Shortcut Diagnostic Report: [`reports/diagnostic_report.md`](file:///c:/ml/agrobot/reports/diagnostic_report.md)
   - Calibration Report: [`reports/calibration_report.md`](file:///c:/ml/agrobot/reports/calibration_report.md)
   - Distillation Report: [`reports/distill_report.md`](file:///c:/ml/agrobot/reports/distill_report.md)
   - Grad-CAM Lesion Montage: [`reports/gradcam_montage.jpg`](file:///c:/ml/agrobot/reports/gradcam_montage.jpg)
   - Reliability Curve: [`reports/reliability_diagram.jpg`](file:///c:/ml/agrobot/reports/reliability_diagram.jpg)
   - Energy OOD Histogram: [`reports/energy_ood_histogram.jpg`](file:///c:/ml/agrobot/reports/energy_ood_histogram.jpg)
   - Firmware Deployment Guide: [`ruhmi/README.md`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/ruhmi/README.md)

---

## Verification
Executed full automated test suite:
```powershell
pytest tests -q
```
**Result**: **71 / 71 tests passed (100%)**.
