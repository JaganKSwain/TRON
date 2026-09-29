# Phase 6 Calibration & Open-Set Rejection Report

## 1. Temperature Scaling
- **Optimal Temperature ($T$)**: **0.5571**
- **Uncalibrated ECE**: **0.7683**
- **Calibrated ECE**: **0.5338** (relative error reduced significantly)
- Reliability Diagram: `C:\ml\agrobot\reports\reliability_diagram.jpg`

## 2. Energy-Based OOD / Non-Leaf Rejection
- **Energy Threshold (95% In-Distribution TPR)**: **-1.6006**
- **OOD Detection AUROC**: **0.9922** (Target $\ge 90\%$)
- **False Positive Rate at 95% TPR (FPR95)**: **0.67%**
- Energy Histogram: `C:\ml\agrobot\reports\energy_ood_histogram.jpg`

Saved configuration: `C:\ml\agrobot\runs\model_a_convnext\calibration.json`
