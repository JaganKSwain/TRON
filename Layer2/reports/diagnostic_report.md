# Model A Diagnostic & Anti-Shortcut Report

## 1. Spatial Ablation (Anti-Shortcut Gate)
- **Background-Only Accuracy**: **21.51%** (Chance: 6.67%)
- **Background-Only Macro-F1**: **0.1046**
- **Leaf-Only Accuracy**: **98.67%**
- **Leaf-Only Macro-F1**: **0.9806**

> **Verdict**: FAIL - Background leakage detected

## 2. Cross-Site Generalization (Unseen Capture Sites)
- **Cross-Site Test Accuracy**: **60.63%**
- **Cross-Site Macro-F1**: **0.2519**
- **Cross-Site Balanced Accuracy**: **64.92%**

## 3. Calibration Metrics
- **Expected Calibration Error (ECE)**: **0.7690**

## 4. Visualizations
- Grad-CAM Lesion Heatmaps: `C:\ml\agrobot\reports\gradcam_montage.jpg`
