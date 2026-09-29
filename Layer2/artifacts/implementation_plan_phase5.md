# Implementation Plan — Phase 5: Diagnostics & Anti-Shortcut Validation

## Goal Description
Execute **Phase 5** of the AgroBot pipeline, subjecting the trained **Model A (ConvNeXt-Tiny + GeM)** to rigorous anti-self-deception checks to verify that its accuracy is derived from plant pathology rather than background photo-session memorization.

---

## Diagnostic Objectives & Gates

### 1. Spatial Background-Only Ablation (Mandatory Hard Gate)
- **Methodology**: Inpaint/blackout the leaf using `generous_mask(mask)`, leaving only the background card and lighting.
- **Hard Gate**: **Accuracy must fall to $\approx$ random chance ($\approx 6.67\%$)**.
- **Rationale**: If the model scores above chance on background-only images, it has learned the capture-site shortcut and cannot be trusted on field plants.

### 2. Spatial Leaf-Only Ablation
- **Methodology**: Neutralize all pixels outside the leaf mask to uniform gray ($[128, 128, 128]$), removing all studio card context.
- **Target**: Macro-F1 and Top-1 accuracy must remain high ($> 90\%$), proving the model relies exclusively on leaf features.

### 3. Cross-Site Holdout Generalization (`test_site` split)
- **Methodology**: Evaluate on the 2,540 images from the 6 multi-site classes whose capture sites were withheld from training entirely (`test_site` split).
- **Target**: High cross-site Macro-F1, directly testing field-transferability proxy.

### 4. Grad-CAM Lesion Heatmap Visualizations
- **Methodology**: Compute Gradient-weighted Class Activation Mapping (Grad-CAM) on the final ConvNeXt stage (`stages[-1]`).
- **Deliverable**: Montage in `C:\ml\agrobot\reports\gradcam_montage.jpg` overlaying heatmaps on leaves to confirm localized attention on necrosis, mold, spots, and chlorosis.

### 5. Expected Calibration Error (ECE) & Reliability Analysis
- **Methodology**: Compute 15-bin Expected Calibration Error and confidence histograms.
- **Deliverable**: Calibration metrics and diagnostic tables in `C:\ml\agrobot\reports\diagnostic_report.md`.

---

## Proposed Work

### 1. Diagnostic Runner Execution
#### [EXECUTE] [`src/agrobot/diagnose.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/diagnose.py)
- Runs spatial ablations (background-only and leaf-only).
- Evaluates `test` and `test_site` splits.
- Computes ECE and Grad-CAM overlays.
- Outputs comprehensive report to `C:\ml\agrobot\reports\diagnostic_report.md`.

---

## Verification Plan

### Automated Execution
```powershell
C:\ml\agrobot\.venv\Scripts\python.exe C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\agrobot\src\agrobot\diagnose.py --checkpoint C:\ml\agrobot\runs\model_a_convnext\best_model.pt
```

### Gate Success Criteria
1. Background-only accuracy $\le 15\%$ (close to $6.67\%$ chance).
2. Leaf-only accuracy $\ge 90\%$.
3. Cross-site macro-F1 $\ge 80\%$.
4. Grad-CAM heatmaps attend to leaf spots and lesions.
