# Implementation Plan — Phase 6: Open-Set Calibration & Abstention System

## Goal Description
Implement **Phase 6** of the AgroBot pipeline, transforming the closed 15-class classifier into a calibrated, open-set aware diagnostic system capable of abstaining when presented with non-leaf objects (soil, sky, hands, weeds, unknown crops, or low-quality views).

---

## Technical Architecture

### 1. Temperature Scaling for Calibration
- Raw softmax outputs are notoriously overconfident.
- **Methodology**: Learn a post-hoc temperature parameter $T > 0$ on the validation set by minimizing Negative Log-Likelihood (NLL) with L-BFGS:
  $$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **Target**: Drastically reduce Expected Calibration Error (ECE from $\approx 0.77$ down to $< 0.05$).

### 2. Energy-Based Out-of-Distribution (OOD) Rejection
- Max softmax probability suffers from high confidence even on far-OOD inputs.
- **Energy Score**: $E(x; T) = -T \cdot \log \sum_{i=1}^K \exp(z_i / T)$
- In-distribution (diseased & healthy crop leaves) exhibit lower free energy (higher negative energy) than out-of-distribution textures, soil, or background clutter.
- **Calibration**: Fit an energy threshold corresponding to **95% True Positive Rate (TPR)** on in-distribution validation data.
- **Metrics**: Measure AUROC and False Positive Rate at 95% TPR (FPR95) against non-leaf / OOD samples.

### 3. Abstention & Prediction Engine (`infer.py`)
- `AgroBotPredictor`:
  - Input: BGR or RGB image (numpy array, PIL image, or path).
  - Evaluates Model A with calibrated temperature and energy threshold.
  - Returns `PredictionResult`:
    - `predicted_class: str`
    - `confidence: float` (calibrated probability)
    - `top_k: list[tuple[str, float]]`
    - `abstain: bool`
    - `abstain_reason: str | None` (e.g. `"energy_score_below_threshold"`, `"low_confidence"`, `"classified_as_not_a_leaf"`)
    - `energy: float`

---

## Proposed Changes

### 1. Calibration & OOD Engine
#### [NEW] [`src/agrobot/calibrate.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/calibrate.py)
- `TemperatureScaler`: L-BFGS optimization on validation set.
- `EnergyOODDetector`: Free energy score computation & threshold fitting.
- Evaluates AUROC and ECE, saving `C:\ml\agrobot\runs\model_a_convnext\calibration.json`.
- Plots reliability diagrams and energy histograms to `C:\ml\agrobot\reports\reliability_diagram.jpg` and `C:\ml\agrobot\reports\energy_ood_histogram.jpg`.

---

### 2. Predictor & Abstention API
#### [NEW] [`src/agrobot/infer.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/src/agrobot/infer.py)
- `AgroBotPredictor`: Production-ready inference engine for single images or batches with integrated abstention logic.

---

### 3. Unit Tests
#### [NEW] [`tests/test_calibrate.py`](file:///C:/Users/jagan/OneDrive/Documents/TRON/Dataset/PlantVillage/agrobot/tests/test_calibrate.py)
- Verifies temperature scaling decreases ECE on miscalibrated logits.
- Verifies energy score separates known distributions from uniform noise.
- Verifies `AgroBotPredictor` correctly triggers `abstain: True` on non-leaf images and produces valid top-k predictions on leaves.

---

## Verification Plan

### Automated Tests
1. **PyTest Suite**:
   ```powershell
   C:\ml\agrobot\.venv\Scripts\pytest.exe C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\agrobot\tests -v
   ```
2. **Calibration Run**:
   ```powershell
   C:\ml\agrobot\.venv\Scripts\python.exe C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\agrobot\src\agrobot\calibrate.py --checkpoint C:\ml\agrobot\runs\model_a_convnext\best_model.pt
   ```
3. **Success Criteria**:
   - ECE drops to $< 0.05$.
   - OOD AUROC on background/non-leaf images $\ge 90\%$.
   - `calibration.json` generated and successfully loaded by `infer.py`.
