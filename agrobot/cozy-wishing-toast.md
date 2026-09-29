# AgroBot Plant Disease Classifier — PlantVillage → Field Deployment

## Context

**Goal:** a 15-class plant-disease classifier that (a) runs on the laptop RTX 4060 for development/inference, and (b) ships as an INT8 model on a **Renesas RA8P1** (Cortex-M85 @1 GHz + Arm Ethos-U55 NPU, 256 GOPS) so AgroBot can diagnose disease from its camera in the field.

**What I found in the dataset (drives the whole design):**

| Finding | Detail | Consequence |
|---|---|---|
| Size | 20,639 images, 15 classes (Pepper/Potato/Tomato), 256×256 JPEG | Small — transfer learning is mandatory |
| **Byte-identical duplicate** | `PlantVillage/PlantVillage/` mirrors all 15 classes (md5-verified) | Must be excluded or every image lands in train *and* test |
| **Capture-site ≈ label** | `Potato___Early_blight` = 100% `RS_Early.B`; `Tomato_Leaf_Mold` = 100% `Crnl_L.Mold`; `Tomato_Spider_mites` = 100% `Com.G_SpM_FL`; 8 of 15 classes are single-site | A CNN can score ~99% by memorising **background/lighting per photo session**, then collapse on real plants. This is the documented PlantVillage shortcut and the #1 risk to the stated goal |
| **Same-leaf series** | `GHLB Leaf 2` appears 16× (Days 1–16); `GH_HL Leaf 495` 4×. 501 redundant images (2.4%), mostly `Tomato_Late_blight` (272) + `Tomato_healthy` (229) | Random split puts near-identical frames in train and test → inflated score |
| Imbalance | 3,209 (Tomato YLCV) → 152 (Potato healthy), 21× | Accuracy is a misleading metric; select on macro-F1 |

**The core tension to be explicit about:** PlantVillage is *detached single leaves on uniform lab backgrounds*. AgroBot's camera sees *whole plants* with soil, sky, clutter, sun/shade, motion blur. Published work (Mohanty 2016) reports 99.35% in-lab dropping to ~31% on field imagery. So the deliverable is not "high test accuracy" — it is **a model whose accuracy is earned from leaf pathology, verified to not depend on background.**

**Environment:** RTX 4060 8 GB; **Python 3.11** (the default 3.14 has no PyTorch wheels); 152 GB free. Toolchain verified installable: `torch 2.13+cu126`, `timm 1.0.28`, `albumentations 2.0.8`, `onnx2tf 1.29.24`, `ethos-u-vela 5.1.0`, `grad-cam 1.5.7`.

**RA8P1 constraints (from Renesas + Arm Vela docs):** Ethos-U55 is **integer-only** — anything not INT8 falls back to the M85 CPU. 2 MB SRAM (tensor arena), 0.5/1 MB MRAM or 4/8 MB flash (weights). Vela **supports** `CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `MEAN`, `FULLY_CONNECTED`, `ADD`, `MUL`, `RELU6`, `HARD_SWISH`, `LOGISTIC`, `SOFTMAX`. It has **no `POW`** → GeM pooling cannot run on the NPU. Per-axis (per-channel) quantisation is allowed *only* for conv/FC. Renesas toolchain is **RUHMI** (EdgeCortix MERA 2.0), accepting `.tflite` / `.onnx` / `.pte` for RA8.

---

## Design

### Layout

Code lives in the working dir (small, syncs fine). Heavy artifacts go **outside OneDrive** — a ~5 GB CUDA venv plus checkpoints would otherwise thrash OneDrive sync. Requires one `request_directory` approval for `C:\ml\agrobot`.

```
<cwd>/agrobot/
  configs/{base,gpu_convnext,mcu_mobilenetv2}.yaml
  src/agrobot/
    data/manifest.py      # scan, exclude nested dup, md5+pHash dedupe, parse site/leaf/shot
    data/splits.py        # leaf-grouped stratified split + cross-site diagnostic split
    data/segment.py       # leaf mask extraction (ExG/HSV+Otsu+GrabCut) + cache + QC
    data/backgrounds.py   # background bank: cross-class harvest, procedural, user field photos
    data/transforms.py    # albumentations stacks incl. RandomBackgroundSwap
    data/dataset.py       # Dataset, weighted sampler, MixUp/CutMix
    models/pooling.py     # GeM, AttentionPool, ConcatPool
    models/classifier.py  # timm backbone + pooling + BN-neck head
    models/mcu.py         # Vela-safe MobileNetV2 variant (ReLU6, AvgPool, no SE)
    train.py  distill.py  calibrate.py  evaluate.py  diagnose.py
    export_onnx.py  export_tflite_int8.py  vela_check.py
    infer.py  camera_demo.py
  ruhmi/README.md         # RA8P1 conversion + deployment steps
C:\ml\agrobot\            # venv, runs/, checkpoints, mask cache  (NOT OneDrive-synced)
```

### Splitting — the honest foundation

Parse each filename `<uuid>___<site> <leaf>[.shot][ Day N].ext` into `(site, leaf_id, shot)`. Group key = `site + leaf_id` (strip `.N` shot index and ` Day N`).

- **Primary:** stratified-by-class, **grouped-by-leaf** split 70/15/15. Guarantees no leaf appears in two splits.
- **Cross-site diagnostic:** for the 6 multi-site classes (`Pepper_Bacterial_spot`, `Tomato_Bacterial_spot`, `Tomato_Late_blight`, `Tomato_Septoria`, `Tomato_YLCV`, `Tomato_healthy`) hold out an entire capture site. Directly measures reliance on capture context. A full site-disjoint split is *impossible* for the 8 single-site classes — holding out their site removes the class.

### Augmentation — where field transfer is won

1. **Leaf segmentation + background swap (highest-impact).** Segment the leaf with classical CV (ExG/HSV + Otsu + GrabCut refine + morphology), cache masks as PNG, QC by mask-area fraction and largest-connected-component ratio. Then `RandomBackgroundSwap` (p≈0.6) composites the leaf onto:
   - backgrounds **harvested from other classes'** images — free, and directly severs the site↔label correlation;
   - procedural textures (Perlin/soil-like gradients, blurred foliage mosaics);
   - any real field photos dropped into `data/backgrounds/`.
2. **Geometric:** `RandomResizedCrop(scale 0.4–1.0)` (camera distance), H/V flip, `Rotate(±180)` (leaves have no canonical orientation), Affine, Perspective.
3. **Photometric / domain randomisation:** ColorJitter (strong), HueSaturationValue, RandomBrightnessContrast, RandomGamma, CLAHE, RandomShadow, RandomSunFlare (low p), RandomToneCurve.
4. **Sensor realism:** GaussNoise, ISONoise, MotionBlur, Defocus, `ImageCompression(q 40–95)`, Downscale — models a cheap MIPI-CSI2 sensor on a moving robot.
5. **Occlusion:** CoarseDropout.
6. **Batch-level:** MixUp + CutMix, label smoothing 0.1.

Exposed as `--aug-level {light,medium,heavy}` so the contribution is measurable, not assumed.

### Model A — laptop GPU (accuracy + robustness)

- **Backbone:** `timm` **ConvNeXt-Tiny** (`convnext_tiny.in12k_ft_in1k`), 224px, `drop_path=0.1`. Chosen over EfficientNet for better out-of-distribution behaviour; broader pretraining helps domain shift.
- **Pooling — GeM (Generalised Mean, learnable p init 3.0)** replacing global average pooling. Rationale specific to this problem: lesions occupy a small fraction of leaf area, and GAP averages that evidence away across the feature map. GeM interpolates between mean (p=1) and max (p→∞), preserving strong localised lesion responses. `AttentionPool` and `GeM+Max concat` implemented for ablation.
- **Head:** BN-neck → Dropout(0.3) → Linear(16). (16 = 15 classes + `not_a_leaf`, below.)
- **Training:** AdamW, layer-wise LR decay, cosine schedule + warmup, AMP bf16, **weight EMA**, grad clip, sqrt-inverse-frequency `WeightedRandomSampler`, early stop on **val macro-F1**.

### Model B — RA8P1 (INT8, Ethos-U55)

- **Vela-safe MobileNetV2**: inverted residuals, **ReLU6 only** (no SE, no SiLU), global **AvgPool** — *not* GeM, since Vela has no `POW`. Width α=0.5, input **160×160** (fallback 128 if the arena is tight). ~0.7 M params ≈ 0.7 MB INT8 → fits MRAM/flash; peak activation ≈ 600 KB ≪ 2 MB SRAM.
- **Knowledge distillation** from Model A (temperature + hard-label mix) — the lever that lets a 0.7 M-param model approach teacher accuracy.
- **QAT** for INT8 (recovers the drop PTQ leaves on small models); PTQ path kept as the fast option.
- **Export chain:** PyTorch → ONNX → `onnx2tf` → TFLite INT8 → **`vela` compile** to report NPU-offload %, SRAM and flash usage → hand `.tflite`/`.vela` to RUHMI. Calibration set = ~500 images drawn from the *augmented* distribution so ranges cover field conditions.

### Open-set rejection — the "no failure cases" requirement

A closed 15-way softmax will confidently label soil, sky, a hand, or an unknown crop as some disease. Three layers:

1. **16th `not_a_leaf` class**, trained from background crops (free — the masks already exist) plus procedural textures.
2. **Temperature scaling** fitted on val → calibrated probabilities; report ECE and a reliability diagram.
3. **Energy score** (`logsumexp` of logits) thresholded at 95% TPR on known classes; report AUROC against the held-out background/OOD set.

`predict()` returns top-k, calibrated confidence, and `abstain: bool` with a reason. On the MCU only max-softmax + energy are used (no extra ops).

### Field-frame inference

PlantVillage images are single centred leaves; AgroBot's frame is a whole plant. `infer.py` therefore supports **tiled multi-scale sliding-window** inference with a cheap green-mask "leafness" gate to skip non-leaf tiles (saves NPU cycles), aggregating tile votes into a plant-level diagnosis plus a disease heatmap. This bridges the framing mismatch without needing bounding-box labels, which PlantVillage does not have.

---

## Phases

Each phase ends in a checkable artifact. P0 deliberately de-risks the export chain *before* spending GPU hours.

| # | Work | Gate |
|---|---|---|
| **P0** | venv on py3.11, torch cu126, deps. Stub model → ONNX → TFLite INT8 → `vela` | Vela reports high NPU offload + arena fit. **If this fails, the RA8P1 design changes — better to know now** |
| **P1** | `manifest.py` + `splits.py`: exclude nested dup, md5+pHash dedupe, parse site/leaf, grouped splits | `dataset_report.md` reproducing the counts above; assert zero leaf overlap across splits |
| **P2** | `segment.py` masks + QC montage; `backgrounds.py` bank | Visual montage; ≥95% masks pass QC |
| **P3** | transforms + dataset + 5-epoch ResNet18 smoke test | Loss decreases; augmentation grid renders |
| **P4** | Model A full train (ConvNeXt-Tiny + GeM + heavy aug + EMA + MixUp) | Best checkpoint; train/val gap small (~1.5–3 h) |
| **P5** | `diagnose.py`: background-only + leaf-only ablation, cross-site eval, corruption sweep, Grad-CAM, calibration | **Hard gate: background-only accuracy must fall to ≈chance (6.7%).** If not, the model is cheating and P4 is re-run with stronger background randomisation |
| **P6** | `not_a_leaf` class + energy/temperature thresholds | AUROC + reliability diagram |
| **P7** | Model B + distillation + QAT | INT8 vs FP32 delta <1–2% |
| **P8** | Exports: ONNX (laptop, `onnxruntime-gpu`), TFLite INT8 + Vela (RA8P1), calibration set, `ruhmi/README.md` | Vela offload/fit report |
| **P9** | `infer.py` predict API, tiled frame inference, `camera_demo.py` | Live webcam demo with abstention |

## Verification

- `pytest` unit tests on: split disjointness (no leaf/group in two splits), manifest dedupe, mask QC thresholds, GeM equals GAP at p=1, MixUp label sums to 1.
- **Metrics reported:** macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, ECE. Accuracy alone is not used for selection.
- **Anti-self-deception diagnostics:** background-only ablation (must ≈ chance), leaf-only ablation (must stay high), cross-site holdout, corruption sweep table, Grad-CAM montage confirming attention on lesions.
- **Deployment checks:** FP32 → INT8 accuracy delta; Vela NPU-offload % and SRAM/flash fit; measured laptop latency via `onnxruntime-gpu`.
- **End-to-end:** `camera_demo.py` on a webcam pointed at a real plant (or a photo on screen), confirming sensible predictions and that non-leaf views trigger abstention.

## Honest expectations

- Grouped-split test macro-F1 **~96–98.5%** — *below* the 99.5% commonly quoted for PlantVillage, because leakage and same-leaf duplicates are removed. The lower number is the trustworthy one.
- **True field accuracy cannot be measured with this data.** Every technique above is a principled proxy. The harness will accept a `data/field_val/` folder, so dropping in even 20–50 real AgroBot photos converts the estimate into a measurement — the single highest-value thing to add later.
