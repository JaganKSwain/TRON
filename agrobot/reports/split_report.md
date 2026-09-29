# Split report

## Overall

| Split | Images | Share |
|---|---:|---:|
| train | 12657 | 61.4% |
| val | 2716 | 13.2% |
| test | 2710 | 13.1% |
| test_site | 2540 | 12.3% |
| **total** | **20623** | |

## Integrity

**PASS** -- no group, duplicate or held-out site crosses a split boundary.

## Cross-site holdout (`test_site`)

These sites were withheld from training entirely. Accuracy here is the best available proxy for field performance: same disease, unseen capture conditions.

| Class | Held-out site | Images |
|---|---|---:|
| Pepper__bell___Bacterial_spot | `NREC_B.Spot` | 408 |
| Tomato_Bacterial_spot | `UF.GRC_BS_Lab` | 371 |
| Tomato_Late_blight | `GHLB_PS` | 146 |
| Tomato_Septoria_leaf_spot | `Keller.St_CG` | 274 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | `YLCV_NREC` | 757 |
| Tomato_healthy | `GH_HL` | 584 |

- Images pulled in by group consistency: 0
- Groups spanning >1 class (near-duplicate unions): 0

The remaining 9 classes have only one capture site, so they cannot contribute a cross-site measurement -- withholding their site would delete the class.

## Per-class breakdown

| Class | train | val | test | test_site |
|---|---:|---:|---:|---:|
| Pepper__bell___Bacterial_spot | 412 | 89 | 88 | 408 |
| Pepper__bell___healthy | 1034 | 222 | 222 | 0 |
| Potato___Early_blight | 700 | 150 | 150 | 0 |
| Potato___Late_blight | 700 | 150 | 150 | 0 |
| Potato___healthy | 106 | 23 | 23 | 0 |
| Tomato_Bacterial_spot | 1229 | 264 | 263 | 371 |
| Tomato_Early_blight | 700 | 150 | 150 | 0 |
| Tomato_Late_blight | 1228 | 263 | 263 | 146 |
| Tomato_Leaf_Mold | 666 | 143 | 143 | 0 |
| Tomato_Septoria_leaf_spot | 1048 | 225 | 224 | 274 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1173 | 252 | 251 | 0 |
| Tomato__Target_Spot | 983 | 211 | 210 | 0 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 1716 | 368 | 367 | 757 |
| Tomato__Tomato_mosaic_virus | 261 | 56 | 56 | 0 |
| Tomato_healthy | 701 | 150 | 150 | 584 |
