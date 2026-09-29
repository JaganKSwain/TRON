# PlantVillage dataset report

## Scan summary

- Files scanned: **20639**
- Unreadable/corrupt: **1**
- Exact duplicates dropped (md5): **14**
- Duplicates found under a *different* label: **0**
- Same-leaf label conflicts: **1** (dropped)
- Images in manifest: **20623** across **15** classes

## Grouping (leakage control)

- Leaf identities (class + leaf_core): **20133**
- Near-duplicate candidates (dhash <= 4 bits): **36** (3 cross-class)
- Confirmed by pixel correlation (r >= 0.9): **13** (1 cross-class)
- **Final split groups: 20126**

- Candidate search exhaustive: **True** (0 of 8 hash chunks skipped as oversized)

A train/val/test split must never straddle a group: repeat shots of one leaf (`Leaf 2 Day 6` vs `Leaf 2 Day 9`) and near-identical frames are forced onto the same side.

Hash-proposed pairs are confirmed against pixels before they are allowed to merge groups. A 64-bit dhash is not decisive on this dataset -- every image is a centred leaf on a flat backdrop, so unrelated species collide -- and an unconfirmed cross-class union would corrupt class stratification.

## Per-class composition

| Class | Images | Groups | Capture sites (count) |
|---|---:|---:|---|
| Pepper__bell___Bacterial_spot | 997 | 997 | `JR_B.Spot` 589, `NREC_B.Spot` 408 |
| Pepper__bell___healthy | 1478 | 1474 | `JR_HL` 1476, `bell-pepper-plant-61726` 1, `Screen` 1 |
| Potato___Early_blight | 1000 | 999 | `RS_Early.B` 1000 |
| Potato___Late_blight | 1000 | 1000 | `RS_LB` 1000 |
| Potato___healthy | 152 | 152 | `RS_HL` 152 |
| Tomato_Bacterial_spot | 2127 | 2127 | `GCREC_Bact.Sp` 1756, `UF.GRC_BS_Lab` 371 |
| Tomato_Early_blight | 1000 | 1000 | `RS_Erly.B` 1000 |
| Tomato_Late_blight | 1900 | 1632 | `RS_Late.B` 920, `GHLB2` 740, `GHLB_PS` 146, `GHLB` 66, +2 more |
| Tomato_Leaf_Mold | 952 | 952 | `Crnl_L.Mold` 952 |
| Tomato_Septoria_leaf_spot | 1771 | 1771 | `Matt.S_CG` 1000, `JR_Sept.L.S` 497, `Keller.St_CG` 274 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1676 | 1676 | `Com.G_SpM_FL` 1676 |
| Tomato__Target_Spot | 1404 | 1404 | `Com.G_TgS_FL` 1404 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 3208 | 3208 | `UF.GRC_YLCV_Lab` 1571, `YLCV_GCREC` 880, `YLCV_NREC` 757 |
| Tomato__Tomato_mosaic_virus | 373 | 372 | `PSU_CG` 373 |
| Tomato_healthy | 1585 | 1362 | `RS_HL` 999, `GH_HL` 584, `CG1` 1, `2700323949_95aa2eaa01_o` 1 |

## Capture-site shortcut

If one capture site dominates a class, background and lighting alone predict the label -- and a CNN will learn that instead of pathology.

| Class | Dominant site | Share |
|---|---|---:|
| Pepper__bell___Bacterial_spot | `JR_B.Spot` | 59.1% |
| Pepper__bell___healthy | `JR_HL` | 99.9% |
| Potato___Early_blight | `RS_Early.B` | 100.0% |
| Potato___Late_blight | `RS_LB` | 100.0% |
| Potato___healthy | `RS_HL` | 100.0% |
| Tomato_Bacterial_spot | `GCREC_Bact.Sp` | 82.6% |
| Tomato_Early_blight | `RS_Erly.B` | 100.0% |
| Tomato_Late_blight | `RS_Late.B` | 48.4% |
| Tomato_Leaf_Mold | `Crnl_L.Mold` | 100.0% |
| Tomato_Septoria_leaf_spot | `Matt.S_CG` | 56.5% |
| Tomato_Spider_mites_Two_spotted_spider_mite | `Com.G_SpM_FL` | 100.0% |
| Tomato__Target_Spot | `Com.G_TgS_FL` | 100.0% |
| Tomato__Tomato_YellowLeaf__Curl_Virus | `UF.GRC_YLCV_Lab` | 49.0% |
| Tomato__Tomato_mosaic_virus | `PSU_CG` | 100.0% |
| Tomato_healthy | `RS_HL` | 63.0% |

- Classes sourced from a **single** site: **8 / 15**
- Accuracy achievable from the capture site alone: **99.3%** (chance is 6.7%)

That figure is the size of the shortcut available to the model. Breaking it is what `data/segment.py` + `RandomBackgroundSwap` and the background-only ablation in `diagnose.py` are for.

## Image properties

- Resolutions: {(256.0, 256.0): 20623}
- Colour modes: {'RGB': 20622, 'RGBA': 1}

## Unreadable files

- `C:\Users\jagan\OneDrive\Documents\TRON\Dataset\PlantVillage\Tomato__Tomato_YellowLeaf__Curl_Virus\svn-r6Yb5c` -- UnidentifiedImageError: cannot identify image file 'C:\\Users\\jagan\\OneDrive\\Documents\\TRON\\Dataset\\PlantVillage\\Tomato__Tomato_YellowLeaf__Curl_Virus\\svn-r6Yb5c'

## Label conflicts (same leaf, two diseases)

Detected structurally: these images share a group -- i.e. the same physical leaf, confirmed by pixel correlation -- with images carrying a different label. The `group_majority` column is the class whose dataset-wide affinity to this capture site is far stronger, so the row's own label is the misfile.

| File | Labelled | Should be | Site | Site affinity (labelled vs correct) |
|---|---|---|---|---:|
| `-c3f53f2c51e4___GH_HL Leaf 170.JPG` | Tomato_Late_blight | Tomato_healthy | `GH_HL` | 2 vs 584 |

These rows are **excluded from** the manifest (`--keep-conflicts` to change).
