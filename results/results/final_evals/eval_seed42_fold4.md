# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed42_fold4/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 4)
**Records evaluated:** 567

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.7237 | 0.8413 | 0.7108 | 0.7446 | 0.8271 | 0.8413 | 542 |
| icdo3_morphology | 0.4143 | 0.9501 | 0.4123 | 0.4318 | 0.5848 | 0.9501 | 541 |
| icdo3_behavior | 0.5586 | 0.9778 | 0.5731 | 0.5534 | 0.7448 | 0.9778 | 541 |
| icdo3_grade | 0.9755 | 0.9834 | 0.9675 | 0.9839 | 0.9755 | 0.9834 | 482 |
| icdo3_laterality | 0.9988 | 0.9982 | 0.9988 | 0.9988 | 0.9988 | 0.9982 | 567 |
| icd11_stem | 0.3231 | 0.9188 | 0.3258 | 0.3357 | 0.4155 | 0.9188 | 554 |
| icd11_ext_laterality | 0.4067 | 0.9622 | 0.3774 | 0.4915 | 0.4067 | 0.9622 | 555 |
| icd11_ext_grading | 0.7543 | 0.9639 | 0.7411 | 0.7687 | 0.9428 | 0.9639 | 554 |
| icd11_ext_anatomy | 0.2261 | 0.6602 | 0.2676 | 0.2464 | 0.5167 | 0.6155 | 567 |
| icd11_ext_histopath | 0.1499 | 0.7077 | 0.1585 | 0.1428 | 0.8620 | 0.9453 | 567 |
| **Mean** | **0.5531** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8413 | 0.9760 | 0.9889 | 0.9889 | 0.0000 |
| icdo3_morphology | 0.9501 | 0.9778 | 0.9871 | 0.9871 | 0.0000 |
| icdo3_behavior | 0.9778 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9834 | 1.0000 | 1.0000 | 1.0000 | 0.9944 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9188 | 0.9639 | 0.9747 | 0.9747 | 0.0000 |
| icd11_ext_laterality | 0.9622 | 0.9892 | 0.9982 | 0.9982 | 0.8413 |
| icd11_ext_grading | 0.9639 | 0.9928 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.8271 | 1 | 7 |
| icdo3_morphology | 0.1154 | 0.7675 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7448 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9755 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9988 | 0 | 3 |
| icd11_stem | 0.0926 | 0.7842 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4880 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9428 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1989 |
| icdo3_morphology | 0.0267 |
| icdo3_behavior | 0.0154 |
| icdo3_grade | 0.0079 |
| icdo3_laterality | 0.0014 |
| icd11_stem | 0.0394 |
| icd11_ext_laterality | 0.0207 |
| icd11_ext_grading | 0.0208 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9522**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6555 | 296 |
| batch_1 | icdo3_morphology | 0.4109 | 295 |
| batch_1 | icdo3_behavior | 0.5728 | 295 |
| batch_1 | icdo3_grade | 0.9817 | 261 |
| batch_1 | icdo3_laterality | 1.0000 | 321 |
| batch_1 | icd11_stem | 0.2724 | 309 |
| batch_1 | icd11_ext_laterality | 0.4031 | 309 |
| batch_1 | icd11_ext_grading | 0.7466 | 308 |
| batch_1 | icd11_ext_anatomy | 0.1226 | 321 |
| batch_1 | icd11_ext_histopath | 0.1552 | 321 |
| batch_2 | icdo3_topography | 0.6823 | 246 |
| batch_2 | icdo3_morphology | 0.2876 | 246 |
| batch_2 | icdo3_behavior | 0.4812 | 246 |
| batch_2 | icdo3_grade | 0.9601 | 221 |
| batch_2 | icdo3_laterality | 0.6640 | 246 |
| batch_2 | icd11_stem | 0.2622 | 245 |
| batch_2 | icd11_ext_laterality | 0.3277 | 246 |
| batch_2 | icd11_ext_grading | 0.7682 | 246 |
| batch_2 | icd11_ext_anatomy | 0.1538 | 246 |
| batch_2 | icd11_ext_histopath | 0.1413 | 246 |
| cancer_primary | icdo3_topography | 0.7237 | 541 |
| cancer_primary | icdo3_morphology | 0.4143 | 541 |
| cancer_primary | icdo3_behavior | 0.5586 | 541 |
| cancer_primary | icdo3_grade | 0.9755 | 481 |
| cancer_primary | icdo3_laterality | 0.9988 | 558 |
| cancer_primary | icd11_stem | 0.2801 | 545 |
| cancer_primary | icd11_ext_laterality | 0.4065 | 546 |
| cancer_primary | icd11_ext_grading | 0.7507 | 546 |
| cancer_primary | icd11_ext_anatomy | 0.2257 | 558 |
| cancer_primary | icd11_ext_histopath | 0.1499 | 558 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 9 |
| non_cancer | icd11_stem | 0.0617 | 9 |
| non_cancer | icd11_ext_laterality | 0.3333 | 9 |
| non_cancer | icd11_ext_grading | 0.2000 | 8 |
| non_cancer | icd11_ext_anatomy | 0.0469 | 9 |
| non_cancer | icd11_ext_histopath | 0.0000 | 9 |
| template | icdo3_topography | 0.6418 | 100 |
| template | icdo3_morphology | 0.1250 | 100 |
| template | icdo3_behavior | 0.4987 | 100 |
| template | icdo3_grade | 0.9678 | 98 |
| template | icdo3_laterality | 0.6667 | 102 |
| template | icd11_stem | 0.1726 | 100 |
| template | icd11_ext_laterality | 0.3249 | 100 |
| template | icd11_ext_grading | 0.8000 | 100 |
| template | icd11_ext_anatomy | 0.2100 | 102 |
| template | icd11_ext_histopath | 0.0290 | 102 |
| non_template | icdo3_topography | 0.7189 | 442 |
| non_template | icdo3_morphology | 0.4399 | 441 |
| non_template | icdo3_behavior | 0.5638 | 441 |
| non_template | icdo3_grade | 0.9760 | 384 |
| non_template | icdo3_laterality | 0.9985 | 465 |
| non_template | icd11_stem | 0.3178 | 454 |
| non_template | icd11_ext_laterality | 0.4064 | 455 |
| non_template | icd11_ext_grading | 0.7509 | 454 |
| non_template | icd11_ext_anatomy | 0.2242 | 465 |
| non_template | icd11_ext_histopath | 0.1526 | 465 |
