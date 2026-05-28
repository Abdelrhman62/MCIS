# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed42_fold0/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 0)
**Records evaluated:** 550

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6552 | 0.7905 | 0.6531 | 0.6748 | 0.6552 | 0.7905 | 525 |
| icdo3_morphology | 0.3823 | 0.9314 | 0.3989 | 0.3781 | 0.4829 | 0.9314 | 525 |
| icdo3_behavior | 0.6364 | 0.9828 | 0.6886 | 0.6019 | 0.8485 | 0.9828 | 524 |
| icdo3_grade | 0.9856 | 0.9872 | 0.9824 | 0.9889 | 0.9856 | 0.9872 | 467 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 549 |
| icd11_stem | 0.2866 | 0.8953 | 0.2780 | 0.3107 | 0.3869 | 0.8953 | 535 |
| icd11_ext_laterality | 0.3241 | 0.9663 | 0.3219 | 0.3265 | 0.3890 | 0.9663 | 534 |
| icd11_ext_grading | 0.7661 | 0.9663 | 0.7637 | 0.7685 | 0.7661 | 0.9663 | 534 |
| icd11_ext_anatomy | 0.2513 | 0.6617 | 0.2462 | 0.2665 | 0.5745 | 0.5982 | 550 |
| icd11_ext_histopath | 0.1694 | 0.7231 | 0.1694 | 0.1731 | 0.7793 | 0.9436 | 550 |
| **Mean** | **0.5457** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7905 | 0.9467 | 0.9867 | 0.9867 | 0.8435 |
| icdo3_morphology | 0.9314 | 0.9752 | 0.9810 | 0.9810 | 0.0000 |
| icdo3_behavior | 0.9828 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9872 | 1.0000 | 1.0000 | 1.0000 | 0.9965 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8953 | 0.9551 | 0.9645 | 0.9645 | 0.0000 |
| icd11_ext_laterality | 0.9663 | 0.9944 | 0.9963 | 0.9963 | 0.0000 |
| icd11_ext_grading | 0.9663 | 0.9963 | 1.0000 | 1.0000 | 0.9524 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7489 | 1 | 7 |
| icdo3_morphology | 0.0769 | 0.7432 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8485 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9856 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.1111 | 0.6376 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3890 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9576 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1587 |
| icdo3_morphology | 0.0462 |
| icdo3_behavior | 0.0126 |
| icdo3_grade | 0.0088 |
| icdo3_laterality | 0.0039 |
| icd11_stem | 0.0652 |
| icd11_ext_laterality | 0.0125 |
| icd11_ext_grading | 0.0238 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9774**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5606 | 285 |
| batch_1 | icdo3_morphology | 0.3624 | 285 |
| batch_1 | icdo3_behavior | 0.6441 | 285 |
| batch_1 | icdo3_grade | 0.9775 | 247 |
| batch_1 | icdo3_laterality | 1.0000 | 309 |
| batch_1 | icd11_stem | 0.2674 | 296 |
| batch_1 | icd11_ext_laterality | 0.3195 | 294 |
| batch_1 | icd11_ext_grading | 0.7559 | 294 |
| batch_1 | icd11_ext_anatomy | 0.1498 | 310 |
| batch_1 | icd11_ext_histopath | 0.1719 | 310 |
| batch_2 | icdo3_topography | 0.6709 | 240 |
| batch_2 | icdo3_morphology | 0.2228 | 240 |
| batch_2 | icdo3_behavior | 0.4896 | 239 |
| batch_2 | icdo3_grade | 0.9953 | 220 |
| batch_2 | icdo3_laterality | 0.6667 | 240 |
| batch_2 | icd11_stem | 0.2071 | 239 |
| batch_2 | icd11_ext_laterality | 0.3298 | 240 |
| batch_2 | icd11_ext_grading | 0.7817 | 240 |
| batch_2 | icd11_ext_anatomy | 0.1553 | 240 |
| batch_2 | icd11_ext_histopath | 0.1627 | 240 |
| cancer_primary | icdo3_topography | 0.6552 | 525 |
| cancer_primary | icdo3_morphology | 0.3823 | 525 |
| cancer_primary | icdo3_behavior | 0.6364 | 524 |
| cancer_primary | icdo3_grade | 0.9856 | 467 |
| cancer_primary | icdo3_laterality | 1.0000 | 542 |
| cancer_primary | icd11_stem | 0.2825 | 528 |
| cancer_primary | icd11_ext_laterality | 0.3244 | 527 |
| cancer_primary | icd11_ext_grading | 0.7644 | 528 |
| cancer_primary | icd11_ext_anatomy | 0.2524 | 543 |
| cancer_primary | icd11_ext_histopath | 0.1694 | 543 |
| non_cancer | icdo3_topography | 0.0000 | 0 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 1.0000 | 7 |
| non_cancer | icd11_stem | 0.0247 | 7 |
| non_cancer | icd11_ext_laterality | 0.3000 | 7 |
| non_cancer | icd11_ext_grading | 0.2000 | 6 |
| non_cancer | icd11_ext_anatomy | 0.0417 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.5508 | 85 |
| template | icdo3_morphology | 0.1429 | 85 |
| template | icdo3_behavior | 0.2500 | 85 |
| template | icdo3_grade | 0.9890 | 83 |
| template | icdo3_laterality | 0.6667 | 85 |
| template | icd11_stem | 0.0970 | 85 |
| template | icd11_ext_laterality | 0.3312 | 85 |
| template | icd11_ext_grading | 0.5878 | 85 |
| template | icd11_ext_anatomy | 0.2201 | 85 |
| template | icd11_ext_histopath | 0.0217 | 85 |
| non_template | icdo3_topography | 0.6314 | 440 |
| non_template | icdo3_morphology | 0.3751 | 440 |
| non_template | icdo3_behavior | 0.6359 | 439 |
| non_template | icdo3_grade | 0.9856 | 384 |
| non_template | icdo3_laterality | 1.0000 | 464 |
| non_template | icd11_stem | 0.2794 | 450 |
| non_template | icd11_ext_laterality | 0.3228 | 449 |
| non_template | icd11_ext_grading | 0.7659 | 449 |
| non_template | icd11_ext_anatomy | 0.2513 | 465 |
| non_template | icd11_ext_histopath | 0.1540 | 465 |
