# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold1/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 1)
**Records evaluated:** 540

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6730 | 0.7861 | 0.6944 | 0.6579 | 0.6730 | 0.7861 | 519 |
| icdo3_morphology | 0.4032 | 0.9456 | 0.4161 | 0.4057 | 0.5377 | 0.9456 | 515 |
| icdo3_behavior | 0.5648 | 0.9748 | 0.6047 | 0.5399 | 0.7530 | 0.9748 | 515 |
| icdo3_grade | 0.9667 | 0.9758 | 0.9566 | 0.9776 | 0.9667 | 0.9758 | 455 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 540 |
| icd11_stem | 0.2835 | 0.9075 | 0.2783 | 0.2901 | 0.3645 | 0.9075 | 530 |
| icd11_ext_laterality | 0.3904 | 0.9584 | 0.4040 | 0.3824 | 0.4685 | 0.9584 | 529 |
| icd11_ext_grading | 0.7471 | 0.9600 | 0.7382 | 0.7569 | 0.9339 | 0.9600 | 525 |
| icd11_ext_anatomy | 0.3371 | 0.6667 | 0.3744 | 0.3213 | 0.5393 | 0.6315 | 540 |
| icd11_ext_histopath | 0.2200 | 0.8031 | 0.2440 | 0.2153 | 0.8433 | 0.9574 | 540 |
| **Mean** | **0.5586** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7861 | 0.9499 | 0.9827 | 0.9827 | 0.9387 |
| icdo3_morphology | 0.9456 | 0.9728 | 0.9748 | 0.9748 | 0.0000 |
| icdo3_behavior | 0.9748 | 0.9981 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9758 | 1.0000 | 1.0000 | 1.0000 | 0.9882 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9075 | 0.9547 | 0.9774 | 0.9774 | 0.0000 |
| icd11_ext_laterality | 0.9584 | 0.9924 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9600 | 0.9886 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7691 | 1 | 7 |
| icdo3_morphology | 0.1538 | 0.6980 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7530 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9667 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0556 | 0.7395 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4685 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9339 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1286 |
| icdo3_morphology | 0.0446 |
| icdo3_behavior | 0.0239 |
| icdo3_grade | 0.0185 |
| icdo3_laterality | 0.0010 |
| icd11_stem | 0.0688 |
| icd11_ext_laterality | 0.0398 |
| icd11_ext_grading | 0.0305 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9474**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5983 | 292 |
| batch_1 | icdo3_morphology | 0.3980 | 288 |
| batch_1 | icdo3_behavior | 0.5545 | 288 |
| batch_1 | icdo3_grade | 0.9759 | 247 |
| batch_1 | icdo3_laterality | 1.0000 | 313 |
| batch_1 | icd11_stem | 0.2633 | 303 |
| batch_1 | icd11_ext_laterality | 0.3185 | 302 |
| batch_1 | icd11_ext_grading | 0.7469 | 298 |
| batch_1 | icd11_ext_anatomy | 0.2089 | 313 |
| batch_1 | icd11_ext_histopath | 0.1915 | 313 |
| batch_2 | icdo3_topography | 0.6587 | 227 |
| batch_2 | icdo3_morphology | 0.2624 | 227 |
| batch_2 | icdo3_behavior | 0.5810 | 227 |
| batch_2 | icdo3_grade | 0.9536 | 208 |
| batch_2 | icdo3_laterality | 0.6667 | 227 |
| batch_2 | icd11_stem | 0.2106 | 227 |
| batch_2 | icd11_ext_laterality | 0.4971 | 227 |
| batch_2 | icd11_ext_grading | 0.7419 | 227 |
| batch_2 | icd11_ext_anatomy | 0.2125 | 227 |
| batch_2 | icd11_ext_histopath | 0.2547 | 227 |
| cancer_primary | icdo3_topography | 0.6729 | 518 |
| cancer_primary | icdo3_morphology | 0.4032 | 515 |
| cancer_primary | icdo3_behavior | 0.5648 | 515 |
| cancer_primary | icdo3_grade | 0.9667 | 454 |
| cancer_primary | icdo3_laterality | 1.0000 | 533 |
| cancer_primary | icd11_stem | 0.2656 | 523 |
| cancer_primary | icd11_ext_laterality | 0.3903 | 523 |
| cancer_primary | icd11_ext_grading | 0.7502 | 520 |
| cancer_primary | icd11_ext_anatomy | 0.3391 | 533 |
| cancer_primary | icd11_ext_histopath | 0.2200 | 533 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0247 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1500 | 5 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.7482 | 78 |
| template | icdo3_morphology | 0.0750 | 78 |
| template | icdo3_behavior | 0.2500 | 78 |
| template | icdo3_grade | 0.6358 | 77 |
| template | icdo3_laterality | 0.6667 | 78 |
| template | icd11_stem | 0.0847 | 78 |
| template | icd11_ext_laterality | 0.3243 | 78 |
| template | icd11_ext_grading | 0.4000 | 78 |
| template | icd11_ext_anatomy | 0.2497 | 78 |
| template | icd11_ext_histopath | 0.0000 | 78 |
| non_template | icdo3_topography | 0.6540 | 441 |
| non_template | icdo3_morphology | 0.4067 | 437 |
| non_template | icdo3_behavior | 0.5641 | 437 |
| non_template | icdo3_grade | 0.9693 | 378 |
| non_template | icdo3_laterality | 1.0000 | 462 |
| non_template | icd11_stem | 0.2832 | 452 |
| non_template | icd11_ext_laterality | 0.3903 | 451 |
| non_template | icd11_ext_grading | 0.7449 | 447 |
| non_template | icd11_ext_anatomy | 0.3337 | 462 |
| non_template | icd11_ext_histopath | 0.2287 | 462 |
