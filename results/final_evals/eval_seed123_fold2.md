# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed123_fold2/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 2)
**Records evaluated:** 548

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.7002 | 0.8212 | 0.7266 | 0.6840 | 0.7002 | 0.8212 | 520 |
| icdo3_morphology | 0.4429 | 0.9362 | 0.4544 | 0.4732 | 0.5594 | 0.9362 | 517 |
| icdo3_behavior | 0.6946 | 0.9884 | 0.6650 | 0.7332 | 0.9261 | 0.9884 | 517 |
| icdo3_grade | 0.9517 | 0.9628 | 0.9394 | 0.9654 | 0.9517 | 0.9628 | 457 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 548 |
| icd11_stem | 0.2597 | 0.9036 | 0.2482 | 0.2757 | 0.3506 | 0.9036 | 529 |
| icd11_ext_laterality | 0.3837 | 0.9792 | 0.3627 | 0.4935 | 0.5755 | 0.9792 | 528 |
| icd11_ext_grading | 0.7601 | 0.9659 | 0.7562 | 0.7652 | 0.7601 | 0.9659 | 528 |
| icd11_ext_anatomy | 0.2196 | 0.6727 | 0.2386 | 0.2229 | 0.5855 | 0.6296 | 548 |
| icd11_ext_histopath | 0.1800 | 0.6942 | 0.1904 | 0.1872 | 0.6898 | 0.9453 | 548 |
| **Mean** | **0.5592** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8212 | 0.9558 | 0.9712 | 0.9712 | 0.9108 |
| icdo3_morphology | 0.9362 | 0.9613 | 0.9710 | 0.9710 | 0.0000 |
| icdo3_behavior | 0.9884 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9628 | 1.0000 | 1.0000 | 1.0000 | 0.9763 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9036 | 0.9565 | 0.9660 | 0.9660 | 0.0000 |
| icd11_ext_laterality | 0.9792 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9659 | 0.9886 | 1.0000 | 1.0000 | 0.9704 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.8002 | 1 | 7 |
| icdo3_morphology | 0.2051 | 0.7239 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.9261 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9517 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0556 | 0.6679 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4604 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9501 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1778 |
| icdo3_morphology | 0.0553 |
| icdo3_behavior | 0.0081 |
| icdo3_grade | 0.0332 |
| icdo3_laterality | 0.0014 |
| icd11_stem | 0.0701 |
| icd11_ext_laterality | 0.0100 |
| icd11_ext_grading | 0.0360 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9151**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6562 | 285 |
| batch_1 | icdo3_morphology | 0.4632 | 282 |
| batch_1 | icdo3_behavior | 0.6991 | 282 |
| batch_1 | icdo3_grade | 0.9748 | 240 |
| batch_1 | icdo3_laterality | 1.0000 | 313 |
| batch_1 | icd11_stem | 0.2596 | 294 |
| batch_1 | icd11_ext_laterality | 0.4102 | 293 |
| batch_1 | icd11_ext_grading | 0.7601 | 293 |
| batch_1 | icd11_ext_anatomy | 0.1006 | 313 |
| batch_1 | icd11_ext_histopath | 0.1842 | 313 |
| batch_2 | icdo3_topography | 0.6508 | 235 |
| batch_2 | icdo3_morphology | 0.2233 | 235 |
| batch_2 | icdo3_behavior | 0.4572 | 235 |
| batch_2 | icdo3_grade | 0.8817 | 217 |
| batch_2 | icdo3_laterality | 0.6667 | 235 |
| batch_2 | icd11_stem | 0.1953 | 235 |
| batch_2 | icd11_ext_laterality | 0.3301 | 235 |
| batch_2 | icd11_ext_grading | 0.7435 | 235 |
| batch_2 | icd11_ext_anatomy | 0.1732 | 235 |
| batch_2 | icd11_ext_histopath | 0.1111 | 235 |
| cancer_primary | icdo3_topography | 0.6998 | 518 |
| cancer_primary | icdo3_morphology | 0.4429 | 517 |
| cancer_primary | icdo3_behavior | 0.6946 | 517 |
| cancer_primary | icdo3_grade | 0.9517 | 457 |
| cancer_primary | icdo3_laterality | 1.0000 | 541 |
| cancer_primary | icd11_stem | 0.2421 | 522 |
| cancer_primary | icd11_ext_laterality | 0.3836 | 522 |
| cancer_primary | icd11_ext_grading | 0.7598 | 522 |
| cancer_primary | icd11_ext_anatomy | 0.2193 | 541 |
| cancer_primary | icd11_ext_histopath | 0.1800 | 541 |
| non_cancer | icdo3_topography | 0.2500 | 2 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0247 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1818 | 6 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.6701 | 79 |
| template | icdo3_morphology | 0.1414 | 79 |
| template | icdo3_behavior | 0.2500 | 79 |
| template | icdo3_grade | 0.9149 | 80 |
| template | icdo3_laterality | 0.6667 | 80 |
| template | icd11_stem | 0.1599 | 79 |
| template | icd11_ext_laterality | 0.3333 | 79 |
| template | icd11_ext_grading | 0.5910 | 79 |
| template | icd11_ext_anatomy | 0.2051 | 80 |
| template | icd11_ext_histopath | 0.0725 | 80 |
| non_template | icdo3_topography | 0.6911 | 441 |
| non_template | icdo3_morphology | 0.4381 | 438 |
| non_template | icdo3_behavior | 0.6943 | 438 |
| non_template | icdo3_grade | 0.9516 | 377 |
| non_template | icdo3_laterality | 1.0000 | 468 |
| non_template | icd11_stem | 0.2492 | 450 |
| non_template | icd11_ext_laterality | 0.3827 | 449 |
| non_template | icd11_ext_grading | 0.7592 | 449 |
| non_template | icd11_ext_anatomy | 0.2150 | 468 |
| non_template | icd11_ext_histopath | 0.1795 | 468 |
