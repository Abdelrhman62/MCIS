# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed123_fold1/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 1)
**Records evaluated:** 540

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.7078 | 0.8131 | 0.7124 | 0.7073 | 0.7078 | 0.8131 | 519 |
| icdo3_morphology | 0.3896 | 0.9379 | 0.3979 | 0.3886 | 0.5195 | 0.9379 | 515 |
| icdo3_behavior | 0.5778 | 0.9709 | 0.5749 | 0.5809 | 0.7704 | 0.9709 | 515 |
| icdo3_grade | 0.9639 | 0.9736 | 0.9553 | 0.9732 | 0.9639 | 0.9736 | 455 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 540 |
| icd11_stem | 0.2916 | 0.9075 | 0.2824 | 0.3139 | 0.3750 | 0.9075 | 530 |
| icd11_ext_laterality | 0.3228 | 0.9565 | 0.3186 | 0.3273 | 0.3874 | 0.9565 | 529 |
| icd11_ext_grading | 0.7518 | 0.9638 | 0.7442 | 0.7606 | 0.9398 | 0.9638 | 525 |
| icd11_ext_anatomy | 0.1993 | 0.6143 | 0.2159 | 0.1949 | 0.5315 | 0.5926 | 540 |
| icd11_ext_histopath | 0.1781 | 0.7603 | 0.2014 | 0.1708 | 0.8193 | 0.9537 | 540 |
| **Mean** | **0.5383** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8131 | 0.9499 | 0.9807 | 0.9807 | 0.9332 |
| icdo3_morphology | 0.9379 | 0.9767 | 0.9845 | 0.9845 | 0.0000 |
| icdo3_behavior | 0.9709 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9736 | 1.0000 | 1.0000 | 1.0000 | 0.9902 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9075 | 0.9623 | 0.9698 | 0.9698 | 0.0000 |
| icd11_ext_laterality | 0.9565 | 0.9924 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9638 | 0.9905 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.8090 | 1 | 7 |
| icdo3_morphology | 0.1538 | 0.6683 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7704 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9639 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.1000 | 0.6749 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3874 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9398 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1733 |
| icdo3_morphology | 0.0390 |
| icdo3_behavior | 0.0237 |
| icdo3_grade | 0.0147 |
| icdo3_laterality | 0.0014 |
| icd11_stem | 0.0552 |
| icd11_ext_laterality | 0.0286 |
| icd11_ext_grading | 0.0321 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9454**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6228 | 292 |
| batch_1 | icdo3_morphology | 0.3778 | 288 |
| batch_1 | icdo3_behavior | 0.5780 | 288 |
| batch_1 | icdo3_grade | 0.9614 | 247 |
| batch_1 | icdo3_laterality | 1.0000 | 313 |
| batch_1 | icd11_stem | 0.2927 | 303 |
| batch_1 | icd11_ext_laterality | 0.3210 | 302 |
| batch_1 | icd11_ext_grading | 0.7468 | 298 |
| batch_1 | icd11_ext_anatomy | 0.0799 | 313 |
| batch_1 | icd11_ext_histopath | 0.1784 | 313 |
| batch_2 | icdo3_topography | 0.7034 | 227 |
| batch_2 | icdo3_morphology | 0.2584 | 227 |
| batch_2 | icdo3_behavior | 0.5810 | 227 |
| batch_2 | icdo3_grade | 0.9681 | 208 |
| batch_2 | icdo3_laterality | 0.6667 | 227 |
| batch_2 | icd11_stem | 0.2096 | 227 |
| batch_2 | icd11_ext_laterality | 0.3252 | 227 |
| batch_2 | icd11_ext_grading | 0.7615 | 227 |
| batch_2 | icd11_ext_anatomy | 0.1535 | 227 |
| batch_2 | icd11_ext_histopath | 0.1590 | 227 |
| cancer_primary | icdo3_topography | 0.7078 | 518 |
| cancer_primary | icdo3_morphology | 0.3896 | 515 |
| cancer_primary | icdo3_behavior | 0.5778 | 515 |
| cancer_primary | icdo3_grade | 0.9639 | 454 |
| cancer_primary | icdo3_laterality | 1.0000 | 533 |
| cancer_primary | icd11_stem | 0.2799 | 523 |
| cancer_primary | icd11_ext_laterality | 0.3227 | 523 |
| cancer_primary | icd11_ext_grading | 0.7515 | 520 |
| cancer_primary | icd11_ext_anatomy | 0.1997 | 533 |
| cancer_primary | icd11_ext_histopath | 0.1781 | 533 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0148 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1778 | 5 |
| non_cancer | icd11_ext_anatomy | 0.0417 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.7333 | 78 |
| template | icdo3_morphology | 0.0750 | 78 |
| template | icdo3_behavior | 0.2500 | 78 |
| template | icdo3_grade | 0.6358 | 77 |
| template | icdo3_laterality | 0.6667 | 78 |
| template | icd11_stem | 0.0958 | 78 |
| template | icd11_ext_laterality | 0.3270 | 78 |
| template | icd11_ext_grading | 0.4000 | 78 |
| template | icd11_ext_anatomy | 0.2087 | 78 |
| template | icd11_ext_histopath | 0.0000 | 78 |
| non_template | icdo3_topography | 0.6970 | 441 |
| non_template | icdo3_morphology | 0.3931 | 437 |
| non_template | icdo3_behavior | 0.5770 | 437 |
| non_template | icdo3_grade | 0.9659 | 378 |
| non_template | icdo3_laterality | 1.0000 | 462 |
| non_template | icd11_stem | 0.2904 | 452 |
| non_template | icd11_ext_laterality | 0.3221 | 451 |
| non_template | icd11_ext_grading | 0.7499 | 447 |
| non_template | icd11_ext_anatomy | 0.1950 | 462 |
| non_template | icd11_ext_histopath | 0.1781 | 462 |
