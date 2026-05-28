# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed42_fold1/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 1)
**Records evaluated:** 540

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6802 | 0.7900 | 0.6871 | 0.6755 | 0.6802 | 0.7900 | 519 |
| icdo3_morphology | 0.3809 | 0.9379 | 0.3815 | 0.3985 | 0.5078 | 0.9379 | 515 |
| icdo3_behavior | 0.5776 | 0.9689 | 0.5900 | 0.5667 | 0.7701 | 0.9689 | 515 |
| icdo3_grade | 0.9609 | 0.9714 | 0.9538 | 0.9688 | 0.9609 | 0.9714 | 455 |
| icdo3_laterality | 0.9987 | 0.9981 | 0.9987 | 0.9988 | 0.9987 | 0.9981 | 540 |
| icd11_stem | 0.3169 | 0.9113 | 0.3365 | 0.3173 | 0.4074 | 0.9113 | 530 |
| icd11_ext_laterality | 0.3228 | 0.9546 | 0.3192 | 0.3265 | 0.3874 | 0.9546 | 529 |
| icd11_ext_grading | 0.7479 | 0.9581 | 0.7403 | 0.7569 | 0.9349 | 0.9581 | 525 |
| icd11_ext_anatomy | 0.2901 | 0.6564 | 0.3169 | 0.2866 | 0.5803 | 0.6056 | 540 |
| icd11_ext_histopath | 0.2030 | 0.7937 | 0.2005 | 0.2081 | 0.9338 | 0.9574 | 540 |
| **Mean** | **0.5479** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7900 | 0.9422 | 0.9692 | 0.9692 | 0.9396 |
| icdo3_morphology | 0.9379 | 0.9786 | 0.9883 | 0.9883 | 0.0000 |
| icdo3_behavior | 0.9689 | 0.9981 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9714 | 1.0000 | 1.0000 | 1.0000 | 0.9916 |
| icdo3_laterality | 0.9981 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9113 | 0.9679 | 0.9774 | 0.9774 | 0.0000 |
| icd11_ext_laterality | 0.9546 | 0.9887 | 0.9962 | 0.9962 | 0.0000 |
| icd11_ext_grading | 0.9581 | 0.9905 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7773 | 1 | 7 |
| icdo3_morphology | 0.1538 | 0.6491 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7701 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9609 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9987 | 0 | 3 |
| icd11_stem | 0.1111 | 0.7284 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3874 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9349 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1279 |
| icdo3_morphology | 0.0475 |
| icdo3_behavior | 0.0267 |
| icdo3_grade | 0.0272 |
| icdo3_laterality | 0.0014 |
| icd11_stem | 0.0659 |
| icd11_ext_laterality | 0.0297 |
| icd11_ext_grading | 0.0347 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9375**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5612 | 292 |
| batch_1 | icdo3_morphology | 0.4087 | 288 |
| batch_1 | icdo3_behavior | 0.5807 | 288 |
| batch_1 | icdo3_grade | 0.9616 | 247 |
| batch_1 | icdo3_laterality | 0.9978 | 313 |
| batch_1 | icd11_stem | 0.2946 | 303 |
| batch_1 | icd11_ext_laterality | 0.3176 | 302 |
| batch_1 | icd11_ext_grading | 0.7427 | 298 |
| batch_1 | icd11_ext_anatomy | 0.1719 | 313 |
| batch_1 | icd11_ext_histopath | 0.2060 | 313 |
| batch_2 | icdo3_topography | 0.6907 | 227 |
| batch_2 | icdo3_morphology | 0.2128 | 227 |
| batch_2 | icdo3_behavior | 0.5810 | 227 |
| batch_2 | icdo3_grade | 0.9597 | 208 |
| batch_2 | icdo3_laterality | 0.6667 | 227 |
| batch_2 | icd11_stem | 0.2041 | 227 |
| batch_2 | icd11_ext_laterality | 0.3296 | 227 |
| batch_2 | icd11_ext_grading | 0.7569 | 227 |
| batch_2 | icd11_ext_anatomy | 0.1702 | 227 |
| batch_2 | icd11_ext_histopath | 0.2025 | 227 |
| cancer_primary | icdo3_topography | 0.6801 | 518 |
| cancer_primary | icdo3_morphology | 0.3809 | 515 |
| cancer_primary | icdo3_behavior | 0.5776 | 515 |
| cancer_primary | icdo3_grade | 0.9608 | 454 |
| cancer_primary | icdo3_laterality | 0.9987 | 533 |
| cancer_primary | icd11_stem | 0.3067 | 523 |
| cancer_primary | icd11_ext_laterality | 0.3227 | 523 |
| cancer_primary | icd11_ext_grading | 0.7476 | 520 |
| cancer_primary | icd11_ext_anatomy | 0.2921 | 533 |
| cancer_primary | icd11_ext_histopath | 0.2030 | 533 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0148 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1778 | 5 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.7006 | 78 |
| template | icdo3_morphology | 0.0750 | 78 |
| template | icdo3_behavior | 0.2500 | 78 |
| template | icdo3_grade | 0.6189 | 77 |
| template | icdo3_laterality | 0.6667 | 78 |
| template | icd11_stem | 0.0847 | 78 |
| template | icd11_ext_laterality | 0.3270 | 78 |
| template | icd11_ext_grading | 0.4000 | 78 |
| template | icd11_ext_anatomy | 0.2259 | 78 |
| template | icd11_ext_histopath | 0.0435 | 78 |
| non_template | icdo3_topography | 0.6690 | 441 |
| non_template | icdo3_morphology | 0.3843 | 437 |
| non_template | icdo3_behavior | 0.5768 | 437 |
| non_template | icdo3_grade | 0.9661 | 378 |
| non_template | icdo3_laterality | 0.9985 | 462 |
| non_template | icd11_stem | 0.3163 | 452 |
| non_template | icd11_ext_laterality | 0.3221 | 451 |
| non_template | icd11_ext_grading | 0.7455 | 447 |
| non_template | icd11_ext_anatomy | 0.2868 | 462 |
| non_template | icd11_ext_histopath | 0.2030 | 462 |
