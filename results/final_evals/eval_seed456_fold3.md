# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold3/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 3)
**Records evaluated:** 554

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6913 | 0.8154 | 0.6945 | 0.6920 | 0.7901 | 0.8154 | 531 |
| icdo3_morphology | 0.4455 | 0.9472 | 0.4537 | 0.4704 | 0.5627 | 0.9472 | 530 |
| icdo3_behavior | 0.5842 | 0.9811 | 0.5621 | 0.6150 | 0.5842 | 0.9811 | 530 |
| icdo3_grade | 0.9881 | 0.9896 | 0.9866 | 0.9895 | 0.9881 | 0.9896 | 479 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 554 |
| icd11_stem | 0.2684 | 0.9131 | 0.2783 | 0.2656 | 0.3451 | 0.9131 | 541 |
| icd11_ext_laterality | 0.3287 | 0.9778 | 0.3266 | 0.3308 | 0.3944 | 0.9778 | 541 |
| icd11_ext_grading | 0.7620 | 0.9647 | 0.7670 | 0.7575 | 0.9525 | 0.9647 | 538 |
| icd11_ext_anatomy | 0.2808 | 0.6884 | 0.3152 | 0.2716 | 0.5616 | 0.6137 | 554 |
| icd11_ext_histopath | 0.1759 | 0.7344 | 0.1867 | 0.1755 | 0.8093 | 0.9513 | 554 |
| **Mean** | **0.5525** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8154 | 0.9529 | 0.9831 | 0.9831 | 0.0000 |
| icdo3_morphology | 0.9472 | 0.9811 | 0.9830 | 0.9830 | 0.0000 |
| icdo3_behavior | 0.9811 | 0.9981 | 1.0000 | 1.0000 | 0.9659 |
| icdo3_grade | 0.9896 | 1.0000 | 1.0000 | 1.0000 | 0.9926 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9131 | 0.9630 | 0.9760 | 0.9760 | 0.0000 |
| icd11_ext_laterality | 0.9778 | 0.9963 | 0.9982 | 0.9982 | 0.0000 |
| icd11_ext_grading | 0.9647 | 0.9981 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7901 | 1 | 7 |
| icdo3_morphology | 0.2051 | 0.7296 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7789 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9881 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0926 | 0.6200 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3944 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9525 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1485 |
| icdo3_morphology | 0.0451 |
| icdo3_behavior | 0.0104 |
| icdo3_grade | 0.0070 |
| icdo3_laterality | 0.0018 |
| icd11_stem | 0.0564 |
| icd11_ext_laterality | 0.0141 |
| icd11_ext_grading | 0.0246 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9859**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6225 | 293 |
| batch_1 | icdo3_morphology | 0.3856 | 292 |
| batch_1 | icdo3_behavior | 0.5936 | 292 |
| batch_1 | icdo3_grade | 0.9906 | 261 |
| batch_1 | icdo3_laterality | 1.0000 | 315 |
| batch_1 | icd11_stem | 0.2445 | 302 |
| batch_1 | icd11_ext_laterality | 0.3266 | 302 |
| batch_1 | icd11_ext_grading | 0.7573 | 300 |
| batch_1 | icd11_ext_anatomy | 0.1201 | 315 |
| batch_1 | icd11_ext_histopath | 0.1798 | 315 |
| batch_2 | icdo3_topography | 0.6469 | 238 |
| batch_2 | icdo3_morphology | 0.2825 | 238 |
| batch_2 | icdo3_behavior | 0.4870 | 238 |
| batch_2 | icdo3_grade | 0.9821 | 218 |
| batch_2 | icdo3_laterality | 0.6667 | 239 |
| batch_2 | icd11_stem | 0.2273 | 239 |
| batch_2 | icd11_ext_laterality | 0.3313 | 239 |
| batch_2 | icd11_ext_grading | 0.7716 | 238 |
| batch_2 | icd11_ext_anatomy | 0.2038 | 239 |
| batch_2 | icd11_ext_histopath | 0.1594 | 239 |
| cancer_primary | icdo3_topography | 0.6913 | 531 |
| cancer_primary | icdo3_morphology | 0.4455 | 530 |
| cancer_primary | icdo3_behavior | 0.5842 | 530 |
| cancer_primary | icdo3_grade | 0.9881 | 478 |
| cancer_primary | icdo3_laterality | 1.0000 | 548 |
| cancer_primary | icd11_stem | 0.2446 | 535 |
| cancer_primary | icd11_ext_laterality | 0.3286 | 535 |
| cancer_primary | icd11_ext_grading | 0.7677 | 534 |
| cancer_primary | icd11_ext_anatomy | 0.2779 | 548 |
| cancer_primary | icd11_ext_histopath | 0.1759 | 548 |
| non_cancer | icdo3_topography | 0.0000 | 0 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 6 |
| non_cancer | icd11_stem | 0.0296 | 6 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.0800 | 4 |
| non_cancer | icd11_ext_anatomy | 0.1250 | 6 |
| non_cancer | icd11_ext_histopath | 0.0000 | 6 |
| template | icdo3_topography | 0.6259 | 95 |
| template | icdo3_morphology | 0.1250 | 95 |
| template | icdo3_behavior | 0.2487 | 95 |
| template | icdo3_grade | 0.9658 | 94 |
| template | icdo3_laterality | 0.6667 | 96 |
| template | icd11_stem | 0.1012 | 95 |
| template | icd11_ext_laterality | 0.3281 | 95 |
| template | icd11_ext_grading | 0.5947 | 95 |
| template | icd11_ext_anatomy | 0.2077 | 96 |
| template | icd11_ext_histopath | 0.0696 | 96 |
| non_template | icdo3_topography | 0.6800 | 436 |
| non_template | icdo3_morphology | 0.4442 | 435 |
| non_template | icdo3_behavior | 0.5951 | 435 |
| non_template | icdo3_grade | 0.9934 | 385 |
| non_template | icdo3_laterality | 1.0000 | 458 |
| non_template | icd11_stem | 0.2680 | 446 |
| non_template | icd11_ext_laterality | 0.3288 | 446 |
| non_template | icd11_ext_grading | 0.7597 | 443 |
| non_template | icd11_ext_anatomy | 0.2777 | 458 |
| non_template | icd11_ext_histopath | 0.1759 | 458 |
