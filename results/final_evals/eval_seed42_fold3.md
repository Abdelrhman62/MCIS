# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed42_fold3/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 3)
**Records evaluated:** 554

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6607 | 0.7815 | 0.6624 | 0.6614 | 0.7551 | 0.7815 | 531 |
| icdo3_morphology | 0.4315 | 0.9453 | 0.4163 | 0.4680 | 0.5450 | 0.9453 | 530 |
| icdo3_behavior | 0.6030 | 0.9811 | 0.5922 | 0.6150 | 0.6030 | 0.9811 | 530 |
| icdo3_grade | 0.9809 | 0.9833 | 0.9786 | 0.9833 | 0.9809 | 0.9833 | 479 |
| icdo3_laterality | 0.9772 | 0.9982 | 0.9987 | 0.9583 | 0.9772 | 0.9982 | 554 |
| icd11_stem | 0.3307 | 0.9187 | 0.3196 | 0.3449 | 0.4251 | 0.9187 | 541 |
| icd11_ext_laterality | 0.4126 | 0.9797 | 0.3950 | 0.4413 | 0.4951 | 0.9797 | 541 |
| icd11_ext_grading | 0.7543 | 0.9591 | 0.7579 | 0.7508 | 0.9428 | 0.9591 | 538 |
| icd11_ext_anatomy | 0.2939 | 0.6708 | 0.3111 | 0.2916 | 0.5225 | 0.6101 | 554 |
| icd11_ext_histopath | 0.2413 | 0.7407 | 0.2411 | 0.2494 | 0.7930 | 0.9513 | 554 |
| **Mean** | **0.5686** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7815 | 0.9699 | 0.9849 | 0.9849 | 0.0000 |
| icdo3_morphology | 0.9453 | 0.9679 | 0.9774 | 0.9774 | 0.0000 |
| icdo3_behavior | 0.9811 | 0.9981 | 1.0000 | 1.0000 | 0.9466 |
| icdo3_grade | 0.9833 | 1.0000 | 1.0000 | 1.0000 | 0.9952 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9187 | 0.9704 | 0.9778 | 0.9778 | 0.0000 |
| icd11_ext_laterality | 0.9797 | 0.9945 | 0.9982 | 0.9982 | 0.0000 |
| icd11_ext_grading | 0.9591 | 0.9907 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7551 | 1 | 7 |
| icdo3_morphology | 0.1923 | 0.7141 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8039 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9809 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9772 | 0 | 3 |
| icd11_stem | 0.1111 | 0.7698 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4951 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9428 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1424 |
| icdo3_morphology | 0.0458 |
| icdo3_behavior | 0.0135 |
| icdo3_grade | 0.0145 |
| icdo3_laterality | 0.0022 |
| icd11_stem | 0.0576 |
| icd11_ext_laterality | 0.0125 |
| icd11_ext_grading | 0.0321 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9753**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5739 | 293 |
| batch_1 | icdo3_morphology | 0.3910 | 292 |
| batch_1 | icdo3_behavior | 0.6411 | 292 |
| batch_1 | icdo3_grade | 0.9821 | 261 |
| batch_1 | icdo3_laterality | 0.9767 | 315 |
| batch_1 | icd11_stem | 0.3365 | 302 |
| batch_1 | icd11_ext_laterality | 0.4230 | 302 |
| batch_1 | icd11_ext_grading | 0.7448 | 300 |
| batch_1 | icd11_ext_anatomy | 0.1360 | 315 |
| batch_1 | icd11_ext_histopath | 0.2273 | 315 |
| batch_2 | icdo3_topography | 0.6330 | 238 |
| batch_2 | icdo3_morphology | 0.2685 | 238 |
| batch_2 | icdo3_behavior | 0.4864 | 238 |
| batch_2 | icdo3_grade | 0.9776 | 218 |
| batch_2 | icdo3_laterality | 0.6667 | 239 |
| batch_2 | icd11_stem | 0.2317 | 239 |
| batch_2 | icd11_ext_laterality | 0.3313 | 239 |
| batch_2 | icd11_ext_grading | 0.7716 | 238 |
| batch_2 | icd11_ext_anatomy | 0.2076 | 239 |
| batch_2 | icd11_ext_histopath | 0.2062 | 239 |
| cancer_primary | icdo3_topography | 0.6607 | 531 |
| cancer_primary | icdo3_morphology | 0.4315 | 530 |
| cancer_primary | icdo3_behavior | 0.6030 | 530 |
| cancer_primary | icdo3_grade | 0.9809 | 478 |
| cancer_primary | icdo3_laterality | 0.9771 | 548 |
| cancer_primary | icd11_stem | 0.2991 | 535 |
| cancer_primary | icd11_ext_laterality | 0.4126 | 535 |
| cancer_primary | icd11_ext_grading | 0.7548 | 534 |
| cancer_primary | icd11_ext_anatomy | 0.2955 | 548 |
| cancer_primary | icd11_ext_histopath | 0.2413 | 548 |
| non_cancer | icdo3_topography | 0.0000 | 0 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 6 |
| non_cancer | icd11_stem | 0.0370 | 6 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1714 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0000 | 6 |
| non_cancer | icd11_ext_histopath | 0.0000 | 6 |
| template | icdo3_topography | 0.5687 | 95 |
| template | icdo3_morphology | 0.1250 | 95 |
| template | icdo3_behavior | 0.2487 | 95 |
| template | icdo3_grade | 0.9396 | 94 |
| template | icdo3_laterality | 0.6667 | 96 |
| template | icd11_stem | 0.1012 | 95 |
| template | icd11_ext_laterality | 0.3281 | 95 |
| template | icd11_ext_grading | 0.5947 | 95 |
| template | icd11_ext_anatomy | 0.2041 | 96 |
| template | icd11_ext_histopath | 0.0696 | 96 |
| non_template | icdo3_topography | 0.6524 | 436 |
| non_template | icdo3_morphology | 0.4301 | 435 |
| non_template | icdo3_behavior | 0.6206 | 435 |
| non_template | icdo3_grade | 0.9901 | 385 |
| non_template | icdo3_laterality | 0.9770 | 458 |
| non_template | icd11_stem | 0.3349 | 446 |
| non_template | icd11_ext_laterality | 0.4129 | 446 |
| non_template | icd11_ext_grading | 0.7503 | 443 |
| non_template | icd11_ext_anatomy | 0.2951 | 458 |
| non_template | icd11_ext_histopath | 0.2411 | 458 |
