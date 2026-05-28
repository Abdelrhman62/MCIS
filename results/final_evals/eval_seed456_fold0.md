# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold0/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 0)
**Records evaluated:** 550

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6718 | 0.7962 | 0.6759 | 0.6730 | 0.6718 | 0.7962 | 525 |
| icdo3_morphology | 0.3682 | 0.9429 | 0.3686 | 0.3739 | 0.4651 | 0.9429 | 525 |
| icdo3_behavior | 0.6583 | 0.9905 | 0.7406 | 0.6176 | 0.8777 | 0.9905 | 524 |
| icdo3_grade | 0.9816 | 0.9850 | 0.9823 | 0.9810 | 0.9816 | 0.9850 | 467 |
| icdo3_laterality | 0.9988 | 0.9982 | 0.9988 | 0.9987 | 0.9988 | 0.9982 | 549 |
| icd11_stem | 0.2961 | 0.9121 | 0.2887 | 0.3131 | 0.3997 | 0.9121 | 535 |
| icd11_ext_laterality | 0.4078 | 0.9682 | 0.4891 | 0.3821 | 0.4894 | 0.9682 | 534 |
| icd11_ext_grading | 0.7643 | 0.9644 | 0.7601 | 0.7690 | 0.7643 | 0.9644 | 534 |
| icd11_ext_anatomy | 0.3470 | 0.6891 | 0.3743 | 0.3357 | 0.6169 | 0.6473 | 550 |
| icd11_ext_histopath | 0.2019 | 0.7368 | 0.2063 | 0.2026 | 0.7740 | 0.9473 | 550 |
| **Mean** | **0.5696** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7962 | 0.9410 | 0.9695 | 0.9695 | 0.9180 |
| icdo3_morphology | 0.9429 | 0.9657 | 0.9752 | 0.9752 | 0.0000 |
| icdo3_behavior | 0.9905 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9850 | 1.0000 | 1.0000 | 1.0000 | 0.9985 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9121 | 0.9607 | 0.9776 | 0.9776 | 0.0000 |
| icd11_ext_laterality | 0.9682 | 0.9981 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9644 | 0.9981 | 1.0000 | 1.0000 | 0.9551 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7678 | 1 | 7 |
| icdo3_morphology | 0.0769 | 0.7125 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8777 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9816 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9988 | 0 | 3 |
| icd11_stem | 0.0926 | 0.7031 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4894 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9554 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1353 |
| icdo3_morphology | 0.0510 |
| icdo3_behavior | 0.0094 |
| icdo3_grade | 0.0120 |
| icdo3_laterality | 0.0018 |
| icd11_stem | 0.0657 |
| icd11_ext_laterality | 0.0307 |
| icd11_ext_grading | 0.0253 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9753**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5944 | 285 |
| batch_1 | icdo3_morphology | 0.3546 | 285 |
| batch_1 | icdo3_behavior | 0.6679 | 285 |
| batch_1 | icdo3_grade | 0.9755 | 247 |
| batch_1 | icdo3_laterality | 0.9978 | 309 |
| batch_1 | icd11_stem | 0.2801 | 296 |
| batch_1 | icd11_ext_laterality | 0.4046 | 294 |
| batch_1 | icd11_ext_grading | 0.7539 | 294 |
| batch_1 | icd11_ext_anatomy | 0.1704 | 310 |
| batch_1 | icd11_ext_histopath | 0.1764 | 310 |
| batch_2 | icdo3_topography | 0.6590 | 240 |
| batch_2 | icdo3_morphology | 0.2232 | 240 |
| batch_2 | icdo3_behavior | 0.4994 | 239 |
| batch_2 | icdo3_grade | 0.9888 | 220 |
| batch_2 | icdo3_laterality | 0.6667 | 240 |
| batch_2 | icd11_stem | 0.2122 | 239 |
| batch_2 | icd11_ext_laterality | 0.3284 | 240 |
| batch_2 | icd11_ext_grading | 0.7843 | 240 |
| batch_2 | icd11_ext_anatomy | 0.2073 | 240 |
| batch_2 | icd11_ext_histopath | 0.1942 | 240 |
| cancer_primary | icdo3_topography | 0.6718 | 525 |
| cancer_primary | icdo3_morphology | 0.3682 | 525 |
| cancer_primary | icdo3_behavior | 0.6583 | 524 |
| cancer_primary | icdo3_grade | 0.9816 | 467 |
| cancer_primary | icdo3_laterality | 0.9988 | 542 |
| cancer_primary | icd11_stem | 0.2912 | 528 |
| cancer_primary | icd11_ext_laterality | 0.4358 | 527 |
| cancer_primary | icd11_ext_grading | 0.7627 | 528 |
| cancer_primary | icd11_ext_anatomy | 0.3506 | 543 |
| cancer_primary | icd11_ext_histopath | 0.2019 | 543 |
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
| template | icdo3_topography | 0.4805 | 85 |
| template | icdo3_morphology | 0.1221 | 85 |
| template | icdo3_behavior | 0.2500 | 85 |
| template | icdo3_grade | 0.9890 | 83 |
| template | icdo3_laterality | 0.6667 | 85 |
| template | icd11_stem | 0.0970 | 85 |
| template | icd11_ext_laterality | 0.3312 | 85 |
| template | icd11_ext_grading | 0.5813 | 85 |
| template | icd11_ext_anatomy | 0.2595 | 85 |
| template | icd11_ext_histopath | 0.0174 | 85 |
| non_template | icdo3_topography | 0.6786 | 440 |
| non_template | icdo3_morphology | 0.3650 | 440 |
| non_template | icdo3_behavior | 0.6581 | 439 |
| non_template | icdo3_grade | 0.9811 | 384 |
| non_template | icdo3_laterality | 0.9985 | 464 |
| non_template | icd11_stem | 0.2916 | 450 |
| non_template | icd11_ext_laterality | 0.4065 | 449 |
| non_template | icd11_ext_grading | 0.7653 | 449 |
| non_template | icd11_ext_anatomy | 0.3533 | 465 |
| non_template | icd11_ext_histopath | 0.1894 | 465 |
