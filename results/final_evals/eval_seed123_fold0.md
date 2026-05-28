# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed123_fold0/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 0)
**Records evaluated:** 550

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6706 | 0.7981 | 0.6864 | 0.6649 | 0.6706 | 0.7981 | 525 |
| icdo3_morphology | 0.3939 | 0.9390 | 0.4418 | 0.3834 | 0.4975 | 0.9390 | 525 |
| icdo3_behavior | 0.6272 | 0.9866 | 0.7396 | 0.5859 | 0.8363 | 0.9866 | 524 |
| icdo3_grade | 0.9884 | 0.9893 | 0.9863 | 0.9909 | 0.9884 | 0.9893 | 467 |
| icdo3_laterality | 0.9772 | 0.9982 | 0.9988 | 0.9583 | 0.9772 | 0.9982 | 549 |
| icd11_stem | 0.3351 | 0.9140 | 0.3185 | 0.3758 | 0.4524 | 0.9140 | 535 |
| icd11_ext_laterality | 0.3245 | 0.9663 | 0.3226 | 0.3265 | 0.3894 | 0.9663 | 534 |
| icd11_ext_grading | 0.7670 | 0.9663 | 0.7607 | 0.7736 | 0.7670 | 0.9663 | 534 |
| icd11_ext_anatomy | 0.2554 | 0.6747 | 0.3046 | 0.2333 | 0.5838 | 0.6345 | 550 |
| icd11_ext_histopath | 0.1914 | 0.7385 | 0.2354 | 0.1839 | 0.7337 | 0.9473 | 550 |
| **Mean** | **0.5531** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7981 | 0.9371 | 0.9714 | 0.9714 | 0.9426 |
| icdo3_morphology | 0.9390 | 0.9714 | 0.9771 | 0.9771 | 0.0000 |
| icdo3_behavior | 0.9866 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9893 | 1.0000 | 1.0000 | 1.0000 | 0.9987 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 0.9995 |
| icd11_stem | 0.9140 | 0.9626 | 0.9776 | 0.9776 | 0.0000 |
| icd11_ext_laterality | 0.9663 | 0.9925 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9663 | 0.9963 | 1.0000 | 1.0000 | 0.9767 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7664 | 1 | 7 |
| icdo3_morphology | 0.1282 | 0.7079 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8363 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9884 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9772 | 0 | 3 |
| icd11_stem | 0.1574 | 0.6905 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3894 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9587 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1621 |
| icdo3_morphology | 0.0519 |
| icdo3_behavior | 0.0105 |
| icdo3_grade | 0.0038 |
| icdo3_laterality | 0.0036 |
| icd11_stem | 0.0721 |
| icd11_ext_laterality | 0.0259 |
| icd11_ext_grading | 0.0298 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9848**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5774 | 285 |
| batch_1 | icdo3_morphology | 0.3597 | 285 |
| batch_1 | icdo3_behavior | 0.6346 | 285 |
| batch_1 | icdo3_grade | 0.9836 | 247 |
| batch_1 | icdo3_laterality | 0.9767 | 309 |
| batch_1 | icd11_stem | 0.3158 | 296 |
| batch_1 | icd11_ext_laterality | 0.3201 | 294 |
| batch_1 | icd11_ext_grading | 0.7547 | 294 |
| batch_1 | icd11_ext_anatomy | 0.1359 | 310 |
| batch_1 | icd11_ext_histopath | 0.1883 | 310 |
| batch_2 | icdo3_topography | 0.6586 | 240 |
| batch_2 | icdo3_morphology | 0.2626 | 240 |
| batch_2 | icdo3_behavior | 0.4994 | 239 |
| batch_2 | icdo3_grade | 0.9935 | 220 |
| batch_2 | icdo3_laterality | 0.6667 | 240 |
| batch_2 | icd11_stem | 0.2187 | 239 |
| batch_2 | icd11_ext_laterality | 0.3298 | 240 |
| batch_2 | icd11_ext_grading | 0.7870 | 240 |
| batch_2 | icd11_ext_anatomy | 0.1465 | 240 |
| batch_2 | icd11_ext_histopath | 0.1604 | 240 |
| cancer_primary | icdo3_topography | 0.6706 | 525 |
| cancer_primary | icdo3_morphology | 0.3939 | 525 |
| cancer_primary | icdo3_behavior | 0.6272 | 524 |
| cancer_primary | icdo3_grade | 0.9884 | 467 |
| cancer_primary | icdo3_laterality | 0.9738 | 542 |
| cancer_primary | icd11_stem | 0.3104 | 528 |
| cancer_primary | icd11_ext_laterality | 0.3247 | 527 |
| cancer_primary | icd11_ext_grading | 0.7658 | 528 |
| cancer_primary | icd11_ext_anatomy | 0.2579 | 543 |
| cancer_primary | icd11_ext_histopath | 0.1914 | 543 |
| non_cancer | icdo3_topography | 0.0000 | 0 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 1.0000 | 7 |
| non_cancer | icd11_stem | 0.0494 | 7 |
| non_cancer | icd11_ext_laterality | 0.3000 | 7 |
| non_cancer | icd11_ext_grading | 0.2000 | 6 |
| non_cancer | icd11_ext_anatomy | 0.0417 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.5245 | 85 |
| template | icdo3_morphology | 0.1221 | 85 |
| template | icdo3_behavior | 0.2500 | 85 |
| template | icdo3_grade | 1.0000 | 83 |
| template | icdo3_laterality | 0.6667 | 85 |
| template | icd11_stem | 0.0970 | 85 |
| template | icd11_ext_laterality | 0.3312 | 85 |
| template | icd11_ext_grading | 0.5813 | 85 |
| template | icd11_ext_anatomy | 0.2024 | 85 |
| template | icd11_ext_histopath | 0.0000 | 85 |
| non_template | icdo3_topography | 0.6638 | 440 |
| non_template | icdo3_morphology | 0.3895 | 440 |
| non_template | icdo3_behavior | 0.6268 | 439 |
| non_template | icdo3_grade | 0.9860 | 384 |
| non_template | icdo3_laterality | 0.9771 | 464 |
| non_template | icd11_stem | 0.3291 | 450 |
| non_template | icd11_ext_laterality | 0.3232 | 449 |
| non_template | icd11_ext_grading | 0.7677 | 449 |
| non_template | icd11_ext_anatomy | 0.2534 | 465 |
| non_template | icd11_ext_histopath | 0.1934 | 465 |
