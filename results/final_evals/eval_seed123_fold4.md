# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed123_fold4/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 4)
**Records evaluated:** 567

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.7171 | 0.8303 | 0.7117 | 0.7340 | 0.8195 | 0.8303 | 542 |
| icdo3_morphology | 0.4523 | 0.9501 | 0.4511 | 0.4660 | 0.6385 | 0.9501 | 541 |
| icdo3_behavior | 0.6015 | 0.9760 | 0.5898 | 0.6144 | 0.8020 | 0.9760 | 541 |
| icdo3_grade | 0.9657 | 0.9772 | 0.9719 | 0.9599 | 0.9657 | 0.9772 | 482 |
| icdo3_laterality | 0.9624 | 0.9982 | 0.9988 | 0.9333 | 0.9624 | 0.9982 | 567 |
| icd11_stem | 0.3169 | 0.9206 | 0.3128 | 0.3289 | 0.4074 | 0.9206 | 554 |
| icd11_ext_laterality | 0.4366 | 0.9694 | 0.4070 | 0.4939 | 0.4366 | 0.9694 | 555 |
| icd11_ext_grading | 0.7650 | 0.9711 | 0.7611 | 0.7693 | 0.9563 | 0.9711 | 554 |
| icd11_ext_anatomy | 0.2162 | 0.6691 | 0.2021 | 0.2376 | 0.5766 | 0.6190 | 567 |
| icd11_ext_histopath | 0.2092 | 0.7519 | 0.2455 | 0.1886 | 0.8021 | 0.9524 | 567 |
| **Mean** | **0.5643** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8303 | 0.9686 | 0.9852 | 0.9852 | 0.0000 |
| icdo3_morphology | 0.9501 | 0.9871 | 0.9926 | 0.9926 | 0.0000 |
| icdo3_behavior | 0.9760 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9772 | 1.0000 | 1.0000 | 1.0000 | 0.9946 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9206 | 0.9639 | 0.9783 | 0.9783 | 0.0000 |
| icd11_ext_laterality | 0.9694 | 0.9874 | 0.9982 | 0.9982 | 0.7983 |
| icd11_ext_grading | 0.9711 | 0.9946 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.8195 | 1 | 7 |
| icdo3_morphology | 0.2051 | 0.7444 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8020 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9657 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9624 | 0 | 3 |
| icd11_stem | 0.0833 | 0.7839 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.5239 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9563 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1823 |
| icdo3_morphology | 0.0316 |
| icdo3_behavior | 0.0160 |
| icdo3_grade | 0.0138 |
| icdo3_laterality | 0.0015 |
| icd11_stem | 0.0539 |
| icd11_ext_laterality | 0.0297 |
| icd11_ext_grading | 0.0214 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9387**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6295 | 296 |
| batch_1 | icdo3_morphology | 0.4141 | 295 |
| batch_1 | icdo3_behavior | 0.6590 | 295 |
| batch_1 | icdo3_grade | 0.9713 | 261 |
| batch_1 | icdo3_laterality | 0.9620 | 321 |
| batch_1 | icd11_stem | 0.2696 | 309 |
| batch_1 | icd11_ext_laterality | 0.4346 | 309 |
| batch_1 | icd11_ext_grading | 0.7554 | 308 |
| batch_1 | icd11_ext_anatomy | 0.1167 | 321 |
| batch_1 | icd11_ext_histopath | 0.2293 | 321 |
| batch_2 | icdo3_topography | 0.6903 | 246 |
| batch_2 | icdo3_morphology | 0.3308 | 246 |
| batch_2 | icdo3_behavior | 0.4714 | 246 |
| batch_2 | icdo3_grade | 0.9426 | 221 |
| batch_2 | icdo3_laterality | 0.6667 | 246 |
| batch_2 | icd11_stem | 0.2519 | 245 |
| batch_2 | icd11_ext_laterality | 0.3277 | 246 |
| batch_2 | icd11_ext_grading | 0.7821 | 246 |
| batch_2 | icd11_ext_anatomy | 0.1431 | 246 |
| batch_2 | icd11_ext_histopath | 0.1413 | 246 |
| cancer_primary | icdo3_topography | 0.7170 | 541 |
| cancer_primary | icdo3_morphology | 0.4523 | 541 |
| cancer_primary | icdo3_behavior | 0.6015 | 541 |
| cancer_primary | icdo3_grade | 0.9657 | 481 |
| cancer_primary | icdo3_laterality | 0.9624 | 558 |
| cancer_primary | icd11_stem | 0.3012 | 545 |
| cancer_primary | icd11_ext_laterality | 0.4364 | 546 |
| cancer_primary | icd11_ext_grading | 0.7632 | 546 |
| cancer_primary | icd11_ext_anatomy | 0.2159 | 558 |
| cancer_primary | icd11_ext_histopath | 0.2092 | 558 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 9 |
| non_cancer | icd11_stem | 0.0212 | 9 |
| non_cancer | icd11_ext_laterality | 0.3333 | 9 |
| non_cancer | icd11_ext_grading | 0.2000 | 8 |
| non_cancer | icd11_ext_anatomy | 0.0469 | 9 |
| non_cancer | icd11_ext_histopath | 0.0000 | 9 |
| template | icdo3_topography | 0.6698 | 100 |
| template | icdo3_morphology | 0.1528 | 100 |
| template | icdo3_behavior | 0.4961 | 100 |
| template | icdo3_grade | 0.9505 | 98 |
| template | icdo3_laterality | 0.6667 | 102 |
| template | icd11_stem | 0.1477 | 100 |
| template | icd11_ext_laterality | 0.3249 | 100 |
| template | icd11_ext_grading | 0.7807 | 100 |
| template | icd11_ext_anatomy | 0.2444 | 102 |
| template | icd11_ext_histopath | 0.0290 | 102 |
| non_template | icdo3_topography | 0.7066 | 442 |
| non_template | icdo3_morphology | 0.4655 | 441 |
| non_template | icdo3_behavior | 0.6262 | 441 |
| non_template | icdo3_grade | 0.9655 | 384 |
| non_template | icdo3_laterality | 0.9623 | 465 |
| non_template | icd11_stem | 0.3142 | 454 |
| non_template | icd11_ext_laterality | 0.4367 | 455 |
| non_template | icd11_ext_grading | 0.7640 | 454 |
| non_template | icd11_ext_anatomy | 0.2093 | 465 |
| non_template | icd11_ext_histopath | 0.2119 | 465 |
