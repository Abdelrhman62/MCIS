# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold4/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 4)
**Records evaluated:** 567

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6889 | 0.8044 | 0.6779 | 0.7115 | 0.7873 | 0.8044 | 542 |
| icdo3_morphology | 0.4641 | 0.9593 | 0.4648 | 0.4970 | 0.6551 | 0.9593 | 541 |
| icdo3_behavior | 0.5946 | 0.9741 | 0.5883 | 0.6017 | 0.7928 | 0.9741 | 541 |
| icdo3_grade | 0.9688 | 0.9793 | 0.9729 | 0.9649 | 0.9688 | 0.9793 | 482 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 567 |
| icd11_stem | 0.3504 | 0.9224 | 0.3571 | 0.3717 | 0.4506 | 0.9224 | 554 |
| icd11_ext_laterality | 0.4347 | 0.9622 | 0.4058 | 0.4915 | 0.4347 | 0.9622 | 555 |
| icd11_ext_grading | 0.7701 | 0.9765 | 0.7671 | 0.7732 | 0.9626 | 0.9765 | 554 |
| icd11_ext_anatomy | 0.3394 | 0.7059 | 0.3446 | 0.3521 | 0.5431 | 0.6720 | 567 |
| icd11_ext_histopath | 0.2031 | 0.7023 | 0.2213 | 0.2019 | 0.7785 | 0.9453 | 567 |
| **Mean** | **0.5814** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8044 | 0.9613 | 0.9852 | 0.9852 | 0.0000 |
| icdo3_morphology | 0.9593 | 0.9815 | 0.9889 | 0.9889 | 0.0000 |
| icdo3_behavior | 0.9741 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9793 | 1.0000 | 1.0000 | 1.0000 | 0.9927 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9224 | 0.9693 | 0.9783 | 0.9783 | 0.0000 |
| icd11_ext_laterality | 0.9622 | 0.9856 | 0.9982 | 0.9982 | 0.7984 |
| icd11_ext_grading | 0.9765 | 0.9964 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7873 | 1 | 7 |
| icdo3_morphology | 0.1795 | 0.8004 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7928 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9688 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.1481 | 0.7551 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.5217 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9626 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1475 |
| icdo3_morphology | 0.0275 |
| icdo3_behavior | 0.0234 |
| icdo3_grade | 0.0142 |
| icdo3_laterality | 0.0003 |
| icd11_stem | 0.0536 |
| icd11_ext_laterality | 0.0305 |
| icd11_ext_grading | 0.0197 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9408**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6036 | 296 |
| batch_1 | icdo3_morphology | 0.4520 | 295 |
| batch_1 | icdo3_behavior | 0.6296 | 295 |
| batch_1 | icdo3_grade | 0.9772 | 261 |
| batch_1 | icdo3_laterality | 1.0000 | 321 |
| batch_1 | icd11_stem | 0.3103 | 309 |
| batch_1 | icd11_ext_laterality | 0.4313 | 309 |
| batch_1 | icd11_ext_grading | 0.7700 | 308 |
| batch_1 | icd11_ext_anatomy | 0.1819 | 321 |
| batch_1 | icd11_ext_histopath | 0.2058 | 321 |
| batch_2 | icdo3_topography | 0.6665 | 246 |
| batch_2 | icdo3_morphology | 0.3031 | 246 |
| batch_2 | icdo3_behavior | 0.4605 | 246 |
| batch_2 | icdo3_grade | 0.9502 | 221 |
| batch_2 | icdo3_laterality | 0.6667 | 246 |
| batch_2 | icd11_stem | 0.2443 | 245 |
| batch_2 | icd11_ext_laterality | 0.3277 | 246 |
| batch_2 | icd11_ext_grading | 0.7681 | 246 |
| batch_2 | icd11_ext_anatomy | 0.2090 | 246 |
| batch_2 | icd11_ext_histopath | 0.1413 | 246 |
| cancer_primary | icdo3_topography | 0.6888 | 541 |
| cancer_primary | icdo3_morphology | 0.4641 | 541 |
| cancer_primary | icdo3_behavior | 0.5946 | 541 |
| cancer_primary | icdo3_grade | 0.9688 | 481 |
| cancer_primary | icdo3_laterality | 1.0000 | 558 |
| cancer_primary | icd11_stem | 0.3247 | 545 |
| cancer_primary | icd11_ext_laterality | 0.4346 | 546 |
| cancer_primary | icd11_ext_grading | 0.7680 | 546 |
| cancer_primary | icd11_ext_anatomy | 0.3425 | 558 |
| cancer_primary | icd11_ext_histopath | 0.2031 | 558 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 9 |
| non_cancer | icd11_stem | 0.0370 | 9 |
| non_cancer | icd11_ext_laterality | 0.3333 | 9 |
| non_cancer | icd11_ext_grading | 0.2000 | 8 |
| non_cancer | icd11_ext_anatomy | 0.0469 | 9 |
| non_cancer | icd11_ext_histopath | 0.0000 | 9 |
| template | icdo3_topography | 0.6237 | 100 |
| template | icdo3_morphology | 0.1528 | 100 |
| template | icdo3_behavior | 0.4974 | 100 |
| template | icdo3_grade | 1.0000 | 98 |
| template | icdo3_laterality | 0.6667 | 102 |
| template | icd11_stem | 0.1726 | 100 |
| template | icd11_ext_laterality | 0.3249 | 100 |
| template | icd11_ext_grading | 0.7807 | 100 |
| template | icd11_ext_anatomy | 0.2621 | 102 |
| template | icd11_ext_histopath | 0.0290 | 102 |
| non_template | icdo3_topography | 0.6802 | 442 |
| non_template | icdo3_morphology | 0.4801 | 441 |
| non_template | icdo3_behavior | 0.6090 | 441 |
| non_template | icdo3_grade | 0.9653 | 384 |
| non_template | icdo3_laterality | 1.0000 | 465 |
| non_template | icd11_stem | 0.3408 | 454 |
| non_template | icd11_ext_laterality | 0.4345 | 455 |
| non_template | icd11_ext_grading | 0.7697 | 454 |
| non_template | icd11_ext_anatomy | 0.3405 | 465 |
| non_template | icd11_ext_histopath | 0.2053 | 465 |
