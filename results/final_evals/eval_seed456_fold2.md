# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold2/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 2)
**Records evaluated:** 548

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.7170 | 0.8288 | 0.7250 | 0.7146 | 0.7170 | 0.8288 | 520 |
| icdo3_morphology | 0.4622 | 0.9420 | 0.4759 | 0.4735 | 0.5839 | 0.9420 | 517 |
| icdo3_behavior | 0.7347 | 0.9923 | 0.7286 | 0.7411 | 0.9796 | 0.9923 | 517 |
| icdo3_grade | 0.9539 | 0.9650 | 0.9463 | 0.9620 | 0.9539 | 0.9650 | 457 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 548 |
| icd11_stem | 0.3150 | 0.9112 | 0.3167 | 0.3284 | 0.4253 | 0.9112 | 529 |
| icd11_ext_laterality | 0.3281 | 0.9811 | 0.3283 | 0.3279 | 0.4922 | 0.9811 | 528 |
| icd11_ext_grading | 0.7589 | 0.9640 | 0.7517 | 0.7670 | 0.7589 | 0.9640 | 528 |
| icd11_ext_anatomy | 0.3387 | 0.7062 | 0.3600 | 0.3565 | 0.6021 | 0.6496 | 548 |
| icd11_ext_histopath | 0.1900 | 0.6949 | 0.2181 | 0.1854 | 0.7284 | 0.9453 | 548 |
| **Mean** | **0.5799** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8288 | 0.9442 | 0.9788 | 0.9788 | 0.8967 |
| icdo3_morphology | 0.9420 | 0.9632 | 0.9710 | 0.9710 | 0.0000 |
| icdo3_behavior | 0.9923 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9650 | 1.0000 | 1.0000 | 1.0000 | 0.9746 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9112 | 0.9471 | 0.9641 | 0.9641 | 0.0000 |
| icd11_ext_laterality | 0.9811 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9640 | 0.9962 | 1.0000 | 1.0000 | 0.9233 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.8194 | 1 | 7 |
| icdo3_morphology | 0.1923 | 0.7812 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.9796 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9539 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.1296 | 0.6858 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3937 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9486 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1706 |
| icdo3_morphology | 0.0499 |
| icdo3_behavior | 0.0088 |
| icdo3_grade | 0.0299 |
| icdo3_laterality | 0.0010 |
| icd11_stem | 0.0664 |
| icd11_ext_laterality | 0.0159 |
| icd11_ext_grading | 0.0260 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9164**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6619 | 285 |
| batch_1 | icdo3_morphology | 0.4842 | 282 |
| batch_1 | icdo3_behavior | 0.7358 | 282 |
| batch_1 | icdo3_grade | 0.9702 | 240 |
| batch_1 | icdo3_laterality | 1.0000 | 313 |
| batch_1 | icd11_stem | 0.3213 | 294 |
| batch_1 | icd11_ext_laterality | 0.3264 | 293 |
| batch_1 | icd11_ext_grading | 0.7602 | 293 |
| batch_1 | icd11_ext_anatomy | 0.1583 | 313 |
| batch_1 | icd11_ext_histopath | 0.2016 | 313 |
| batch_2 | icdo3_topography | 0.6901 | 235 |
| batch_2 | icdo3_morphology | 0.2253 | 235 |
| batch_2 | icdo3_behavior | 0.4767 | 235 |
| batch_2 | icdo3_grade | 0.8964 | 217 |
| batch_2 | icdo3_laterality | 0.6667 | 235 |
| batch_2 | icd11_stem | 0.1693 | 235 |
| batch_2 | icd11_ext_laterality | 0.3303 | 235 |
| batch_2 | icd11_ext_grading | 0.7405 | 235 |
| batch_2 | icd11_ext_anatomy | 0.2332 | 235 |
| batch_2 | icd11_ext_histopath | 0.0761 | 235 |
| cancer_primary | icdo3_topography | 0.7167 | 518 |
| cancer_primary | icdo3_morphology | 0.4622 | 517 |
| cancer_primary | icdo3_behavior | 0.7347 | 517 |
| cancer_primary | icdo3_grade | 0.9539 | 457 |
| cancer_primary | icdo3_laterality | 1.0000 | 541 |
| cancer_primary | icd11_stem | 0.2910 | 522 |
| cancer_primary | icd11_ext_laterality | 0.3281 | 522 |
| cancer_primary | icd11_ext_grading | 0.7562 | 522 |
| cancer_primary | icd11_ext_anatomy | 0.3394 | 541 |
| cancer_primary | icd11_ext_histopath | 0.1900 | 541 |
| non_cancer | icdo3_topography | 0.2500 | 2 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0395 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.2000 | 6 |
| non_cancer | icd11_ext_anatomy | 0.0312 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.6621 | 79 |
| template | icdo3_morphology | 0.1525 | 79 |
| template | icdo3_behavior | 0.2500 | 79 |
| template | icdo3_grade | 0.9149 | 80 |
| template | icdo3_laterality | 0.6667 | 80 |
| template | icd11_stem | 0.1352 | 79 |
| template | icd11_ext_laterality | 0.3333 | 79 |
| template | icd11_ext_grading | 0.5910 | 79 |
| template | icd11_ext_anatomy | 0.2006 | 80 |
| template | icd11_ext_histopath | 0.0725 | 80 |
| non_template | icdo3_topography | 0.7095 | 441 |
| non_template | icdo3_morphology | 0.4581 | 438 |
| non_template | icdo3_behavior | 0.7345 | 438 |
| non_template | icdo3_grade | 0.9543 | 377 |
| non_template | icdo3_laterality | 1.0000 | 468 |
| non_template | icd11_stem | 0.3060 | 450 |
| non_template | icd11_ext_laterality | 0.3272 | 449 |
| non_template | icd11_ext_grading | 0.7577 | 449 |
| non_template | icd11_ext_anatomy | 0.3368 | 468 |
| non_template | icd11_ext_histopath | 0.1892 | 468 |
