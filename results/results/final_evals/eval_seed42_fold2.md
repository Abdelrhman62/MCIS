# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed42_fold2/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 2)
**Records evaluated:** 548

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6829 | 0.8019 | 0.7033 | 0.6725 | 0.6829 | 0.8019 | 520 |
| icdo3_morphology | 0.3975 | 0.9304 | 0.3899 | 0.4361 | 0.5021 | 0.9304 | 517 |
| icdo3_behavior | 0.7264 | 0.9884 | 0.7264 | 0.7264 | 0.9685 | 0.9884 | 517 |
| icdo3_grade | 0.9517 | 0.9628 | 0.9394 | 0.9654 | 0.9517 | 0.9628 | 457 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 548 |
| icd11_stem | 0.3191 | 0.9074 | 0.3265 | 0.3224 | 0.4308 | 0.9074 | 529 |
| icd11_ext_laterality | 0.4390 | 0.9811 | 0.4115 | 0.4943 | 0.6584 | 0.9811 | 528 |
| icd11_ext_grading | 0.7540 | 0.9621 | 0.7481 | 0.7606 | 0.7540 | 0.9621 | 528 |
| icd11_ext_anatomy | 0.3401 | 0.7173 | 0.3338 | 0.3636 | 0.6046 | 0.6533 | 548 |
| icd11_ext_histopath | 0.1523 | 0.6724 | 0.1704 | 0.1564 | 0.7008 | 0.9416 | 548 |
| **Mean** | **0.5763** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8019 | 0.9462 | 0.9750 | 0.9750 | 0.9067 |
| icdo3_morphology | 0.9304 | 0.9652 | 0.9691 | 0.9691 | 0.0000 |
| icdo3_behavior | 0.9884 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9628 | 1.0000 | 1.0000 | 1.0000 | 0.9718 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9074 | 0.9603 | 0.9716 | 0.9716 | 0.0000 |
| icd11_ext_laterality | 0.9811 | 0.9981 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9621 | 0.9943 | 1.0000 | 1.0000 | 0.8611 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7805 | 1 | 7 |
| icdo3_morphology | 0.2051 | 0.6249 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.9685 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9517 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.1481 | 0.6610 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.5268 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9426 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1378 |
| icdo3_morphology | 0.0578 |
| icdo3_behavior | 0.0074 |
| icdo3_grade | 0.0349 |
| icdo3_laterality | 0.0004 |
| icd11_stem | 0.0776 |
| icd11_ext_laterality | 0.0109 |
| icd11_ext_grading | 0.0304 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9151**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6278 | 285 |
| batch_1 | icdo3_morphology | 0.3937 | 282 |
| batch_1 | icdo3_behavior | 0.7256 | 282 |
| batch_1 | icdo3_grade | 0.9703 | 240 |
| batch_1 | icdo3_laterality | 1.0000 | 313 |
| batch_1 | icd11_stem | 0.3239 | 294 |
| batch_1 | icd11_ext_laterality | 0.4936 | 293 |
| batch_1 | icd11_ext_grading | 0.7516 | 293 |
| batch_1 | icd11_ext_anatomy | 0.1943 | 313 |
| batch_1 | icd11_ext_histopath | 0.1637 | 313 |
| batch_2 | icdo3_topography | 0.6538 | 235 |
| batch_2 | icdo3_morphology | 0.2141 | 235 |
| batch_2 | icdo3_behavior | 0.4767 | 235 |
| batch_2 | icdo3_grade | 0.8912 | 217 |
| batch_2 | icdo3_laterality | 0.6667 | 235 |
| batch_2 | icd11_stem | 0.1706 | 235 |
| batch_2 | icd11_ext_laterality | 0.3292 | 235 |
| batch_2 | icd11_ext_grading | 0.7540 | 235 |
| batch_2 | icd11_ext_anatomy | 0.2200 | 235 |
| batch_2 | icd11_ext_histopath | 0.0761 | 235 |
| cancer_primary | icdo3_topography | 0.6826 | 518 |
| cancer_primary | icdo3_morphology | 0.3975 | 517 |
| cancer_primary | icdo3_behavior | 0.7264 | 517 |
| cancer_primary | icdo3_grade | 0.9517 | 457 |
| cancer_primary | icdo3_laterality | 1.0000 | 541 |
| cancer_primary | icd11_stem | 0.2623 | 522 |
| cancer_primary | icd11_ext_laterality | 0.4389 | 522 |
| cancer_primary | icd11_ext_grading | 0.7510 | 522 |
| cancer_primary | icd11_ext_anatomy | 0.3411 | 541 |
| cancer_primary | icd11_ext_histopath | 0.1523 | 541 |
| non_cancer | icdo3_topography | 0.2500 | 2 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 7 |
| non_cancer | icd11_stem | 0.0648 | 7 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.2000 | 6 |
| non_cancer | icd11_ext_anatomy | 0.0312 | 7 |
| non_cancer | icd11_ext_histopath | 0.0000 | 7 |
| template | icdo3_topography | 0.6402 | 79 |
| template | icdo3_morphology | 0.1414 | 79 |
| template | icdo3_behavior | 0.2500 | 79 |
| template | icdo3_grade | 0.9149 | 80 |
| template | icdo3_laterality | 0.6667 | 80 |
| template | icd11_stem | 0.1297 | 79 |
| template | icd11_ext_laterality | 0.3333 | 79 |
| template | icd11_ext_grading | 0.5910 | 79 |
| template | icd11_ext_anatomy | 0.1949 | 80 |
| template | icd11_ext_histopath | 0.0725 | 80 |
| non_template | icdo3_topography | 0.6748 | 441 |
| non_template | icdo3_morphology | 0.3935 | 438 |
| non_template | icdo3_behavior | 0.7261 | 438 |
| non_template | icdo3_grade | 0.9516 | 377 |
| non_template | icdo3_laterality | 1.0000 | 468 |
| non_template | icd11_stem | 0.3117 | 450 |
| non_template | icd11_ext_laterality | 0.4380 | 449 |
| non_template | icd11_ext_grading | 0.7528 | 449 |
| non_template | icd11_ext_anatomy | 0.3381 | 468 |
| non_template | icd11_ext_histopath | 0.1517 | 468 |
