# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed123_fold3/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** val_fold (fold 3)
**Records evaluated:** 554

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6750 | 0.7947 | 0.6678 | 0.6859 | 0.7715 | 0.7947 | 531 |
| icdo3_morphology | 0.4307 | 0.9491 | 0.4260 | 0.4444 | 0.5440 | 0.9491 | 530 |
| icdo3_behavior | 0.6058 | 0.9830 | 0.6027 | 0.6091 | 0.6058 | 0.9830 | 530 |
| icdo3_grade | 0.9829 | 0.9854 | 0.9843 | 0.9814 | 0.9829 | 0.9854 | 479 |
| icdo3_laterality | 0.9772 | 0.9982 | 0.9987 | 0.9583 | 0.9772 | 0.9982 | 554 |
| icd11_stem | 0.3183 | 0.9205 | 0.3136 | 0.3260 | 0.4092 | 0.9205 | 541 |
| icd11_ext_laterality | 0.4126 | 0.9797 | 0.4945 | 0.3864 | 0.4952 | 0.9797 | 541 |
| icd11_ext_grading | 0.7642 | 0.9684 | 0.7688 | 0.7603 | 0.9552 | 0.9684 | 538 |
| icd11_ext_anatomy | 0.2403 | 0.6830 | 0.2259 | 0.2683 | 0.5493 | 0.5903 | 554 |
| icd11_ext_histopath | 0.1846 | 0.7218 | 0.1871 | 0.1874 | 0.7075 | 0.9477 | 554 |
| **Mean** | **0.5591** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7947 | 0.9548 | 0.9812 | 0.9812 | 0.0000 |
| icdo3_morphology | 0.9491 | 0.9736 | 0.9755 | 0.9755 | 0.0000 |
| icdo3_behavior | 0.9830 | 0.9981 | 1.0000 | 1.0000 | 0.9336 |
| icdo3_grade | 0.9854 | 1.0000 | 1.0000 | 1.0000 | 0.9920 |
| icdo3_laterality | 0.9982 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.9205 | 0.9649 | 0.9741 | 0.9741 | 0.0000 |
| icd11_ext_laterality | 0.9797 | 0.9963 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9684 | 0.9944 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7715 | 1 | 7 |
| icdo3_morphology | 0.1538 | 0.7578 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8078 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9829 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9772 | 0 | 3 |
| icd11_stem | 0.1111 | 0.7325 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.4952 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9552 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1396 |
| icdo3_morphology | 0.0426 |
| icdo3_behavior | 0.0160 |
| icdo3_grade | 0.0156 |
| icdo3_laterality | 0.0021 |
| icd11_stem | 0.0596 |
| icd11_ext_laterality | 0.0205 |
| icd11_ext_grading | 0.0306 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9716**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5901 | 293 |
| batch_1 | icdo3_morphology | 0.3845 | 292 |
| batch_1 | icdo3_behavior | 0.6208 | 292 |
| batch_1 | icdo3_grade | 0.9782 | 261 |
| batch_1 | icdo3_laterality | 0.9767 | 315 |
| batch_1 | icd11_stem | 0.3219 | 302 |
| batch_1 | icd11_ext_laterality | 0.4400 | 302 |
| batch_1 | icd11_ext_grading | 0.7586 | 300 |
| batch_1 | icd11_ext_anatomy | 0.1207 | 315 |
| batch_1 | icd11_ext_histopath | 0.1777 | 315 |
| batch_2 | icdo3_topography | 0.6417 | 238 |
| batch_2 | icdo3_morphology | 0.2825 | 238 |
| batch_2 | icdo3_behavior | 0.4870 | 238 |
| batch_2 | icdo3_grade | 0.9868 | 218 |
| batch_2 | icdo3_laterality | 0.6667 | 239 |
| batch_2 | icd11_stem | 0.2377 | 239 |
| batch_2 | icd11_ext_laterality | 0.3299 | 239 |
| batch_2 | icd11_ext_grading | 0.7767 | 238 |
| batch_2 | icd11_ext_anatomy | 0.1629 | 239 |
| batch_2 | icd11_ext_histopath | 0.1677 | 239 |
| cancer_primary | icdo3_topography | 0.6750 | 531 |
| cancer_primary | icdo3_morphology | 0.4307 | 530 |
| cancer_primary | icdo3_behavior | 0.6058 | 530 |
| cancer_primary | icdo3_grade | 0.9829 | 478 |
| cancer_primary | icdo3_laterality | 0.9771 | 548 |
| cancer_primary | icd11_stem | 0.2937 | 535 |
| cancer_primary | icd11_ext_laterality | 0.4126 | 535 |
| cancer_primary | icd11_ext_grading | 0.7674 | 534 |
| cancer_primary | icd11_ext_anatomy | 0.2383 | 548 |
| cancer_primary | icd11_ext_histopath | 0.1846 | 548 |
| non_cancer | icdo3_topography | 0.0000 | 0 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.3333 | 1 |
| non_cancer | icdo3_laterality | 0.6667 | 6 |
| non_cancer | icd11_stem | 0.0296 | 6 |
| non_cancer | icd11_ext_laterality | 0.3333 | 6 |
| non_cancer | icd11_ext_grading | 0.1333 | 4 |
| non_cancer | icd11_ext_anatomy | 0.1042 | 6 |
| non_cancer | icd11_ext_histopath | 0.0000 | 6 |
| template | icdo3_topography | 0.5991 | 95 |
| template | icdo3_morphology | 0.1250 | 95 |
| template | icdo3_behavior | 0.2487 | 95 |
| template | icdo3_grade | 0.9570 | 94 |
| template | icdo3_laterality | 0.6667 | 96 |
| template | icd11_stem | 0.1012 | 95 |
| template | icd11_ext_laterality | 0.3264 | 95 |
| template | icd11_ext_grading | 0.5947 | 95 |
| template | icd11_ext_anatomy | 0.1941 | 96 |
| template | icd11_ext_histopath | 0.0696 | 96 |
| non_template | icdo3_topography | 0.6638 | 436 |
| non_template | icdo3_morphology | 0.4296 | 435 |
| non_template | icdo3_behavior | 0.6235 | 435 |
| non_template | icdo3_grade | 0.9871 | 385 |
| non_template | icdo3_laterality | 0.9770 | 458 |
| non_template | icd11_stem | 0.3217 | 446 |
| non_template | icd11_ext_laterality | 0.4133 | 446 |
| non_template | icd11_ext_grading | 0.7623 | 443 |
| non_template | icd11_ext_anatomy | 0.2399 | 458 |
| non_template | icd11_ext_histopath | 0.1844 | 458 |
