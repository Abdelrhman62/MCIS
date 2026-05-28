# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold1/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/tcga_brca_external_model_ready/tcga_brca_external_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 1025

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.1217 | 0.9482 | 0.1242 | 0.1193 | 0.1948 | 0.9482 | 1023 |
| icdo3_morphology | 0.1043 | 0.8508 | 0.1829 | 0.1117 | 0.1788 | 0.8508 | 992 |
| icdo3_behavior | 0.2413 | 0.9327 | 0.2500 | 0.2332 | 0.9652 | 0.9327 | 1025 |
| icdo3_grade | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| icdo3_laterality | 0.5873 | 0.8762 | 0.5941 | 0.5974 | 0.8810 | 0.8762 | 961 |
| icd11_stem | 0.0722 | 0.7580 | 0.0765 | 0.0685 | 0.3248 | 0.7580 | 1025 |
| icd11_ext_laterality | 0.2925 | 0.8783 | 0.2947 | 0.2993 | 0.8776 | 0.8783 | 961 |
| icd11_ext_grading | 0.3042 | 0.4848 | 0.5383 | 0.2476 | 0.5069 | 0.4848 | 788 |
| icd11_ext_anatomy | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9902 | 1025 |
| icd11_ext_histopath | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.9766 | 1025 |
| **Mean** | **0.1724** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.9482 | 0.9951 | 0.9980 | 0.9980 | 0.0000 |
| icdo3_morphology | 0.8508 | 0.9214 | 0.9466 | 0.9466 | 0.0000 |
| icdo3_behavior | 0.9327 | 0.9990 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| icdo3_laterality | 0.8762 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icd11_stem | 0.7580 | 0.9180 | 0.9395 | 0.9395 | 0.0000 |
| icd11_ext_laterality | 0.8783 | 0.9938 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.4848 | 0.9454 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.1391 | 1 | 7 |
| icdo3_morphology | 0.0103 | 0.2154 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.3217 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.0000 | 0 | 0 |
| icdo3_laterality | 0.0000 | 0.5873 | 0 | 3 |
| icd11_stem | 0.0000 | 0.2165 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3511 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.3802 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.4352 |
| icdo3_morphology | 0.0534 |
| icdo3_behavior | 0.0637 |
| icdo3_grade | 0.0000 |
| icdo3_laterality | 0.0813 |
| icd11_stem | 0.1059 |
| icd11_ext_laterality | 0.0866 |
| icd11_ext_grading | 0.3449 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.0000**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| cancer_primary | icdo3_topography | 0.1217 | 1023 |
| cancer_primary | icdo3_morphology | 0.1043 | 992 |
| cancer_primary | icdo3_behavior | 0.2413 | 1025 |
| cancer_primary | icdo3_grade | 0.0000 | 0 |
| cancer_primary | icdo3_laterality | 0.5873 | 961 |
| cancer_primary | icd11_stem | 0.0722 | 1025 |
| cancer_primary | icd11_ext_laterality | 0.2925 | 961 |
| cancer_primary | icd11_ext_grading | 0.3042 | 788 |
| cancer_primary | icd11_ext_anatomy | 0.0000 | 1025 |
| cancer_primary | icd11_ext_histopath | 0.0000 | 1025 |
| non_template | icdo3_topography | 0.1217 | 1023 |
| non_template | icdo3_morphology | 0.1043 | 992 |
| non_template | icdo3_behavior | 0.2413 | 1025 |
| non_template | icdo3_grade | 0.0000 | 0 |
| non_template | icdo3_laterality | 0.5873 | 961 |
| non_template | icd11_stem | 0.0722 | 1025 |
| non_template | icd11_ext_laterality | 0.2925 | 961 |
| non_template | icd11_ext_grading | 0.3042 | 788 |
| non_template | icd11_ext_anatomy | 0.0000 | 1025 |
| non_template | icd11_ext_histopath | 0.0000 | 1025 |
