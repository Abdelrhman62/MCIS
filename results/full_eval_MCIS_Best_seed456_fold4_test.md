# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold4/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6413 | 0.7766 | 0.6715 | 0.6266 | 0.7329 | 0.7766 | 282 |
| icdo3_morphology | 0.2881 | 0.9075 | 0.2860 | 0.3102 | 0.4939 | 0.9075 | 281 |
| icdo3_behavior | 0.6163 | 0.9644 | 0.6084 | 0.6251 | 0.8218 | 0.9644 | 281 |
| icdo3_grade | 0.9491 | 0.9567 | 0.9602 | 0.9404 | 0.9491 | 0.9567 | 254 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 296 |
| icd11_stem | 0.2263 | 0.8646 | 0.2366 | 0.2241 | 0.3818 | 0.8646 | 288 |
| icd11_ext_laterality | 0.3285 | 0.9826 | 0.3285 | 0.3285 | 0.6569 | 0.9826 | 287 |
| icd11_ext_grading | 0.7543 | 0.9582 | 0.7442 | 0.7674 | 0.9429 | 0.9582 | 287 |
| icd11_ext_anatomy | 0.2966 | 0.6348 | 0.3576 | 0.2713 | 0.5933 | 0.5912 | 296 |
| icd11_ext_histopath | 0.1488 | 0.6818 | 0.1445 | 0.1551 | 0.8558 | 0.9155 | 296 |
| **Official Mean (Agg)** | **0.6294** | | | | | | |
| **Flat Correct Mean** | **0.6120** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7766 | 0.9433 | 0.9681 | 0.9681 | 0.9350 |
| icdo3_morphology | 0.9075 | 0.9573 | 0.9715 | 0.9715 | 0.9064 |
| icdo3_behavior | 0.9644 | 1.0000 | 1.0000 | 1.0000 | 0.9875 |
| icdo3_grade | 0.9567 | 1.0000 | 1.0000 | 1.0000 | 0.9877 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8646 | 0.9444 | 0.9583 | 0.9583 | 0.9463 |
| icd11_ext_laterality | 0.9826 | 0.9965 | 1.0000 | 1.0000 | 0.9739 |
| icd11_ext_grading | 0.9582 | 0.9965 | 1.0000 | 1.0000 | 0.9829 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7329 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.6286 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8218 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9491 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0556 | 0.5677 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3941 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9429 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1356 |
| icdo3_morphology | 0.0722 |
| icdo3_behavior | 0.0304 |
| icdo3_grade | 0.0340 |
| icdo3_laterality | 0.0006 |
| icd11_stem | 0.1035 |
| icd11_ext_laterality | 0.0201 |
| icd11_ext_grading | 0.0389 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9298**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | F1-Present | N |
|---|---|---|---|---|
| batch_1 | icdo3_topography | 0.5884 | 0.6724 | 169 |
| batch_1 | icdo3_morphology | 0.2661 | 0.6387 | 168 |
| batch_1 | icdo3_behavior | 0.6300 | 0.8400 | 168 |
| batch_1 | icdo3_grade | 0.9517 | 0.9517 | 150 |
| batch_1 | icdo3_laterality | 1.0000 | 1.0000 | 183 |
| batch_1 | icd11_stem | 0.1916 | 0.3695 | 175 |
| batch_1 | icd11_ext_laterality | 0.3253 | 0.6507 | 175 |
| batch_1 | icd11_ext_grading | 0.7407 | 0.9259 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1843 | 0.5898 | 183 |
| batch_1 | icd11_ext_histopath | 0.1536 | 0.8832 | 183 |
| batch_2 | icdo3_topography | 0.5707 | 0.6522 | 113 |
| batch_2 | icdo3_morphology | 0.1880 | 0.5013 | 113 |
| batch_2 | icdo3_behavior | 0.4988 | 0.9977 | 113 |
| batch_2 | icdo3_grade | 0.9455 | 0.9455 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 1.0000 | 113 |
| batch_2 | icd11_stem | 0.1820 | 0.5459 | 113 |
| batch_2 | icd11_ext_laterality | 0.3333 | 1.0000 | 112 |
| batch_2 | icd11_ext_grading | 0.7888 | 0.9860 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1483 | 0.7907 | 113 |
| batch_2 | icd11_ext_histopath | 0.0942 | 0.7222 | 113 |
| cancer_primary | icdo3_topography | 0.6408 | 0.7323 | 281 |
| cancer_primary | icdo3_morphology | 0.2881 | 0.4939 | 281 |
| cancer_primary | icdo3_behavior | 0.6163 | 0.8218 | 281 |
| cancer_primary | icdo3_grade | 0.9491 | 0.9491 | 254 |
| cancer_primary | icdo3_laterality | 1.0000 | 1.0000 | 292 |
| cancer_primary | icd11_stem | 0.2016 | 0.4186 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3284 | 0.6568 | 283 |
| cancer_primary | icd11_ext_grading | 0.7511 | 0.9389 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.2969 | 0.5939 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1488 | 0.8558 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1.0000 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 1.0000 | 4 |
| non_cancer | icd11_stem | 0.0247 | 0.2222 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 1.0000 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 1.0000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0000 | 0.0000 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 0.0000 | 4 |
| template | icdo3_topography | 0.6509 | 0.8679 | 33 |
| template | icdo3_morphology | 0.1533 | 0.7359 | 33 |
| template | icdo3_behavior | 0.2500 | 1.0000 | 33 |
| template | icdo3_grade | 0.9634 | 0.9634 | 34 |
| template | icdo3_laterality | 0.6667 | 1.0000 | 34 |
| template | icd11_stem | 0.1159 | 0.6259 | 33 |
| template | icd11_ext_laterality | 0.3299 | 0.9898 | 33 |
| template | icd11_ext_grading | 0.5779 | 0.9632 | 33 |
| template | icd11_ext_anatomy | 0.2205 | 0.7055 | 34 |
| template | icd11_ext_histopath | 0.1304 | 1.0000 | 34 |
| non_template | icdo3_topography | 0.6277 | 0.7174 | 249 |
| non_template | icdo3_morphology | 0.2865 | 0.4912 | 248 |
| non_template | icdo3_behavior | 0.6156 | 0.8208 | 248 |
| non_template | icdo3_grade | 0.9498 | 0.9498 | 220 |
| non_template | icdo3_laterality | 1.0000 | 1.0000 | 262 |
| non_template | icd11_stem | 0.2239 | 0.3777 | 255 |
| non_template | icd11_ext_laterality | 0.3285 | 0.6570 | 254 |
| non_template | icd11_ext_grading | 0.7558 | 0.9448 | 254 |
| non_template | icd11_ext_anatomy | 0.2940 | 0.5881 | 262 |
| non_template | icd11_ext_histopath | 0.1385 | 0.7963 | 262 |
