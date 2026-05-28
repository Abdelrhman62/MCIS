# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold2/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6910 | 0.8156 | 0.7025 | 0.6843 | 0.7897 | 0.8156 | 282 |
| icdo3_morphology | 0.2989 | 0.9217 | 0.3072 | 0.3096 | 0.5123 | 0.9217 | 281 |
| icdo3_behavior | 0.5973 | 0.9644 | 0.5756 | 0.6251 | 0.7964 | 0.9644 | 281 |
| icdo3_grade | 0.9647 | 0.9724 | 0.9681 | 0.9639 | 0.9647 | 0.9724 | 254 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 296 |
| icd11_stem | 0.1826 | 0.8681 | 0.1743 | 0.1934 | 0.3081 | 0.8681 | 288 |
| icd11_ext_laterality | 0.3265 | 0.9791 | 0.3256 | 0.3275 | 0.6531 | 0.9791 | 287 |
| icd11_ext_grading | 0.7392 | 0.9443 | 0.7289 | 0.7534 | 0.9240 | 0.9443 | 287 |
| icd11_ext_anatomy | 0.2793 | 0.6238 | 0.3455 | 0.2576 | 0.5586 | 0.5811 | 296 |
| icd11_ext_histopath | 0.1571 | 0.7174 | 0.1469 | 0.1700 | 0.9034 | 0.9223 | 296 |
| **Official Mean (Agg)** | **0.6353** | | | | | | |
| **Flat Correct Mean** | **0.6141** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8156 | 0.9504 | 0.9894 | 0.9894 | 0.9474 |
| icdo3_morphology | 0.9217 | 0.9680 | 0.9751 | 0.9751 | 0.9214 |
| icdo3_behavior | 0.9644 | 1.0000 | 1.0000 | 1.0000 | 0.9809 |
| icdo3_grade | 0.9724 | 1.0000 | 1.0000 | 1.0000 | 0.9848 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8681 | 0.9340 | 0.9514 | 0.9514 | 0.9470 |
| icd11_ext_laterality | 0.9791 | 0.9895 | 1.0000 | 1.0000 | 0.9182 |
| icd11_ext_grading | 0.9443 | 0.9895 | 1.0000 | 1.0000 | 0.9812 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7897 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.6520 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.7964 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9647 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0000 | 0.5477 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3918 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9240 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1690 |
| icdo3_morphology | 0.0714 |
| icdo3_behavior | 0.0258 |
| icdo3_grade | 0.0238 |
| icdo3_laterality | 0.0004 |
| icd11_stem | 0.1048 |
| icd11_ext_laterality | 0.0151 |
| icd11_ext_grading | 0.0515 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9550**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | F1-Present | N |
|---|---|---|---|---|
| batch_1 | icdo3_topography | 0.6283 | 0.7181 | 169 |
| batch_1 | icdo3_morphology | 0.3010 | 0.7225 | 168 |
| batch_1 | icdo3_behavior | 0.6027 | 0.8036 | 168 |
| batch_1 | icdo3_grade | 0.9718 | 0.9718 | 150 |
| batch_1 | icdo3_laterality | 1.0000 | 1.0000 | 183 |
| batch_1 | icd11_stem | 0.1851 | 0.3571 | 175 |
| batch_1 | icd11_ext_laterality | 0.3222 | 0.6444 | 175 |
| batch_1 | icd11_ext_grading | 0.7288 | 0.9110 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1640 | 0.5249 | 183 |
| batch_1 | icd11_ext_histopath | 0.1600 | 0.9198 | 183 |
| batch_2 | icdo3_topography | 0.6312 | 0.7214 | 113 |
| batch_2 | icdo3_morphology | 0.1672 | 0.4458 | 113 |
| batch_2 | icdo3_behavior | 0.4988 | 0.9977 | 113 |
| batch_2 | icdo3_grade | 0.9556 | 0.9556 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 1.0000 | 113 |
| batch_2 | icd11_stem | 0.1546 | 0.4637 | 113 |
| batch_2 | icd11_ext_laterality | 0.3333 | 1.0000 | 112 |
| batch_2 | icd11_ext_grading | 0.7596 | 0.9495 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1941 | 0.7765 | 113 |
| batch_2 | icd11_ext_histopath | 0.1072 | 0.8222 | 113 |
| cancer_primary | icdo3_topography | 0.6905 | 0.7891 | 281 |
| cancer_primary | icdo3_morphology | 0.2989 | 0.5123 | 281 |
| cancer_primary | icdo3_behavior | 0.5973 | 0.7964 | 281 |
| cancer_primary | icdo3_grade | 0.9647 | 0.9647 | 254 |
| cancer_primary | icdo3_laterality | 1.0000 | 1.0000 | 292 |
| cancer_primary | icd11_stem | 0.1640 | 0.3407 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3265 | 0.6529 | 283 |
| cancer_primary | icd11_ext_grading | 0.7356 | 0.9195 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.2651 | 0.6060 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1571 | 0.9034 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1.0000 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 1.0000 | 4 |
| non_cancer | icd11_stem | 0.0247 | 0.2222 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 1.0000 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 1.0000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.1250 | 1.0000 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 0.0000 | 4 |
| template | icdo3_topography | 0.6345 | 0.8460 | 33 |
| template | icdo3_morphology | 0.1589 | 0.7627 | 33 |
| template | icdo3_behavior | 0.2500 | 1.0000 | 33 |
| template | icdo3_grade | 1.0000 | 1.0000 | 34 |
| template | icdo3_laterality | 0.6667 | 1.0000 | 34 |
| template | icd11_stem | 0.1429 | 0.7714 | 33 |
| template | icd11_ext_laterality | 0.3333 | 1.0000 | 33 |
| template | icd11_ext_grading | 0.5779 | 0.9632 | 33 |
| template | icd11_ext_anatomy | 0.2385 | 0.7631 | 34 |
| template | icd11_ext_histopath | 0.1304 | 1.0000 | 34 |
| non_template | icdo3_topography | 0.6847 | 0.7825 | 249 |
| non_template | icdo3_morphology | 0.2994 | 0.5132 | 248 |
| non_template | icdo3_behavior | 0.5966 | 0.7954 | 248 |
| non_template | icdo3_grade | 0.9607 | 0.9607 | 220 |
| non_template | icdo3_laterality | 1.0000 | 1.0000 | 262 |
| non_template | icd11_stem | 0.1768 | 0.2983 | 255 |
| non_template | icd11_ext_laterality | 0.3258 | 0.6516 | 254 |
| non_template | icd11_ext_grading | 0.7393 | 0.9241 | 254 |
| non_template | icd11_ext_anatomy | 0.2739 | 0.5479 | 262 |
| non_template | icd11_ext_histopath | 0.1504 | 0.8648 | 262 |
