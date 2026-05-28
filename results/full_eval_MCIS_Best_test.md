# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold1/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6921 | 0.7837 | 0.6995 | 0.6910 | 0.7910 | 0.7837 | 282 |
| icdo3_morphology | 0.3067 | 0.9075 | 0.3253 | 0.3242 | 0.5258 | 0.9075 | 281 |
| icdo3_behavior | 0.6680 | 0.9786 | 0.7130 | 0.6399 | 0.8907 | 0.9786 | 281 |
| icdo3_grade | 0.9692 | 0.9764 | 0.9701 | 0.9703 | 0.9692 | 0.9764 | 254 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 296 |
| icd11_stem | 0.2309 | 0.8611 | 0.2247 | 0.2395 | 0.3897 | 0.8611 | 288 |
| icd11_ext_laterality | 0.3238 | 0.9652 | 0.3242 | 0.3236 | 0.6477 | 0.9652 | 287 |
| icd11_ext_grading | 0.7440 | 0.9512 | 0.7313 | 0.7593 | 0.9300 | 0.9512 | 287 |
| icd11_ext_anatomy | 0.3063 | 0.6420 | 0.3557 | 0.2873 | 0.6126 | 0.6284 | 296 |
| icd11_ext_histopath | 0.1501 | 0.6897 | 0.1467 | 0.1551 | 0.8633 | 0.9189 | 296 |
| **Mean** | **0.5391** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7837 | 0.9610 | 0.9787 | 0.9787 | 0.0000 |
| icdo3_morphology | 0.9075 | 0.9680 | 0.9786 | 0.9786 | 0.0000 |
| icdo3_behavior | 0.9786 | 0.9964 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9764 | 1.0000 | 1.0000 | 1.0000 | 0.9902 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8611 | 0.9444 | 0.9653 | 0.9653 | 0.0000 |
| icd11_ext_laterality | 0.9652 | 0.9930 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9512 | 0.9965 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7910 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.6692 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8907 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9692 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0556 | 0.5817 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3886 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9300 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1362 |
| icdo3_morphology | 0.0832 |
| icdo3_behavior | 0.0220 |
| icdo3_grade | 0.0253 |
| icdo3_laterality | 0.0006 |
| icd11_stem | 0.1132 |
| icd11_ext_laterality | 0.0256 |
| icd11_ext_grading | 0.0455 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9585**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.6435 | 169 |
| batch_1 | icdo3_morphology | 0.2824 | 168 |
| batch_1 | icdo3_behavior | 0.6547 | 168 |
| batch_1 | icdo3_grade | 0.9718 | 150 |
| batch_1 | icdo3_laterality | 1.0000 | 183 |
| batch_1 | icd11_stem | 0.1780 | 175 |
| batch_1 | icd11_ext_laterality | 0.3197 | 175 |
| batch_1 | icd11_ext_grading | 0.7206 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1849 | 183 |
| batch_1 | icd11_ext_histopath | 0.1550 | 183 |
| batch_2 | icdo3_topography | 0.6399 | 113 |
| batch_2 | icdo3_morphology | 0.1966 | 113 |
| batch_2 | icdo3_behavior | 0.5000 | 113 |
| batch_2 | icdo3_grade | 0.9654 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 113 |
| batch_2 | icd11_stem | 0.2172 | 113 |
| batch_2 | icd11_ext_laterality | 0.3302 | 112 |
| batch_2 | icd11_ext_grading | 0.7945 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1955 | 113 |
| batch_2 | icd11_ext_histopath | 0.0942 | 113 |
| cancer_primary | icdo3_topography | 0.6916 | 281 |
| cancer_primary | icdo3_morphology | 0.3067 | 281 |
| cancer_primary | icdo3_behavior | 0.6680 | 281 |
| cancer_primary | icdo3_grade | 0.9692 | 254 |
| cancer_primary | icdo3_laterality | 1.0000 | 292 |
| cancer_primary | icd11_stem | 0.2124 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3237 | 283 |
| cancer_primary | icd11_ext_grading | 0.7401 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.3057 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1501 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 4 |
| non_cancer | icd11_stem | 0.0185 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 4 |
| template | icdo3_topography | 0.5920 | 33 |
| template | icdo3_morphology | 0.1450 | 33 |
| template | icdo3_behavior | 0.2500 | 33 |
| template | icdo3_grade | 1.0000 | 34 |
| template | icdo3_laterality | 0.6667 | 34 |
| template | icd11_stem | 0.1219 | 33 |
| template | icd11_ext_laterality | 0.3264 | 33 |
| template | icd11_ext_grading | 0.5779 | 33 |
| template | icd11_ext_anatomy | 0.1888 | 34 |
| template | icd11_ext_histopath | 0.1159 | 34 |
| non_template | icdo3_topography | 0.6898 | 249 |
| non_template | icdo3_morphology | 0.3083 | 248 |
| non_template | icdo3_behavior | 0.6676 | 248 |
| non_template | icdo3_grade | 0.9656 | 220 |
| non_template | icdo3_laterality | 1.0000 | 262 |
| non_template | icd11_stem | 0.2262 | 255 |
| non_template | icd11_ext_laterality | 0.3239 | 254 |
| non_template | icd11_ext_grading | 0.7445 | 254 |
| non_template | icd11_ext_anatomy | 0.3048 | 262 |
| non_template | icd11_ext_histopath | 0.1448 | 262 |
