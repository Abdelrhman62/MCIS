# Full Evaluation Report

**Checkpoint:** `checkpoints/E1/best.pt`
**Config:** `configs/E1.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6551 | 0.7730 | 0.6787 | 0.6589 | 0.7487 | 0.7730 | 282 |
| icdo3_morphology | 0.2688 | 0.9075 | 0.2716 | 0.2808 | 0.4608 | 0.9075 | 281 |
| icdo3_behavior | 0.6287 | 0.9715 | 0.6202 | 0.6380 | 0.8382 | 0.9715 | 281 |
| icdo3_grade | 0.9579 | 0.9646 | 0.9694 | 0.9489 | 0.9579 | 0.9646 | 254 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 296 |
| icd11_stem | 0.2521 | 0.8889 | 0.2482 | 0.2766 | 0.4254 | 0.8889 | 288 |
| icd11_ext_laterality | 0.3265 | 0.9791 | 0.3256 | 0.3275 | 0.6531 | 0.9791 | 287 |
| icd11_ext_grading | 0.7529 | 0.9547 | 0.7455 | 0.7637 | 0.9411 | 0.9547 | 287 |
| icd11_ext_anatomy | 0.1927 | 0.6165 | 0.2085 | 0.1863 | 0.5140 | 0.6014 | 296 |
| icd11_ext_histopath | 0.1025 | 0.6517 | 0.0992 | 0.1102 | 0.7861 | 0.9088 | 296 |
| **Mean** | **0.5137** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7730 | 0.9291 | 0.9752 | 0.9752 | 0.0000 |
| icdo3_morphology | 0.9075 | 0.9680 | 0.9680 | 0.9680 | 0.0000 |
| icdo3_behavior | 0.9715 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| icdo3_grade | 0.9646 | 1.0000 | 1.0000 | 1.0000 | 0.9955 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8889 | 0.9444 | 0.9618 | 0.9618 | 0.0000 |
| icd11_ext_laterality | 0.9791 | 0.9965 | 1.0000 | 1.0000 | 0.0000 |
| icd11_ext_grading | 0.9547 | 0.9930 | 1.0000 | 1.0000 | 0.0000 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7487 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.5864 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8382 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9579 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0741 | 0.6081 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3918 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9411 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1656 |
| icdo3_morphology | 0.0714 |
| icdo3_behavior | 0.0236 |
| icdo3_grade | 0.0337 |
| icdo3_laterality | 0.0008 |
| icd11_stem | 0.0779 |
| icd11_ext_laterality | 0.0148 |
| icd11_ext_grading | 0.0384 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9368**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | N |
|---|---|---|---|
| batch_1 | icdo3_topography | 0.5788 | 169 |
| batch_1 | icdo3_morphology | 0.2456 | 168 |
| batch_1 | icdo3_behavior | 0.6473 | 168 |
| batch_1 | icdo3_grade | 0.9680 | 150 |
| batch_1 | icdo3_laterality | 1.0000 | 183 |
| batch_1 | icd11_stem | 0.2301 | 175 |
| batch_1 | icd11_ext_laterality | 0.3222 | 175 |
| batch_1 | icd11_ext_grading | 0.7386 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1138 | 183 |
| batch_1 | icd11_ext_histopath | 0.1064 | 183 |
| batch_2 | icdo3_topography | 0.6288 | 113 |
| batch_2 | icdo3_morphology | 0.1966 | 113 |
| batch_2 | icdo3_behavior | 0.4988 | 113 |
| batch_2 | icdo3_grade | 0.9455 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 113 |
| batch_2 | icd11_stem | 0.1817 | 113 |
| batch_2 | icd11_ext_laterality | 0.3333 | 112 |
| batch_2 | icd11_ext_grading | 0.7885 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1090 | 113 |
| batch_2 | icd11_ext_histopath | 0.0942 | 113 |
| cancer_primary | icdo3_topography | 0.6545 | 281 |
| cancer_primary | icdo3_morphology | 0.2688 | 281 |
| cancer_primary | icdo3_behavior | 0.6287 | 281 |
| cancer_primary | icdo3_grade | 0.9579 | 254 |
| cancer_primary | icdo3_laterality | 1.0000 | 292 |
| cancer_primary | icd11_stem | 0.2033 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3265 | 283 |
| cancer_primary | icd11_ext_grading | 0.7497 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.1925 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1025 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 4 |
| non_cancer | icd11_stem | 0.0617 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 4 |
| template | icdo3_topography | 0.6028 | 33 |
| template | icdo3_morphology | 0.1589 | 33 |
| template | icdo3_behavior | 0.2500 | 33 |
| template | icdo3_grade | 0.9634 | 34 |
| template | icdo3_laterality | 0.6667 | 34 |
| template | icd11_stem | 0.1297 | 33 |
| template | icd11_ext_laterality | 0.3333 | 33 |
| template | icd11_ext_grading | 0.5779 | 33 |
| template | icd11_ext_anatomy | 0.1894 | 34 |
| template | icd11_ext_histopath | 0.1159 | 34 |
| non_template | icdo3_topography | 0.6484 | 249 |
| non_template | icdo3_morphology | 0.2674 | 248 |
| non_template | icdo3_behavior | 0.6281 | 248 |
| non_template | icdo3_grade | 0.9593 | 220 |
| non_template | icdo3_laterality | 1.0000 | 262 |
| non_template | icd11_stem | 0.2485 | 255 |
| non_template | icd11_ext_laterality | 0.3258 | 254 |
| non_template | icd11_ext_grading | 0.7542 | 254 |
| non_template | icd11_ext_anatomy | 0.1903 | 262 |
| non_template | icd11_ext_histopath | 0.0969 | 262 |
