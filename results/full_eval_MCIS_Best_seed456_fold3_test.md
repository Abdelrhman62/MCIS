# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold3/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6927 | 0.8298 | 0.7158 | 0.6859 | 0.7917 | 0.8298 | 282 |
| icdo3_morphology | 0.2546 | 0.9039 | 0.2618 | 0.2584 | 0.4365 | 0.9039 | 281 |
| icdo3_behavior | 0.6183 | 0.9644 | 0.6037 | 0.6360 | 0.8244 | 0.9644 | 281 |
| icdo3_grade | 0.9574 | 0.9646 | 0.9539 | 0.9641 | 0.9574 | 0.9646 | 254 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 296 |
| icd11_stem | 0.2184 | 0.8646 | 0.2128 | 0.2282 | 0.3686 | 0.8646 | 288 |
| icd11_ext_laterality | 0.3265 | 0.9791 | 0.3256 | 0.3275 | 0.6531 | 0.9791 | 287 |
| icd11_ext_grading | 0.7493 | 0.9512 | 0.7390 | 0.7625 | 0.9366 | 0.9512 | 287 |
| icd11_ext_anatomy | 0.1924 | 0.6024 | 0.2805 | 0.1826 | 0.5130 | 0.5676 | 296 |
| icd11_ext_histopath | 0.1584 | 0.7333 | 0.1492 | 0.1700 | 0.9109 | 0.9291 | 296 |
| **Official Mean (Agg)** | **0.6350** | | | | | | |
| **Flat Correct Mean** | **0.6153** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.8298 | 0.9681 | 0.9929 | 0.9929 | 0.9562 |
| icdo3_morphology | 0.9039 | 0.9609 | 0.9822 | 0.9822 | 0.9530 |
| icdo3_behavior | 0.9644 | 1.0000 | 1.0000 | 1.0000 | 0.9571 |
| icdo3_grade | 0.9646 | 1.0000 | 1.0000 | 1.0000 | 0.9879 |
| icdo3_laterality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8646 | 0.9306 | 0.9514 | 0.9514 | 0.9476 |
| icd11_ext_laterality | 0.9791 | 0.9965 | 1.0000 | 1.0000 | 0.9191 |
| icd11_ext_grading | 0.9512 | 0.9895 | 1.0000 | 1.0000 | 0.9871 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7917 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.5555 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8244 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9574 | 0 | 3 |
| icdo3_laterality | 0.0000 | 1.0000 | 0 | 3 |
| icd11_stem | 0.0556 | 0.5442 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3918 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9366 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1602 |
| icdo3_morphology | 0.0741 |
| icdo3_behavior | 0.0289 |
| icdo3_grade | 0.0281 |
| icdo3_laterality | 0.0011 |
| icd11_stem | 0.1010 |
| icd11_ext_laterality | 0.0141 |
| icd11_ext_grading | 0.0419 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9180**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | F1-Present | N |
|---|---|---|---|---|
| batch_1 | icdo3_topography | 0.6763 | 0.7729 | 169 |
| batch_1 | icdo3_morphology | 0.2636 | 0.6326 | 168 |
| batch_1 | icdo3_behavior | 0.6336 | 0.8448 | 168 |
| batch_1 | icdo3_grade | 0.9691 | 0.9691 | 150 |
| batch_1 | icdo3_laterality | 1.0000 | 1.0000 | 183 |
| batch_1 | icd11_stem | 0.1840 | 0.3549 | 175 |
| batch_1 | icd11_ext_laterality | 0.3222 | 0.6444 | 175 |
| batch_1 | icd11_ext_grading | 0.7365 | 0.9206 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1242 | 0.3975 | 183 |
| batch_1 | icd11_ext_histopath | 0.1614 | 0.9278 | 183 |
| batch_2 | icdo3_topography | 0.5824 | 0.6656 | 113 |
| batch_2 | icdo3_morphology | 0.1530 | 0.4081 | 113 |
| batch_2 | icdo3_behavior | 0.4988 | 0.9977 | 113 |
| batch_2 | icdo3_grade | 0.9407 | 0.9407 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 1.0000 | 113 |
| batch_2 | icd11_stem | 0.1779 | 0.5337 | 113 |
| batch_2 | icd11_ext_laterality | 0.3333 | 1.0000 | 112 |
| batch_2 | icd11_ext_grading | 0.7830 | 0.9787 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1007 | 0.8055 | 113 |
| batch_2 | icd11_ext_histopath | 0.1072 | 0.8222 | 113 |
| cancer_primary | icdo3_topography | 0.6923 | 0.7912 | 281 |
| cancer_primary | icdo3_morphology | 0.2546 | 0.4365 | 281 |
| cancer_primary | icdo3_behavior | 0.6183 | 0.8244 | 281 |
| cancer_primary | icdo3_grade | 0.9574 | 0.9574 | 254 |
| cancer_primary | icdo3_laterality | 1.0000 | 1.0000 | 292 |
| cancer_primary | icd11_stem | 0.2006 | 0.4166 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3265 | 0.6529 | 283 |
| cancer_primary | icd11_ext_grading | 0.7461 | 0.9326 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.1919 | 0.5117 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1584 | 0.9109 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1.0000 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 1.0000 | 4 |
| non_cancer | icd11_stem | 0.0185 | 0.1667 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 1.0000 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 1.0000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 1.0000 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 0.0000 | 4 |
| template | icdo3_topography | 0.6328 | 0.8437 | 33 |
| template | icdo3_morphology | 0.1450 | 0.6961 | 33 |
| template | icdo3_behavior | 0.2500 | 1.0000 | 33 |
| template | icdo3_grade | 1.0000 | 1.0000 | 34 |
| template | icdo3_laterality | 0.6667 | 1.0000 | 34 |
| template | icd11_stem | 0.1297 | 0.7005 | 33 |
| template | icd11_ext_laterality | 0.3333 | 1.0000 | 33 |
| template | icd11_ext_grading | 0.5779 | 0.9632 | 33 |
| template | icd11_ext_anatomy | 0.2292 | 0.7333 | 34 |
| template | icd11_ext_histopath | 0.1304 | 1.0000 | 34 |
| non_template | icdo3_topography | 0.6878 | 0.7861 | 249 |
| non_template | icdo3_morphology | 0.2571 | 0.4408 | 248 |
| non_template | icdo3_behavior | 0.6176 | 0.8234 | 248 |
| non_template | icdo3_grade | 0.9516 | 0.9516 | 220 |
| non_template | icdo3_laterality | 1.0000 | 1.0000 | 262 |
| non_template | icd11_stem | 0.2143 | 0.3617 | 255 |
| non_template | icd11_ext_laterality | 0.3258 | 0.6516 | 254 |
| non_template | icd11_ext_grading | 0.7504 | 0.9380 | 254 |
| non_template | icd11_ext_anatomy | 0.1835 | 0.4894 | 262 |
| non_template | icd11_ext_histopath | 0.1521 | 0.8747 | 262 |
