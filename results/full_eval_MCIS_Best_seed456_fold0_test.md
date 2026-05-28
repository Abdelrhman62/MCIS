# Full Evaluation Report

**Checkpoint:** `checkpoints/MCIS_Best_seed456_fold0/best.pt`
**Config:** `configs/MCIS_Best.yaml`
**Data:** `data/frozen/m1_model_ready/m1_model_ready.parquet`
**Split:** test (fold 0)
**Records evaluated:** 296

## 1. Per-Axis Metrics

| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |
|---|---|---|---|---|---|---|---|
| icdo3_topography | 0.6839 | 0.7872 | 0.7018 | 0.6842 | 0.7816 | 0.7872 | 282 |
| icdo3_morphology | 0.3377 | 0.9146 | 0.3372 | 0.3570 | 0.5789 | 0.9146 | 281 |
| icdo3_behavior | 0.6216 | 0.9680 | 0.6173 | 0.6261 | 0.8288 | 0.9680 | 281 |
| icdo3_grade | 0.9672 | 0.9724 | 0.9681 | 0.9682 | 0.9672 | 0.9724 | 254 |
| icdo3_laterality | 0.9977 | 0.9966 | 0.9973 | 0.9981 | 0.9977 | 0.9966 | 296 |
| icd11_stem | 0.3111 | 0.8681 | 0.3054 | 0.3183 | 0.5250 | 0.8681 | 288 |
| icd11_ext_laterality | 0.3265 | 0.9791 | 0.3256 | 0.3275 | 0.6531 | 0.9791 | 287 |
| icd11_ext_grading | 0.7380 | 0.9408 | 0.7309 | 0.7480 | 0.9225 | 0.9408 | 287 |
| icd11_ext_anatomy | 0.3117 | 0.6091 | 0.3493 | 0.2901 | 0.6235 | 0.5777 | 296 |
| icd11_ext_histopath | 0.1861 | 0.7391 | 0.1687 | 0.2134 | 0.8561 | 0.9291 | 296 |
| **Official Mean (Agg)** | **0.6485** | | | | | | |
| **Flat Correct Mean** | **0.6332** | | | | | | |

## 2. Ranking Metrics (Single-Pick Only)

| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |
|---|---|---|---|---|---|
| icdo3_topography | 0.7872 | 0.9468 | 0.9752 | 0.9752 | 0.9483 |
| icdo3_morphology | 0.9146 | 0.9644 | 0.9751 | 0.9751 | 0.9263 |
| icdo3_behavior | 0.9680 | 1.0000 | 1.0000 | 1.0000 | 0.9442 |
| icdo3_grade | 0.9724 | 1.0000 | 1.0000 | 1.0000 | 0.9852 |
| icdo3_laterality | 0.9966 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| icd11_stem | 0.8681 | 0.9514 | 0.9653 | 0.9653 | 0.9625 |
| icd11_ext_laterality | 0.9791 | 0.9965 | 1.0000 | 1.0000 | 0.9437 |
| icd11_ext_grading | 0.9408 | 0.9965 | 1.0000 | 1.0000 | 0.9848 |

## 3. Rare vs Common Code F1

| Axis | Rare F1 | Common F1 | #Rare | #Common |
|---|---|---|---|---|
| icdo3_topography | 0.0000 | 0.7816 | 1 | 7 |
| icdo3_morphology | 0.0000 | 0.7368 | 13 | 11 |
| icdo3_behavior | 0.0000 | 0.8288 | 1 | 3 |
| icdo3_grade | 0.0000 | 0.9672 | 0 | 3 |
| icdo3_laterality | 0.0000 | 0.9977 | 0 | 3 |
| icd11_stem | 0.1667 | 0.6000 | 18 | 9 |
| icd11_ext_laterality | 0.0000 | 0.3918 | 1 | 5 |
| icd11_ext_grading | 0.0000 | 0.9225 | 1 | 4 |

## 4. Calibration (ECE)

| Axis | ECE |
|---|---|
| icdo3_topography | 0.1266 |
| icdo3_morphology | 0.0689 |
| icdo3_behavior | 0.0271 |
| icdo3_grade | 0.0229 |
| icdo3_laterality | 0.0020 |
| icd11_stem | 0.1105 |
| icd11_ext_laterality | 0.0100 |
| icd11_ext_grading | 0.0456 |

## 5. Grade Ordinal — Cohen's κ (quadratic)

**κ = 0.9446**

## 6. Subgroup-Stratified F1-Macro

| Subgroup | Axis | F1-Macro | F1-Present | N |
|---|---|---|---|---|
| batch_1 | icdo3_topography | 0.6643 | 0.7592 | 169 |
| batch_1 | icdo3_morphology | 0.3163 | 0.7592 | 168 |
| batch_1 | icdo3_behavior | 0.6369 | 0.8492 | 168 |
| batch_1 | icdo3_grade | 0.9842 | 0.9842 | 150 |
| batch_1 | icdo3_laterality | 0.9962 | 0.9962 | 183 |
| batch_1 | icd11_stem | 0.2609 | 0.5032 | 175 |
| batch_1 | icd11_ext_laterality | 0.3222 | 0.6444 | 175 |
| batch_1 | icd11_ext_grading | 0.7210 | 0.9013 | 174 |
| batch_1 | icd11_ext_anatomy | 0.1928 | 0.6168 | 183 |
| batch_1 | icd11_ext_histopath | 0.1600 | 0.9198 | 183 |
| batch_2 | icdo3_topography | 0.5929 | 0.6775 | 113 |
| batch_2 | icdo3_morphology | 0.1984 | 0.5290 | 113 |
| batch_2 | icdo3_behavior | 0.4988 | 0.9977 | 113 |
| batch_2 | icdo3_grade | 0.9407 | 0.9407 | 104 |
| batch_2 | icdo3_laterality | 0.6667 | 1.0000 | 113 |
| batch_2 | icd11_stem | 0.2161 | 0.6482 | 113 |
| batch_2 | icd11_ext_laterality | 0.3333 | 1.0000 | 112 |
| batch_2 | icd11_ext_grading | 0.7654 | 0.9568 | 113 |
| batch_2 | icd11_ext_anatomy | 0.1614 | 0.8606 | 113 |
| batch_2 | icd11_ext_histopath | 0.1507 | 0.8667 | 113 |
| cancer_primary | icdo3_topography | 0.6833 | 0.7809 | 281 |
| cancer_primary | icdo3_morphology | 0.3377 | 0.5789 | 281 |
| cancer_primary | icdo3_behavior | 0.6216 | 0.8288 | 281 |
| cancer_primary | icdo3_grade | 0.9672 | 0.9672 | 254 |
| cancer_primary | icdo3_laterality | 0.9976 | 0.9976 | 292 |
| cancer_primary | icd11_stem | 0.2556 | 0.5308 | 284 |
| cancer_primary | icd11_ext_laterality | 0.3265 | 0.6529 | 283 |
| cancer_primary | icd11_ext_grading | 0.7348 | 0.9184 | 283 |
| cancer_primary | icd11_ext_anatomy | 0.3112 | 0.6224 | 292 |
| cancer_primary | icd11_ext_histopath | 0.1861 | 0.8561 | 292 |
| non_cancer | icdo3_topography | 0.1250 | 1.0000 | 1 |
| non_cancer | icdo3_morphology | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_behavior | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_grade | 0.0000 | 0.0000 | 0 |
| non_cancer | icdo3_laterality | 0.6667 | 1.0000 | 4 |
| non_cancer | icd11_stem | 0.0556 | 0.5000 | 4 |
| non_cancer | icd11_ext_laterality | 0.3333 | 1.0000 | 4 |
| non_cancer | icd11_ext_grading | 0.2000 | 1.0000 | 4 |
| non_cancer | icd11_ext_anatomy | 0.0625 | 1.0000 | 4 |
| non_cancer | icd11_ext_histopath | 0.0000 | 0.0000 | 4 |
| template | icdo3_topography | 0.6273 | 0.8364 | 33 |
| template | icdo3_morphology | 0.1521 | 0.7301 | 33 |
| template | icdo3_behavior | 0.2500 | 1.0000 | 33 |
| template | icdo3_grade | 1.0000 | 1.0000 | 34 |
| template | icdo3_laterality | 0.6667 | 1.0000 | 34 |
| template | icd11_stem | 0.1237 | 0.6679 | 33 |
| template | icd11_ext_laterality | 0.3333 | 1.0000 | 33 |
| template | icd11_ext_grading | 0.5779 | 0.9632 | 33 |
| template | icd11_ext_anatomy | 0.2308 | 0.7385 | 34 |
| template | icd11_ext_histopath | 0.1304 | 1.0000 | 34 |
| non_template | icdo3_topography | 0.6781 | 0.7750 | 249 |
| non_template | icdo3_morphology | 0.3394 | 0.5819 | 248 |
| non_template | icdo3_behavior | 0.6209 | 0.8279 | 248 |
| non_template | icdo3_grade | 0.9631 | 0.9631 | 220 |
| non_template | icdo3_laterality | 0.9974 | 0.9974 | 262 |
| non_template | icd11_stem | 0.3080 | 0.5198 | 255 |
| non_template | icd11_ext_laterality | 0.3258 | 0.6516 | 254 |
| non_template | icd11_ext_grading | 0.7374 | 0.9218 | 254 |
| non_template | icd11_ext_anatomy | 0.3063 | 0.6125 | 262 |
| non_template | icd11_ext_histopath | 0.1794 | 0.8252 | 262 |
