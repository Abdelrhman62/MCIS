# Step 1 — Multi-Metric Eval (fold 1)

**Checkpoint:** `/Users/abdo/Desktop/MCIS/checkpoints/B0_fold1/best.pt`  
**Best epoch:** 9  
**Best metric saved:** 0.5844  
**Generated:** 2026-05-13T220814Z

## Summary across axes

| Metric | Value |
|---|---:|
| mean_f1_macro_singlepick | 0.6158 |
| mean_f1_macro_present_singlepick | 0.7008 |
| mean_f1_micro_singlepick | 0.9398 |
| mean_accuracy_singlepick | 0.9398 |
| mean_f1_micro_multilabel | 0.5531 |
| mean_f1_macro_multilabel | 0.0966 |
| mean_accuracy_exact_set_multilabel | 0.7333 |
| early_stop_metric_current | 0.5844 |
| early_stop_metric_present_only | 0.6269 |
| early_stop_metric_accuracy_based | 0.8366 |

## Per-axis (single-pick)

| Axis | K | n_valid | macro | macro_present | micro | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| icdo3_topography | 8 | 519 | 0.6806 | 0.6806 | 0.7919 | 0.7919 |
| icdo3_morphology | 24 | 515 | 0.4318 | 0.5757 | 0.9495 | 0.9495 |
| icdo3_behavior | 4 | 515 | 0.5669 | 0.7559 | 0.9728 | 0.9728 |
| icdo3_grade | 3 | 455 | 0.9742 | 0.9742 | 0.9802 | 0.9802 |
| icdo3_laterality | 3 | 540 | 0.8674 | 0.8674 | 0.9889 | 0.9889 |
| icd11_stem | 27 | 530 | 0.3280 | 0.4217 | 0.9151 | 0.9151 |
| icd11_ext_laterality | 6 | 529 | 0.3228 | 0.3873 | 0.9565 | 0.9565 |
| icd11_ext_grading | 5 | 525 | 0.7548 | 0.9435 | 0.9638 | 0.9638 |

## Per-axis (multi-label)

| Axis | K | N | n_with_any | micro | macro | exact-set acc |
|---|---:|---:|---:|---:|---:|---:|
| icd11_ext_anatomy | 16 | 540 | 388 | 0.5318 | 0.1177 | 0.5407 |
| icd11_ext_histopath | 23 | 540 | 54 | 0.5743 | 0.0756 | 0.9259 |