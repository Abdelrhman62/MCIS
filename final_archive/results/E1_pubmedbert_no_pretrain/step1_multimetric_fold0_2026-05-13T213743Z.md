# Step 1 — Multi-Metric Eval (fold 0)

**Checkpoint:** `/Users/abdo/Desktop/MCIS/checkpoints/B0/best.pt`  
**Best epoch:** 8  
**Best metric saved:** 0.5305  
**Generated:** 2026-05-13T213743Z

## Summary across axes

| Metric | Value |
|---|---:|
| mean_f1_macro_singlepick | 0.5978 |
| mean_f1_macro_present_singlepick | 0.6546 |
| mean_f1_micro_singlepick | 0.9417 |
| mean_accuracy_singlepick | 0.9417 |
| mean_f1_micro_multilabel | 0.4632 |
| mean_f1_macro_multilabel | 0.0730 |
| mean_accuracy_exact_set_multilabel | 0.7173 |
| early_stop_metric_current | 0.5305 |
| early_stop_metric_present_only | 0.5589 |
| early_stop_metric_accuracy_based | 0.8295 |

## Per-axis (single-pick)

| Axis | K | n_valid | macro | macro_present | micro | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| icdo3_topography | 8 | 525 | 0.6679 | 0.6679 | 0.8000 | 0.8000 |
| icdo3_morphology | 24 | 525 | 0.3632 | 0.4588 | 0.9276 | 0.9276 |
| icdo3_behavior | 4 | 524 | 0.6120 | 0.8160 | 0.9847 | 0.9847 |
| icdo3_grade | 3 | 467 | 0.9884 | 0.9884 | 0.9893 | 0.9893 |
| icdo3_laterality | 3 | 549 | 0.7950 | 0.7950 | 0.9872 | 0.9872 |
| icd11_stem | 27 | 535 | 0.2589 | 0.3495 | 0.9065 | 0.9065 |
| icd11_ext_laterality | 6 | 534 | 0.3242 | 0.3890 | 0.9663 | 0.9663 |
| icd11_ext_grading | 5 | 534 | 0.7726 | 0.7726 | 0.9719 | 0.9719 |

## Per-axis (multi-label)

| Axis | K | N | n_with_any | micro | macro | exact-set acc |
|---|---:|---:|---:|---:|---:|---:|
| icd11_ext_anatomy | 16 | 550 | 402 | 0.5097 | 0.1030 | 0.5291 |
| icd11_ext_histopath | 23 | 550 | 56 | 0.4167 | 0.0430 | 0.9055 |