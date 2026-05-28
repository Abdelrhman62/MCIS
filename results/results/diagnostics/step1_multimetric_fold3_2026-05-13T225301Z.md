# Step 1 — Multi-Metric Eval (fold 3)

**Checkpoint:** `/Users/abdo/Desktop/MCIS/checkpoints/B0_fold3/best.pt`  
**Best epoch:** 8  
**Best metric saved:** 0.5721  
**Generated:** 2026-05-13T225301Z

## Summary across axes

| Metric | Value |
|---|---:|
| mean_f1_macro_singlepick | 0.6197 |
| mean_f1_macro_present_singlepick | 0.6880 |
| mean_f1_micro_singlepick | 0.9451 |
| mean_accuracy_singlepick | 0.9451 |
| mean_f1_micro_multilabel | 0.5245 |
| mean_f1_macro_multilabel | 0.0873 |
| mean_accuracy_exact_set_multilabel | 0.7247 |
| early_stop_metric_current | 0.5721 |
| early_stop_metric_present_only | 0.6063 |
| early_stop_metric_accuracy_based | 0.8349 |

## Per-axis (single-pick)

| Axis | K | n_valid | macro | macro_present | micro | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| icdo3_topography | 8 | 531 | 0.6511 | 0.7441 | 0.7910 | 0.7910 |
| icdo3_morphology | 24 | 530 | 0.4310 | 0.5445 | 0.9472 | 0.9472 |
| icdo3_behavior | 4 | 530 | 0.5877 | 0.5877 | 0.9830 | 0.9830 |
| icdo3_grade | 3 | 479 | 0.9832 | 0.9832 | 0.9854 | 0.9854 |
| icdo3_laterality | 3 | 554 | 0.9212 | 0.9212 | 0.9946 | 0.9946 |
| icd11_stem | 27 | 541 | 0.2964 | 0.3810 | 0.9168 | 0.9168 |
| icd11_ext_laterality | 6 | 541 | 0.3290 | 0.3948 | 0.9797 | 0.9797 |
| icd11_ext_grading | 5 | 538 | 0.7582 | 0.9477 | 0.9628 | 0.9628 |

## Per-axis (multi-label)

| Axis | K | N | n_with_any | micro | macro | exact-set acc |
|---|---:|---:|---:|---:|---:|---:|
| icd11_ext_anatomy | 16 | 554 | 421 | 0.5589 | 0.1187 | 0.5307 |
| icd11_ext_histopath | 23 | 554 | 54 | 0.4902 | 0.0558 | 0.9188 |