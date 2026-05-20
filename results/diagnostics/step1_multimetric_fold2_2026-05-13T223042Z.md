# Step 1 — Multi-Metric Eval (fold 2)

**Checkpoint:** `/Users/abdo/Desktop/MCIS/checkpoints/B0_fold2/best.pt`  
**Best epoch:** 9  
**Best metric saved:** 0.5814  
**Generated:** 2026-05-13T223042Z

## Summary across axes

| Metric | Value |
|---|---:|
| mean_f1_macro_singlepick | 0.5992 |
| mean_f1_macro_present_singlepick | 0.6745 |
| mean_f1_micro_singlepick | 0.9433 |
| mean_accuracy_singlepick | 0.9433 |
| mean_f1_micro_multilabel | 0.5637 |
| mean_f1_macro_multilabel | 0.0912 |
| mean_accuracy_exact_set_multilabel | 0.7628 |
| early_stop_metric_current | 0.5814 |
| early_stop_metric_present_only | 0.6191 |
| early_stop_metric_accuracy_based | 0.8530 |

## Per-axis (single-pick)

| Axis | K | n_valid | macro | macro_present | micro | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| icdo3_topography | 8 | 520 | 0.7040 | 0.7040 | 0.8173 | 0.8173 |
| icdo3_morphology | 24 | 517 | 0.3724 | 0.4704 | 0.9265 | 0.9265 |
| icdo3_behavior | 4 | 517 | 0.7221 | 0.9627 | 0.9865 | 0.9865 |
| icdo3_grade | 3 | 457 | 0.9572 | 0.9572 | 0.9672 | 0.9672 |
| icdo3_laterality | 3 | 548 | 0.6631 | 0.6631 | 0.9927 | 0.9927 |
| icd11_stem | 27 | 529 | 0.2860 | 0.3862 | 0.9093 | 0.9093 |
| icd11_ext_laterality | 6 | 528 | 0.3276 | 0.4914 | 0.9811 | 0.9811 |
| icd11_ext_grading | 5 | 528 | 0.7609 | 0.7609 | 0.9659 | 0.9659 |

## Per-axis (multi-label)

| Axis | K | N | n_with_any | micro | macro | exact-set acc |
|---|---:|---:|---:|---:|---:|---:|
| icd11_ext_anatomy | 16 | 548 | 393 | 0.6022 | 0.1205 | 0.6040 |
| icd11_ext_histopath | 23 | 548 | 51 | 0.5253 | 0.0618 | 0.9215 |