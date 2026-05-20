# Step 1 — Multi-Metric Eval (fold 4)

**Checkpoint:** `/Users/abdo/Desktop/MCIS/checkpoints/B0_fold4/best.pt`  
**Best epoch:** 9  
**Best metric saved:** 0.5943  
**Generated:** 2026-05-14T021222Z

## Summary across axes

| Metric | Value |
|---|---:|
| mean_f1_macro_singlepick | 0.6185 |
| mean_f1_macro_present_singlepick | 0.7104 |
| mean_f1_micro_singlepick | 0.9515 |
| mean_accuracy_singlepick | 0.9515 |
| mean_f1_micro_multilabel | 0.5701 |
| mean_f1_macro_multilabel | 0.0958 |
| mean_accuracy_exact_set_multilabel | 0.7637 |
| early_stop_metric_current | 0.5943 |
| early_stop_metric_present_only | 0.6403 |
| early_stop_metric_accuracy_based | 0.8576 |

## Per-axis (single-pick)

| Axis | K | n_valid | macro | macro_present | micro | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| icdo3_topography | 8 | 542 | 0.7232 | 0.8265 | 0.8321 | 0.8321 |
| icdo3_morphology | 24 | 541 | 0.3925 | 0.5541 | 0.9556 | 0.9556 |
| icdo3_behavior | 4 | 541 | 0.5934 | 0.7912 | 0.9760 | 0.9760 |
| icdo3_grade | 3 | 482 | 0.9784 | 0.9784 | 0.9855 | 0.9855 |
| icdo3_laterality | 3 | 567 | 0.8847 | 0.8847 | 0.9912 | 0.9912 |
| icd11_stem | 27 | 554 | 0.2805 | 0.3606 | 0.9278 | 0.9278 |
| icd11_ext_laterality | 6 | 555 | 0.3243 | 0.3243 | 0.9658 | 0.9658 |
| icd11_ext_grading | 5 | 554 | 0.7706 | 0.9633 | 0.9783 | 0.9783 |

## Per-axis (multi-label)

| Axis | K | N | n_with_any | micro | macro | exact-set acc |
|---|---:|---:|---:|---:|---:|---:|
| icd11_ext_anatomy | 16 | 567 | 399 | 0.5997 | 0.1259 | 0.6049 |
| icd11_ext_histopath | 23 | 567 | 56 | 0.5405 | 0.0656 | 0.9224 |