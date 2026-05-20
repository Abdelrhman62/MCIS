# Step 2 — 5-Fold E1 Summary (5/5 folds)

**Folds aggregated:** [0, 1, 2, 3, 4]  
**Generated:** 2026-05-14T021222Z

## Cross-axis summary (mean ± std across folds)

| Metric | Mean | Std | Per-fold |
|---|---:|---:|---|
| mean_f1_macro_singlepick | 0.6102 | 0.0097 | 0.5978, 0.6158, 0.5992, 0.6197, 0.6185 |
| mean_f1_macro_present_singlepick | 0.6857 | 0.0197 | 0.6546, 0.7008, 0.6745, 0.6880, 0.7104 |
| mean_f1_micro_singlepick | 0.9443 | 0.0040 | 0.9417, 0.9398, 0.9433, 0.9451, 0.9515 |
| mean_accuracy_singlepick | 0.9443 | 0.0040 | 0.9417, 0.9398, 0.9433, 0.9451, 0.9515 |
| mean_f1_micro_multilabel | 0.5349 | 0.0391 | 0.4632, 0.5531, 0.5637, 0.5245, 0.5701 |
| mean_f1_macro_multilabel | 0.0888 | 0.0086 | 0.0730, 0.0966, 0.0912, 0.0873, 0.0958 |
| mean_accuracy_exact_set_multilabel | 0.7404 | 0.0194 | 0.7173, 0.7333, 0.7628, 0.7247, 0.7637 |
| early_stop_metric_current | 0.5726 | 0.0222 | 0.5305, 0.5844, 0.5814, 0.5721, 0.5943 |
| early_stop_metric_present_only | 0.6103 | 0.0280 | 0.5589, 0.6269, 0.6191, 0.6063, 0.6403 |
| early_stop_metric_accuracy_based | 0.8423 | 0.0110 | 0.8295, 0.8366, 0.8530, 0.8349, 0.8576 |

## Per-axis (single-pick) F1-Macro across folds

| Axis | macro mean | macro std | macro_present mean | micro mean | accuracy mean |
|---|---:|---:|---:|---:|---:|
| icdo3_topography | 0.6854 | 0.0256 | 0.7246 | 0.8065 | 0.8065 |
| icdo3_morphology | 0.3982 | 0.0287 | 0.5207 | 0.9413 | 0.9413 |
| icdo3_behavior | 0.6164 | 0.0547 | 0.7827 | 0.9806 | 0.9806 |
| icdo3_grade | 0.9763 | 0.0107 | 0.9763 | 0.9815 | 0.9815 |
| icdo3_laterality | 0.8263 | 0.0914 | 0.8263 | 0.9909 | 0.9909 |
| icd11_stem | 0.2899 | 0.0226 | 0.3798 | 0.9151 | 0.9151 |
| icd11_ext_laterality | 0.3256 | 0.0023 | 0.3974 | 0.9699 | 0.9699 |
| icd11_ext_grading | 0.7634 | 0.0070 | 0.8776 | 0.9686 | 0.9686 |

## Per-axis (multi-label) across folds

| Axis | micro mean | micro std | macro mean | exact-set acc |
|---|---:|---:|---:|---:|
| icd11_ext_anatomy | 0.5605 | 0.0366 | 0.1172 | 0.5619 |
| icd11_ext_histopath | 0.5094 | 0.0537 | 0.0604 | 0.9188 |