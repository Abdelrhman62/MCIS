# External Validation Report

**Checkpoint:** `checkpoints/E7_hierarchical_fold0/best.pt`
**Data:** `data/frozen/tcga_brca_external_model_ready/tcga_brca_external_model_ready.parquet`
**Records:** 1025
**Axes evaluated:** ICD-O-3 only (TCGA ICD-11 excluded — silver labels)

## Results

**Mean ICD-O-3 F1-Macro: 0.2073**

| Axis | F1-Macro | F1-Macro (present) | N valid | N classes |
|---|---|---|---|---|
| icdo3_topography | 0.1194 | 0.1911 | 1023 | 5 |
| icdo3_morphology | 0.1451 | 0.2487 | 992 | 14 |
| icdo3_behavior | 0.1877 | 0.7508 | 1025 | 1 |
| icdo3_grade | 0.0000 | 0.0000 | 0 | 0 |
| icdo3_laterality | 0.5842 | 0.8763 | 961 | 2 |

## Notes
- ICD-O-3 F1-Macro computed across all K classes (zero-fill for unseen classes).
- Records with NULL_TARGET_SENTINEL (missing label) excluded per axis.
- TCGA ICD-11 codes are silver (mapped from ICD-O-3, not natively coded) — never scored.