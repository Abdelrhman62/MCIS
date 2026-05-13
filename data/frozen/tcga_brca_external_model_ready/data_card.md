# TCGA-BRCA External — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `tcga_brca_external_model_ready/`
- **Built at:** 2026-05-07T12:49:00.015417+00:00
- **Purpose:** External validation only. Never used in training.

## 2. Row counts
- **Total rows:** 1,025
- **Split:** all `brca_external` (single label; v1 internal `brca_train`/`brca_val`/`brca_test` discarded)

## 3. Schema (canonical, 34 cols)
Matches `tcga_pretrain_model_ready/`, `m1_model_ready/`, and `seer_rare_augmentation_model_ready/`. See handoff §2.1.

## 4. Label coverage

### 4.1 Native value distributions
| Axis | Coverage | Notes |
|---|---|---|
| `icdo3_topography` | 1025/1025 (100.0%) | 6 unique codes |
| `icdo3_morphology` | 1025/1025 (100.0%) | 21 unique codes |
| `icdo3_behavior` | 1025/1025 (100.0%) | values: ['3'] |
| `icdo3_grade` | 0/1025 (0.0%) | TCGA-BRCA has no curated grade |
| `icdo3_laterality` | 961/1025 (93.8%) | values: ['L', 'R'] |
| `icd11_ext_anatomy` | 1025/1025 (100.0%) have ≥1 positive; max per row=1 | parsed from postcoord |
| `icd11_ext_histopath` | 44/1025 (4.3%) have ≥1 positive; max per row=1 | parsed from postcoord |
| `icd11_ext_laterality` | 961/1025 (93.8%) have ≥1 positive; max per row=1 | parsed from postcoord |
| `icd11_ext_grading` | 1017/1025 (99.2%) have ≥1 positive; max per row=2 | parsed from postcoord |

### 4.2 Baheya-vocab coverage (Ext-1 evaluation)
This BRCA artifact is evaluated against the **Baheya M1 vocab** (Ext-1) and against the **TCGA-pretrain vocab** (Ext-2). Coverage under Baheya M1:

#### `icdo3_topography` (Baheya K=8, BRCA has 6 unique)
- **In Baheya vocab:** ['C50.2', 'C50.3', 'C50.4', 'C50.5', 'C50.9'] (1,023/1,025 rows = 99.8% of populated rows)
- **OOV under Baheya:** ['C50.8']

#### `icdo3_morphology` (Baheya K=24, BRCA has 21 unique)
- **In Baheya vocab (14 codes):** ['8010', '8050', '8200', '8211', '8480', '8500', '8502', '8503', '8507', '8510', '8520', '8522', '8575', '9020']
- **OOV under Baheya (7 codes):** ['8013', '8022', '8090', '8401', '8523', '8524', '8541']
- **Row-level:** 992/1,025 rows in vocab = 96.8%

#### `icdo3_laterality` (after L/R/B normalization)
- **BRCA values:** ['L', 'R']
- **Baheya vocab covers:** ['L', 'R'] → 100% coverage

#### `icdo3_behavior`
- **BRCA values:** ['3']; **Baheya values:** ['0', '2', '3', '6']
- **OOV:** []

#### `icdo3_grade`
- BRCA has 0% grade populated. Coverage analysis is moot.

## 5. Encoding decisions
Same as TCGA pretrain (see that data card §6).

## 6. Hard rules
- This is **external validation only**. No training. No fold structure.
- Eval uses two protocols:
  - **Ext-1:** Baheya M1 vocab (this artifact's labels intersected with Baheya vocab).
  - **Ext-2:** TCGA-pretrain vocab.
- ICD-11 columns are silver; never used in eval (per architecture v6 §2.4).

## 7. Files
- `tcga_brca_external_model_ready.parquet` — eval data
- `baheya_vocab_coverage.json` — Ext-1 coverage details
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts
