# TCGA Pretrain — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `tcga_pretrain_model_ready/`
- **Built at:** 2026-05-07T12:49:00.015417+00:00
- **Vocab version:** v2 (canonical schema; supersedes v1)
- **Built from:** `pretrain_train` split

## 2. Row counts
- **Total rows:** 8,453
- **Splits:** {'pretrain_train': 5926, 'pretrain_val': 1268, 'pretrain_test': 1259}
- **Cancer types (top 10):** {'UCEC': 545, 'KIRC': 525, 'HNSC': 520, 'LUAD': 487, 'THCA': 487, 'LGG': 465, 'LUSC': 462, 'PRAD': 446, 'COAD': 417, 'GBM': 386}

## 3. Schema (canonical, 34 cols; matches Baheya M1 + SEER + TCGA-BRCA)
See `MCIS_Dataset_Unification_Handoff.md` §2.1 for full column list.

## 4. Label coverage
| Axis | Coverage | Type |
|---|---|---|
| `icdo3_topography` | 8453/8453 (100.0%) | single-pick |
| `icdo3_morphology` | 8453/8453 (100.0%) | single-pick |
| `icdo3_behavior` | 8453/8453 (100.0%) | single-pick |
| `icdo3_grade` | 3643/8453 (43.1%) | single-pick (string) |
| `icdo3_laterality` | 3071/8453 (36.3%) | single-pick (L/R/B) |
| `icd11_stem` | 8453/8453 (100.0%) | silver, never label |
| `icd11_ext_anatomy` | 417/8453 (4.9%) have ≥1 positive; max per row=1 | multi-label list[str] |
| `icd11_ext_histopath` | 470/8453 (5.6%) have ≥1 positive; max per row=1 | multi-label list[str] |
| `icd11_ext_laterality` | 3071/8453 (36.3%) have ≥1 positive; max per row=1 | multi-label list[str] |
| `icd11_ext_grading` | 6085/8453 (72.0%) have ≥1 positive; max per row=2 | multi-label list[str] |

## 5. Vocab (built from `pretrain_train` only)
| Axis | K | First codes |
|---|---:|---|
| `icdo3_topography` | 118 | ['C64.9', 'C54.1', 'C34.1', 'C73.9', 'C61.9'] |
| `icdo3_morphology` | 108 | ['8140', '8070', '8260', '8310', '8441'] |
| `icdo3_behavior` | 3 | ['3', '0', '1'] |
| `icdo3_grade` | 4 | ['3', '2', '1', '4'] (string-encoded) |
| `icdo3_laterality` | 3 | ['L', 'R', 'B'] |

ICD-11 extension axes are **not** part of the pretrain vocab (silver-only, never used as supervision).
A head built from this artifact has no ICD-11 component.

## 6. Encoding decisions
- **`icdo3_laterality`:** L/R/B single-pick. Original X-codes preserved in `icdo3_laterality_raw`. Mapping: `XK8G→L, XK9K→R, XK9J→B`.
- **`icdo3_grade`:** string `'1'`–`'4'`. Null = mask in loss (not treated as Grade-Unknown class).
- **ICD-11 extension axes:** parsed from `icd11_full_postcoord` into `list[str]`. Empty list = no positives. Stored for schema parity only; not used in pretraining.
- **Splits:** 70/15/15 iterative-stratified (cancer_type × morphology × behavior), group-aware on `duplicate_group_id`, seed=42. Identical to v1.

## 7. Hard rules
- `icd11_*` columns are silver. Never used as supervision. Never used in eval.
- The `text_section_tagged` and `text_concatenated` columns are auditable — no label-derived fields injected.
- Vocab built from `pretrain_train` only. OOV codes in `pretrain_val`/`pretrain_test` are excluded from F1 (footnoted, not silently dropped).

## 8. Changelog v1 → v2
- icdo3_laterality re-keyed: X-codes → L/R/B (matches Baheya).
- icdo3_grade encoding: int → string.
- ICD-11 extension axes added as list[str] columns: anatomy, histopath, laterality, grading.
- Bookkeeping cols added: fold (=-1 for TCGA), batch (=-1), template_flag (=False), is_cancer_primary (=True), coding_source (='gdc_curated'), data_quality_flag (=None).
- Aux cols added (=None): nuclear_grade, tubular_score, mitosis_score, lvi, dcis_in_specimen.
- icdo3_laterality_raw added.
- icd11_full_postcoord retained (TCGA-only reference).

## 9. Files
- `tcga_pretrain_model_ready.parquet` — model-ready data
- `label_vocab.json` — frozen vocab (v2)
- `splits.csv` — record_id → split mapping
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts
