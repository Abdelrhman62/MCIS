# MCIS — Baheya M1 Model-Ready Data Card

Generated: build_m1_model_ready.py
Source: baheya_m1_primary_ready.csv (3,056 × 43)

This artifact freezes the model-ready Baheya M1 dataset for the OCE Phase 2
fine-tuning loop and all M1 experiments (B0–B2, E1, E1-fmt, E2, E5a, E5b,
MCIS-Best, Ext-1). Per architecture v6 §4.4, the model receives section-tagged
text input plus 10 axis labels. Two text formats are emitted (`text_section_tagged`
and `text_concatenated`) so E1-fmt can A/B them without regenerating the dataset.

## 1. Decisions locked at build time

| # | Decision | Rationale |
|---|---|---|
| Q1 | Keep all 2,760 trainable rows; mask null axes in loss. | Architecture §6.3 already commits to per-axis masked CE. Drop-on-any-null wastes labeled signal on all other axes and silently biases the kept set. |
| Q2 | Keep all 441 templated reports in training. | These are real, gold-coded Baheya patients written in stereotyped reporting style — not synthetic SEER augmentation. Templated style is what the model will see at deployment. Subgroup-stratified eval (§11.3) catches any over-reliance. |
| Q3 | Vocabularies built from trainable split only. Test-only codes excluded from F1. | Standard treatment in PLM-ICD/CAML/LAAT. Including test-only codes in the head adds random weights at test time that depress macro-F1 without informative signal. Reported transparently in §4 below. |
| Q4 | CV folds regenerated on full 2,760 trainable. | Consistent with Q1 keep-rule. Same three acceptance bars as before. |

## 2. Final row counts

| Subset | Rows |
|---|---|
| Source CSV | 3,056 |
| Dropped (zero supervision on all 10 axes) | 1 |
| Total in artifact | 3,055 |
| Trainable | 2,759 |
| Test | 296 |
| Trainable patients | 2,759 |
| Test patients | 296 |
| Patients spanning splits | 0 |

## 3. Per-axis label coverage (trainable split)

Single-pick axes: `non_null` is the count of records with a label on this axis.
Loss is masked when null. Multi-label axes: `n_with_any` is the count of records
with ≥1 code on this axis.

| Axis | K (trainable) | Coverage in trainable |
|---|---|---|
| icdo3_topography | 8 | 2637 (95.6%) |
| icdo3_morphology | 24 | 2628 (95.3%) |
| icdo3_behavior | 4 | 2627 (95.2%) |
| icdo3_grade | 3 | 2340 (84.8%) |
| icdo3_laterality | 3 | 2758 (100.0%) |
| icd11_stem | 27 | 2689 (97.5%) |
| icd11_ext_anatomy (multi) | 16 | 2003 (72.6%) |
| icd11_ext_histopath (multi) | 23 | 271 (9.8%) |
| icd11_ext_laterality (multi) | 6 | 2687 (97.4%) |
| icd11_ext_grading (multi) | 5 | 2679 (97.1%) |
| icd11_ext_staging | excluded | 0 examples in M1 biopsy data (per architecture §6.2) |

## 4. Test-only label audit (excluded from F1 per Q3)

Codes present in test but absent from the trainable vocabulary. These are excluded
from per-axis F1 computation but reported transparently here. Affected test rows
still receive credit for whatever they correctly predict on the trainable vocab.

| Axis | # test-only codes | # test rows affected | Codes |
|---|---|---|---|
| icd11_ext_histopath | 3 | 3 | XH4JA4, XH5ZH7, XH9Z29 |


## 5. Subgroup composition (trainable, for §11.3 stratified reporting)

| Subgroup | N | % of trainable |
|---|---|---|
| Batch 1 | 1,572 | 57.0% |
| Batch 2 | 1,187 | 43.0% |
| Templated reports | 441 | 16.0% |
| Free-text reports | 2,318 | 84.0% |
| Cancer primary | 2,723 | 98.7% |
| Non-cancer primary | 36 | 1.3% |
| NOS topography (C50.9) | 681 | 24.7% |
| Specific topography (not C50.9) | 1,956 | 70.9% |
| Rare stems (<10 train examples) | 18 stems | — |
| Rare morphology codes (<10 train examples) | 13 codes | — |

## 6. Auxiliary histology coverage by batch (E3 toggle gate)

| Field | Batch 1 (1572) | Batch 2 (1187) |
|---|---|---|
| nuclear_grade | 1163 (74%) | 1082 (91%) |
| tubular_score | 1244 (79%) | 1098 (93%) |
| mitosis_score | 1244 (79%) | 1096 (92%) |
| lvi | 1188 (76%) | 1006 (85%) |
| dcis_in_specimen | 920 (59%) | 613 (52%) |

Both batches show populated aux fields → E3 (auxiliary heads) is data-feasible.
Whether E3 ships is decided by the §12.5 acceptance bars (global lift + both
batch subgroups holding + lift on hard axes).

## 7. Text-length distribution (trainable, words)

| Format | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| section_tagged | 64.8 | 61 | 82 | 100 | 130 | 242 |
| concatenated | 59.2 | 55 | 77 | 93 | 122 | 237 |

Architecture §6.2 specifies 512-token truncation for Baheya. Word counts above
are an upper bound on token counts before WordPiece subword splitting. With
typical PubMedBERT subword inflation factor ~1.4–1.6×, the p99 of ~130 words
maps to ~195 tokens — comfortably within 512. Truncation impact for M1
will be confirmed by the truncation_audit script (Phase 0a).

## 8. CV fold acceptance report

5-fold, group-aware (`duplicate_group_id`), iterative multilabel stratified
on (icd11_stem ⊕ icdo3_morphology) one-hot blocks. Seed=42.

| Fold | Size | Batch1 % | IDC % | Unique stems present | Common stems missing |
|---|---|---|---|---|---|
| 0 | 550 | 56.36 | 72.91 | 20 | 0 / 16 |
| 1 | 540 | 57.96 | 75.00 | 21 | 0 / 16 |
| 2 | 548 | 57.12 | 73.18 | 20 | 0 / 16 |
| 3 | 554 | 56.86 | 75.27 | 21 | 0 / 16 |
| 4 | 567 | 56.61 | 75.13 | 21 | 0 / 16 |


**Acceptance bars (architecture §13.2 / Build Plan §3.3):**
- ✅ Zero duplicate-group leakage across folds.
- ✅ Common stems (≥5 trainable examples) present in every fold.
- ✅ Per-fold Batch 1 within ±5pp of overall (57.0%). Range: [56.4, 58.0].

## 9. Artifact files

| File | Purpose |
|---|---|
| m1_model_ready.parquet | The model-ready table. One row per record, columns described in §10. |
| label_vocab.json | All 10 axis vocabularies, frequency-descending. Built from trainable only. |
| cv_folds.csv | record_id → fold ∈ {0..4}, only for trainable rows. |
| split_verification_report.csv | Per-fold acceptance metrics (machine-readable). |
| data_card.md | This document. |
| CHECKSUMS.txt | SHA256 of all of the above. |

## 10. Parquet column dictionary

**Bookkeeping (NEVER passed to the model):**
- `record_id` (str): primary key, e.g. `BAH-P-0001`.
- `patient_id` (int): unique patient.
- `batch` (int): 1 or 2; subgroup-stratified eval.
- `split` (str): `trainable` or `test`.
- `fold` (int): 0..4 for trainable, -1 for test.
- `duplicate_group_id` (int): -1 = singleton; ≥0 = real near-duplicate cluster.
- `template_flag` (bool): stylistically templated Baheya report.
- `is_cancer_primary` (bool): non-cancer primaries are kept (deployment-realistic).
- `data_quality_flag` (str|None): `missing_icd11`, `missing_icdo3`, etc.
- `text_length_words` (int): word count of `text_combined` source.

**Model input (PASS to encoder):**
- `text_section_tagged` (str): architecture §4.4 format.
- `text_concatenated` (str): the curation-pipeline format from `text_combined`.

**Labels (PASS to loss):**
- `icdo3_topography` (str|None): single-pick or null. Mask in loss when null.
- `icdo3_morphology` (str|None): same.
- `icdo3_behavior` (str|None): same.
- `icdo3_grade` (str|None): same.
- `icdo3_laterality` (str|None): same.
- `icd11_stem` (str|None): same.
- `icd11_ext_anatomy` (list[str]): multi-hot. Empty list = no positive labels.
- `icd11_ext_histopath` (list[str]): same.
- `icd11_ext_laterality` (list[str]): same.
- `icd11_ext_grading` (list[str]): same.

**Auxiliary structured (PASS only when E3 toggle on):**
- `nuclear_grade`, `tubular_score`, `mitosis_score`, `lvi`, `dcis_in_specimen`.

## 11. Reproducibility

- Seed: 42.
- iterstrat 0.2.0 has a known `random_state` bug; we set both `np.random.seed`
  and Python's `random.seed` immediately before `mskf.split()`.
- All checksums in CHECKSUMS.txt are SHA-256 of the bytes on disk.
