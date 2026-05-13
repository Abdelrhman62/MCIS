# SEER Rare Augmentation — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `seer_rare_augmentation_model_ready/`
- **Built at:** 2026-05-07T13:42:35.665827+00:00
- **Purpose:** Synthetic template augmentation targeting Baheya M1 v2 rare-morphology codes.
  All templates carry `template_flag=True`, `coding_source='seer_template'`, `dataset='seer_augment'`.

## 2. Row counts
- **Total rows:** 619
- **Kept from v1 SEER (640 - 431 obsolete drop):** 431
- **Newly generated (4 v2-rare codes):** 188

## 3. Per-code distribution
| Morphology | Count | Source |
|---|---:|---|
| 8032 | 49 | kept from v1 |
| 8050 | 48 | kept from v1 |
| 8200 | 49 | new (joint-sampled from SEER reference) |
| 8211 | 48 | kept from v1 |
| 8453 | 47 | kept from v1 |
| 8480 | 41 | new (joint-sampled from SEER reference) |
| 8501 | 49 | kept from v1 |
| 8502 | 49 | new (joint-sampled from SEER reference) |
| 8509 | 49 | kept from v1 |
| 8540 | 47 | kept from v1 |
| 8550 | 49 | new (joint-sampled from SEER reference) |
| 8575 | 45 | kept from v1 |
| 9020 | 49 | kept from v1 |

## 4. v2 rare-code coverage
All 13 codes from Baheya M1 v2's `<10 train` rare list are covered:

`8032, 8050, 8200, 8211, 8453, 8480, 8501, 8502, 8509, 8540, 8550, 8575, 9020`

Removed from v1 because no longer rare in v2: `8010, 8140, 8503, 8504, 8522` (see step 1 stdout templates dropped — exact count in build log).

## 5. Schema
Canonical 34-column schema (matches Baheya M1, TCGA pretrain, TCGA-BRCA). See `MCIS_Dataset_Unification_Handoff.md` §2.1.

## 6. Encoding decisions
- **Tag scheme (text_section_tagged):** Literal Baheya format:
  ```
  [REPORT_TYPE] biopsy_report
  [DIAGNOSIS] {morphology}, {topography}, {laterality}, Grade {Roman}, {stage} stage.
  [SPECIMEN] tru-cut biopsy
  [LATERALITY] {Left|Right|Bilateral}     (only when non-null)
  ```
  No `[GRADE]` tag — grade is embedded in `[DIAGNOSIS]` per Baheya convention. `[MICROSCOPIC]` and `[CLINICAL_INFO]` omitted (no synthetic content for those).
- **`icdo3_laterality`:** L/R/B; raw English (`Left/Right/Bilateral`) preserved in `icdo3_laterality_raw`.
- **`icdo3_grade`:** string `'1'`–`'4'`; null = mask in loss.
- **`icdo3_behavior`:** all `'3'` (SEER breast malignant primary).
- **WHO fallback names:** Lookup table did not have entries for new codes 8200/8480/8502/8550. We used WHO ICD-O-3 standard names:
  - 8200 → Adenoid cystic carcinoma
  - 8480 → Mucinous adenocarcinoma
  - 8502 → Secretory carcinoma of breast
  - 8550 → Acinar cell carcinoma
- **ICD-11 columns:** All None / empty list. SEER has no ICD-11 mapping.
- **Aux columns:** All None (Baheya-only fields).
- **Splits/folds:** All rows have `split='train'`, `fold=-1`. Templates are training-time augmentation; never appear in validation or test.

## 7. Sample rendered templates
### Code 8032
```
[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Spindle cell carcinoma, right lower-outer quadrant of breast.
[SPECIMEN] tru-cut biopsy
[LATERALITY] Right
```

### Code 8050
```
[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Papillary carcinoma, NOS, left breast, Grade I, Localized stage.
[SPECIMEN] tru-cut biopsy
[LATERALITY] Left
```

### Code 8200
```
[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Adenoid cystic carcinoma, left upper-outer quadrant of breast, Grade I, Localized stage.
[SPECIMEN] tru-cut biopsy
[LATERALITY] Left
```

### Code 8211
```
[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Tubular adenocarcinoma, left upper-inner quadrant of breast, Grade I, Localized stage.
[SPECIMEN] tru-cut biopsy
[LATERALITY] Left
```

### Code 8453
```
[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Intraductal papillary-mucinous carcinoma, right central portion of breast, Grade II.
[SPECIMEN] tru-cut biopsy
[LATERALITY] Right
```

## 8. Hard rules
- All rows have `template_flag=True`. The dataloader can use this to apply different sampling weights or to exclude templates from certain training stages.
- `text_length_bert_tokens` is approximated as `1.3 * word_count`. Re-compute with the actual PubMedBERT tokenizer at training time if exact counts are needed.
- Templates are training-time only. No fold structure. No external validation use.
- Per architecture v6 §12.6, augmentation is subject to E5b ablation; if it doesn't lift rare-code F1 by ≥+1.0, templates are dropped from final pipeline.

## 9. Files
- `seer_rare_augmentation_model_ready.parquet` — model-ready data
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts

## 10. Provenance
- v1 source: `seer_rare_augmentation.csv` (640 templates, 11 cols, generated against Baheya M1 v1 rare list)
- New templates: joint-sampled from `seer_breast_reference.csv` (933,520 SEER 17 Registries 2000-2022 rows; per-code pools verified non-zero in `seer_breast_distributions.json`)
- Baheya v2 rare list: computed from `m1_model_ready.parquet` trainable split (2,759 rows), `<10 train` threshold per data card §5
- Seed: 42
