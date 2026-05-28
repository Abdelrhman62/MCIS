# MCIS Final Archive — Locked Results & Checkpoints

**Created:** 2026-05-25  
**Repo:** `Abdelrhman62/MCIS` | **W&B:** `MCIS_AI`  
**Primary metric:** Mean F1 (mean of F1-Macro single-pick + F1-Micro multi-label)

---

## Checkpoints

| Folder | Experiment | Score | Best Epoch | Notes |
|---|---|---|---|---|
| `checkpoints/B0/` | Majority class baseline | 0.0702 ± 0.0011 | 8 | 5-fold, M4 CPU |
| `checkpoints/B2_plmicd/` | PLM-ICD flat LAAT baseline | 0.6431 ± 0.0125 | n/a | 5-fold, A100 |
| `checkpoints/E1/` | PubMedBERT no pretrain (canonical) | 0.6416 ± 0.0116 | 21 | 5-fold, A100 |
| `checkpoints/Phase1_TCGA_pubmedbert_25ep/` | Phase 1 Pretrain (PubMedBERT) | 0.7002 | 22 | TCGA-pretrain, A100 |
| `checkpoints/E2_baheya_from_tcga/` | Baheya from TCGA (Phase 1 5ep) | 0.6589 | 25 | Fold 0 (from slovenia runs) |
| `checkpoints/E2_segmentation_ablations/E2_fixed_overlap32/` | E2 Fixed Boundaries + 32 Overlap | 0.6585 | 27 | Fold 0 |
| `checkpoints/E2_segmentation_ablations/E2_sa_no_overlap/` | E2 Sentence-Aware | 0.6509 | 17 | Fold 0 |
| `checkpoints/E2_segmentation_ablations/E2_sa_overlap32/` | E2 Sentence-Aware + 32 Overlap | 0.6488 | 18 | Fold 0 |

> **Note:** B1 (TF-IDF) does not produce `.pt` checkpoints — results only.

---

## Results

| Folder | Experiment | Score | Status |
|---|---|---|---|
| `results/B0_majority_baseline/` | Majority class 5-fold | 0.0702 ± 0.0011 | ✅ Locked |
| `results/B1_tfidf_baseline/` | TF-IDF + LogReg 5-fold **(GATE = 0.6351)** | 0.6351 ± 0.0124 | ✅ Locked |
| `results/B2_plmicd_canonical/` | PLM-ICD flat LAAT 5-fold | 0.6431 ± 0.0125 | ✅ Canonical |
| `results/E1_pubmedbert_no_pretrain/` | PubMedBERT 5-fold (all fold JSONs + summary) | 0.6416 ± 0.0116 | ✅ Canonical |
| `results/E1_fmt_ablation/` | E1 text_concat vs section_tagged logs | E1=0.641, E1_fmt=0.630 | ✅ Done |
| `results/ext_val/` | External validation on TCGA-BRCA (E1 & E2) | E1=0.14, E2=0.19 | ✅ Done |
| `results/data_analysis/` | Truncation audit and unknown token reports | n/a | ✅ Done |

---

## Pending (will be added after completion)

- `checkpoints/E2_baheya_from_tcga/` — E2 with 25-epoch Phase 1 (pending)
- `results/E2_baheya_from_tcga/` — E2 5-fold results (pending)
- `results/ext_val_normalized/` — Ext-val with section normalizer + temperature (pending)
