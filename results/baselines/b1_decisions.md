# B1 TF-IDF Baseline — Decisions & Findings

**Date:** 2026-05-11
**Script:** `scripts/run_tfidf_baseline.py`
**Latest run:** `results/baselines/tfidf_2026-05-11T221423Z.json`
**Companion to:** B0 majority baseline (`majority_2026-05-11T151836Z.json`)

---

## Summary

| Metric | B0 majority | B1 TF-IDF | Lift |
|---|---:|---:|---:|
| Early-stop (mean ± std) | 0.0702 ± 0.0011 | **0.6351 ± 0.0124** | +0.5649 |

B1 establishes the realistic floor. E1 (PubMedBERT) must clear B1 by
≥0.10 absolute on the early-stop metric, AND show meaningful lift on the
**lift-meaningful axes** (§3 below), to justify its compute cost.

---

## 1. Method

- TF-IDF: `ngram_range=(1,2)`, `max_features=30000`, `min_df=2`, lowercase.
  Fit per fold on train, transform val (no leakage).
- Single-pick axes (8): `LogisticRegression(class_weight='balanced',
  solver='lbfgs', max_iter=1000)`. Multinomial softmax — matches production
  masked-CE head theoretically.
- Multi-label axes (2): per-code binary LR (`solver='liblinear'`,
  `class_weight='balanced'`). Codes with <2 train positives → predict 0.
- Threshold tuning: global per axis, picked by max mean F1-Micro across
  folds on grid `[0.1..0.9]` step 0.1.
- 5-fold CV, same `cv_folds.csv` as production. Same metric module
  (`src.eval.metrics`) as the trainer.
- Text column: `text_section_tagged` (encoder's input per v6 §4.4).

---

## 2. Text-leak finding (architecture-by-design, NOT a bug)

Sample row from trainable split:

```
GRADE LABEL: 2
LAT  LABEL: L

[REPORT_TYPE] biopsy_report
[DIAGNOSIS] Lt. breast, Tru-cut biopsy:Invasive duct carcinoma, grade II.
[MICROSCOPIC] Invasive duct carcinoma grade II with marked desmoplastic stroma...
[CLINICAL_INFO] Lt. breast UQ suspicious lesion ( BIRADS V ).
[TUMOUR_TYPE] IDC
[SPECIMEN] tru cut biopsy
[LATERALITY] Left
```

`text_section_tagged` literally contains the structured-field values
verbatim: `[LATERALITY] Left`, `grade II`, `IDC`, `tru cut biopsy`.

**Consequence:** TF-IDF trivially copies these tokens to predict the
corresponding labels.

| Axis | B1 F1 | Why so high |
|---|---:|---|
| icdo3_laterality | 0.9065 | `[LATERALITY] Left` token directly maps to `L` |
| icdo3_grade | 0.9481 | `grade II` substring → label `2` |
| icd11_ext_grading | 0.7222 | Same source field as icdo3_grade |
| icdo3_behavior | 0.5625 | `Invasive` / `in situ` / `DCIS` tokens |

This is **how the production model also sees the data**. PubMedBERT will
match B1 on these axes by the same mechanism. **High F1 here ≠ encoder
adding value**; it reflects that the structured field is already in the
text input.

**Decision:** Do NOT rebuild `text_section_tagged` to strip these fields.
Per v6 §4.4 the section-tagged format is the locked encoder input. The
leak is recorded here transparently and judgement of E1 lift will weight
the **lift-meaningful axes** (§3) heavier than the saturated axes.

---

## 3. Lift-meaningful axes (where E1 must show value)

Axes where bag-of-ngrams cannot solve the task on its own:

| Axis | B1 F1 | Why bag-of-ngrams is weak |
|---|---:|---|
| icdo3_morphology | 0.4035 | K=24, requires morphology disambiguation |
| icd11_stem | 0.2901 | K=27, most heterogeneous axis, semantic reasoning needed |
| icd11_ext_anatomy | 0.6282 (F1-Micro) | post-coord anatomy, multi-axis |
| icd11_ext_histopath | 0.6811 (F1-Micro) | rare codes (most <10 train pos) |
| icd11_ext_laterality | 0.4250 | extension code requires WHO ECT mapping |
| icdo3_topography | 0.6659 | mostly lexical but C50.x sub-codes need quadrant reasoning |

**E1 success criteria (lift-meaningful subset):**

- icd11_stem: B1 = 0.29. E1 must hit ≥0.45 to claim PubMedBERT value.
- icdo3_morphology: B1 = 0.40. E1 must hit ≥0.55.
- icd11_ext_anatomy / histopath: B1 = 0.63 / 0.68 (F1-Micro). E1 should
  reach ≥0.70 each, but these are saturated by `class_weight='balanced'`
  threshold inflation (§4) — interpret with care.

**Saturated axes (B1 already ≥0.55) — do not over-index on E1 lift here:**
- icdo3_laterality (0.91), icdo3_grade (0.95), icd11_ext_grading (0.72),
  icdo3_behavior (0.56), icdo3_topography (0.67).

---

## 4. Threshold sweep finding

Initial grid `[0.1..0.5]` was boundary-pegged — both multi-label axes
peaked at t=0.5 with monotonically increasing sweep. Extended to
`[0.1..0.9]`. Peaks found:

| Axis | Locked t | F1-Micro at peak | Curve |
|---|---:|---:|---|
| icd11_ext_anatomy | 0.60 | 0.6282 | Peaks at 0.6, drops past |
| icd11_ext_histopath | 0.80 | 0.6811 | Peaks at 0.8, drops slightly at 0.9 |

**Why thresholds > 0.5:** `class_weight='balanced'` inflates positive-class
probas for rare codes (XH histopath codes mostly <10 train positives).
True posterior is small but the rebalancing pulls predicted proba up. The
optimal decision boundary therefore sits well above 0.5.

**Production note:** the deployment trainer uses **masked BCE without
class_weight rebalancing** for multi-label axes. Probabilities will be
better calibrated, so production optimal threshold likely sits in
[0.3, 0.5]. B1's t=0.6 / t=0.8 are not directly transferable as defaults.

---

## 5. Sanity floor for trainer

For any E* run, the **per-axis B1 F1** is the realistic floor. Any axis
where E* scores below B1 indicates one of:

- text input was changed and the trivial field-copy patterns were broken
- loss isn't flowing for that axis
- label encoding mismatch
- under-training (more epochs needed)

The **early-stop B1 floor of 0.6351** is what E1 must clear at convergence
(epoch ≥3 typically). If E1 early-stop fails to exceed 0.6351, do NOT
proceed to E2 — investigate trainer first.

---

## 6. Open items recorded

1. **Threshold rebalancing in E1 eval pipeline** — production heads
   compute F1-Micro at t=0.5 by default in `f1_multilabel`. Run a one-off
   threshold sweep on E1 fold 0 val predictions to confirm production
   optimal threshold matches the proba calibration of masked BCE.
2. **Rare-code subset evaluation** — neither B0 nor B1 reports F1 stratified
   by code-frequency band (rare <10, common ≥10). E1 should add this
   stratification per v6 §11 to show rare-code lift.
3. **Subgroup eval (Batch 1 vs Batch 2; templated vs free-text)** —
   B1 numbers are global. E1 needs §11.3 stratified reporting.

---

## 7. Runtime

~9 seconds wall on M4 CPU. TF-IDF + LR at this scale is essentially free.
No GPU needed for any B baseline.

---

## 8. Files

- Script: `scripts/run_tfidf_baseline.py`
- Output JSON: `results/baselines/tfidf_2026-05-11T221423Z.json`
- Output MD: `results/baselines/tfidf_2026-05-11T221423Z.md`
- This doc: `results/baselines/b1_decisions.md`
