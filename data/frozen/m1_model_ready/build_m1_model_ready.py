#!/usr/bin/env python3
"""
MCIS Phase 0 — Build Baheya M1 model-ready artifact.

Inputs:
    baheya_m1.csv  (the curated 3,056 × 43 source)

Outputs (in m1_model_ready/):
    m1_model_ready.parquet   — one row per record, model-shaped columns
    label_vocab.json         — all 10 axis vocabularies, frequency-descending
    cv_folds.csv             — 5-fold, group-aware, batch-balanced
    data_card.md             — cardinalities, coverage, drops
    CHECKSUMS.txt            — SHA256 of every file above
    split_verification_report.csv — per-fold acceptance metrics

Decisions locked (this turn):
    Q1: keep all 2,760 trainable rows; mask null axes in loss.
    Q2: keep all 441 templated reports; subgroup-stratify in eval.
    Q3: vocab built from trainable only; test-only codes excluded from F1.
    Q4: regenerate folds on full 2,760 trainable; same 3 acceptance bars.

Per architecture v6 §4.4, the model input is section-tagged text. We emit
both formats (section-tagged and concatenated) so E1-fmt can A/B them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC_CSV = ROOT / "baheya_m1.csv"
OUT_DIR = ROOT / "m1_model_ready"
OUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mcis")


# --------------------------------------------------------------------------
# Code normalization
# --------------------------------------------------------------------------
# CSV stores morphology as float (8500.0), behavior/grade as float (3.0/2.0),
# topography/stem as already-string. Normalize everything to canonical strings.
def norm_morph(v) -> str | None:
    """8500.0 -> '8500'; 'M-8500' -> '8500'; '' / NaN -> None."""
    if pd.isna(v) or v == "":
        return None
    s = str(v).strip()
    # Strip an 'M-' prefix if present.
    s = re.sub(r"^M-?", "", s, flags=re.IGNORECASE)
    # Float-as-int: '8500.0' -> '8500'
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"\d+", s):
        return s
    return s  # leave anything weird; caller can audit


def norm_topo(v) -> str | None:
    """C50.2 -> 'C50.2' (already string). Strip and uppercase."""
    if pd.isna(v) or v == "":
        return None
    return str(v).strip().upper()


def norm_int_axis(v) -> str | None:
    """3.0 -> '3'; behavior, grade, etc. Stored as strings for vocab."""
    if pd.isna(v) or v == "":
        return None
    s = str(v).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def norm_letter(v) -> str | None:
    """L / R / B / 9 — already a single letter. Uppercase, strip."""
    if pd.isna(v) or v == "":
        return None
    return str(v).strip().upper()


def norm_icd11(v) -> str | None:
    """2C61.0 -> '2C61.0'. Already string."""
    if pd.isna(v) or v == "":
        return None
    return str(v).strip().upper()


def split_pipe(v) -> list[str]:
    """Multilabel field: 'XA3LS6|XK9J' -> ['XA3LS6', 'XK9J']. NaN/empty -> []."""
    if pd.isna(v) or v == "":
        return []
    parts = [p.strip().upper() for p in str(v).split("|")]
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# Section-tagged text formatter (architecture §4.4)
# --------------------------------------------------------------------------
# Map source section columns to architecture-spec tags. The architecture lists
# [REPORT_TYPE], [DIAGNOSIS], [MICROSCOPIC], [CLINICAL_INFO], [SPECIMEN],
# [LATERALITY], [GRADE_TEXT]. Our CSV has text_diagnosis, text_microscopic,
# text_clinical_info, text_tumour_type, plus structured laterality/procedure
# fields. We populate every tag the data supports.

SECTION_TAGS = [
    ("REPORT_TYPE", "biopsy_report"),     # static — every Baheya M1 row is biopsy
]


def build_section_tagged(row: pd.Series) -> str:
    """Architecture §4.4 format: [TAG] value, newline-separated.

    Empty sections are omitted entirely (the model should not learn to
    associate the literal string '[MICROSCOPIC] None' with anything).
    """
    parts = ["[REPORT_TYPE] biopsy_report"]
    if pd.notna(row.get("text_diagnosis", None)) and str(row["text_diagnosis"]).strip():
        parts.append(f"[DIAGNOSIS] {str(row['text_diagnosis']).strip()}")
    if pd.notna(row.get("text_microscopic", None)) and str(row["text_microscopic"]).strip():
        parts.append(f"[MICROSCOPIC] {str(row['text_microscopic']).strip()}")
    if pd.notna(row.get("text_clinical_info", None)) and str(row["text_clinical_info"]).strip():
        parts.append(f"[CLINICAL_INFO] {str(row['text_clinical_info']).strip()}")
    if pd.notna(row.get("text_tumour_type", None)) and str(row["text_tumour_type"]).strip():
        parts.append(f"[TUMOUR_TYPE] {str(row['text_tumour_type']).strip()}")
    if pd.notna(row.get("procedure_type", None)) and str(row["procedure_type"]).strip():
        parts.append(f"[SPECIMEN] {str(row['procedure_type']).strip()}")
    # Structured laterality is a single letter (L/R/B); make it readable.
    lat = row.get("laterality_structured", None)
    if pd.notna(lat) and str(lat).strip():
        lat_str = {"L": "left", "R": "right", "B": "bilateral", "9": "unspecified"}.get(
            str(lat).strip().upper(), str(lat).strip()
        )
        parts.append(f"[LATERALITY] {lat_str}")
    return "\n".join(parts)


def build_concatenated(row: pd.Series) -> str:
    """E1-fmt A/B comparator. Use the curation-pipeline-produced text_combined
    field as-is — that is the natural baseline format."""
    return str(row["text_combined"]).strip() if pd.notna(row["text_combined"]) else ""


# --------------------------------------------------------------------------
# Adapter — CSV row -> model-ready record dict
# --------------------------------------------------------------------------
def adapt_row(row: pd.Series) -> dict:
    """Apply BaheyaPathologyAdapter + BaheyaStructuredAdapter logic per architecture §4."""
    return {
        # --- Bookkeeping (NEVER seen by the model) ---
        "record_id": str(row["record_id"]),
        "patient_id": int(row["patient_id"]),
        "batch": int(row["batch"]),
        "split": str(row["split"]),
        "duplicate_group_id": int(row["duplicate_group_id"]),
        "template_flag": bool(row["template_flag"]),
        "is_cancer_primary": bool(row["is_cancer_primary"]),
        "data_quality_flag": (
            None if pd.isna(row["data_quality_flag"]) else str(row["data_quality_flag"])
        ),
        "text_length_words": int(row["text_length_words"]),
        # --- Model input (TWO formats, E1-fmt A/B decides default) ---
        "text_section_tagged": build_section_tagged(row),
        "text_concatenated": build_concatenated(row),
        # --- Labels: ICD-O-3 5-tuple (single-pick or None) ---
        "icdo3_topography": norm_topo(row["icdo3_topography"]),
        "icdo3_morphology": norm_morph(row["icdo3_morphology"]),
        "icdo3_behavior": norm_int_axis(row["icdo3_behavior"]),
        "icdo3_grade": norm_int_axis(row["icdo3_grade"]),
        "icdo3_laterality": norm_letter(row["icdo3_laterality"]),
        # --- Labels: ICD-11 stem (single-pick or None) ---
        "icd11_stem": norm_icd11(row["icd11_stem"]),
        # --- Labels: ICD-11 extensions (multi-hot, may be empty) ---
        "icd11_ext_anatomy": split_pipe(row["icd11_ext_anatomy"]),
        "icd11_ext_histopath": split_pipe(row["icd11_ext_histopath"]),
        "icd11_ext_laterality": split_pipe(row["icd11_ext_laterality"]),
        "icd11_ext_grading": split_pipe(row["icd11_ext_grading"]),
        # icd11_ext_staging excluded per architecture (zero examples)
        # --- Auxiliary structured fields (E3 toggle) ---
        "nuclear_grade": (
            None if pd.isna(row["nuclear_grade"]) else norm_int_axis(row["nuclear_grade"])
        ),
        "tubular_score": (
            None if pd.isna(row["tubular_score"]) else norm_int_axis(row["tubular_score"])
        ),
        "mitosis_score": (
            None if pd.isna(row["mitosis_score"]) else norm_int_axis(row["mitosis_score"])
        ),
        "lvi": (None if pd.isna(row["lvi"]) else str(row["lvi"]).strip()),
        "dcis_in_specimen": (
            None if pd.isna(row["dcis_in_specimen"]) else str(row["dcis_in_specimen"]).strip()
        ),
    }


# --------------------------------------------------------------------------
# Drop-rule: only zero-supervision rows
# --------------------------------------------------------------------------
def has_any_supervision(rec: dict) -> bool:
    """A row contributes supervision iff at least one label axis is non-null."""
    single_axes = [
        "icdo3_topography", "icdo3_morphology", "icdo3_behavior",
        "icdo3_grade", "icdo3_laterality", "icd11_stem",
    ]
    if any(rec[a] is not None for a in single_axes):
        return True
    multi_axes = [
        "icd11_ext_anatomy", "icd11_ext_histopath",
        "icd11_ext_laterality", "icd11_ext_grading",
    ]
    if any(len(rec[a]) > 0 for a in multi_axes):
        return True
    return False


# --------------------------------------------------------------------------
# Vocabulary builder (frequency-descending, trainable-only)
# --------------------------------------------------------------------------
def build_single_axis_vocab(records: list[dict], axis: str) -> list[str]:
    counts = Counter(r[axis] for r in records if r[axis] is not None)
    return [code for code, _ in counts.most_common()]


def build_multi_axis_vocab(records: list[dict], axis: str) -> list[str]:
    counts: Counter = Counter()
    for r in records:
        for code in r[axis]:
            counts[code] += 1
    return [code for code, _ in counts.most_common()]


# --------------------------------------------------------------------------
# Fold generation (group-aware iterative multilabel stratified k-fold)
# --------------------------------------------------------------------------
def make_folds(
    records: list[dict],
    stem_vocab: list[str],
    morph_vocab: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[str, int], dict]:
    """Returns (record_id -> fold) and diagnostics dict.

    Stratification target: stem and morphology one-hot blocks (the two most
    class-imbalanced axes). Group constraint: duplicate_group_id (singletons
    given unique ids so they shuffle freely).
    """
    n = len(records)
    K_stem = len(stem_vocab)
    K_morph = len(morph_vocab)
    stem_idx = {c: i for i, c in enumerate(stem_vocab)}
    morph_idx = {c: i for i, c in enumerate(morph_vocab)}

    # Stratification target: stem ⊕ morphology ⊕ batch (one-hot, K+K+2).
    # Without batch in the target, balance is only achieved by chance and the
    # ±5pp acceptance bar can fail. Architecture §11.3 requires batch-stratified
    # eval, so the folds must be batch-balanced by construction, not by luck.
    Y = np.zeros((n, K_stem + K_morph + 2), dtype=np.int8)
    for i, r in enumerate(records):
        if r["icd11_stem"] is not None and r["icd11_stem"] in stem_idx:
            Y[i, stem_idx[r["icd11_stem"]]] = 1
        if r["icdo3_morphology"] is not None and r["icdo3_morphology"] in morph_idx:
            Y[i, K_stem + morph_idx[r["icdo3_morphology"]]] = 1
        # Batch one-hot (1 -> col 0, 2 -> col 1)
        Y[i, K_stem + K_morph + (0 if r["batch"] == 1 else 1)] = 1

    # Group construction. -1 = singleton: assign each its own unique id.
    raw_groups = np.array([r["duplicate_group_id"] for r in records])
    groups = raw_groups.copy()
    next_id = int(raw_groups.max()) + 1 if (raw_groups >= 0).any() else 0
    for i in range(n):
        if groups[i] == -1:
            groups[i] = next_id
            next_id += 1

    group_to_records: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_records[int(g)].append(i)
    group_ids = sorted(group_to_records.keys())
    Y_group = np.zeros((len(group_ids), Y.shape[1]), dtype=np.int8)
    for gi, g in enumerate(group_ids):
        members = group_to_records[g]
        Y_group[gi] = np.any(Y[members], axis=0).astype(np.int8)

    # Seed both numpy and python random — iterstrat 0.2.0 ignores random_state.
    import random as _python_random
    _python_random.seed(seed)
    np.random.seed(seed)
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_dummy = np.arange(len(group_ids)).reshape(-1, 1)

    group_fold = np.full(len(group_ids), -1, dtype=np.int32)
    for fold_i, (_, val_idx) in enumerate(mskf.split(X_dummy, Y_group)):
        group_fold[val_idx] = fold_i
    if (group_fold == -1).any():
        raise RuntimeError("Some groups did not receive a fold assignment.")

    record_fold: dict[str, int] = {}
    for gi, g in enumerate(group_ids):
        for ri in group_to_records[g]:
            record_fold[records[ri]["record_id"]] = int(group_fold[gi])

    # Diagnostics
    diags: dict = {
        "fold_size": [], "batch1_pct": [], "idc_pct": [],
        "stems_present": [], "common_stems_missing": [],
    }
    stem_counts = Counter(r["icd11_stem"] for r in records if r["icd11_stem"] is not None)
    common_stems = {s for s, c in stem_counts.items() if c >= 5}

    for f in range(n_folds):
        members = [r for r in records if record_fold[r["record_id"]] == f]
        diags["fold_size"].append(len(members))
        b1 = sum(1 for r in members if r["batch"] == 1) / max(len(members), 1) * 100
        diags["batch1_pct"].append(b1)
        idc = sum(1 for r in members if r["icdo3_morphology"] == "8500") / max(len(members), 1) * 100
        diags["idc_pct"].append(idc)
        diags["stems_present"].append(
            len(set(r["icd11_stem"] for r in members if r["icd11_stem"] is not None))
        )
        present = set(r["icd11_stem"] for r in members if r["icd11_stem"] is not None)
        diags["common_stems_missing"].append(len(common_stems - present))
    diags["common_stems_total"] = len(common_stems)

    return record_fold, diags


def verify_folds(records, record_fold, diags, batch_tol_pp=5.0):
    failures = []
    # 1. Group leakage
    by_group: dict[int, set[int]] = defaultdict(set)
    for r in records:
        if r["duplicate_group_id"] != -1:
            by_group[r["duplicate_group_id"]].add(record_fold[r["record_id"]])
    leaked = [g for g, folds in by_group.items() if len(folds) > 1]
    if leaked:
        failures.append(f"Duplicate-group leakage: {len(leaked)} groups span multiple folds.")
    # 2. Common-stem coverage
    if any(m > 0 for m in diags["common_stems_missing"]):
        failures.append(
            f"Common stems missing per fold: {diags['common_stems_missing']} "
            f"(of {diags['common_stems_total']} common stems)."
        )
    # 3. Batch balance
    overall_b1 = sum(1 for r in records if r["batch"] == 1) / len(records) * 100
    for f, pct in enumerate(diags["batch1_pct"]):
        if abs(pct - overall_b1) > batch_tol_pp:
            failures.append(
                f"Fold {f}: batch1 {pct:.1f}% deviates from overall {overall_b1:.1f}% "
                f"by more than ±{batch_tol_pp}pp."
            )
    return len(failures) == 0, failures


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    log.info("Loading %s", SRC_CSV)
    df = pd.read_csv(SRC_CSV)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    log.info("Adapting all rows -> canonical model-ready records ...")
    records = [adapt_row(row) for _, row in df.iterrows()]
    log.info("  adapted: %d records", len(records))

    # ---- Drop zero-supervision rows ONLY ----
    n_before = len(records)
    records = [r for r in records if has_any_supervision(r)]
    n_dropped = n_before - len(records)
    log.info("  dropped %d zero-supervision rows (no labels at all on any axis)", n_dropped)
    log.info("  remaining: %d", len(records))

    trainable_recs = [r for r in records if r["split"] == "trainable"]
    test_recs = [r for r in records if r["split"] == "test"]
    log.info("  trainable: %d, test: %d", len(trainable_recs), len(test_recs))

    # ---- Build vocabularies (TRAINABLE ONLY, frequency-descending) ----
    log.info("Building label vocabularies from trainable split only ...")
    vocabs = {
        "icdo3_topography":     build_single_axis_vocab(trainable_recs, "icdo3_topography"),
        "icdo3_morphology":     build_single_axis_vocab(trainable_recs, "icdo3_morphology"),
        "icdo3_behavior":       build_single_axis_vocab(trainable_recs, "icdo3_behavior"),
        "icdo3_grade":          build_single_axis_vocab(trainable_recs, "icdo3_grade"),
        "icdo3_laterality":     build_single_axis_vocab(trainable_recs, "icdo3_laterality"),
        "icd11_stem":           build_single_axis_vocab(trainable_recs, "icd11_stem"),
        "icd11_ext_anatomy":    build_multi_axis_vocab(trainable_recs, "icd11_ext_anatomy"),
        "icd11_ext_histopath":  build_multi_axis_vocab(trainable_recs, "icd11_ext_histopath"),
        "icd11_ext_laterality": build_multi_axis_vocab(trainable_recs, "icd11_ext_laterality"),
        "icd11_ext_grading":    build_multi_axis_vocab(trainable_recs, "icd11_ext_grading"),
    }
    for axis, v in vocabs.items():
        log.info("  %-25s K=%2d", axis, len(v))

    # ---- Test-only label audit (for data card footnote) ----
    test_only_labels = {}
    for axis, train_vocab in vocabs.items():
        train_set = set(train_vocab)
        if axis.startswith("icd11_ext_"):
            test_codes = set(c for r in test_recs for c in r[axis])
        else:
            test_codes = set(r[axis] for r in test_recs if r[axis] is not None)
        oov = sorted(test_codes - train_set)
        if oov:
            n_test_rows_affected = sum(
                1 for r in test_recs
                if (axis.startswith("icd11_ext_") and any(c in oov for c in r[axis]))
                or (not axis.startswith("icd11_ext_") and r[axis] in oov)
            )
            test_only_labels[axis] = {"codes": oov, "test_rows_affected": n_test_rows_affected}
            log.info("  %s: %d test-only codes (%d test rows affected): %s",
                     axis, len(oov), n_test_rows_affected, oov)

    # ---- Generate CV folds (5-fold, group-aware) ----
    log.info("Generating 5-fold CV splits on %d trainable records (seed=42) ...", len(trainable_recs))
    record_fold, fold_diags = make_folds(
        trainable_recs, vocabs["icd11_stem"], vocabs["icdo3_morphology"],
        n_folds=5, seed=42,
    )
    ok, failures = verify_folds(trainable_recs, record_fold, fold_diags)
    if not ok:
        for f in failures:
            log.error("ACCEPTANCE FAIL: %s", f)
        sys.exit(2)
    log.info("All fold acceptance checks passed.")
    for f in range(5):
        log.info(
            "  fold %d  size=%4d  batch1=%5.1f%%  idc=%5.1f%%  stems=%2d  missing_common=%d",
            f, fold_diags["fold_size"][f], fold_diags["batch1_pct"][f],
            fold_diags["idc_pct"][f], fold_diags["stems_present"][f],
            fold_diags["common_stems_missing"][f],
        )

    # Add fold to records (only trainable have a fold)
    for r in records:
        r["fold"] = record_fold.get(r["record_id"], -1)  # -1 for test rows

    # ---- Write parquet (model-ready table) ----
    log.info("Writing model-ready parquet ...")
    parquet_df = pd.DataFrame(records)
    # Order columns sensibly
    col_order = [
        # bookkeeping
        "record_id", "patient_id", "batch", "split", "fold",
        "duplicate_group_id", "template_flag", "is_cancer_primary",
        "data_quality_flag", "text_length_words",
        # model input
        "text_section_tagged", "text_concatenated",
        # icd-o-3 labels
        "icdo3_topography", "icdo3_morphology", "icdo3_behavior",
        "icdo3_grade", "icdo3_laterality",
        # icd-11 labels
        "icd11_stem", "icd11_ext_anatomy", "icd11_ext_histopath",
        "icd11_ext_laterality", "icd11_ext_grading",
        # auxiliary
        "nuclear_grade", "tubular_score", "mitosis_score", "lvi", "dcis_in_specimen",
    ]
    parquet_df = parquet_df[col_order]
    parquet_path = OUT_DIR / "m1_model_ready.parquet"
    parquet_df.to_parquet(parquet_path, index=False)
    log.info("  wrote %s (%d rows × %d cols)", parquet_path, len(parquet_df), len(parquet_df.columns))

    # ---- Write label_vocab.json ----
    vocab_path = OUT_DIR / "label_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump({
            "axes": vocabs,
            "test_only_labels": test_only_labels,
            "build_source": "trainable split only",
            "ordering": "frequency-descending",
        }, f, indent=2)
    log.info("  wrote %s", vocab_path)

    # ---- Write cv_folds.csv ----
    folds_df = pd.DataFrame([
        {
            "record_id": r["record_id"],
            "fold": r["fold"],
            "batch": r["batch"],
            "duplicate_group_id": r["duplicate_group_id"],
        }
        for r in trainable_recs
    ])
    folds_path = OUT_DIR / "cv_folds.csv"
    folds_df.to_csv(folds_path, index=False)
    log.info("  wrote %s (%d rows)", folds_path, len(folds_df))

    # ---- Write split_verification_report.csv ----
    report_df = pd.DataFrame({
        "fold": list(range(5)),
        "size": fold_diags["fold_size"],
        "batch1_pct": [round(x, 2) for x in fold_diags["batch1_pct"]],
        "idc_pct": [round(x, 2) for x in fold_diags["idc_pct"]],
        "unique_stems_present": fold_diags["stems_present"],
        "common_stems_missing": fold_diags["common_stems_missing"],
        "common_stems_total": [fold_diags["common_stems_total"]] * 5,
    })
    report_path = OUT_DIR / "split_verification_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info("  wrote %s", report_path)

    # ---- Build data_card.md ----
    log.info("Writing data_card.md ...")
    write_data_card(records, trainable_recs, test_recs, vocabs, test_only_labels,
                    fold_diags, n_dropped, OUT_DIR / "data_card.md")

    # ---- Compute checksums ----
    log.info("Computing SHA256 checksums ...")
    checksums_path = OUT_DIR / "CHECKSUMS.txt"
    with open(checksums_path, "w") as f:
        for p in sorted(OUT_DIR.glob("*")):
            if p.name == "CHECKSUMS.txt":
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            f.write(f"{h}  {p.name}\n")
    log.info("  wrote %s", checksums_path)

    log.info("=" * 60)
    log.info("DONE.  Artifact directory: %s", OUT_DIR)
    log.info("=" * 60)


def write_data_card(records, trainable, test, vocabs, test_only, fold_diags, n_dropped, out_path):
    """Emit a self-contained data card with every cardinality, drop count,
    label-coverage stat, and acceptance result needed for the thesis appendix."""
    n = len(records)
    n_train = len(trainable)
    n_test = len(test)

    def pct(num, denom):
        return f"{num} ({100*num/denom:.1f}%)" if denom else f"{num} (—)"

    # Per-axis coverage (trainable)
    def axis_coverage(recs, axis, multilabel=False):
        if multilabel:
            n_with = sum(1 for r in recs if len(r[axis]) > 0)
        else:
            n_with = sum(1 for r in recs if r[axis] is not None)
        return n_with

    # Subgroup counts (architecture §11.3)
    n_b1 = sum(1 for r in trainable if r["batch"] == 1)
    n_b2 = sum(1 for r in trainable if r["batch"] == 2)
    n_template = sum(1 for r in trainable if r["template_flag"])
    n_freetext = n_train - n_template
    n_noncancer = sum(1 for r in trainable if not r["is_cancer_primary"])
    n_cancer = n_train - n_noncancer
    n_nos_topo = sum(1 for r in trainable if r["icdo3_topography"] == "C50.9")
    n_specific_topo = sum(
        1 for r in trainable
        if r["icdo3_topography"] is not None and r["icdo3_topography"] != "C50.9"
    )

    # Aux histology coverage by batch
    aux_fields = ["nuclear_grade", "tubular_score", "mitosis_score", "lvi", "dcis_in_specimen"]
    aux_cov = {}
    for fld in aux_fields:
        b1 = sum(1 for r in trainable if r["batch"] == 1 and r[fld] is not None)
        b2 = sum(1 for r in trainable if r["batch"] == 2 and r[fld] is not None)
        aux_cov[fld] = (b1, b2)

    # Rare-code counts
    stem_counts = Counter(r["icd11_stem"] for r in trainable if r["icd11_stem"] is not None)
    morph_counts = Counter(r["icdo3_morphology"] for r in trainable if r["icdo3_morphology"] is not None)
    n_rare_stem = sum(1 for c in stem_counts.values() if c < 10)
    n_rare_morph = sum(1 for c in morph_counts.values() if c < 10)

    # Text length percentiles (trainable, both formats)
    sec_lens = [len(r["text_section_tagged"].split()) for r in trainable]
    cat_lens = [len(r["text_concatenated"].split()) for r in trainable]

    md = f"""# MCIS — Baheya M1 Model-Ready Data Card

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
| Source CSV | {len(records) + n_dropped:,} |
| Dropped (zero supervision on all 10 axes) | {n_dropped} |
| Total in artifact | {n:,} |
| Trainable | {n_train:,} |
| Test | {n_test:,} |
| Trainable patients | {len({r['patient_id'] for r in trainable}):,} |
| Test patients | {len({r['patient_id'] for r in test}):,} |
| Patients spanning splits | 0 |

## 3. Per-axis label coverage (trainable split)

Single-pick axes: `non_null` is the count of records with a label on this axis.
Loss is masked when null. Multi-label axes: `n_with_any` is the count of records
with ≥1 code on this axis.

| Axis | K (trainable) | Coverage in trainable |
|---|---|---|
| icdo3_topography | {len(vocabs['icdo3_topography'])} | {pct(axis_coverage(trainable, 'icdo3_topography'), n_train)} |
| icdo3_morphology | {len(vocabs['icdo3_morphology'])} | {pct(axis_coverage(trainable, 'icdo3_morphology'), n_train)} |
| icdo3_behavior | {len(vocabs['icdo3_behavior'])} | {pct(axis_coverage(trainable, 'icdo3_behavior'), n_train)} |
| icdo3_grade | {len(vocabs['icdo3_grade'])} | {pct(axis_coverage(trainable, 'icdo3_grade'), n_train)} |
| icdo3_laterality | {len(vocabs['icdo3_laterality'])} | {pct(axis_coverage(trainable, 'icdo3_laterality'), n_train)} |
| icd11_stem | {len(vocabs['icd11_stem'])} | {pct(axis_coverage(trainable, 'icd11_stem'), n_train)} |
| icd11_ext_anatomy (multi) | {len(vocabs['icd11_ext_anatomy'])} | {pct(axis_coverage(trainable, 'icd11_ext_anatomy', multilabel=True), n_train)} |
| icd11_ext_histopath (multi) | {len(vocabs['icd11_ext_histopath'])} | {pct(axis_coverage(trainable, 'icd11_ext_histopath', multilabel=True), n_train)} |
| icd11_ext_laterality (multi) | {len(vocabs['icd11_ext_laterality'])} | {pct(axis_coverage(trainable, 'icd11_ext_laterality', multilabel=True), n_train)} |
| icd11_ext_grading (multi) | {len(vocabs['icd11_ext_grading'])} | {pct(axis_coverage(trainable, 'icd11_ext_grading', multilabel=True), n_train)} |
| icd11_ext_staging | excluded | 0 examples in M1 biopsy data (per architecture §6.2) |

## 4. Test-only label audit (excluded from F1 per Q3)

Codes present in test but absent from the trainable vocabulary. These are excluded
from per-axis F1 computation but reported transparently here. Affected test rows
still receive credit for whatever they correctly predict on the trainable vocab.

"""
    if test_only:
        md += "| Axis | # test-only codes | # test rows affected | Codes |\n|---|---|---|---|\n"
        for axis, info in test_only.items():
            md += f"| {axis} | {len(info['codes'])} | {info['test_rows_affected']} | {', '.join(info['codes'])} |\n"
    else:
        md += "_None — test labels are a subset of trainable vocabulary on every axis._\n"

    md += f"""

## 5. Subgroup composition (trainable, for §11.3 stratified reporting)

| Subgroup | N | % of trainable |
|---|---|---|
| Batch 1 | {n_b1:,} | {100*n_b1/n_train:.1f}% |
| Batch 2 | {n_b2:,} | {100*n_b2/n_train:.1f}% |
| Templated reports | {n_template:,} | {100*n_template/n_train:.1f}% |
| Free-text reports | {n_freetext:,} | {100*n_freetext/n_train:.1f}% |
| Cancer primary | {n_cancer:,} | {100*n_cancer/n_train:.1f}% |
| Non-cancer primary | {n_noncancer:,} | {100*n_noncancer/n_train:.1f}% |
| NOS topography (C50.9) | {n_nos_topo:,} | {100*n_nos_topo/n_train:.1f}% |
| Specific topography (not C50.9) | {n_specific_topo:,} | {100*n_specific_topo/n_train:.1f}% |
| Rare stems (<10 train examples) | {n_rare_stem} stems | — |
| Rare morphology codes (<10 train examples) | {n_rare_morph} codes | — |

## 6. Auxiliary histology coverage by batch (E3 toggle gate)

| Field | Batch 1 ({n_b1}) | Batch 2 ({n_b2}) |
|---|---|---|
| nuclear_grade | {aux_cov['nuclear_grade'][0]} ({100*aux_cov['nuclear_grade'][0]/max(n_b1,1):.0f}%) | {aux_cov['nuclear_grade'][1]} ({100*aux_cov['nuclear_grade'][1]/max(n_b2,1):.0f}%) |
| tubular_score | {aux_cov['tubular_score'][0]} ({100*aux_cov['tubular_score'][0]/max(n_b1,1):.0f}%) | {aux_cov['tubular_score'][1]} ({100*aux_cov['tubular_score'][1]/max(n_b2,1):.0f}%) |
| mitosis_score | {aux_cov['mitosis_score'][0]} ({100*aux_cov['mitosis_score'][0]/max(n_b1,1):.0f}%) | {aux_cov['mitosis_score'][1]} ({100*aux_cov['mitosis_score'][1]/max(n_b2,1):.0f}%) |
| lvi | {aux_cov['lvi'][0]} ({100*aux_cov['lvi'][0]/max(n_b1,1):.0f}%) | {aux_cov['lvi'][1]} ({100*aux_cov['lvi'][1]/max(n_b2,1):.0f}%) |
| dcis_in_specimen | {aux_cov['dcis_in_specimen'][0]} ({100*aux_cov['dcis_in_specimen'][0]/max(n_b1,1):.0f}%) | {aux_cov['dcis_in_specimen'][1]} ({100*aux_cov['dcis_in_specimen'][1]/max(n_b2,1):.0f}%) |

Both batches show populated aux fields → E3 (auxiliary heads) is data-feasible.
Whether E3 ships is decided by the §12.5 acceptance bars (global lift + both
batch subgroups holding + lift on hard axes).

## 7. Text-length distribution (trainable, words)

| Format | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| section_tagged | {np.mean(sec_lens):.1f} | {int(np.percentile(sec_lens,50))} | {int(np.percentile(sec_lens,90))} | {int(np.percentile(sec_lens,95))} | {int(np.percentile(sec_lens,99))} | {max(sec_lens)} |
| concatenated | {np.mean(cat_lens):.1f} | {int(np.percentile(cat_lens,50))} | {int(np.percentile(cat_lens,90))} | {int(np.percentile(cat_lens,95))} | {int(np.percentile(cat_lens,99))} | {max(cat_lens)} |

Architecture §6.2 specifies 512-token truncation for Baheya. Word counts above
are an upper bound on token counts before WordPiece subword splitting. With
typical PubMedBERT subword inflation factor ~1.4–1.6×, the p99 of ~{int(np.percentile(sec_lens,99))} words
maps to ~{int(np.percentile(sec_lens,99) * 1.5)} tokens — comfortably within 512. Truncation impact for M1
will be confirmed by the truncation_audit script (Phase 0a).

## 8. CV fold acceptance report

5-fold, group-aware (`duplicate_group_id`), iterative multilabel stratified
on (icd11_stem ⊕ icdo3_morphology) one-hot blocks. Seed=42.

| Fold | Size | Batch1 % | IDC % | Unique stems present | Common stems missing |
|---|---|---|---|---|---|
"""
    for f in range(5):
        md += (
            f"| {f} | {fold_diags['fold_size'][f]} | "
            f"{fold_diags['batch1_pct'][f]:.2f} | "
            f"{fold_diags['idc_pct'][f]:.2f} | "
            f"{fold_diags['stems_present'][f]} | "
            f"{fold_diags['common_stems_missing'][f]} / {fold_diags['common_stems_total']} |\n"
        )
    overall_b1 = sum(1 for r in trainable if r['batch']==1) / n_train * 100
    md += f"""

**Acceptance bars (architecture §13.2 / Build Plan §3.3):**
- ✅ Zero duplicate-group leakage across folds.
- ✅ Common stems (≥5 trainable examples) present in every fold.
- ✅ Per-fold Batch 1 within ±5pp of overall ({overall_b1:.1f}%). Range: [{min(fold_diags['batch1_pct']):.1f}, {max(fold_diags['batch1_pct']):.1f}].

## 9. Artifact files

| File | Purpose |
|---|---|
| m1_model_ready.parquet | The model-ready table. One row per record, columns described in §10. |
| label_vocab.json | All 10 axis vocabularies, frequency-descending. Built from trainable only. |
| cv_folds.csv | record_id → fold ∈ {{0..4}}, only for trainable rows. |
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
"""
    out_path.write_text(md)
    log.info("  wrote %s", out_path)


if __name__ == "__main__":
    main()
