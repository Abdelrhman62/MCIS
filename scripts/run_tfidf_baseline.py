#!/usr/bin/env python3
"""TF-IDF + Logistic Regression baseline for M1 (B1) — realistic floor.

Architecture v6 §13.2 Phase 0a deliverable. Companion to B0 majority baseline.
Where B0 establishes the absolute sanity floor (~0.07 early-stop), B1
establishes the *realistic* floor PubMedBERT must clear by a substantial
margin to justify its compute cost. If E1 doesn't beat B1 by >0.10 on the
early-stop metric, the encoder is not adding meaningful language priors.

What this script does:
    1. Load m1_model_ready.parquet + label_vocab.json + cv_folds.csv.
    2. For each of 5 CV folds:
       - Split into train / val using cv_folds.csv.
       - Fit TfidfVectorizer on train text_section_tagged. Transform val.
       - For each single-pick axis: fit LogisticRegression(class_weight=
         'balanced', solver='liblinear') on rows where label != null. Predict
         val; null val rows are masked by the metric module.
       - For each multi-label axis: fit per-code LogisticRegression OvR
         (skipping codes with <2 positives in train). Store probas + multi-hot
         gold per fold.
    3. After all folds: per multi-label axis, sweep threshold ∈ [0.1..0.5],
       pick global threshold maximizing mean F1-Micro across folds.
    4. Recompute per-fold multi-label metrics at locked threshold.
    5. Aggregate across folds: mean ± std per axis.
    6. Compute the early-stop metric per fold + mean ± std overall.
    7. Write JSON + markdown.

Design choices (vs B0):
    - Vectorizer is fit per-fold, not globally (no val→train leakage).
    - text_section_tagged is the encoder's input text (architecture v6 §4.4);
      using it here gives a fair comparison floor.
    - Threshold is global per axis (one t for all folds at deployment),
      picked by max mean F1-Micro. Per-fold tuning would overfit val.
    - class_weight='balanced' is on by default — required to make rare codes
      recoverable from TF-IDF features.
    - liblinear solver: fast, one-vs-rest native, K≤27. lbfgs would be more
      theoretically right for multinomial but speed differences dominate.
    - Fake-logits trick: sklearn predict / predict_proba is wrapped into
      [N, K] arrays compatible with src.eval.metrics.f1_macro_singlepick
      and f1_multilabel, so the same metric module the trainer uses scores
      this baseline. Zero duplicate metric code.

Usage:
    python scripts/run_tfidf_baseline.py
    python scripts/run_tfidf_baseline.py --folds 0
    python scripts/run_tfidf_baseline.py \\
        --parquet data/frozen/m1_model_ready/m1_model_ready.parquet \\
        --vocab data/frozen/m1_model_ready/label_vocab.json \\
        --folds-csv data/frozen/m1_model_ready/cv_folds.csv

Runtime: ~5-8 min on M4 CPU. No GPU.

Exit codes:
    0  success
    1  setup error (missing file/column)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.label_vocab import LabelVocab  # noqa: E402
from src.data.loaders import split_trainable_test  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    NULL_TARGET_SENTINEL,
    aggregate_axis_metrics,
    f1_macro_singlepick,
    f1_multilabel,
)


# ---------------------------------------------------------------------------
# Config (tunable but locked here for reproducibility)
# ---------------------------------------------------------------------------

TFIDF_NGRAM = (1, 2)
TFIDF_MAX_FEATURES = 30000
TFIDF_MIN_DF = 2
LR_MAX_ITER = 1000
# Single-pick axes are multiclass (K up to 27). `liblinear` is binary-only in
# sklearn ≥1.5; `lbfgs` is the multinomial solver and the theoretically
# correct match for the production masked-CE softmax head.
LR_SOLVER_SINGLEPICK = "lbfgs"
# Multi-label fits one binary classifier per code — liblinear is fine + fast.
LR_SOLVER_MULTILABEL = "liblinear"
LR_CLASS_WEIGHT = "balanced"
LR_RANDOM_STATE = 42
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_POS_FOR_MULTILABEL_FIT = 2  # codes with <2 train positives → predict 0
TEXT_COLUMN = "text_section_tagged"
LOGIT_CLIP = 15.0  # clip logit(p) into [-15, 15] to avoid inf at p∈{0,1}


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("tfidf_baseline")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Encoding helpers (mirror B0 conventions)
# ---------------------------------------------------------------------------


def _encode_singlepick_column(series: pd.Series, axis_vocab: Any) -> np.ndarray:
    """Encode a single-pick column to class indices [N], -1 for null."""
    out = np.full(len(series), NULL_TARGET_SENTINEL, dtype=np.int64)
    for i, raw in enumerate(series.tolist()):
        idx = axis_vocab.encode(raw)
        if idx is not None:
            out[i] = idx
    return out


def _encode_multilabel_column(series: pd.Series, axis_vocab: Any) -> np.ndarray:
    """Encode a multi-label column to multi-hot float [N, K]."""
    K = axis_vocab.num_classes
    out = np.zeros((len(series), K), dtype=np.float32)
    for i, raw in enumerate(series.tolist()):
        out[i] = axis_vocab.encode_multi(raw)
    return out


def _to_fake_singlepick_logits(
    preds: np.ndarray, n_records: int, num_classes: int
) -> np.ndarray:
    """Build logits [N, K] s.t. argmax(logits[i]) == preds[i]."""
    logits = np.zeros((n_records, num_classes), dtype=np.float32)
    rows = np.arange(n_records)
    logits[rows, preds] = 10.0
    return logits


def _probas_to_logits(probas: np.ndarray, clip: float = LOGIT_CLIP) -> np.ndarray:
    """Convert [N, K] probabilities → logits via inverse-sigmoid (clipped).

    Allows feeding sklearn probas through f1_multilabel, which applies
    sigmoid internally. sigmoid(logit(p)) = p, so threshold-on-sigmoid is
    equivalent to threshold-on-probability.
    """
    p = np.clip(probas.astype(np.float64), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1.0 - p))
    logits = np.clip(logits, -clip, clip)
    return logits.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-fold training + prediction
# ---------------------------------------------------------------------------


def _fit_predict_singlepick(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    n_val: int,
    num_classes: int,
    log: logging.Logger,
    axis_name: str,
) -> np.ndarray:
    """Fit LR on non-null train rows; predict for all val rows.

    Returns predicted class indices [N_val] for ALL val rows. Null val rows
    are masked downstream by f1_macro_singlepick.
    """
    valid_train = y_train != NULL_TARGET_SENTINEL
    if valid_train.sum() == 0:
        log.warning("  %s: no valid train labels, predicting 0 for all val", axis_name)
        return np.zeros(n_val, dtype=np.int64)

    X_train_valid = X_train[valid_train]
    y_train_valid = y_train[valid_train]

    # If only one class in train, sklearn fits trivially. Predict that class.
    n_unique = len(np.unique(y_train_valid))
    if n_unique == 1:
        only_cls = int(y_train_valid[0])
        return np.full(n_val, only_cls, dtype=np.int64)

    clf = LogisticRegression(
        class_weight=LR_CLASS_WEIGHT,
        solver=LR_SOLVER_SINGLEPICK,
        max_iter=LR_MAX_ITER,
        random_state=LR_RANDOM_STATE,
    )
    clf.fit(X_train_valid, y_train_valid)
    preds = clf.predict(X_val).astype(np.int64)
    return preds


def _fit_predict_multilabel(
    X_train: Any,
    Y_train: np.ndarray,
    X_val: Any,
    num_classes: int,
    log: logging.Logger,
    axis_name: str,
) -> np.ndarray:
    """Fit per-code binary LR on train; predict probas for val.

    Codes with <MIN_POS_FOR_MULTILABEL_FIT positives in train get all-zero
    probas (cannot fit binary classifier with one class).

    Returns [N_val, K] probabilities.
    """
    n_val = X_val.shape[0]
    probas = np.zeros((n_val, num_classes), dtype=np.float32)

    pos_counts = Y_train.sum(axis=0).astype(int)
    skipped = []
    for k in range(num_classes):
        n_pos = int(pos_counts[k])
        if n_pos < MIN_POS_FOR_MULTILABEL_FIT:
            skipped.append(k)
            continue
        y_k = Y_train[:, k].astype(int)
        clf = LogisticRegression(
            class_weight=LR_CLASS_WEIGHT,
            solver=LR_SOLVER_MULTILABEL,
            max_iter=LR_MAX_ITER,
            random_state=LR_RANDOM_STATE,
        )
        clf.fit(X_train, y_k)
        # predict_proba columns are sorted by clf.classes_; column for "1" may
        # be index 0 or 1. Locate it.
        pos_col = int(np.where(clf.classes_ == 1)[0][0])
        probas[:, k] = clf.predict_proba(X_val)[:, pos_col].astype(np.float32)

    if skipped:
        log.debug(
            "  %s: skipped %d/%d codes with <%d train positives",
            axis_name, len(skipped), num_classes, MIN_POS_FOR_MULTILABEL_FIT,
        )
    return probas


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_across_folds(
    per_fold: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Mean ± std per metric per axis across folds. Same as B0."""
    if not per_fold:
        return {}

    axis_to_values: dict[str, dict[str, list[float]]] = {}
    for fold_metrics in per_fold:
        for axis, m in fold_metrics.items():
            axis_to_values.setdefault(axis, {})
            for k, v in m.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    axis_to_values[axis].setdefault(k, []).append(float(v))

    out: dict[str, dict[str, float]] = {}
    for axis, kvs in axis_to_values.items():
        out[axis] = {}
        for k, vals in kvs.items():
            arr = np.array(vals)
            out[axis][f"{k}_mean"] = float(arr.mean())
            out[axis][f"{k}_std"] = float(arr.std(ddof=0))
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# B1 TF-IDF + LogisticRegression Baseline — M1")
    lines.append("")
    lines.append(f"- **Run at:** `{report['run_at']}`")
    lines.append(f"- **Folds:** {report['folds_run']}")
    lines.append(f"- **Trainable rows:** {report['n_trainable']}")
    lines.append(f"- **Text column:** `{report['text_column']}`")
    lines.append("")

    cfg = report["config"]
    lines.append("## Config")
    lines.append("")
    lines.append(f"- TF-IDF: ngram={tuple(cfg['tfidf_ngram'])}, "
                 f"max_features={cfg['tfidf_max_features']}, "
                 f"min_df={cfg['tfidf_min_df']}")
    lines.append(f"- LR: class_weight={cfg['lr_class_weight']!r}, "
                 f"solver_singlepick={cfg['lr_solver_singlepick']!r}, "
                 f"solver_multilabel={cfg['lr_solver_multilabel']!r}, "
                 f"max_iter={cfg['lr_max_iter']}, "
                 f"random_state={cfg['lr_random_state']}")
    lines.append(f"- Multi-label threshold grid: {cfg['threshold_grid']}")
    lines.append(f"- Multi-label min positives for fit: "
                 f"{cfg['min_pos_for_multilabel_fit']}")
    lines.append("")

    agg = report["aggregated"]
    early = report["early_stop_per_fold"]
    early_mean = float(np.mean(early)) if early else 0.0
    early_std = float(np.std(early)) if early else 0.0

    lines.append("## Early-stop metric")
    lines.append("")
    lines.append(f"**mean ± std:** {early_mean:.4f} ± {early_std:.4f}")
    lines.append(f"- per-fold: {[round(x, 4) for x in early]}")
    lines.append("")

    # Single-pick table
    sp_rows = [
        (axis, m) for axis, m in agg.items()
        if axis in report["single_pick_axes"]
    ]
    if sp_rows:
        lines.append("## Single-pick axes (target = F1-Macro)")
        lines.append("")
        lines.append("| axis | F1-Macro | F1-Macro (present) | Accuracy | n_valid |")
        lines.append("|---|---:|---:|---:|---:|")
        for axis, m in sp_rows:
            f1m = m.get("f1_macro_mean", 0.0)
            f1ms = m.get("f1_macro_std", 0.0)
            f1mp = m.get("f1_macro_present_mean", 0.0)
            acc = m.get("accuracy_mean", 0.0)
            nv = m.get("n_valid_mean", 0.0)
            lines.append(
                f"| {axis} | {f1m:.4f} ± {f1ms:.4f} | "
                f"{f1mp:.4f} | {acc:.4f} | {nv:.0f} |"
            )
        lines.append("")

    # Multi-label table
    ml_rows = [
        (axis, m) for axis, m in agg.items()
        if axis in report["multi_label_axes"]
    ]
    if ml_rows:
        thresholds = report["multi_label_thresholds"]
        lines.append("## Multi-label axes (target = F1-Micro)")
        lines.append("")
        lines.append(
            "Thresholds are global per axis, chosen by max mean F1-Micro "
            "across folds on the sweep grid."
        )
        lines.append("")
        lines.append("| axis | locked t | F1-Micro | F1-Macro | n_with_any_positive |")
        lines.append("|---|---:|---:|---:|---:|")
        for axis, m in ml_rows:
            t = thresholds.get(axis, 0.5)
            fmi = m.get("f1_micro_mean", 0.0)
            fmis = m.get("f1_micro_std", 0.0)
            fma = m.get("f1_macro_mean", 0.0)
            nwap = m.get("n_with_any_positive_mean", 0.0)
            lines.append(
                f"| {axis} | {t:.2f} | {fmi:.4f} ± {fmis:.4f} | "
                f"{fma:.4f} | {nwap:.0f} |"
            )
        lines.append("")

        # Threshold sweep table
        lines.append("### Threshold sweep (mean F1-Micro across folds)")
        lines.append("")
        sweep = report["threshold_sweep"]
        grid = report["config"]["threshold_grid"]
        hdr = "| axis | " + " | ".join(f"t={t}" for t in grid) + " |"
        sep = "|---|" + "|".join(["---:"] * len(grid)) + "|"
        lines.append(hdr)
        lines.append(sep)
        for axis in report["multi_label_axes"]:
            scores = sweep.get(axis, {})
            cells = [f"{scores.get(str(t), 0.0):.4f}" for t in grid]
            lines.append(f"| {axis} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "B1 is the **realistic floor**. TF-IDF + LogReg captures the lexical "
        "patterns of pathology language without semantic understanding. "
        "PubMedBERT (E1) must clear B1 by a substantial margin on the "
        "early-stop metric (>0.10 absolute) to justify its compute cost."
    )
    lines.append("")
    lines.append(
        "Per-axis expectations: lexical axes (topography, morphology, stem) "
        "should see large lifts over B0 — TF-IDF can match `'left breast'` "
        "or `'invasive ductal'`. Contextual axes (grade, behavior) and "
        "rare multi-label codes should remain low — these need semantic "
        "understanding the bag-of-ngrams cannot provide."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet", type=Path,
        default=REPO_ROOT / "data/frozen/m1_model_ready/m1_model_ready.parquet",
    )
    parser.add_argument(
        "--vocab", type=Path,
        default=REPO_ROOT / "data/frozen/m1_model_ready/label_vocab.json",
    )
    parser.add_argument(
        "--folds-csv", type=Path,
        default=REPO_ROOT / "data/frozen/m1_model_ready/cv_folds.csv",
    )
    parser.add_argument(
        "--folds", type=int, nargs="*", default=None,
        help="Specific folds to run (default: all 5)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path (default results/baselines/tfidf_<utc>.json)",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Suppress markdown summary to stdout",
    )
    parser.add_argument(
        "--text-column", default=TEXT_COLUMN,
        help=f"Text column to vectorize (default: {TEXT_COLUMN})",
    )

    args = parser.parse_args()
    log = _setup_logger()

    # Load artifacts
    for p in (args.parquet, args.vocab, args.folds_csv):
        if not p.exists():
            log.error("Not found: %s", p)
            return 1

    log.info("Loading parquet: %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    log.info("  loaded %d rows × %d cols", len(df), len(df.columns))

    if args.text_column not in df.columns:
        log.error("Text column %r not found. Available: %s",
                  args.text_column, [c for c in df.columns if c.startswith("text")])
        return 1

    log.info("Loading vocab: %s", args.vocab)
    vocab = LabelVocab.load_json(args.vocab)
    log.info("  axes: %s", list(vocab.axes.keys()))

    log.info("Loading CV folds: %s", args.folds_csv)
    folds_df = pd.read_csv(args.folds_csv)
    log.info("  fold counts: %s",
             dict(folds_df["fold"].value_counts().sort_index()))

    # Determine axis families
    single_pick_axes = [
        name for name, ax in vocab.axes.items() if not ax.multilabel
    ]
    multi_label_axes = [
        name for name, ax in vocab.axes.items() if ax.multilabel
    ]
    log.info("  %d single-pick axes: %s", len(single_pick_axes), single_pick_axes)
    log.info("  %d multi-label axes: %s", len(multi_label_axes), multi_label_axes)

    # Trainable pool
    trainable_df, _ = split_trainable_test(df)
    log.info("  trainable pool: %d rows", len(trainable_df))

    # Validate single-pick storage for ext_laterality + ext_grading.
    # Per v6 §6.2 fix, these are single-pick. If storage is list[str], the
    # vocab's encode() must be able to handle them — but len > 1 would mean
    # the data violates the spec contract.
    for axis_name in ("icd11_ext_laterality", "icd11_ext_grading"):
        if axis_name not in trainable_df.columns:
            continue
        max_len = 0
        for raw in trainable_df[axis_name].tolist():
            if isinstance(raw, (list, tuple, np.ndarray)):
                if len(raw) > max_len:
                    max_len = len(raw)
        if max_len > 1:
            log.error(
                "Spec violation: %s has rows with >1 code (max=%d). "
                "Per v6 §6.2 + audit, must be single-pick (0/3055 multi). "
                "Rebuild m1_model_ready or fix data.",
                axis_name, max_len,
            )
            return 1

    # Merge fold assignments
    if "fold" not in trainable_df.columns:
        if "record_id" not in folds_df.columns or "fold" not in folds_df.columns:
            log.error("folds_csv must have 'record_id' and 'fold' columns")
            return 1
        fold_map = dict(zip(folds_df["record_id"], folds_df["fold"]))
        trainable_df = trainable_df.copy()
        trainable_df["fold"] = trainable_df["record_id"].map(fold_map)
        if trainable_df["fold"].isna().any():
            log.error("%d records missing fold assignment",
                      int(trainable_df["fold"].isna().sum()))
            return 1

    folds_to_run = (
        args.folds if args.folds is not None
        else sorted(trainable_df["fold"].unique().tolist())
    )
    log.info("Running folds: %s", folds_to_run)

    # Fail loud on empty text — truncation audit confirmed Baheya M1 always
    # populates text_section_tagged. If this fires, regenerate parquet.
    empty_text = trainable_df[args.text_column].astype(str).str.strip() == ""
    if empty_text.any():
        log.error("Empty text in %d trainable rows for column %r",
                  int(empty_text.sum()), args.text_column)
        return 1

    # -----------------------------------------------------------------
    # Phase 1: per-fold fit + predict.
    # Single-pick: convert predictions to fake logits and compute metrics now.
    # Multi-label: STORE probas + multi-hot, defer metric computation until
    # after global threshold sweep.
    # -----------------------------------------------------------------
    per_fold_singlepick: list[dict[str, dict[str, Any]]] = []
    per_fold_multilabel_cache: list[dict[str, dict[str, np.ndarray]]] = []

    for fold_idx in folds_to_run:
        log.info("=== Fold %d ===", fold_idx)
        val_mask = trainable_df["fold"] == fold_idx
        train_fold = trainable_df[~val_mask].reset_index(drop=True)
        val_fold = trainable_df[val_mask].reset_index(drop=True)
        log.info("  train=%d val=%d", len(train_fold), len(val_fold))

        # TF-IDF: fit on train, transform val
        log.info("  fitting TF-IDF (ngram=%s, max_features=%d, min_df=%d)",
                 TFIDF_NGRAM, TFIDF_MAX_FEATURES, TFIDF_MIN_DF)
        vectorizer = TfidfVectorizer(
            ngram_range=TFIDF_NGRAM,
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            lowercase=True,
        )
        X_train = vectorizer.fit_transform(train_fold[args.text_column].astype(str))
        X_val = vectorizer.transform(val_fold[args.text_column].astype(str))
        log.info("  X_train=%s X_val=%s", X_train.shape, X_val.shape)

        fold_sp_metrics: dict[str, dict[str, Any]] = {}
        fold_ml_cache: dict[str, dict[str, np.ndarray]] = {}

        # Single-pick axes
        for axis_name in single_pick_axes:
            axis_vocab = vocab.axes[axis_name]
            K = axis_vocab.num_classes
            y_train = _encode_singlepick_column(train_fold[axis_name], axis_vocab)
            y_val = _encode_singlepick_column(val_fold[axis_name], axis_vocab)

            preds = _fit_predict_singlepick(
                X_train, y_train, X_val, len(val_fold), K, log, axis_name,
            )
            logits = _to_fake_singlepick_logits(preds, len(val_fold), K)
            m = f1_macro_singlepick(logits, y_val, num_classes=K)

            # Diagnostic: what code did the LR predict most often on val?
            if len(preds):
                top_pred = int(Counter(preds.tolist()).most_common(1)[0][0])
                m["top_predicted_code"] = axis_vocab.codes[top_pred]
            fold_sp_metrics[axis_name] = m
            log.info("    %s: F1-Macro=%.4f acc=%.4f n_valid=%d",
                     axis_name, m["f1_macro"], m["accuracy"], m["n_valid"])

        # Multi-label axes (cache probas + multihot; defer metric to global t sweep)
        for axis_name in multi_label_axes:
            axis_vocab = vocab.axes[axis_name]
            K = axis_vocab.num_classes
            Y_train = _encode_multilabel_column(train_fold[axis_name], axis_vocab)
            Y_val = _encode_multilabel_column(val_fold[axis_name], axis_vocab)

            probas = _fit_predict_multilabel(
                X_train, Y_train, X_val, K, log, axis_name,
            )
            fold_ml_cache[axis_name] = {"probas": probas, "multihot": Y_val}
            log.info("    %s: cached probas %s, gold multihot sum=%d",
                     axis_name, probas.shape, int(Y_val.sum()))

        per_fold_singlepick.append(fold_sp_metrics)
        per_fold_multilabel_cache.append(fold_ml_cache)

    # -----------------------------------------------------------------
    # Phase 2: global threshold sweep per multi-label axis.
    # Pick the t that maximizes mean F1-Micro across folds.
    # -----------------------------------------------------------------
    log.info("=== Threshold sweep ===")
    locked_thresholds: dict[str, float] = {}
    threshold_sweep: dict[str, dict[str, float]] = {}

    for axis_name in multi_label_axes:
        best_t = 0.5
        best_score = -1.0
        sweep_scores: dict[str, float] = {}

        for t in THRESHOLD_GRID:
            fold_scores: list[float] = []
            for fold_cache in per_fold_multilabel_cache:
                cache = fold_cache[axis_name]
                logits = _probas_to_logits(cache["probas"])
                m = f1_multilabel(logits, cache["multihot"], threshold=t)
                fold_scores.append(m["f1_micro"])
            mean_score = float(np.mean(fold_scores))
            sweep_scores[str(t)] = mean_score
            if mean_score > best_score:
                best_score = mean_score
                best_t = t

        locked_thresholds[axis_name] = best_t
        threshold_sweep[axis_name] = sweep_scores
        log.info("  %s: locked t=%.2f (mean F1-Micro=%.4f)",
                 axis_name, best_t, best_score)
        log.info("    sweep: %s",
                 {t: round(s, 4) for t, s in sweep_scores.items()})

    # -----------------------------------------------------------------
    # Phase 3: recompute per-fold multi-label metrics at locked threshold;
    # combine with single-pick metrics; compute per-fold early-stop.
    # -----------------------------------------------------------------
    per_fold: list[dict[str, dict[str, Any]]] = []
    early_stop_per_fold: list[float] = []

    for fold_idx_pos, fold_idx in enumerate(folds_to_run):
        sp = per_fold_singlepick[fold_idx_pos]
        ml_cache = per_fold_multilabel_cache[fold_idx_pos]

        combined: dict[str, dict[str, Any]] = dict(sp)
        for axis_name in multi_label_axes:
            t = locked_thresholds[axis_name]
            cache = ml_cache[axis_name]
            logits = _probas_to_logits(cache["probas"])
            m = f1_multilabel(logits, cache["multihot"], threshold=t)
            m["locked_threshold"] = t
            combined[axis_name] = m

        per_fold.append(combined)

        agg = aggregate_axis_metrics(
            combined,
            single_pick_axes=single_pick_axes,
            multi_label_axes=multi_label_axes,
        )
        early_stop_per_fold.append(float(agg.early_stop_metric))
        log.info("  fold %d early_stop = %.4f", fold_idx, agg.early_stop_metric)

    # -----------------------------------------------------------------
    # Aggregate across folds
    # -----------------------------------------------------------------
    aggregated = aggregate_across_folds(per_fold)

    run_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    report = {
        "run_at": run_at,
        "parquet": str(args.parquet),
        "vocab": str(args.vocab),
        "folds_csv": str(args.folds_csv),
        "folds_run": folds_to_run,
        "n_trainable": int(len(trainable_df)),
        "text_column": args.text_column,
        "single_pick_axes": single_pick_axes,
        "multi_label_axes": multi_label_axes,
        "config": {
            "tfidf_ngram": list(TFIDF_NGRAM),
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_min_df": TFIDF_MIN_DF,
            "lr_class_weight": LR_CLASS_WEIGHT,
            "lr_solver_singlepick": LR_SOLVER_SINGLEPICK,
            "lr_solver_multilabel": LR_SOLVER_MULTILABEL,
            "lr_max_iter": LR_MAX_ITER,
            "lr_random_state": LR_RANDOM_STATE,
            "threshold_grid": THRESHOLD_GRID,
            "min_pos_for_multilabel_fit": MIN_POS_FOR_MULTILABEL_FIT,
            "logit_clip": LOGIT_CLIP,
        },
        "per_fold": per_fold,
        "aggregated": aggregated,
        "early_stop_per_fold": early_stop_per_fold,
        "multi_label_thresholds": locked_thresholds,
        "threshold_sweep": threshold_sweep,
    }

    out_path = args.out or (
        REPO_ROOT / "results" / "baselines"
        / f"tfidf_{run_at.replace(':', '').replace('+0000', 'Z')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote JSON report to %s", out_path)

    # Also write markdown alongside JSON
    md_path = out_path.with_suffix(".md")
    md_path.write_text(render_markdown(report))
    log.info("Wrote markdown report to %s", md_path)

    if not args.no_markdown:
        print()
        print(render_markdown(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
