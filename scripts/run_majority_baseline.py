#!/usr/bin/env python3
"""Majority-class baseline for M1 (B0) — sanity floor for E1 and beyond.

Architecture v6 §13.2 Phase 0a deliverable. Establishes the minimum acceptable
F1 floor for the trainer. If any E* run scores worse than this on any axis,
something is broken in the model or training loop — NOT a "the data is hard"
problem.

What this script does:
    1. Load m1_model_ready.parquet + label_vocab.json + cv_folds.csv.
    2. For each of 5 CV folds:
       - Split into train / val using cv_folds.csv.
       - For each single-pick axis: predict the mode class observed on train.
       - For each multi-label axis: report TWO baselines per fold:
            (a) all_zeros — predict empty set for every record
            (b) per_label_mode — predict label k iff prevalence(k) > 0.5 on train
       - Compute per-axis F1-Macro / F1-Micro / accuracy using the SAME
         metric module (src.eval.metrics) the trainer will use.
    3. Aggregate across folds: mean ± std per axis.
    4. Compute the early-stop metric per fold + mean ± std overall.
    5. Write JSON + markdown.

Why two multi-label flavors:
    - all_zeros is the absolute minimum (F1 = 0 when targets are non-empty;
      could be >0 on axes with mostly-empty targets like ext_anatomy where
      5%/16 codes appear, but realistic floor).
    - per_label_mode is the "smart dummy" — handles axes where some codes
      appear in >50% of records (icd11_ext_grading: XS58 appears in ~60%).
      This is the realistic floor any trainer must clear.

The early-stop metric per architecture v6 §11:
    (mean F1-Macro single-pick + mean F1-Micro multi-label) / 2

Usage:
    # Default — runs all 5 folds, writes results/baselines/majority_<utc>.json
    python scripts/run_majority_baseline.py

    # Single fold for quick check
    python scripts/run_majority_baseline.py --folds 0

    # Different parquet/vocab/folds paths
    python scripts/run_majority_baseline.py \\
        --parquet data/frozen/m1_model_ready/m1_model_ready.parquet \\
        --vocab data/frozen/m1_model_ready/label_vocab.json \\
        --folds-csv data/frozen/m1_model_ready/cv_folds.csv

Runtime: ~30 seconds on M4 CPU. No GPU needed.

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


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("majority_baseline")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Per-fold computation
# ---------------------------------------------------------------------------


def _encode_singlepick_column(
    series: pd.Series, vocab: Any
) -> np.ndarray:
    """Encode a single-pick column to class indices [N] with -1 for null."""
    out = np.full(len(series), NULL_TARGET_SENTINEL, dtype=np.int64)
    for i, raw in enumerate(series.tolist()):
        idx = vocab.encode(raw)
        if idx is not None:
            out[i] = idx
    return out


def _encode_multilabel_column(series: pd.Series, vocab: Any) -> np.ndarray:
    """Encode a multi-label column to multi-hot float [N, K]."""
    K = vocab.num_classes
    out = np.zeros((len(series), K), dtype=np.float32)
    for i, raw in enumerate(series.tolist()):
        out[i] = vocab.encode_multi(raw)
    return out


def _fake_logits_for_singlepick_majority(
    n_records: int, num_classes: int, mode_idx: int
) -> np.ndarray:
    """Build logits that argmax to `mode_idx` for every record."""
    logits = np.zeros((n_records, num_classes), dtype=np.float32)
    logits[:, mode_idx] = 10.0
    return logits


def _fake_logits_for_multilabel(
    n_records: int, num_classes: int, predicted_mask: np.ndarray
) -> np.ndarray:
    """Build logits where label k is +10 if predicted_mask[k] else -10.

    Sigmoid(10) ≈ 1, sigmoid(-10) ≈ 0 → threshold-0.5 gives back the mask.
    """
    logits = np.full((n_records, num_classes), -10.0, dtype=np.float32)
    for k in range(num_classes):
        if predicted_mask[k]:
            logits[:, k] = 10.0
    return logits


def run_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: LabelVocab,
    single_pick_axes: list[str],
    multi_label_axes: list[str],
    multi_strategy: str,
    log: logging.Logger,
) -> dict[str, dict[str, Any]]:
    """Compute per-axis metrics for one fold.

    Args:
        train_df: Training rows (used only to determine mode/prevalence).
        val_df: Validation rows (used to compute metrics).
        vocab: Fitted LabelVocab.
        single_pick_axes: Names of single-pick axes.
        multi_label_axes: Names of multi-label axes.
        multi_strategy: 'all_zeros' or 'per_label_mode'.

    Returns:
        per_axis dict: axis_name -> metric dict matching the contract of
        aggregate_axis_metrics.
    """
    per_axis: dict[str, dict[str, Any]] = {}

    for axis in single_pick_axes:
        axis_vocab = vocab[axis]
        K = axis_vocab.num_classes

        # Mode class on train (skipping nulls)
        train_indices = _encode_singlepick_column(train_df[axis], axis_vocab)
        valid_mask = train_indices != NULL_TARGET_SENTINEL
        if not valid_mask.any():
            log.warning("  %s: no valid train labels, skipping", axis)
            continue
        mode_idx = int(Counter(train_indices[valid_mask].tolist()).most_common(1)[0][0])

        # Predict on val
        val_indices = _encode_singlepick_column(val_df[axis], axis_vocab)
        n_val = len(val_indices)
        logits = _fake_logits_for_singlepick_majority(n_val, K, mode_idx)
        m = f1_macro_singlepick(logits, val_indices, num_classes=K)
        m["mode_code"] = axis_vocab.codes[mode_idx]
        per_axis[axis] = m

    for axis in multi_label_axes:
        axis_vocab = vocab[axis]
        K = axis_vocab.num_classes

        # Determine predicted mask
        train_multihot = _encode_multilabel_column(train_df[axis], axis_vocab)
        if multi_strategy == "all_zeros":
            predicted_mask = np.zeros(K, dtype=bool)
        elif multi_strategy == "per_label_mode":
            # Predict label if it appears in > 50% of training records
            prevalence = train_multihot.mean(axis=0)
            predicted_mask = prevalence > 0.5
        else:
            raise ValueError(f"Unknown multi_strategy: {multi_strategy!r}")

        # Predict on val
        val_multihot = _encode_multilabel_column(val_df[axis], axis_vocab)
        n_val = len(val_multihot)
        logits = _fake_logits_for_multilabel(n_val, K, predicted_mask)
        m = f1_multilabel(logits, val_multihot, threshold=0.5)
        m["strategy"] = multi_strategy
        m["predicted_codes"] = [
            axis_vocab.codes[k] for k in range(K) if predicted_mask[k]
        ]
        per_axis[axis] = m

    return per_axis


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------


def aggregate_across_folds(
    per_fold: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Mean ± std per metric per axis across folds."""
    if not per_fold:
        return {}

    # Collect numeric keys per axis across folds
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
# Markdown render
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# B0 Majority Baseline — M1")
    lines.append("")
    lines.append(f"- **Run at:** `{report['run_at']}`")
    lines.append(f"- **Folds:** {report['folds_run']}")
    lines.append(f"- **Trainable rows:** {report['n_trainable']}")
    lines.append("")

    for strategy in report["strategies"]:
        lines.append(f"## Strategy: `{strategy}`")
        lines.append("")
        agg = report["strategies"][strategy]["aggregated"]
        early = report["strategies"][strategy]["early_stop_per_fold"]
        early_mean = float(np.mean(early)) if early else 0.0
        early_std = float(np.std(early)) if early else 0.0

        lines.append(
            f"**Early-stop metric (mean ± std):** {early_mean:.4f} ± {early_std:.4f}"
        )
        lines.append(f"  - per-fold: {[round(x, 4) for x in early]}")
        lines.append("")

        # Single-pick axes table
        sp_rows = [
            (axis, m)
            for axis, m in agg.items()
            if axis in report["single_pick_axes"]
        ]
        if sp_rows:
            lines.append("**Single-pick axes (target = F1-Macro):**")
            lines.append("")
            lines.append(
                "| axis | F1-Macro | F1-Macro (present) | Accuracy | n_valid |"
            )
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

        # Multi-label axes table
        ml_rows = [
            (axis, m)
            for axis, m in agg.items()
            if axis in report["multi_label_axes"]
        ]
        if ml_rows:
            lines.append("**Multi-label axes (target = F1-Micro):**")
            lines.append("")
            lines.append(
                "| axis | F1-Micro | F1-Macro | n_with_any_positive |"
            )
            lines.append("|---|---:|---:|---:|")
            for axis, m in ml_rows:
                fmi = m.get("f1_micro_mean", 0.0)
                fmis = m.get("f1_micro_std", 0.0)
                fma = m.get("f1_macro_mean", 0.0)
                nwap = m.get("n_with_any_positive_mean", 0.0)
                lines.append(
                    f"| {axis} | {fmi:.4f} ± {fmis:.4f} | "
                    f"{fma:.4f} | {nwap:.0f} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "These numbers are the SANITY FLOOR. Any E* run that scores below "
        "the higher of the two strategies on any axis indicates a broken "
        "trainer (loss not flowing, label encoding mismatch, etc.) — NOT "
        "that 'the data is hard'."
    )
    lines.append("")
    lines.append(
        "Note: single-pick F1-Macro for majority baseline ≈ 1/K (rough "
        "lower bound); axes with a strongly dominant class (e.g. behavior "
        "where '3' = 88%) will score higher in absolute accuracy but still "
        "near-zero F1-Macro because the model never predicts the minority."
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
        help="Output JSON path (default results/baselines/majority_<utc>.json)",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Suppress markdown summary to stdout",
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

    log.info("Loading vocab: %s", args.vocab)
    vocab = LabelVocab.load_json(args.vocab)
    log.info("  axes: %s", list(vocab.axes.keys()))

    log.info("Loading CV folds: %s", args.folds_csv)
    folds_df = pd.read_csv(args.folds_csv)
    log.info("  fold counts: %s", dict(folds_df["fold"].value_counts().sort_index()))

    # Determine axis families from vocab axis_meta (post-fix: 8 single-pick + 2 multi)
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

    # Merge fold assignments
    if "fold" not in trainable_df.columns:
        # Merge from folds_df
        if "record_id" not in folds_df.columns or "fold" not in folds_df.columns:
            log.error("folds_csv must have 'record_id' and 'fold' columns")
            return 1
        fold_map = dict(zip(folds_df["record_id"], folds_df["fold"]))
        trainable_df = trainable_df.copy()
        trainable_df["fold"] = trainable_df["record_id"].map(fold_map)
        if trainable_df["fold"].isna().any():
            log.error("%d records missing fold assignment",
                     trainable_df["fold"].isna().sum())
            return 1

    folds_to_run = args.folds if args.folds is not None else sorted(
        trainable_df["fold"].unique().tolist()
    )
    log.info("Running folds: %s", folds_to_run)

    # Two strategies for multi-label baselines
    strategies = ["all_zeros", "per_label_mode"]
    strategy_results: dict[str, dict[str, Any]] = {}

    for strategy in strategies:
        log.info("=== Strategy: %s ===", strategy)
        per_fold: list[dict[str, dict[str, Any]]] = []
        early_stop_per_fold: list[float] = []

        for fold_idx in folds_to_run:
            log.info("  fold %d ...", fold_idx)
            val_mask = trainable_df["fold"] == fold_idx
            train_fold = trainable_df[~val_mask].reset_index(drop=True)
            val_fold = trainable_df[val_mask].reset_index(drop=True)
            log.info("    train=%d val=%d", len(train_fold), len(val_fold))

            metrics = run_fold(
                train_df=train_fold,
                val_df=val_fold,
                vocab=vocab,
                single_pick_axes=single_pick_axes,
                multi_label_axes=multi_label_axes,
                multi_strategy=strategy,
                log=log,
            )
            per_fold.append(metrics)

            # Compute the early-stop metric for this fold
            agg = aggregate_axis_metrics(
                metrics,
                single_pick_axes=single_pick_axes,
                multi_label_axes=multi_label_axes,
            )
            early_stop_per_fold.append(float(agg.early_stop_metric))
            log.info("    fold %d early_stop = %.4f", fold_idx, agg.early_stop_metric)

        # Aggregate across folds
        aggregated = aggregate_across_folds(per_fold)
        strategy_results[strategy] = {
            "per_fold": per_fold,
            "aggregated": aggregated,
            "early_stop_per_fold": early_stop_per_fold,
        }

    run_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    report = {
        "run_at": run_at,
        "parquet": str(args.parquet),
        "vocab": str(args.vocab),
        "folds_run": folds_to_run,
        "n_trainable": int(len(trainable_df)),
        "single_pick_axes": single_pick_axes,
        "multi_label_axes": multi_label_axes,
        "strategies": strategy_results,
    }

    out_path = args.out or (
        REPO_ROOT / "results" / "baselines"
        / f"majority_{run_at.replace(':', '').replace('+0000', 'Z')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote JSON report to %s", out_path)

    if not args.no_markdown:
        print()
        print(render_markdown(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
