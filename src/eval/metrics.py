"""Per-axis metrics for MCIS M1.

Two metric families:
- single-pick axes (ICD-O-3 + ICD-11 stem) → F1-Macro per axis, masked CE.
  Records with NULL_TARGET_SENTINEL (=-1) are excluded from the metric.
- multi-label axes (ICD-11 ext anatomy/histopath/laterality/grading) → F1-Micro
  per axis. Empty positive set = valid "no positives" prediction.

Per architecture v6 §11:
- F1-Macro is the primary metric for single-pick (per-class then averaged).
- F1-Micro is the primary metric for multi-label (per-instance counts).
- Test-only labels (codes in test but absent from build vocab) are NOT scored.

Early-stopping metric (per chat lock Q3): mean of {mean F1-Macro over single-pick,
mean F1-Micro over multi-label}. Larger = better.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

NULL_TARGET_SENTINEL = -1


# ---------------------------------------------------------------------------
# Single-pick (masked F1-Macro)
# ---------------------------------------------------------------------------


def f1_macro_singlepick(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    num_classes: int,
    null_sentinel: int = NULL_TARGET_SENTINEL,
) -> dict[str, float | int]:
    """Compute F1-Macro for a single-pick axis with masked nulls.

    Args:
        logits: [N, K] pre-softmax logits.
        targets: [N] int class indices, or null_sentinel for masked records.
        num_classes: K. Must match logits.shape[-1].
        null_sentinel: Records with this target value are excluded.

    Returns:
        {
          "f1_macro":      macro-averaged F1 across all K classes (zero-fill missing),
          "f1_macro_present": macro-averaged F1 across only K_present classes seen in y_true,
          "n_valid":       int, records that contributed,
          "n_classes_present": int, distinct classes seen in y_true (post-mask),
          "accuracy":      micro-accuracy = correct / n_valid (informational),
        }
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    assert logits.shape[-1] == num_classes, (
        f"logits.shape[-1]={logits.shape[-1]} != num_classes={num_classes}"
    )

    valid = targets != null_sentinel
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "f1_macro": 0.0,
            "f1_macro_present": 0.0,
            "n_valid": 0,
            "n_classes_present": 0,
            "accuracy": 0.0,
        }

    y_true = targets[valid]
    preds = logits[valid].argmax(axis=-1)
    accuracy = float((preds == y_true).mean())

    classes_present = np.unique(y_true)
    n_classes_present = int(len(classes_present))

    # Per-class F1: precision & recall over the full K-space, then macro-mean.
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    for k in range(num_classes):
        tp = int(((preds == k) & (y_true == k)).sum())
        fp = int(((preds == k) & (y_true != k)).sum())
        fn = int(((preds != k) & (y_true == k)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[k] = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )

    f1_macro_all = float(per_class_f1.mean())
    f1_macro_present = float(per_class_f1[classes_present].mean()) if n_classes_present else 0.0

    return {
        "f1_macro": f1_macro_all,
        "f1_macro_present": f1_macro_present,
        "n_valid": n_valid,
        "n_classes_present": n_classes_present,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Multi-label (F1-Micro and F1-Macro)
# ---------------------------------------------------------------------------


def f1_multilabel(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute F1-Micro + F1-Macro for a multi-label axis.

    Args:
        logits: [N, K] pre-sigmoid logits.
        targets: [N, K] float multi-hot ∈ {0, 1}.
        threshold: Probability cutoff. 0.5 default; could become a tuned hyperparam later.

    Returns:
        {
          "f1_micro":  TP-pooled F1 across all (instance, class) pairs.
          "f1_macro":  Per-class F1 averaged across K classes (zero-fill).
          "n_records": int.
          "n_with_any_positive": int, records with ≥1 positive in y_true.
        }

    Multi-label has no "null" mask — empty positive list = valid no-positives target.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    n_records, num_classes = logits.shape
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    y_true = (targets > 0.5).astype(np.int32)

    # Micro
    tp_micro = int(((preds == 1) & (y_true == 1)).sum())
    fp_micro = int(((preds == 1) & (y_true == 0)).sum())
    fn_micro = int(((preds == 0) & (y_true == 1)).sum())
    prec_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    rec_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = (
        2 * prec_micro * rec_micro / (prec_micro + rec_micro)
        if (prec_micro + rec_micro) > 0
        else 0.0
    )

    # Macro
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    for k in range(num_classes):
        tp = int(((preds[:, k] == 1) & (y_true[:, k] == 1)).sum())
        fp = int(((preds[:, k] == 1) & (y_true[:, k] == 0)).sum())
        fn = int(((preds[:, k] == 0) & (y_true[:, k] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_f1[k] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f1_macro = float(per_class_f1.mean())

    n_with_any = int((y_true.sum(axis=1) > 0).sum())

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": f1_macro,
        "n_records": int(n_records),
        "n_with_any_positive": n_with_any,
    }


# ---------------------------------------------------------------------------
# Aggregation across axes (early-stop metric)
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """Cross-axis summary used for early stopping + W&B logging."""

    per_axis: dict[str, dict[str, float | int]]  # axis -> metric dict
    mean_f1_macro_singlepick: float
    mean_f1_micro_multilabel: float
    early_stop_metric: float  # mean of the two means

    def flatten_for_wandb(self, prefix: str = "val") -> dict[str, float]:
        """Turn nested per-axis metrics into a flat W&B-loggable dict."""
        out: dict[str, float] = {}
        for axis, metrics in self.per_axis.items():
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    out[f"{prefix}/{axis}/{k}"] = float(v)
        out[f"{prefix}/mean_f1_macro_singlepick"] = self.mean_f1_macro_singlepick
        out[f"{prefix}/mean_f1_micro_multilabel"] = self.mean_f1_micro_multilabel
        out[f"{prefix}/early_stop_metric"] = self.early_stop_metric
        return out


def aggregate_axis_metrics(
    per_axis: dict[str, dict[str, float | int]],
    single_pick_axes: list[str],
    multi_label_axes: list[str],
) -> AggregateMetrics:
    """Combine per-axis metrics into the early-stop scalar.

    Per chat lock Q3:
        early_stop_metric = mean(
            mean(F1-Macro over single-pick axes),
            mean(F1-Micro over multi-label axes)
        )

    Axes missing from per_axis are skipped (e.g. Phase 1 has no ICD-11 axes).
    Axes with n_valid == 0 are also skipped (no records contributed).
    """
    sp_scores: list[float] = []
    for axis in single_pick_axes:
        m = per_axis.get(axis)
        if m is None or m.get("n_valid", 0) == 0:
            continue
        sp_scores.append(float(m["f1_macro"]))

    ml_scores: list[float] = []
    for axis in multi_label_axes:
        m = per_axis.get(axis)
        if m is None or m.get("n_records", 0) == 0:
            continue
        ml_scores.append(float(m["f1_micro"]))

    mean_sp = float(np.mean(sp_scores)) if sp_scores else 0.0
    mean_ml = float(np.mean(ml_scores)) if ml_scores else 0.0

    if sp_scores and ml_scores:
        early_stop = (mean_sp + mean_ml) / 2.0
    elif sp_scores:
        early_stop = mean_sp
    elif ml_scores:
        early_stop = mean_ml
    else:
        early_stop = 0.0

    return AggregateMetrics(
        per_axis=per_axis,
        mean_f1_macro_singlepick=mean_sp,
        mean_f1_micro_multilabel=mean_ml,
        early_stop_metric=early_stop,
    )
