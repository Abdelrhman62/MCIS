"""Tests for src.eval.metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.eval.metrics import (
    NULL_TARGET_SENTINEL,
    AggregateMetrics,
    aggregate_axis_metrics,
    f1_macro_singlepick,
    f1_multilabel,
)


# ---------------------------------------------------------------------------
# Single-pick (masked F1-Macro)
# ---------------------------------------------------------------------------


class TestF1MacroSinglepick:
    def test_perfect_predictions_score_1(self) -> None:
        # K=3, 4 records, all correct
        logits = torch.tensor([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
        ])
        targets = torch.tensor([0, 1, 2, 0])
        m = f1_macro_singlepick(logits, targets, num_classes=3)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_valid"] == 4
        assert m["n_classes_present"] == 3

    def test_null_targets_excluded(self) -> None:
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
        targets = torch.tensor([0, 1, NULL_TARGET_SENTINEL])
        m = f1_macro_singlepick(logits, targets, num_classes=2)
        assert m["n_valid"] == 2  # last record dropped

    def test_all_null_returns_zero(self) -> None:
        logits = torch.zeros(3, 5)
        targets = torch.full((3,), NULL_TARGET_SENTINEL)
        m = f1_macro_singlepick(logits, targets, num_classes=5)
        assert m["n_valid"] == 0
        assert m["f1_macro"] == 0.0

    def test_random_predictions_below_perfect(self) -> None:
        # Adversarial: all wrong
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        targets = torch.tensor([0, 1])
        m = f1_macro_singlepick(logits, targets, num_classes=2)
        assert m["accuracy"] == 0.0
        assert m["f1_macro"] == 0.0

    def test_present_vs_all_classes(self) -> None:
        # Vocab K=4, but only classes 0 and 1 appear in y_true.
        # f1_macro spreads over all 4 (zero-fill the absent ones).
        # f1_macro_present averages only over the 2 present.
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
        targets = torch.tensor([0, 1])
        m = f1_macro_singlepick(logits, targets, num_classes=4)
        assert m["f1_macro_present"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(2.0 / 4)  # 2 perfect + 2 zero
        assert m["n_classes_present"] == 2

    def test_numpy_input_works(self) -> None:
        logits = np.array([[10.0, 0.0], [0.0, 10.0]])
        targets = np.array([0, 1])
        m = f1_macro_singlepick(logits, targets, num_classes=2)
        assert m["f1_macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-label (F1-Micro / Macro)
# ---------------------------------------------------------------------------


class TestF1Multilabel:
    def test_perfect_multilabel(self) -> None:
        # K=3, 2 records, all correct
        logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        m = f1_multilabel(logits, targets)
        assert m["f1_micro"] == pytest.approx(1.0)
        assert m["f1_macro"] == pytest.approx(1.0)
        assert m["n_records"] == 2

    def test_no_positives_predictable(self) -> None:
        # Empty targets, predict empty → no TP/FP/FN, f1 = 0 by convention
        logits = torch.full((2, 3), -10.0)  # all negative predictions
        targets = torch.zeros(2, 3)
        m = f1_multilabel(logits, targets)
        assert m["f1_micro"] == 0.0
        assert m["n_with_any_positive"] == 0

    def test_threshold_works(self) -> None:
        # Logit 0.0 → prob 0.5; default threshold 0.5 picks it up
        logits = torch.tensor([[0.1, -0.1]])
        targets = torch.tensor([[1.0, 0.0]])
        m = f1_multilabel(logits, targets, threshold=0.5)
        assert m["f1_micro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_dual_metric_average(self) -> None:
        per_axis = {
            "icdo3_morphology": {"f1_macro": 0.6, "n_valid": 100, "n_classes_present": 24, "accuracy": 0.7, "f1_macro_present": 0.6},
            "icdo3_grade":      {"f1_macro": 0.8, "n_valid": 80,  "n_classes_present": 3,  "accuracy": 0.85, "f1_macro_present": 0.8},
            "icd11_ext_anatomy":   {"f1_micro": 0.5, "f1_macro": 0.4, "n_records": 100, "n_with_any_positive": 50},
            "icd11_ext_histopath": {"f1_micro": 0.7, "f1_macro": 0.6, "n_records": 100, "n_with_any_positive": 80},
        }
        agg = aggregate_axis_metrics(
            per_axis,
            single_pick_axes=["icdo3_morphology", "icdo3_grade"],
            multi_label_axes=["icd11_ext_anatomy", "icd11_ext_histopath"],
        )
        # mean(f1_macro single) = (0.6 + 0.8) / 2 = 0.7
        # mean(f1_micro multi)  = (0.5 + 0.7) / 2 = 0.6
        # early_stop = (0.7 + 0.6) / 2 = 0.65
        assert agg.mean_f1_macro_singlepick == pytest.approx(0.7)
        assert agg.mean_f1_micro_multilabel == pytest.approx(0.6)
        assert agg.early_stop_metric == pytest.approx(0.65)

    def test_phase1_no_multilabel_falls_back(self) -> None:
        # Phase 1: TCGA pretrain has no ICD-11 axes
        per_axis = {
            "icdo3_morphology": {"f1_macro": 0.5, "n_valid": 100, "n_classes_present": 24, "accuracy": 0.6, "f1_macro_present": 0.5},
            "icdo3_topography": {"f1_macro": 0.7, "n_valid": 100, "n_classes_present": 8, "accuracy": 0.8, "f1_macro_present": 0.7},
        }
        agg = aggregate_axis_metrics(
            per_axis,
            single_pick_axes=["icdo3_morphology", "icdo3_topography"],
            multi_label_axes=[],
        )
        assert agg.mean_f1_micro_multilabel == 0.0
        # Falls back to single-pick mean
        assert agg.early_stop_metric == pytest.approx(0.6)

    def test_zero_valid_records_skipped(self) -> None:
        # An axis with n_valid=0 (e.g. all targets null in this batch)
        per_axis = {
            "icdo3_morphology": {"f1_macro": 0.5, "n_valid": 100, "n_classes_present": 24, "accuracy": 0.6, "f1_macro_present": 0.5},
            "icdo3_grade":      {"f1_macro": 0.0, "n_valid": 0,   "n_classes_present": 0,  "accuracy": 0.0, "f1_macro_present": 0.0},
        }
        agg = aggregate_axis_metrics(
            per_axis,
            single_pick_axes=["icdo3_morphology", "icdo3_grade"],
            multi_label_axes=[],
        )
        # Grade has n_valid=0 → skipped; morphology mean = 0.5
        assert agg.mean_f1_macro_singlepick == pytest.approx(0.5)

    def test_flatten_for_wandb_keys(self) -> None:
        per_axis = {
            "icdo3_morphology": {"f1_macro": 0.6, "n_valid": 100, "n_classes_present": 24, "accuracy": 0.7, "f1_macro_present": 0.6},
        }
        agg = aggregate_axis_metrics(
            per_axis,
            single_pick_axes=["icdo3_morphology"],
            multi_label_axes=[],
        )
        flat = agg.flatten_for_wandb(prefix="val")
        assert "val/icdo3_morphology/f1_macro" in flat
        assert "val/mean_f1_macro_singlepick" in flat
        assert "val/early_stop_metric" in flat
        # Strings excluded; only floats
        assert all(isinstance(v, float) for v in flat.values())

    def test_no_metrics_returns_zero(self) -> None:
        agg = aggregate_axis_metrics(
            per_axis={},
            single_pick_axes=["x"],
            multi_label_axes=["y"],
        )
        assert agg.early_stop_metric == 0.0
