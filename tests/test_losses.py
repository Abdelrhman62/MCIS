"""Tests for src.losses.{class_weighted, focal, builder}.

Coverage:
  weight computation:
    - inverse-freq weights correct shape, sum, clipping
    - null masking (single-pick targets == -1 don't count)
    - smoothing handles zero-count classes
  ClassWeightedCrossEntropy:
    - signature matches SinglePickHead.compute_loss
    - reduces to F.cross_entropy when weights = uniform
    - mask null sentinels
    - returns (loss=0, n_valid=0) when all targets are null
  WeightedBCEWithLogits:
    - signature matches MultiLabelHead.compute_loss
    - reduces to F.binary_cross_entropy_with_logits when pos_weight = 1
  FocalCrossEntropy:
    - gamma=0 reduces to standard CE
    - gamma>0: easy examples weighted down
    - alpha works as per-class weight
    - null masking
  FocalBCEWithLogits:
    - gamma=0 reduces to standard BCE (modulo alpha)
    - alpha=0.5 + gamma=0 reduces to 0.5 * BCE
  attach_loss_fns_to_heads:
    - head.compute_loss is overridden
    - other axes' heads NOT affected
    - device-following works (strategy buffers move with model.to())
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.losses.class_weighted import (
    ClassWeightedCrossEntropy,
    WeightedBCEWithLogits,
    compute_inverse_freq_weights_multi_label,
    compute_inverse_freq_weights_single_pick,
)
from src.losses.focal import FocalBCEWithLogits, FocalCrossEntropy
from src.losses.builder import attach_loss_fns_to_heads
from src.models.heads import NULL_TARGET_SENTINEL


# ============================================================================
# Weight computation
# ============================================================================


def test_inverse_freq_single_pick_basic() -> None:
    # 4 classes; counts: 8, 4, 2, 1 (rare). n_valid=15. K=4.
    # weights ~ n / (K * (count + 1))
    targets = np.array([0]*8 + [1]*4 + [2]*2 + [3]*1)
    w = compute_inverse_freq_weights_single_pick(targets, num_classes=4)
    assert w.shape == (4,)
    assert (w >= 0).all()
    # Rare class (3, count=1) should get the highest weight
    assert w.argmax().item() == 3
    # Common class (0, count=8) should get the lowest
    assert w.argmin().item() == 0


def test_inverse_freq_single_pick_masks_null() -> None:
    targets = np.array([0]*5 + [1]*5 + [NULL_TARGET_SENTINEL]*100)
    w = compute_inverse_freq_weights_single_pick(targets, num_classes=2)
    # Nulls shouldn't influence weights — should look like balanced 5/5
    assert w.shape == (2,)
    assert torch.allclose(w[0], w[1], atol=1e-4)


def test_inverse_freq_single_pick_zero_count_class() -> None:
    # Class 2 has zero examples; with smooth=1 it should still get finite weight
    targets = np.array([0]*10 + [1]*10)
    w = compute_inverse_freq_weights_single_pick(targets, num_classes=3)
    assert w.shape == (3,)
    assert torch.isfinite(w).all()
    # Zero-count class gets the highest weight (rarest)
    assert w.argmax().item() == 2


def test_inverse_freq_single_pick_clip() -> None:
    # Extreme imbalance: 1000 vs 1; without clip, weight ratio ≈ 1000;
    # with default clip=10.0, rare class capped.
    targets = np.array([0]*1000 + [1]*1)
    w = compute_inverse_freq_weights_single_pick(targets, num_classes=2)
    assert w.max() <= 10.0 + 1e-6


def test_inverse_freq_single_pick_all_null_uniform() -> None:
    # Degenerate: all null → uniform.
    targets = np.array([NULL_TARGET_SENTINEL] * 5)
    w = compute_inverse_freq_weights_single_pick(targets, num_classes=4)
    assert torch.allclose(w, torch.ones(4))


def test_inverse_freq_multi_label_basic() -> None:
    # 4 classes; class 0 = 9 pos / 1 neg (low pos_weight, common positive);
    # class 2 = 5 pos / 5 neg (balanced); class 3 = 1 pos / 9 neg (high pos_weight).
    # NOTE: classes with 0 positives get pos_weight = (n-0)/(0+smooth) = n/1
    # = clipped at max — we explicitly avoid 0-positive classes here so the
    # test asserts the expected ordering.
    targets = np.zeros((10, 4), dtype=np.int64)
    targets[:9, 0] = 1
    targets[:5, 1] = 1
    targets[:5, 2] = 1
    targets[:1, 3] = 1
    w = compute_inverse_freq_weights_multi_label(targets)
    assert w.shape == (4,)
    # Class 3 (1 positive, 9 negative) → highest pos_weight
    assert w.argmax().item() == 3
    # Class 0 (9 positive, 1 negative) → lowest
    assert w.argmin().item() == 0


def test_inverse_freq_multi_label_clip() -> None:
    # 1000 vs 1 imbalance → ratio ≈ 999; should be clipped at 10.
    n = 1000
    targets = np.zeros((n, 1), dtype=np.int64)
    targets[:1, 0] = 1
    w = compute_inverse_freq_weights_multi_label(targets)
    assert w.max() <= 10.0 + 1e-6


# ============================================================================
# ClassWeightedCrossEntropy
# ============================================================================


def test_class_weighted_ce_signature() -> None:
    fn = ClassWeightedCrossEntropy(torch.tensor([1.0, 1.0, 1.0]))
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 1, 2, 0])
    loss, n_valid = fn(logits, targets)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert n_valid == B


def test_class_weighted_ce_uniform_equals_F_ce() -> None:
    """Uniform weights → identical to F.cross_entropy."""
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 1, 2, 0])
    fn = ClassWeightedCrossEntropy(torch.ones(K))
    loss, _ = fn(logits, targets)
    expected = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-6)


def test_class_weighted_ce_masks_null() -> None:
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, NULL_TARGET_SENTINEL, 2, NULL_TARGET_SENTINEL])
    fn = ClassWeightedCrossEntropy(torch.ones(K))
    loss, n_valid = fn(logits, targets)
    assert n_valid == 2
    # Should equal CE on the 2 valid rows
    valid_mask = targets != NULL_TARGET_SENTINEL
    expected = F.cross_entropy(logits[valid_mask], targets[valid_mask])
    assert torch.allclose(loss, expected, atol=1e-6)


def test_class_weighted_ce_all_null() -> None:
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.full((B,), NULL_TARGET_SENTINEL, dtype=torch.long)
    fn = ClassWeightedCrossEntropy(torch.ones(K))
    loss, n_valid = fn(logits, targets)
    assert n_valid == 0
    assert float(loss) == 0.0


def test_class_weighted_ce_applies_weight() -> None:
    """Non-uniform weights produce different loss than uniform."""
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 1, 2, 0])
    fn_uniform = ClassWeightedCrossEntropy(torch.ones(K))
    fn_weighted = ClassWeightedCrossEntropy(torch.tensor([1.0, 5.0, 1.0]))
    loss_u, _ = fn_uniform(logits, targets)
    loss_w, _ = fn_weighted(logits, targets)
    assert not torch.allclose(loss_u, loss_w)


# ============================================================================
# WeightedBCEWithLogits
# ============================================================================


def test_weighted_bce_uniform_equals_F_bce() -> None:
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, 2, (B, K)).float()
    fn = WeightedBCEWithLogits(torch.ones(K))
    loss, n_valid = fn(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-6)
    assert n_valid == B


def test_weighted_bce_applies_pos_weight() -> None:
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, 2, (B, K)).float()
    fn_u = WeightedBCEWithLogits(torch.ones(K))
    fn_w = WeightedBCEWithLogits(torch.tensor([1.0, 10.0, 1.0]))
    loss_u, _ = fn_u(logits, targets)
    loss_w, _ = fn_w(logits, targets)
    assert not torch.allclose(loss_u, loss_w)


# ============================================================================
# FocalCrossEntropy
# ============================================================================


def test_focal_ce_gamma0_equals_ce() -> None:
    """γ=0 reduces to standard CE."""
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 1, 2, 0])
    fn = FocalCrossEntropy(gamma=0.0)
    loss, _ = fn(logits, targets)
    expected = F.cross_entropy(logits, targets, reduction="mean")
    assert torch.allclose(loss, expected, atol=1e-5)


def test_focal_ce_gamma_positive_downweights_easy() -> None:
    """Confident-correct logits → focal loss << CE loss."""
    B, K = 1, 3
    # Make class 0 obvious: logit huge
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0])
    fn_ce = FocalCrossEntropy(gamma=0.0)
    fn_focal = FocalCrossEntropy(gamma=2.0)
    loss_ce, _ = fn_ce(logits, targets)
    loss_focal, _ = fn_focal(logits, targets)
    assert float(loss_focal) < float(loss_ce)
    # And focal should be tiny because (1-p)^2 ≈ 0
    assert float(loss_focal) < 1e-3


def test_focal_ce_masks_null() -> None:
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, NULL_TARGET_SENTINEL, 2, NULL_TARGET_SENTINEL])
    fn = FocalCrossEntropy(gamma=2.0)
    loss, n_valid = fn(logits, targets)
    assert n_valid == 2


def test_focal_ce_alpha_works() -> None:
    """Per-class alpha changes the loss."""
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 1, 2, 0])
    fn_no_alpha = FocalCrossEntropy(gamma=2.0)
    fn_with_alpha = FocalCrossEntropy(gamma=2.0, alpha=torch.tensor([1.0, 5.0, 1.0]))
    l1, _ = fn_no_alpha(logits, targets)
    l2, _ = fn_with_alpha(logits, targets)
    assert not torch.allclose(l1, l2)


# ============================================================================
# FocalBCEWithLogits
# ============================================================================


def test_focal_bce_gamma0_alpha05_equals_half_bce() -> None:
    """γ=0, α=0.5 → loss = 0.5 * BCE (alpha_t = 0.5 for all classes)."""
    torch.manual_seed(0)
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, 2, (B, K)).float()
    fn = FocalBCEWithLogits(gamma=0.0, alpha=0.5)
    loss, _ = fn(logits, targets)
    expected = 0.5 * F.binary_cross_entropy_with_logits(
        logits, targets, reduction="mean"
    )
    assert torch.allclose(loss, expected, atol=1e-5)


def test_focal_bce_signature() -> None:
    B, K = 4, 3
    logits = torch.randn(B, K)
    targets = torch.randint(0, 2, (B, K)).float()
    fn = FocalBCEWithLogits(gamma=2.0, alpha=0.25)
    loss, n_valid = fn(logits, targets)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert n_valid == B


def test_focal_bce_gamma_positive_downweights() -> None:
    """Confident-correct predictions → very small focal BCE loss."""
    # All targets = 1, all logits = +10 (sigmoid → 0.9999)
    logits = torch.full((4, 3), 10.0)
    targets = torch.ones(4, 3)
    fn_bce = FocalBCEWithLogits(gamma=0.0, alpha=0.5)
    fn_focal = FocalBCEWithLogits(gamma=2.0, alpha=0.5)
    loss_bce, _ = fn_bce(logits, targets)
    loss_focal, _ = fn_focal(logits, targets)
    assert float(loss_focal) < float(loss_bce)


# ============================================================================
# attach_loss_fns_to_heads
# ============================================================================


class _MockMultiTaskHeads(nn.Module):
    """Mirror of MultiTaskHeads.heads attribute structure."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for axis in ["topo", "morph"]:
            head = _MockHead(axis)
            self.heads[axis] = head


class _MockHead(nn.Module):
    def __init__(self, axis: str) -> None:
        super().__init__()
        self.axis = axis
        self.proj = nn.Linear(8, 1)
        self._original_called = False

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        # Default behavior — should be replaced by attach.
        self._original_called = True
        return torch.tensor(99.0), 1


class _MockOCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = _MockMultiTaskHeads()


def test_attach_overrides_compute_loss() -> None:
    model = _MockOCE()
    strategy = ClassWeightedCrossEntropy(torch.ones(3), axis="topo")
    attach_loss_fns_to_heads(model, {"topo": strategy})

    head_topo = model.heads.heads["topo"]
    # The original method should no longer fire — instead the strategy runs.
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    loss, n_valid = head_topo.compute_loss(logits, targets)
    assert not head_topo._original_called, \
        "Original compute_loss should not have been called"
    # Loss should be sensible (not 99.0)
    assert float(loss) != 99.0
    assert n_valid == 2


def test_attach_leaves_other_heads_untouched() -> None:
    """Attaching to 'topo' must NOT affect 'morph'."""
    model = _MockOCE()
    strategy = ClassWeightedCrossEntropy(torch.ones(3), axis="topo")
    attach_loss_fns_to_heads(model, {"topo": strategy})

    morph = model.heads.heads["morph"]
    logits = torch.randn(2, 3)
    targets = torch.tensor([0, 1])
    loss, _ = morph.compute_loss(logits, targets)
    assert morph._original_called  # original fired
    assert float(loss) == 99.0  # original sentinel return


def test_attach_strategy_is_a_child_module() -> None:
    """Strategy should appear in named_modules so .to(device) works."""
    model = _MockOCE()
    strategy = ClassWeightedCrossEntropy(torch.ones(3), axis="topo")
    attach_loss_fns_to_heads(model, {"topo": strategy})

    names = [name for name, _ in model.named_modules()]
    assert any("_loss_strategy" in n for n in names), \
        f"_loss_strategy not in named_modules: {names}"


def test_attach_strategy_buffer_moves_with_to() -> None:
    """Class weights move when model.to(device)."""
    # Skip on CPU-only environments — just test it doesn't crash on CPU.
    model = _MockOCE()
    strategy = ClassWeightedCrossEntropy(torch.tensor([1.0, 2.0, 3.0]), axis="topo")
    attach_loss_fns_to_heads(model, {"topo": strategy})
    model.to("cpu")
    head = model.heads.heads["topo"]
    # Strategy stashed as child module
    assert hasattr(head, "_loss_strategy")
    # Run forward to ensure end-to-end works
    loss, _ = head.compute_loss(torch.randn(2, 3), torch.tensor([0, 1]))
    assert torch.isfinite(loss)


def test_attach_empty_loss_fns_is_noop() -> None:
    """attach with empty dict (cfg.train.loss='bce' case) does nothing."""
    model = _MockOCE()
    attach_loss_fns_to_heads(model, {})
    # Original behavior preserved
    head = model.heads.heads["topo"]
    loss, _ = head.compute_loss(torch.randn(2, 3), torch.tensor([0, 1]))
    assert head._original_called
    assert float(loss) == 99.0
