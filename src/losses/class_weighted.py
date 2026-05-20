"""Class-weighted losses for grid pass A1.

Two strategies:
  - ClassWeightedCrossEntropy: per-class weights for single-pick axes.
    Mirrors sklearn's class_weight='balanced' which is what B1 uses; this
    is the apples-to-apples comparison for E1 grid A1.
  - WeightedBCEWithLogits: per-class pos_weight for multi-label axes,
    computed as (#negatives / #positives) per class.

Weights are computed from the *training fold's* label distribution (NEVER
from full data — would leak val into train). The trainer is responsible
for passing the right train_df.

Clipping: weights are clipped at MAX_CLASS_WEIGHT (default 10.0) to prevent
ultra-rare classes (1-2 examples) from dominating the loss. This matches
sklearn 'balanced' which uses n_samples/(n_classes*np.bincount(y)) and is
typically bounded by the inverse of the rarest class proportion.

NULL handling: single-pick targets == -1 are masked out before computing
the loss AND before computing the inverse-freq weights (they're not real
examples of any class).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.heads import NULL_TARGET_SENTINEL


# Default clip for class weights. Equivalent to capping at 10× the mean
# class frequency. Anything more aggressive starts to overpower the
# encoder's representation learning on common classes.
DEFAULT_MAX_CLASS_WEIGHT = 10.0


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_inverse_freq_weights_single_pick(
    targets: Sequence[int] | np.ndarray | torch.Tensor,
    num_classes: int,
    *,
    max_weight: float = DEFAULT_MAX_CLASS_WEIGHT,
    smooth: float = 1.0,
    null_sentinel: int = NULL_TARGET_SENTINEL,
) -> torch.Tensor:
    """Compute per-class inverse-frequency weights for a single-pick axis.

    Formula: w_c = clip( n_valid / (K * (count_c + smooth)), max=max_weight )

    This matches sklearn's class_weight='balanced' with Laplace smoothing
    (smooth=1.0). Zero-count classes get weight = n_valid/K, then clipped.

    Args:
        targets: 1D array of class indices (-1 = null, masked out).
        num_classes: K (axis cardinality).
        max_weight: Clip weights at this value.
        smooth: Laplace-smoothing additive constant (avoid div-by-zero).
        null_sentinel: Value indicating null (default -1).

    Returns:
        FloatTensor [K] of per-class weights.

    Raises:
        ValueError: If num_classes <= 0.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if isinstance(targets, torch.Tensor):
        arr = targets.detach().cpu().numpy()
    else:
        arr = np.asarray(targets)

    valid = arr[arr != null_sentinel]
    n_valid = len(valid)
    if n_valid == 0:
        # No valid targets — return uniform weights (degenerate but safe).
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    weights = n_valid / (num_classes * (counts + smooth))
    weights = np.clip(weights, a_min=None, a_max=max_weight)
    return torch.from_numpy(weights).float()


def compute_inverse_freq_weights_multi_label(
    targets: np.ndarray | torch.Tensor,
    *,
    max_weight: float = DEFAULT_MAX_CLASS_WEIGHT,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute per-class pos_weight for a multi-label axis.

    Formula: pos_weight_c = clip( (n - n_pos_c) / (n_pos_c + smooth), max )

    BCEWithLogits's pos_weight scales the positive term by this factor,
    which is the standard imbalance correction for multi-label BCE.

    Args:
        targets: [N, K] binary array.
        max_weight: Clip pos_weights at this value.
        smooth: Additive smoothing to avoid div-by-zero on zero-positive
            classes.

    Returns:
        FloatTensor [K] of per-class positive-class weights.
    """
    if isinstance(targets, torch.Tensor):
        arr = targets.detach().cpu().numpy()
    else:
        arr = np.asarray(targets)

    if arr.ndim != 2:
        raise ValueError(f"multi-label targets must be 2D, got shape {arr.shape}")

    n = arr.shape[0]
    n_pos = arr.sum(axis=0).astype(np.float64)
    pos_weight = (n - n_pos) / (n_pos + smooth)
    pos_weight = np.clip(pos_weight, a_min=None, a_max=max_weight)
    return torch.from_numpy(pos_weight).float()


# ---------------------------------------------------------------------------
# Loss strategies
# ---------------------------------------------------------------------------


class ClassWeightedCrossEntropy(nn.Module):
    """Drop-in replacement for SinglePickHead's masked CE with class weights.

    Signature matches ``SinglePickHead.compute_loss``: takes (logits, targets)
    and returns (loss_scalar, n_valid). Honors NULL_TARGET_SENTINEL=-1.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        axis: str = "<unknown>",
    ) -> None:
        super().__init__()
        if class_weights.dim() != 1:
            raise ValueError(
                f"class_weights must be 1D, got shape {class_weights.shape}"
            )
        # Buffer so it moves with .to(device) and round-trips state_dict.
        # We DON'T want it in state_dict though (it's data-derived, not
        # learned). Use a non-persistent buffer.
        self.register_buffer(
            "class_weights", class_weights.float(), persistent=False
        )
        self.axis = axis

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        # Mirror SinglePickHead.compute_loss shape checks
        if targets.dim() != 1:
            raise ValueError(
                f"axis {self.axis!r}: single-pick targets must be 1D, "
                f"got shape {targets.shape}"
            )
        K = self.class_weights.shape[0]
        if logits.shape != (targets.shape[0], K):
            raise ValueError(
                f"axis {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} mismatch (expected [B, K] with K={K})"
            )

        valid_mask = targets != NULL_TARGET_SENTINEL
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype), 0

        loss = F.cross_entropy(
            logits[valid_mask],
            targets[valid_mask],
            weight=self.class_weights.to(logits.device).to(logits.dtype),
            reduction="mean",
        )
        return loss, n_valid


class WeightedBCEWithLogits(nn.Module):
    """Drop-in for MultiLabelHead's BCE with per-class pos_weight.

    Same signature as ``MultiLabelHead.compute_loss``: (loss_scalar, n_valid).
    No null masking — multi-label has none.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        *,
        axis: str = "<unknown>",
    ) -> None:
        super().__init__()
        if pos_weight.dim() != 1:
            raise ValueError(
                f"pos_weight must be 1D, got shape {pos_weight.shape}"
            )
        self.register_buffer("pos_weight", pos_weight.float(), persistent=False)
        self.axis = axis

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        K = self.pos_weight.shape[0]
        if logits.shape != targets.shape or logits.shape[-1] != K:
            raise ValueError(
                f"axis {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} (expected matching [B, K] with K={K})"
            )
        targets_f = targets.to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_f,
            pos_weight=self.pos_weight.to(logits.device).to(logits.dtype),
            reduction="mean",
        )
        return loss, targets.shape[0]
