"""Focal loss for grid pass A2 (Lin et al. 2017, RetinaNet).

Two strategies:
  - FocalCrossEntropy: focal-modulated softmax CE for single-pick axes.
    Loss = -alpha_c * (1 - p_target)^gamma * log(p_target)
    where p_target = softmax(logits)[target_class].
  - FocalBCEWithLogits: focal BCE for multi-label axes (per-class).
    Loss_c = -alpha_c * (1 - p_c)^gamma * log(p_c)   if y_c = 1
             -(1-alpha_c) * p_c^gamma   * log(1-p_c) if y_c = 0
    summed over classes, mean over batch.

alpha defaults:
  - Single-pick: None (uniform = 1.0 per class). Pass per-axis class
    weights via the `alpha` argument to combine focal + class-weighting
    (grid A2 stacks on A1 if desired).
  - Multi-label: 0.25 (RetinaNet default for rare-positive scenarios,
    where most labels per record are 0).

gamma default: 2.0 (RetinaNet default; matches B0.yaml.train.focal_gamma).

NULL handling: single-pick targets == -1 are masked out.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.heads import NULL_TARGET_SENTINEL


class FocalCrossEntropy(nn.Module):
    """Focal-modulated softmax CE for single-pick axes."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        axis: str = "<unknown>",
    ) -> None:
        """Construct focal CE.

        Args:
            gamma: Focusing parameter (γ). gamma=0 reduces to standard CE.
                gamma=2.0 is RetinaNet default.
            alpha: Optional [K] tensor of per-class weights. If None,
                uniform. Combines focal modulation with class re-weighting.
            axis: Axis name (for error messages).
        """
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        self.axis = axis
        if alpha is not None:
            if alpha.dim() != 1:
                raise ValueError(
                    f"alpha must be 1D, got shape {alpha.shape}"
                )
            self.register_buffer("alpha", alpha.float(), persistent=False)
            self._has_alpha = True
        else:
            self._has_alpha = False

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        if targets.dim() != 1:
            raise ValueError(
                f"axis {self.axis!r}: single-pick targets must be 1D, "
                f"got shape {targets.shape}"
            )
        if logits.dim() != 2 or logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"axis {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} mismatch (expected logits=[B, K])"
            )

        valid_mask = targets != NULL_TARGET_SENTINEL
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype), 0

        valid_logits = logits[valid_mask]    # [n_valid, K]
        valid_targets = targets[valid_mask]  # [n_valid]

        # log_softmax for numerical stability — softmax then log is unstable.
        log_probs = F.log_softmax(valid_logits, dim=-1)  # [n_valid, K]
        # Gather log p(target) per row.
        log_p_target = log_probs.gather(
            1, valid_targets.unsqueeze(1)
        ).squeeze(1)  # [n_valid]
        p_target = log_p_target.exp()  # [n_valid]

        # Focal modulator (1 - p)^gamma. Use clamp on (1-p) for numerical
        # safety when p≈1 and gamma is not an integer.
        focal_weight = (1.0 - p_target).clamp(min=1e-8) ** self.gamma

        # alpha weighting per target class
        if self._has_alpha:
            alpha = self.alpha.to(logits.device).to(logits.dtype)
            alpha_per_row = alpha[valid_targets]
            loss_per_row = -alpha_per_row * focal_weight * log_p_target
        else:
            loss_per_row = -focal_weight * log_p_target

        return loss_per_row.mean(), n_valid


class FocalBCEWithLogits(nn.Module):
    """Focal BCE for multi-label axes (per-class focal modulation).

    Standard RetinaNet formulation generalized to multi-label:
        For each (record, class) pair:
            p = sigmoid(logit)
            if y=1: loss = -alpha * (1-p)^gamma * log(p)
            if y=0: loss = -(1-alpha) * p^gamma * log(1-p)
    Summed over classes, mean over batch.
    """

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: float = 0.25,
        axis: str = "<unknown>",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.gamma = gamma
        self.alpha = alpha
        self.axis = axis

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        if logits.shape != targets.shape:
            raise ValueError(
                f"axis {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} mismatch"
            )

        targets_f = targets.to(logits.dtype)

        # Stable per-element BCE: F.binary_cross_entropy_with_logits with
        # reduction='none' to get per-element loss, then apply focal weight.
        bce_per_elt = F.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="none"
        )  # = -[ y log p + (1-y) log(1-p) ]

        # p_t = p where y=1, (1-p) where y=0
        # Computed as: y*sigmoid + (1-y)*(1-sigmoid) but we want it stable.
        # exp(-bce_per_elt) == p_t (this is the standard trick).
        p_t = torch.exp(-bce_per_elt).clamp(min=1e-8, max=1.0)

        focal_weight = (1.0 - p_t) ** self.gamma  # [B, K]

        # alpha_t = alpha where y=1, (1-alpha) where y=0
        alpha_t = targets_f * self.alpha + (1.0 - targets_f) * (1.0 - self.alpha)

        loss_per_elt = alpha_t * focal_weight * bce_per_elt
        # Mean over (B, K) — matches BCE-with-logits default reduction.
        loss = loss_per_elt.mean()
        return loss, targets.shape[0]
