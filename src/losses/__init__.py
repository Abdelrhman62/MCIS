"""MCIS loss functions — pluggable per-axis loss strategy.

Default behavior (no overrides): ``SinglePickHead`` uses masked cross-entropy,
``MultiLabelHead`` uses BCE-with-logits. These live inside the head classes
in src/models/heads.py for backward compatibility.

For grid-pass experiments (E5 grid: A1 class-weighted, A2 focal, A3 APL),
use ``build_loss_fn(cfg, vocab, train_df)`` to construct the strategy
objects, then call ``attach_loss_fns_to_heads(model, fns)`` to swap them
in. The trainer calls these two steps from ``Trainer.from_config``.

Loss-function contract:

  single-pick:  fn(logits[B, K], targets[B]) -> (loss_scalar, n_valid)
                Must respect NULL_TARGET_SENTINEL=-1 masking.
  multi-label:  fn(logits[B, K], targets[B, K]) -> (loss_scalar, n_valid)
                No null masking; n_valid = B.

Symmetric signature with SinglePickHead.compute_loss / MultiLabelHead.compute_loss
so they slot in directly via head.loss_fn_override = fn.
"""
from __future__ import annotations

from src.losses.class_weighted import (
    ClassWeightedCrossEntropy,
    WeightedBCEWithLogits,
    compute_inverse_freq_weights_single_pick,
    compute_inverse_freq_weights_multi_label,
)
from src.losses.focal import FocalCrossEntropy, FocalBCEWithLogits
from src.losses.hierarchical import HierarchicalCrossEntropyLoss
from src.losses.builder import (
    LossKind,
    build_loss_fns,
    attach_loss_fns_to_heads,
)

__all__ = [
    "ClassWeightedCrossEntropy",
    "WeightedBCEWithLogits",
    "FocalCrossEntropy",
    "FocalBCEWithLogits",
    "HierarchicalCrossEntropyLoss",
    "compute_inverse_freq_weights_single_pick",
    "compute_inverse_freq_weights_multi_label",
    "LossKind",
    "build_loss_fns",
    "attach_loss_fns_to_heads",
]
