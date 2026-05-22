"""Loss builder — wires cfg.train.loss + train_df into per-axis loss objects.

Usage in Trainer.from_config:

    from src.losses import build_loss_fns, attach_loss_fns_to_heads

    if cfg.train.loss != "bce":
        loss_fns = build_loss_fns(cfg, vocab, train_df)
        attach_loss_fns_to_heads(model, loss_fns, log=log)

Backward compatibility: when cfg.train.loss == "bce" (default), this module
is not invoked at all and heads use their built-in CE / BCE.

Supported loss kinds (cfg.train.loss):
  - "bce"             → no override; built-in CE / BCE (default)
  - "class_weighted"  → grid A1: inverse-freq weights, clipped at 10×
  - "focal"           → grid A2: focal CE / focal BCE, gamma from cfg
  - "focal_weighted"  → focal CE + class weights as alpha (A1 ⊕ A2)
  - "apl"             → grid A3: NOT IMPLEMENTED yet (deferred, see plan)
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.label_vocab import LabelVocab
from src.losses.class_weighted import (
    ClassWeightedCrossEntropy,
    WeightedBCEWithLogits,
    compute_inverse_freq_weights_multi_label,
    compute_inverse_freq_weights_single_pick,
)
from src.losses.focal import FocalBCEWithLogits, FocalCrossEntropy
from src.models.heads import NULL_TARGET_SENTINEL


class LossKind(str, Enum):
    BCE = "bce"
    CLASS_WEIGHTED = "class_weighted"
    FOCAL = "focal"
    FOCAL_WEIGHTED = "focal_weighted"
    APL = "apl"


def _extract_single_pick_targets(
    train_df: pd.DataFrame, axis: str, vocab: "LabelVocab | None" = None
) -> np.ndarray:
    """Get [N] array of class indices (-1 = null) for a single-pick axis.

    The parquet schema (per dataset loaders.py) uses one of these patterns:
      - column `{axis}_idx` (int, -1 for null)  ← preferred
      - column `{axis}` (str class label, NaN/None for null) — fallback,
        requires vocab.label_to_idx mapping

    This builder prefers the indexed form because it's cheaper and matches
    what BaheyaM1Dataset.__getitem__ produces. If unavailable, falls back.

    Returns:
        np.ndarray of shape [N], dtype int64, -1 for null.
    """
    idx_col = f"{axis}_idx"
    if idx_col in train_df.columns:
        arr = train_df[idx_col].to_numpy()
        # pandas may carry int64 with NaN as float; coerce to int64.
        if arr.dtype != np.int64:
            arr = pd.to_numeric(arr, errors="coerce").to_numpy()
            arr = np.where(np.isnan(arr), NULL_TARGET_SENTINEL, arr).astype(np.int64)
        return arr
    # Fallback: raw string column — encode using vocab
    if axis in train_df.columns and vocab is not None:
        codes = train_df[axis].tolist()
        arr = np.array(
            [vocab.axes[axis].code_to_idx.get(c, NULL_TARGET_SENTINEL)
             if isinstance(c, str) else NULL_TARGET_SENTINEL
             for c in codes],
            dtype=np.int64,
        )
        return arr
    raise KeyError(
        f"single-pick axis {axis!r}: neither {idx_col!r} nor a labeled string "
        f"column found in train_df. Available cols: "
        f"{sorted(train_df.columns)[:20]}..."
    )


def _extract_multi_label_targets(
    train_df: pd.DataFrame, axis: str, K: int,
    vocab: "LabelVocab | None" = None
) -> np.ndarray:
    """Get [N, K] binary matrix for a multi-label axis."""
    col = f"{axis}_multihot"
    if col not in train_df.columns:
        # Fallback: pipe-delimited string column
        if axis in train_df.columns and vocab is not None:
            matrix = np.zeros((len(train_df), K), dtype=np.int64)
            c2i = vocab.axes[axis].code_to_idx
            delim = vocab.axes[axis].delimiter or "|"
            for i, val in enumerate(train_df[axis].tolist()):
                if not isinstance(val, str) or not val.strip():
                    continue
                for code in val.split(delim):
                    code = code.strip()
                    if code in c2i:
                        matrix[i, c2i[code]] = 1
            return matrix
        raise KeyError(
            f"multi-label axis {axis!r}: column {col!r} not found. "
            f"Available cols: {sorted(train_df.columns)[:20]}..."
        )
    rows = train_df[col].tolist()
    matrix = np.zeros((len(rows), K), dtype=np.int64)
    for i, row in enumerate(rows):
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.int64)
        if arr.shape != (K,):
            raise ValueError(
                f"multi-label axis {axis!r}: row {i} has shape {arr.shape}, "
                f"expected ({K},)"
            )
        matrix[i] = arr
    return matrix


def build_loss_fns(
    cfg: Any,  # BenchmarkConfig — typed loosely to avoid circular import
    vocab: LabelVocab,
    train_df: pd.DataFrame,
    *,
    log: logging.Logger | None = None,
) -> dict[str, torch.nn.Module]:
    """Build per-axis loss objects per cfg.train.loss.

    Args:
        cfg: BenchmarkConfig. Reads cfg.train.loss, cfg.train.focal_gamma,
            cfg.axis_types.single_pick, cfg.axis_types.multilabel.
        vocab: LabelVocab — gives num_classes per axis.
        train_df: Training fold DataFrame. Source of class frequencies.
        log: Logger (optional).

    Returns:
        dict mapping axis -> nn.Module. Empty dict for kind="bce"
        (caller should skip the attach step in that case).

    Raises:
        NotImplementedError: For kind="apl".
        ValueError: For unknown kind.
    """
    if log is None:
        log = logging.getLogger("mcis.losses.builder")

    try:
        kind = LossKind(cfg.train.loss)
    except ValueError:
        raise ValueError(
            f"Unknown cfg.train.loss={cfg.train.loss!r}. "
            f"Valid: {[k.value for k in LossKind]}"
        )

    if kind == LossKind.BCE:
        log.info("Loss: bce (default heads' CE / BCE) — no override")
        return {}

    if kind == LossKind.APL:
        raise NotImplementedError(
            "APL (Niemi 2025) loss not yet implemented — see plan §2 prereqs"
        )

    gamma = float(getattr(cfg.train, "focal_gamma", 2.0))
    single_pick_axes = list(cfg.axis_types.single_pick)
    multilabel_axes = list(cfg.axis_types.multilabel)

    log.info(
        "Loss: %s (gamma=%.2f; clip class_weight at 10×)",
        kind.value, gamma,
    )

    fns: dict[str, torch.nn.Module] = {}

    # Single-pick axes
    for axis in single_pick_axes:
        if axis not in vocab.axes:
            log.warning("Axis %r not in vocab — skipping loss build", axis)
            continue
        K = vocab[axis].num_classes
        targets = _extract_single_pick_targets(train_df, axis, vocab)
        weights = compute_inverse_freq_weights_single_pick(
            targets, num_classes=K
        )
        log.info(
            "  axis=%-22s K=%2d  weight_range=[%.2f, %.2f]  null=%d/%d",
            axis, K, float(weights.min()), float(weights.max()),
            int((targets == NULL_TARGET_SENTINEL).sum()), len(targets),
        )

        if kind == LossKind.CLASS_WEIGHTED:
            fns[axis] = ClassWeightedCrossEntropy(weights, axis=axis)
        elif kind == LossKind.FOCAL:
            fns[axis] = FocalCrossEntropy(gamma=gamma, axis=axis)
        elif kind == LossKind.FOCAL_WEIGHTED:
            fns[axis] = FocalCrossEntropy(
                gamma=gamma, alpha=weights, axis=axis
            )

    # Multi-label axes
    for axis in multilabel_axes:
        if axis not in vocab.axes:
            log.warning("Axis %r not in vocab — skipping loss build", axis)
            continue
        K = vocab[axis].num_classes
        targets = _extract_multi_label_targets(train_df, axis, K, vocab)
        pos_weight = compute_inverse_freq_weights_multi_label(targets)
        log.info(
            "  axis=%-22s K=%2d  pos_weight_range=[%.2f, %.2f]  n_pos_mean=%.1f",
            axis, K, float(pos_weight.min()), float(pos_weight.max()),
            float(targets.sum(axis=0).mean()),
        )

        if kind == LossKind.CLASS_WEIGHTED:
            fns[axis] = WeightedBCEWithLogits(pos_weight, axis=axis)
        elif kind == LossKind.FOCAL:
            fns[axis] = FocalBCEWithLogits(gamma=gamma, alpha=0.25, axis=axis)
        elif kind == LossKind.FOCAL_WEIGHTED:
            # For multi-label, we don't stack pos_weight under focal because
            # the focal alpha already serves a similar purpose. Just use focal.
            fns[axis] = FocalBCEWithLogits(gamma=gamma, alpha=0.25, axis=axis)

    return fns


def attach_loss_fns_to_heads(
    model: torch.nn.Module,
    loss_fns: dict[str, torch.nn.Module],
    *,
    log: logging.Logger | None = None,
) -> None:
    """Override per-axis ``head.compute_loss`` with the provided strategies.

    Mechanism: monkey-patch ``head.compute_loss`` (bound method) to call the
    strategy. We also stash the strategy as a child module on the head so it
    moves with .to(device) and is included in state_dict (the strategy's
    buffers are non-persistent so the actual class weights don't leak into
    the checkpoint — only the strategy module structure does, which is fine).

    Args:
        model: The OncologyCodingEngine (or any model with ``.heads.heads``
            ModuleDict of head modules per src/models/heads.py).
        loss_fns: Mapping axis -> nn.Module strategy.
        log: Logger (optional).
    """
    if log is None:
        log = logging.getLogger("mcis.losses.builder")

    if not loss_fns:
        return

    # Discover heads container. In OCE: model.heads is MultiTaskHeads; its
    # ModuleDict of per-axis heads is model.heads.heads.
    try:
        heads_dict = model.heads.heads  # ModuleDict[axis -> head]
    except AttributeError as e:
        raise AttributeError(
            "attach_loss_fns_to_heads requires model.heads.heads "
            "(MultiTaskHeads.heads ModuleDict). Did the model schema change?"
        ) from e

    attached = 0
    for axis, fn in loss_fns.items():
        if axis not in heads_dict:
            log.warning(
                "Loss fn for axis %r built but no matching head present — "
                "skipping attach", axis
            )
            continue
        head = heads_dict[axis]
        # Register strategy as a child module so it moves with .to(device)
        # and any (non-persistent) buffers go along.
        head.add_module("_loss_strategy", fn)
        # Patch compute_loss to delegate to the strategy.
        def _make_compute_loss(strategy: torch.nn.Module):
            def compute_loss(
                self_head, logits: torch.Tensor, targets: torch.Tensor
            ) -> tuple[torch.Tensor, int]:
                return strategy(logits, targets)
            return compute_loss
        # Bind to the instance (not the class) so other axes are unaffected.
        import types
        head.compute_loss = types.MethodType(_make_compute_loss(fn), head)
        attached += 1
        log.info("Attached loss strategy %s to head %r",
                 type(fn).__name__, axis)

    log.info("attach_loss_fns_to_heads: attached=%d of %d",
             attached, len(loss_fns))
