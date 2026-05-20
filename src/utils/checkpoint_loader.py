"""Selective checkpoint loading for MCIS — Phase 1 → Phase 2 transfer.

Architecture v6 §4.4 mandates encoder-only transfer from Phase 1 (TCGA
pretrain, native TCGA ICD-O-3 vocab) to Phase 2 (Baheya fine-tune, full
Baheya vocab with ICD-11). Heads MUST reset between phases because:

  1. Phase 1 has no ICD-11 heads at all (TCGA ICD-11 is silver, dropped).
  2. Phase 2 ICD-O-3 head K may differ from Phase 1 K (different vocab).
  3. Contaminating Phase 1 with Baheya vocab structure breaks the transfer
     claim ("encoder learned oncology vocabulary diversity from TCGA, not
     Baheya's label distribution").

This module also gets used later for E5a/E5b/E6 stacking on top of E2 ckpts,
where we DO want heads loaded (same vocab) — toggle via flags.

Key-pattern filter (matches the OCE param naming in src/models/oce.py):
  - 'encoder.*'   → always loaded if load_encoder=True
  - 'attention.*' → loaded if load_attention=True (default True for E2;
                    safe because attention is per-axis indexed and uses the
                    same axis names across phases — only K_per_axis differs,
                    which is caught by shape mismatch)
  - 'heads.*'     → loaded if load_heads=True (default False for Phase1→2;
                    True for E5a/E5b/E6 stacking on E2 base)

On shape mismatch (e.g. Phase 1 attention has K=8 morph, Phase 2 has K=24):
  - The offending key is skipped with a warning.
  - load_state_dict is called with strict=False so the rest still loads.
  - A summary log line reports total loaded / skipped / shape-mismatched.

Usage:
    from src.utils.checkpoint_loader import load_checkpoint_partial

    report = load_checkpoint_partial(
        model=model,
        ckpt_path="checkpoints/Phase1_TCGA/best.pt",
        load_encoder=True,
        load_attention=True,
        load_heads=False,   # Phase1→Phase2: reset heads
        log=log,
    )
    # report.loaded_keys, report.skipped_keys, report.shape_mismatched

The Trainer should NOT call set_active_axes / set_phase from a Phase 1 ckpt
when initialising Phase 2 — Phase 2's `cfg.data.module='M1'` will already
drive `_set_phase_for_module(...)` to activate all 10 axes after this loader
returns.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


# Param name prefixes — match src/models/oce.py composition.
ENCODER_PREFIX = "encoder."
ATTENTION_PREFIX = "attention."
HEADS_PREFIX = "heads."


@dataclass
class LoadReport:
    """Detailed report of which keys loaded, skipped, or shape-mismatched."""

    loaded_keys: list[str] = field(default_factory=list)
    skipped_by_filter: list[str] = field(default_factory=list)
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    missing_in_ckpt: list[str] = field(default_factory=list)
    unexpected_in_ckpt: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded_keys)} "
            f"filter_skipped={len(self.skipped_by_filter)} "
            f"shape_mismatched={len(self.shape_mismatched)} "
            f"missing={len(self.missing_in_ckpt)} "
            f"unexpected={len(self.unexpected_in_ckpt)}"
        )


def _classify_key(key: str) -> str:
    """Return 'encoder' | 'attention' | 'heads' | 'other'."""
    if key.startswith(ENCODER_PREFIX):
        return "encoder"
    if key.startswith(ATTENTION_PREFIX):
        return "attention"
    if key.startswith(HEADS_PREFIX):
        return "heads"
    return "other"


def _state_hash(state: dict[str, torch.Tensor], prefix: str) -> str:
    """SHA1 over concatenated bytes of all tensors with the given prefix.

    Used for sanity assertions in tests (encoder hash should change after
    load; head hash should NOT change when load_heads=False).
    """
    h = hashlib.sha1()
    for k in sorted(state.keys()):
        if not k.startswith(prefix):
            continue
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def load_checkpoint_partial(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    load_encoder: bool = True,
    load_attention: bool = True,
    load_heads: bool = False,
    map_location: str | torch.device = "cpu",
    log: logging.Logger | None = None,
) -> LoadReport:
    """Load a subset of a checkpoint's state_dict into ``model``.

    Args:
        model: Target model (typically OncologyCodingEngine).
        ckpt_path: Path to a torch.save'd dict with 'model_state_dict' key.
        load_encoder: Load 'encoder.*' params. Default True.
        load_attention: Load 'attention.*' params. Default True. Shape
            mismatches (e.g. different K_per_axis) are skipped with warning.
        load_heads: Load 'heads.*' params. Default False (Phase1→2 resets
            heads). Set True when stacking on a same-vocab base (E5a/E5b/E6).
        map_location: Passed to torch.load.
        log: Optional logger. If None, a module-level logger is used.

    Returns:
        LoadReport with detailed breakdown.

    Raises:
        FileNotFoundError: If ckpt_path doesn't exist.
        KeyError: If checkpoint has no 'model_state_dict' key.
    """
    if log is None:
        log = logging.getLogger("mcis.checkpoint_loader")

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info("Loading checkpoint: %s", ckpt_path)
    log.info(
        "  Flags: load_encoder=%s load_attention=%s load_heads=%s",
        load_encoder, load_attention, load_heads,
    )

    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no 'model_state_dict' key. "
            f"Found keys: {sorted(ckpt.keys())}"
        )

    ckpt_state: dict[str, torch.Tensor] = ckpt["model_state_dict"]
    model_state: dict[str, torch.Tensor] = model.state_dict()

    # Snapshot pre-load hashes for telemetry.
    encoder_hash_before = _state_hash(model_state, ENCODER_PREFIX)
    heads_hash_before = _state_hash(model_state, HEADS_PREFIX)

    # Build the subset to load.
    flag_by_section = {
        "encoder": load_encoder,
        "attention": load_attention,
        "heads": load_heads,
    }
    report = LoadReport()
    subset: dict[str, torch.Tensor] = {}

    for key, ckpt_tensor in ckpt_state.items():
        section = _classify_key(key)
        if section == "other":
            # Unknown prefix — be conservative, skip and log.
            report.skipped_by_filter.append(key)
            log.warning("Skipping unknown-prefix key: %s", key)
            continue

        if not flag_by_section[section]:
            report.skipped_by_filter.append(key)
            continue

        # Filter passed; check shape against model.
        if key not in model_state:
            report.unexpected_in_ckpt.append(key)
            log.warning("Key in checkpoint but not in model: %s", key)
            continue

        model_tensor = model_state[key]
        if model_tensor.shape != ckpt_tensor.shape:
            report.shape_mismatched.append(
                (key, tuple(ckpt_tensor.shape), tuple(model_tensor.shape))
            )
            log.warning(
                "Shape mismatch on %s: ckpt %s vs model %s — skipping",
                key, tuple(ckpt_tensor.shape), tuple(model_tensor.shape),
            )
            continue

        subset[key] = ckpt_tensor
        report.loaded_keys.append(key)

    # Find keys in model but neither in subset nor in ckpt (missing).
    expected_to_load: set[str] = set()
    for key in model_state:
        section = _classify_key(key)
        if section == "other":
            continue
        if flag_by_section[section]:
            expected_to_load.add(key)
    actually_loaded = set(report.loaded_keys)
    # 'missing' = expected to load (by flag) but not in ckpt at all
    missing = expected_to_load - set(ckpt_state.keys())
    report.missing_in_ckpt = sorted(missing)
    if missing:
        log.warning(
            "Keys requested by load flags but missing from ckpt: %d (e.g. %s)",
            len(missing), sorted(missing)[:3],
        )

    # Apply the subset. strict=False because we deliberately skipped a lot.
    incompatible = model.load_state_dict(subset, strict=False)
    # incompatible is a NamedTuple: (missing_keys, unexpected_keys) — these
    # report against the FULL model state_dict, not our subset. They will
    # always have entries here because we filtered. Don't surface them as
    # warnings (we already logged the real story).
    _ = incompatible

    # Post-load sanity hashes.
    post_state = model.state_dict()
    encoder_hash_after = _state_hash(post_state, ENCODER_PREFIX)
    heads_hash_after = _state_hash(post_state, HEADS_PREFIX)

    log.info("Load report: %s", report.summary())
    log.info(
        "  encoder hash: %s -> %s (%s)",
        encoder_hash_before[:8], encoder_hash_after[:8],
        "CHANGED" if encoder_hash_before != encoder_hash_after else "UNCHANGED",
    )
    log.info(
        "  heads hash:   %s -> %s (%s)",
        heads_hash_before[:8], heads_hash_after[:8],
        "CHANGED" if heads_hash_before != heads_hash_after else "UNCHANGED",
    )

    # Sanity gates — surface obvious misconfiguration.
    if load_encoder and encoder_hash_before == encoder_hash_after:
        log.warning(
            "load_encoder=True but encoder weights unchanged after load. "
            "Possible issues: ckpt has no 'encoder.*' keys, or model already "
            "had identical weights. Investigate."
        )
    if not load_heads and heads_hash_before != heads_hash_after:
        log.error(
            "load_heads=False but head weights changed. "
            "This is a logic bug in load_checkpoint_partial — heads should "
            "have been filter-skipped. Failing fast."
        )
        raise RuntimeError("Heads modified despite load_heads=False")

    return report
