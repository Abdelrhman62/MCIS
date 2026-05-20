"""Tests for src.utils.checkpoint_loader.

Strategy: build a tiny nn.Module mirroring the OCE param-naming convention
(self.encoder.*, self.attention.*, self.heads.*) — small enough to run in
milliseconds, large enough to exercise every code path.

What we verify:
  1. Phase1→Phase2 default: encoder + attention load, heads reset
  2. Same-vocab stacking: load_heads=True loads everything
  3. Shape mismatch on attention/heads is skipped, not crashed
  4. encoder hash CHANGES when loaded; heads hash UNCHANGED when not loaded
  5. Hard fail if loader accidentally mutates heads when load_heads=False
  6. Missing ckpt path raises FileNotFoundError
  7. Checkpoint without 'model_state_dict' key raises KeyError
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.utils.checkpoint_loader import (
    LoadReport,
    load_checkpoint_partial,
)


class _TinyOCE(nn.Module):
    """Mirrors OCE.{encoder, attention, heads} naming for testing.

    Allows per-axis K to be configured so we can simulate vocab mismatches.
    """

    def __init__(
        self,
        encoder_hidden: int = 16,
        attn_axis_K: dict[str, int] | None = None,
        head_axis_K: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        attn_axis_K = attn_axis_K or {"topo": 8, "morph": 24}
        head_axis_K = head_axis_K or {"topo": 8, "morph": 24}

        # Tiny encoder: a couple of Linear layers
        self.encoder = nn.Sequential(
            nn.Linear(32, encoder_hidden),
            nn.Linear(encoder_hidden, encoder_hidden),
        )
        # Attention: one Linear per axis, output K
        self.attention = nn.ModuleDict({
            axis: nn.Linear(encoder_hidden, K)
            for axis, K in attn_axis_K.items()
        })
        # Heads: one Linear per axis, output K
        self.heads = nn.ModuleDict({
            axis: nn.Linear(encoder_hidden, K)
            for axis, K in head_axis_K.items()
        })


def _save_ckpt(model: nn.Module, path: Path) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


def _all_params_equal(m1: nn.Module, m2: nn.Module, prefix: str) -> bool:
    s1, s2 = m1.state_dict(), m2.state_dict()
    keys = [k for k in s1 if k.startswith(prefix)]
    return all(torch.equal(s1[k], s2[k]) for k in keys)


# ---------------------------------------------------------------------------
# Test 1: Phase1→Phase2 default (encoder + attention load; heads reset)
# ---------------------------------------------------------------------------


def test_phase1_to_phase2_default(tmp_path: Path) -> None:
    """Encoder + attention load; heads must remain at fresh init."""
    torch.manual_seed(0)
    source = _TinyOCE()
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)  # fresh init for target — different from source
    target = _TinyOCE()

    # Sanity: source and target must start with DIFFERENT weights everywhere
    assert not _all_params_equal(source, target, "encoder.")
    assert not _all_params_equal(source, target, "attention.")
    assert not _all_params_equal(source, target, "heads.")

    report = load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=True,
        load_heads=False,  # the critical default
    )

    # Encoder + attention should now match source
    assert _all_params_equal(source, target, "encoder."), \
        "encoder weights should match source after load"
    assert _all_params_equal(source, target, "attention."), \
        "attention weights should match source after load"
    # Heads should still differ from source
    assert not _all_params_equal(source, target, "heads."), \
        "heads must NOT have been loaded (load_heads=False)"

    # Report invariants
    enc_keys = [k for k in source.state_dict() if k.startswith("encoder.")]
    attn_keys = [k for k in source.state_dict() if k.startswith("attention.")]
    head_keys = [k for k in source.state_dict() if k.startswith("heads.")]
    assert len(report.loaded_keys) == len(enc_keys) + len(attn_keys)
    assert len(report.skipped_by_filter) == len(head_keys)
    assert len(report.shape_mismatched) == 0


# ---------------------------------------------------------------------------
# Test 2: Same-vocab stacking (load_heads=True loads everything)
# ---------------------------------------------------------------------------


def test_same_vocab_stacking_loads_all(tmp_path: Path) -> None:
    """E5a/E5b on E2 base: full load including heads."""
    torch.manual_seed(0)
    source = _TinyOCE()
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)
    target = _TinyOCE()

    report = load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=True,
        load_heads=True,
    )

    assert _all_params_equal(source, target, "encoder.")
    assert _all_params_equal(source, target, "attention.")
    assert _all_params_equal(source, target, "heads.")
    assert len(report.shape_mismatched) == 0
    assert len(report.skipped_by_filter) == 0


# ---------------------------------------------------------------------------
# Test 3: Shape mismatch on attention/heads is skipped, not crashed
# ---------------------------------------------------------------------------


def test_shape_mismatch_attention_skipped(tmp_path: Path) -> None:
    """Phase 1 had attention K=8 for topo; Phase 2 has K=12. Skip + warn."""
    torch.manual_seed(0)
    source = _TinyOCE(attn_axis_K={"topo": 8, "morph": 24})
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)
    target = _TinyOCE(attn_axis_K={"topo": 12, "morph": 24})

    report = load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=True,
        load_heads=False,
    )

    # encoder should load fine
    assert _all_params_equal(source, target, "encoder.")
    # morph attention should load (same K=24)
    assert torch.equal(
        source.state_dict()["attention.morph.weight"],
        target.state_dict()["attention.morph.weight"],
    )
    # topo attention should be SKIPPED (shape mismatch)
    assert not torch.equal(
        source.state_dict()["attention.topo.weight"],
        target.state_dict()["attention.topo.weight"],
    )
    # Report should list the topo keys as shape_mismatched
    mismatched_keys = [k for k, _, _ in report.shape_mismatched]
    assert any("attention.topo" in k for k in mismatched_keys)
    # Specifically, topo.weight should be there with the right shapes
    weight_entry = [
        (k, src_sh, tgt_sh)
        for k, src_sh, tgt_sh in report.shape_mismatched
        if k == "attention.topo.weight"
    ]
    assert len(weight_entry) == 1
    _, src_sh, tgt_sh = weight_entry[0]
    assert src_sh == (8, 16)   # source: K=8, hidden=16
    assert tgt_sh == (12, 16)  # target: K=12, hidden=16


# ---------------------------------------------------------------------------
# Test 4: Heads must NOT be mutated when load_heads=False
#         (the hard-fail safety check inside the loader)
# ---------------------------------------------------------------------------


def test_heads_invariance_when_load_heads_false(tmp_path: Path) -> None:
    """Snapshot heads, run loader with load_heads=False, assert byte-identical."""
    torch.manual_seed(0)
    source = _TinyOCE()
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)
    target = _TinyOCE()

    # Snapshot
    head_snapshot = {
        k: v.clone() for k, v in target.state_dict().items() if k.startswith("heads.")
    }

    load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=True,
        load_heads=False,
    )

    for k, v_before in head_snapshot.items():
        v_after = target.state_dict()[k]
        assert torch.equal(v_before, v_after), \
            f"Head param {k} changed despite load_heads=False"


# ---------------------------------------------------------------------------
# Test 5: Encoder-only load (load_attention=False, load_heads=False)
# ---------------------------------------------------------------------------


def test_encoder_only_load(tmp_path: Path) -> None:
    """Strictest transfer: only encoder loaded, attention and heads stay fresh."""
    torch.manual_seed(0)
    source = _TinyOCE()
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)
    target = _TinyOCE()

    load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=False,
        load_heads=False,
    )

    assert _all_params_equal(source, target, "encoder.")
    assert not _all_params_equal(source, target, "attention.")
    assert not _all_params_equal(source, target, "heads.")


# ---------------------------------------------------------------------------
# Test 6: Missing file raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_ckpt_raises(tmp_path: Path) -> None:
    target = _TinyOCE()
    with pytest.raises(FileNotFoundError):
        load_checkpoint_partial(
            model=target,
            ckpt_path=tmp_path / "does_not_exist.pt",
        )


# ---------------------------------------------------------------------------
# Test 7: Checkpoint without 'model_state_dict' key raises KeyError
# ---------------------------------------------------------------------------


def test_malformed_ckpt_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pt"
    torch.save({"not_a_state_dict": {}}, bad_path)
    target = _TinyOCE()
    with pytest.raises(KeyError, match="model_state_dict"):
        load_checkpoint_partial(model=target, ckpt_path=bad_path)


# ---------------------------------------------------------------------------
# Test 8: LoadReport summary string is informative
# ---------------------------------------------------------------------------


def test_load_report_summary(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = _TinyOCE()
    ckpt_path = tmp_path / "source.pt"
    _save_ckpt(source, ckpt_path)

    torch.manual_seed(42)
    target = _TinyOCE()
    report = load_checkpoint_partial(
        model=target,
        ckpt_path=ckpt_path,
        load_encoder=True,
        load_attention=True,
        load_heads=False,
    )
    s = report.summary()
    assert "loaded=" in s
    assert "filter_skipped=" in s
    assert "shape_mismatched=" in s
