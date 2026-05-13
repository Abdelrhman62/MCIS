"""Unit tests for src/models/heads.py."""

from __future__ import annotations

import pytest
import torch

from src.models.heads import (
    NULL_TARGET_SENTINEL,
    HeadOutput,
    MultiLabelHead,
    MultiTaskHeads,
    SinglePickHead,
)


# -----------------------------------------------------------------------------
# SinglePickHead
# -----------------------------------------------------------------------------


class TestSinglePickHead:

    def test_construction_validates_dims(self):
        with pytest.raises(ValueError, match="num_labels"):
            SinglePickHead(axis="x", num_labels=0, hidden_dim=16)
        with pytest.raises(ValueError, match="hidden_dim"):
            SinglePickHead(axis="x", num_labels=5, hidden_dim=0)

    def test_forward_shape(self):
        head = SinglePickHead(axis="morphology", num_labels=24, hidden_dim=32)
        ctx = torch.randn(4, 24, 32)
        out = head(ctx)
        assert isinstance(out, HeadOutput)
        assert out.axis == "morphology"
        assert out.logits.shape == (4, 24)

    def test_forward_wrong_K_raises(self):
        head = SinglePickHead(axis="x", num_labels=5, hidden_dim=16)
        with pytest.raises(ValueError, match="expected K=5"):
            head(torch.randn(2, 8, 16))

    def test_forward_wrong_d_raises(self):
        head = SinglePickHead(axis="x", num_labels=5, hidden_dim=16)
        with pytest.raises(ValueError, match="expected hidden_dim=16"):
            head(torch.randn(2, 5, 32))

    def test_loss_no_nulls(self):
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        # Use head's own forward so logits are properly part of the autograd graph.
        ctx = torch.randn(3, 4, 8, requires_grad=True)
        logits = head(ctx).logits
        targets = torch.tensor([0, 2, 1])
        loss, n_valid = head.compute_loss(logits, targets)
        assert n_valid == 3
        assert loss.item() > 0
        assert loss.requires_grad is True

    def test_loss_with_some_nulls_masks_them(self):
        """Records with NULL target should not contribute to loss."""
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(4, 4)
        targets = torch.tensor([0, NULL_TARGET_SENTINEL, 2, NULL_TARGET_SENTINEL])
        loss, n_valid = head.compute_loss(logits, targets)
        assert n_valid == 2

        # Verify loss equals what we'd get computing CE on just the valid records
        valid_logits = logits[[0, 2]]
        valid_targets = torch.tensor([0, 2])
        expected = torch.nn.functional.cross_entropy(valid_logits, valid_targets)
        assert torch.allclose(loss, expected)

    def test_loss_all_nulls_returns_zero(self):
        """All-null batch should return scalar zero loss with n_valid=0."""
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(3, 4)
        targets = torch.tensor([NULL_TARGET_SENTINEL] * 3)
        loss, n_valid = head.compute_loss(logits, targets)
        assert n_valid == 0
        assert loss.item() == 0.0

    def test_loss_target_shape_must_be_1d(self):
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(3, 4)
        targets = torch.tensor([[0], [1], [2]])  # 2D
        with pytest.raises(ValueError, match="must be 1D"):
            head.compute_loss(logits, targets)

    def test_loss_logits_targets_batch_mismatch(self):
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(3, 4)
        targets = torch.tensor([0, 1])  # B=2 but logits has B=3
        with pytest.raises(ValueError, match="mismatch"):
            head.compute_loss(logits, targets)

    def test_gradient_flows_when_valid_records(self):
        head = SinglePickHead(axis="x", num_labels=4, hidden_dim=8)
        ctx = torch.randn(3, 4, 8, requires_grad=True)
        out = head(ctx)
        targets = torch.tensor([0, 1, 2])
        loss, _ = head.compute_loss(out.logits, targets)
        loss.backward()
        assert head.projection.weight.grad is not None
        assert head.projection.weight.grad.abs().sum() > 0


# -----------------------------------------------------------------------------
# MultiLabelHead
# -----------------------------------------------------------------------------


class TestMultiLabelHead:

    def test_forward_shape(self):
        head = MultiLabelHead(axis="ext_anatomy", num_labels=16, hidden_dim=32)
        ctx = torch.randn(4, 16, 32)
        out = head(ctx)
        assert out.logits.shape == (4, 16)
        assert out.axis == "ext_anatomy"

    def test_loss_basic(self):
        head = MultiLabelHead(axis="x", num_labels=5, hidden_dim=8)
        ctx = torch.randn(3, 5, 8, requires_grad=True)
        logits = head(ctx).logits
        targets = torch.tensor(
            [[1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
            dtype=torch.float,
        )
        loss, n_valid = head.compute_loss(logits, targets)
        assert n_valid == 3
        assert loss.item() > 0
        assert loss.requires_grad is True

    def test_loss_all_zero_targets_is_valid(self):
        """Empty list of positive labels = all zeros = valid training signal.
        The model should learn 'no positives for this axis on this record'."""
        head = MultiLabelHead(axis="x", num_labels=5, hidden_dim=8)
        logits = torch.randn(2, 5)
        targets = torch.zeros(2, 5)
        loss, n_valid = head.compute_loss(logits, targets)
        assert n_valid == 2
        assert loss.item() > 0  # Still nonzero — sigmoid outputs aren't exactly 0

    def test_loss_accepts_long_targets(self):
        """Targets sometimes come in as Long from dataloaders; head should
        cast them up to logits.dtype rather than failing."""
        head = MultiLabelHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(2, 4)
        targets = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.long)
        loss, _ = head.compute_loss(logits, targets)
        assert loss.item() > 0

    def test_loss_shape_mismatch_raises(self):
        head = MultiLabelHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(2, 4)
        targets = torch.zeros(2, 5)
        with pytest.raises(ValueError, match="mismatch"):
            head.compute_loss(logits, targets)

    def test_loss_target_shape_must_be_2d(self):
        head = MultiLabelHead(axis="x", num_labels=4, hidden_dim=8)
        logits = torch.randn(2, 4)
        targets = torch.zeros(2)  # 1D
        with pytest.raises(ValueError, match="must be 2D"):
            head.compute_loss(logits, targets)


# -----------------------------------------------------------------------------
# MultiTaskHeads container
# -----------------------------------------------------------------------------


def _make_full_m1_heads(hidden_dim: int = 16) -> MultiTaskHeads:
    """Build the full M1 head set per architecture v6 §6.2."""
    return MultiTaskHeads(
        hidden_dim=hidden_dim,
        single_pick_spec={
            "icdo3_topography": 8,
            "icdo3_morphology": 24,
            "icdo3_behavior": 4,
            "icdo3_grade": 3,
            "icdo3_laterality": 3,
            "icd11_stem": 27,
            "icd11_ext_laterality": 6,
            "icd11_ext_grading": 5,
        },
        multi_label_spec={
            "icd11_ext_anatomy": 16,
            "icd11_ext_histopath": 23,
        },
    )


class TestMultiTaskHeadsConstruction:

    def test_full_m1_construction(self):
        heads = _make_full_m1_heads()
        assert len(heads.heads) == 10
        assert heads.axis_kinds["icdo3_morphology"] == "single"
        assert heads.axis_kinds["icd11_ext_anatomy"] == "multi"
        # Lock v6 §6.2 contract: ext_laterality + ext_grading are single-pick
        # (WHO ECT axis rules; 0/3055 Baheya multi rows).
        assert heads.axis_kinds["icd11_ext_laterality"] == "single"
        assert heads.axis_kinds["icd11_ext_grading"] == "single"
        assert heads.axis_kinds["icd11_ext_histopath"] == "multi"

    def test_empty_specs_rejected(self):
        with pytest.raises(ValueError, match="At least one"):
            MultiTaskHeads(hidden_dim=16, single_pick_spec={}, multi_label_spec={})

    def test_axis_overlap_rejected(self):
        with pytest.raises(ValueError, match="appear in both"):
            MultiTaskHeads(
                hidden_dim=16,
                single_pick_spec={"x": 5},
                multi_label_spec={"x": 5},
            )

    def test_single_only_construction(self):
        heads = MultiTaskHeads(
            hidden_dim=16,
            single_pick_spec={"a": 5},
            multi_label_spec={},
        )
        assert len(heads.heads) == 1


class TestMultiTaskHeadsForward:

    def test_forward_runs_all_heads(self):
        heads = _make_full_m1_heads(hidden_dim=16)
        # Build label_contexts dict with correct shape per axis.
        contexts = {
            "icdo3_topography": torch.randn(2, 8, 16),
            "icdo3_morphology": torch.randn(2, 24, 16),
            "icdo3_behavior": torch.randn(2, 4, 16),
            "icdo3_grade": torch.randn(2, 3, 16),
            "icdo3_laterality": torch.randn(2, 3, 16),
            "icd11_stem": torch.randn(2, 27, 16),
            "icd11_ext_anatomy": torch.randn(2, 16, 16),
            "icd11_ext_histopath": torch.randn(2, 23, 16),
            "icd11_ext_laterality": torch.randn(2, 6, 16),
            "icd11_ext_grading": torch.randn(2, 5, 16),
        }
        outputs = heads(contexts)
        assert set(outputs.keys()) == set(contexts.keys())
        assert outputs["icdo3_morphology"].logits.shape == (2, 24)
        assert outputs["icd11_ext_anatomy"].logits.shape == (2, 16)

    def test_forward_missing_context_raises(self):
        heads = _make_full_m1_heads(hidden_dim=16)
        contexts = {"icdo3_topography": torch.randn(2, 8, 16)}  # missing 9 axes
        with pytest.raises(KeyError, match="missing axes"):
            heads(contexts)


class TestMultiTaskHeadsLoss:

    def _full_contexts(self, B: int = 2, d: int = 16) -> dict[str, torch.Tensor]:
        return {
            "icdo3_topography": torch.randn(B, 8, d, requires_grad=True),
            "icdo3_morphology": torch.randn(B, 24, d, requires_grad=True),
            "icdo3_behavior": torch.randn(B, 4, d, requires_grad=True),
            "icdo3_grade": torch.randn(B, 3, d, requires_grad=True),
            "icdo3_laterality": torch.randn(B, 3, d, requires_grad=True),
            "icd11_stem": torch.randn(B, 27, d, requires_grad=True),
            "icd11_ext_anatomy": torch.randn(B, 16, d, requires_grad=True),
            "icd11_ext_histopath": torch.randn(B, 23, d, requires_grad=True),
            "icd11_ext_laterality": torch.randn(B, 6, d, requires_grad=True),
            "icd11_ext_grading": torch.randn(B, 5, d, requires_grad=True),
        }

    def _full_targets(self, B: int = 2) -> dict[str, torch.Tensor]:
        return {
            "icdo3_topography": torch.zeros(B, dtype=torch.long),
            "icdo3_morphology": torch.zeros(B, dtype=torch.long),
            "icdo3_behavior": torch.zeros(B, dtype=torch.long),
            "icdo3_grade": torch.zeros(B, dtype=torch.long),
            "icdo3_laterality": torch.zeros(B, dtype=torch.long),
            "icd11_stem": torch.zeros(B, dtype=torch.long),
            "icd11_ext_laterality": torch.zeros(B, dtype=torch.long),
            "icd11_ext_grading": torch.zeros(B, dtype=torch.long),
            "icd11_ext_anatomy": torch.zeros(B, 16, dtype=torch.float),
            "icd11_ext_histopath": torch.zeros(B, 23, dtype=torch.float),
        }

    def test_compute_losses_all_axes(self):
        heads = _make_full_m1_heads(hidden_dim=16)
        contexts = self._full_contexts()
        targets = self._full_targets()
        outputs = heads(contexts)
        losses = heads.compute_losses(outputs, targets)
        assert set(losses.keys()) == set(contexts.keys())
        for axis, (loss, n_valid) in losses.items():
            assert loss.item() >= 0

    def test_compute_losses_missing_target_for_active_axis_raises(self):
        heads = _make_full_m1_heads(hidden_dim=16)
        outputs = heads(self._full_contexts())
        targets = self._full_targets()
        del targets["icdo3_morphology"]
        with pytest.raises(KeyError, match="targets missing"):
            heads.compute_losses(outputs, targets)

    def test_inactive_axes_excluded_from_losses(self):
        """When set_phase('icdo3_only'), ICD-11 axes must NOT appear in loss dict."""
        heads = _make_full_m1_heads(hidden_dim=16)
        heads.set_phase("icdo3_only")
        outputs = heads(self._full_contexts())
        # Provide only ICD-O-3 targets — ICD-11 are inactive so missing is OK.
        targets = {
            k: v for k, v in self._full_targets().items()
            if k.startswith("icdo3_")
        }
        losses = heads.compute_losses(outputs, targets)
        assert set(losses.keys()) == {
            "icdo3_topography", "icdo3_morphology", "icdo3_behavior",
            "icdo3_grade", "icdo3_laterality",
        }
        # icd11_stem and all ext_* must be absent
        assert "icd11_stem" not in losses
        assert "icd11_ext_anatomy" not in losses


class TestPhaseControl:

    def test_set_phase_icdo3_only_excludes_icd11(self):
        heads = _make_full_m1_heads()
        heads.set_phase("icdo3_only")
        active = set(heads.active_axes)
        # icd11_stem is NOT in the icdo3_only group per PHASE_AXIS_GROUPS
        assert "icd11_stem" not in active
        assert "icd11_ext_anatomy" not in active
        # All ICD-O-3 axes ARE active
        assert "icdo3_morphology" in active
        assert "icdo3_topography" in active

    def test_set_phase_all_activates_everything(self):
        heads = _make_full_m1_heads()
        heads.set_phase("icdo3_only")
        heads.set_phase("all")
        assert set(heads.active_axes) == set(heads.heads.keys())

    def test_set_phase_unknown_rejected(self):
        heads = _make_full_m1_heads()
        with pytest.raises(ValueError, match="Unknown phase"):
            heads.set_phase("phase3_with_dragons")

    def test_set_active_axes_explicit(self):
        heads = _make_full_m1_heads()
        heads.set_active_axes({"icdo3_morphology", "icd11_stem"})
        assert set(heads.active_axes) == {"icdo3_morphology", "icd11_stem"}

    def test_set_active_axes_ignores_unknown_names(self):
        """Allow configs to list axes not present in this model — useful for
        config templates that span M1/M2 axes."""
        heads = _make_full_m1_heads()
        heads.set_active_axes({"icdo3_morphology", "this_axis_does_not_exist"})
        assert heads.active_axes == ["icdo3_morphology"]

    def test_set_active_axes_empty_rejected(self):
        heads = _make_full_m1_heads()
        with pytest.raises(ValueError, match="empty set"):
            heads.set_active_axes(set())

    def test_inactive_head_gradients_stay_zero_after_full_pass(self):
        """The critical correctness test for the head_active mechanism:
        if a head is inactive, a forward+loss+backward must NOT update its
        projection weights.
        """
        heads = _make_full_m1_heads(hidden_dim=16)
        heads.set_phase("icdo3_only")  # icd11_stem and ext_* are inactive

        contexts = {
            "icdo3_topography": torch.randn(2, 8, 16),
            "icdo3_morphology": torch.randn(2, 24, 16),
            "icdo3_behavior": torch.randn(2, 4, 16),
            "icdo3_grade": torch.randn(2, 3, 16),
            "icdo3_laterality": torch.randn(2, 3, 16),
            "icd11_stem": torch.randn(2, 27, 16),
            "icd11_ext_anatomy": torch.randn(2, 16, 16),
            "icd11_ext_histopath": torch.randn(2, 23, 16),
            "icd11_ext_laterality": torch.randn(2, 6, 16),
            "icd11_ext_grading": torch.randn(2, 5, 16),
        }
        targets = {
            "icdo3_topography": torch.zeros(2, dtype=torch.long),
            "icdo3_morphology": torch.zeros(2, dtype=torch.long),
            "icdo3_behavior": torch.zeros(2, dtype=torch.long),
            "icdo3_grade": torch.zeros(2, dtype=torch.long),
            "icdo3_laterality": torch.zeros(2, dtype=torch.long),
        }

        # Snapshot inactive head weights BEFORE backward
        icd11_stem_before = heads.heads["icd11_stem"].projection.weight.detach().clone()
        ext_anatomy_before = heads.heads["icd11_ext_anatomy"].projection.weight.detach().clone()

        # Forward + loss + backward
        outputs = heads(contexts)
        losses = heads.compute_losses(outputs, targets)
        total = sum(loss for loss, _ in losses.values())
        total.backward()

        # Manually apply a SGD step on every parameter that received gradients
        # to simulate what a real trainer would do.
        with torch.no_grad():
            for p in heads.parameters():
                if p.grad is not None:
                    p.sub_(p.grad * 0.01)

        # Inactive heads' projection weights MUST be unchanged
        icd11_stem_after = heads.heads["icd11_stem"].projection.weight.detach()
        ext_anatomy_after = heads.heads["icd11_ext_anatomy"].projection.weight.detach()
        assert torch.equal(icd11_stem_before, icd11_stem_after), (
            "Inactive icd11_stem weights were updated — head_active mechanism failed"
        )
        assert torch.equal(ext_anatomy_before, ext_anatomy_after), (
            "Inactive ext_anatomy weights were updated — head_active mechanism failed"
        )

    def test_state_dict_preserved_across_phase_changes(self):
        """Phase changes must not affect parameter values — only loss participation."""
        heads = _make_full_m1_heads()
        before = {k: v.clone() for k, v in heads.state_dict().items()}
        heads.set_phase("icdo3_only")
        heads.set_phase("all")
        after = heads.state_dict()
        for k in before:
            assert torch.equal(before[k], after[k]), (
                f"Parameter {k} changed across phase toggles"
            )
