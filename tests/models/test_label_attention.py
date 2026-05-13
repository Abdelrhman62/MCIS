"""Unit tests for src/models/label_attention.py."""

from __future__ import annotations

import pytest
import torch

from src.models.label_attention import (
    AxisAttentionOutput,
    AxisLabelAttention,
    MultiAxisLabelAttention,
)


# -----------------------------------------------------------------------------
# Single-axis attention
# -----------------------------------------------------------------------------


class TestAxisLabelAttention:

    def test_construction_validates_dims(self):
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            AxisLabelAttention(hidden_dim=0, num_labels=5)
        with pytest.raises(ValueError, match="num_labels must be positive"):
            AxisLabelAttention(hidden_dim=32, num_labels=0)

    def test_default_attn_dim_equals_hidden_dim(self):
        m = AxisLabelAttention(hidden_dim=64, num_labels=10)
        assert m.attn_dim == 64

    def test_explicit_attn_dim(self):
        m = AxisLabelAttention(hidden_dim=64, num_labels=10, attn_dim=128)
        assert m.attn_dim == 128
        assert m.input_projection.weight.shape == (128, 64)
        assert m.label_queries.shape == (10, 128)

    def test_output_shapes(self):
        B, T, d, K = 4, 32, 64, 10
        m = AxisLabelAttention(hidden_dim=d, num_labels=K)
        H = torch.randn(B, T, d)
        mask = torch.ones(B, T, dtype=torch.long)

        out = m(H, mask)
        assert isinstance(out, AxisAttentionOutput)
        assert out.context.shape == (B, K, d)
        assert out.weights.shape == (B, K, T)

    def test_attention_weights_sum_to_one_over_unmasked_tokens(self):
        """Per-label attention weights must form a probability distribution
        over the valid tokens."""
        B, T, d, K = 2, 16, 32, 5
        m = AxisLabelAttention(hidden_dim=d, num_labels=K)
        H = torch.randn(B, T, d)
        mask = torch.ones(B, T, dtype=torch.long)

        out = m(H, mask)
        # Sum over T should be 1.0 for every (b, k)
        sums = out.weights.sum(dim=2)  # [B, K]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_padded_positions_get_zero_weight(self):
        """Tokens with attention_mask == 0 must receive zero attention weight,
        regardless of how favorable their hidden states would otherwise be."""
        B, T, d, K = 2, 16, 32, 5
        m = AxisLabelAttention(hidden_dim=d, num_labels=K)
        H = torch.randn(B, T, d)

        # Mask out the second half of all sequences in the batch.
        mask = torch.cat(
            [torch.ones(B, T // 2), torch.zeros(B, T // 2)], dim=1
        ).long()

        out = m(H, mask)
        # Weights on padded positions must be exactly zero.
        padded_weights = out.weights[:, :, T // 2:]
        assert torch.all(padded_weights == 0), (
            "Padded positions must receive zero attention weight"
        )
        # Weights on valid positions still sum to 1.
        valid_sums = out.weights[:, :, :T // 2].sum(dim=2)
        assert torch.allclose(valid_sums, torch.ones_like(valid_sums), atol=1e-5)

    def test_per_record_masking_independent(self):
        """Different records in a batch can have different mask patterns;
        each record's softmax must respect its own mask."""
        d, K = 16, 4
        m = AxisLabelAttention(hidden_dim=d, num_labels=K)
        H = torch.randn(3, 8, d)
        # Record 0: all valid. Record 1: only first 4 valid. Record 2: only first 2.
        mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.long)

        out = m(H, mask)
        # Record 0: all weights nonzero.
        assert (out.weights[0] > 0).all()
        # Record 1: positions 4..7 are zero.
        assert (out.weights[1, :, 4:] == 0).all()
        assert (out.weights[1, :, :4] > 0).all()
        # Record 2: positions 2..7 are zero.
        assert (out.weights[2, :, 2:] == 0).all()
        assert (out.weights[2, :, :2] > 0).all()

    def test_dim_mismatch_raises(self):
        m = AxisLabelAttention(hidden_dim=64, num_labels=5)
        H = torch.randn(2, 16, 32)  # wrong d
        mask = torch.ones(2, 16, dtype=torch.long)
        with pytest.raises(ValueError, match="hidden_dim"):
            m(H, mask)

    def test_mask_shape_mismatch_raises(self):
        m = AxisLabelAttention(hidden_dim=64, num_labels=5)
        H = torch.randn(2, 16, 64)
        mask = torch.ones(2, 8, dtype=torch.long)  # wrong T
        with pytest.raises(ValueError, match="attention_mask shape"):
            m(H, mask)

    def test_gradients_flow_to_label_queries(self):
        """Backprop should produce nonzero gradients on label_queries
        (otherwise the labels wouldn't learn what to attend to)."""
        m = AxisLabelAttention(hidden_dim=16, num_labels=4)
        H = torch.randn(2, 8, 16, requires_grad=True)
        mask = torch.ones(2, 8, dtype=torch.long)

        out = m(H, mask)
        loss = out.context.sum()
        loss.backward()

        assert m.label_queries.grad is not None
        assert m.label_queries.grad.abs().sum() > 0

    def test_context_is_weighted_sum_of_hidden_states(self):
        """Manually verify the output equals the formula context_k = sum_t w_kt * h_t.

        This catches transposition bugs that would otherwise be silent.
        """
        m = AxisLabelAttention(hidden_dim=8, num_labels=3)
        H = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, dtype=torch.long)

        out = m(H, mask)
        # Reconstruct context manually from weights and H.
        # weights: [1, 3, 5], H: [1, 5, 8] -> manual: [1, 3, 8]
        manual_context = out.weights @ H
        assert torch.allclose(out.context, manual_context, atol=1e-6)


# -----------------------------------------------------------------------------
# Multi-axis container
# -----------------------------------------------------------------------------


class TestMultiAxisLabelAttention:

    def test_construction_with_multiple_axes(self):
        axes = {"morphology": 24, "icd11_stem": 27, "topography": 8}
        m = MultiAxisLabelAttention(hidden_dim=64, axis_label_counts=axes)
        assert set(m.axis_names) == {"morphology", "icd11_stem", "topography"}
        assert m.attentions["morphology"].num_labels == 24
        assert m.attentions["icd11_stem"].num_labels == 27
        assert m.attentions["topography"].num_labels == 8

    def test_empty_axis_dict_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            MultiAxisLabelAttention(hidden_dim=64, axis_label_counts={})

    def test_forward_returns_dict_per_axis(self):
        axes = {"a": 5, "b": 3}
        m = MultiAxisLabelAttention(hidden_dim=16, axis_label_counts=axes)
        H = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.long)

        out = m(H, mask)
        assert set(out.keys()) == {"a", "b"}
        assert isinstance(out["a"], AxisAttentionOutput)
        assert out["a"].context.shape == (2, 5, 16)
        assert out["b"].context.shape == (2, 3, 16)

    def test_axis_attentions_have_independent_parameters(self):
        """Each axis should have its own label_queries; updating one should
        not change the other."""
        axes = {"a": 5, "b": 3}
        m = MultiAxisLabelAttention(hidden_dim=16, axis_label_counts=axes)

        a_queries = m.attentions["a"].label_queries.data.clone()
        b_queries = m.attentions["b"].label_queries.data.clone()

        # Modify axis 'a' parameters
        with torch.no_grad():
            m.attentions["a"].label_queries.add_(1.0)

        # Axis 'b' parameters must be unchanged
        assert torch.equal(m.attentions["b"].label_queries.data, b_queries)
        # Axis 'a' parameters must have changed
        assert not torch.equal(m.attentions["a"].label_queries.data, a_queries)

    def test_shared_attn_dim_across_axes(self):
        """If attn_dim is provided, all axes use it."""
        m = MultiAxisLabelAttention(
            hidden_dim=64, axis_label_counts={"a": 5, "b": 3}, attn_dim=128
        )
        assert m.attentions["a"].attn_dim == 128
        assert m.attentions["b"].attn_dim == 128
