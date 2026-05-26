"""Per-axis LAAT-style label-wise attention.

For each prediction axis (e.g., ICD-O-3 morphology, ICD-11 stem), this module
learns per-label query vectors and produces per-label context representations
by attending over the encoder's token outputs.

Design decisions (locked in planning):
- One AxisLabelAttention module per axis, not a shared module across axes.
  Each axis has different label semantics; per-axis attention lets each axis
  learn what tokens matter for *that axis* rather than averaging.
- LAAT formulation (Vu et al. 2020), not CAML (Mullenbach 2018):
    attention scores  = tanh(H @ W) @ U^T        (LAAT — uses an MLP first)
    vs. CAML:
    attention scores  = H @ U^T                   (CAML — direct dot product)
  LAAT's intermediate projection consistently outperformed CAML on MIMIC and
  is what PLM-ICD inherits. We follow PLM-ICD.
- Mask-aware softmax: padded positions (attention_mask == 0) are excluded
  from the softmax via -inf logits. Without this, padding tokens get
  non-zero attention weight and pollute the per-label representation.
- Returns both the per-label context AND the attention weights. The weights
  are needed at inference for the reviewer-dashboard evidence-span feature
  (per architecture v6 §3.3).

Output contract for each axis:
    forward(H: [B, T, d], mask: [B, T])
        -> (context: [B, K_axis, d], weights: [B, K_axis, T])

where K_axis is the per-axis label count (e.g., K=24 for morphology).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AxisAttentionOutput:
    """Output of one AxisLabelAttention.forward call.

    Attributes:
        context: float tensor [B, K, d]. Per-label attended representation.
        weights: float tensor [B, K, T]. Attention weights, one distribution
            per label over input tokens. Sums to 1 along T (excluding padded
            positions which are 0). Used for evidence-span extraction.
    """

    context: torch.Tensor
    weights: torch.Tensor


class AxisLabelAttention(nn.Module):
    """LAAT-style label-wise attention for one prediction axis.

    For axis with K labels, hidden_dim d, and an attention projection of size
    attn_dim a:
        W in R^{d x a}     — input projection (shared across labels)
        U in R^{K x a}     — per-label query (one per label in the axis)

    Score computation:
        Z = tanh(H @ W)                           [B, T, a]
        scores = Z @ U^T                          [B, T, K]
        weights = softmax(scores, dim=token)      [B, T, K]
        context = weights^T @ H                   [B, K, d]

    The shared W is what differs from raw CAML; it gives the model an
    intermediate non-linear transformation before computing per-label
    similarity, and is the empirically motivated LAAT contribution.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        attn_dim: int | None = None,
        description_embeddings: torch.Tensor | None = None,
    ) -> None:
        """Initialize attention for one axis.

        Args:
            hidden_dim: Encoder hidden dimension d (e.g., 768 for BERT-base).
            num_labels: Number of labels K in this axis (e.g., 24 for
                morphology). Must be positive.
            attn_dim: Attention projection dimension a. Defaults to
                hidden_dim. PLM-ICD uses hidden_dim; LAAT paper used 512
                regardless of encoder. We follow PLM-ICD's default.
            description_embeddings: Optional [K, attn_dim] tensor of
                pre-computed embeddings (e.g., PubMedBERT [CLS] vectors
                of label descriptions). When provided, label_queries are
                initialized from these instead of Xavier uniform. The
                embeddings are still trainable — this only changes init.

        Raises:
            ValueError: If hidden_dim or num_labels is non-positive.
        """
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {num_labels}")

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.attn_dim = attn_dim if attn_dim is not None else hidden_dim

        # Shared projection W: applied to encoder outputs before similarity.
        self.input_projection = nn.Linear(hidden_dim, self.attn_dim, bias=False)

        # Per-label queries U: one row per label in this axis.
        self.label_queries = nn.Parameter(
            torch.empty(num_labels, self.attn_dim)
        )

        if description_embeddings is not None:
            # E9: Initialize from pre-computed description embeddings
            assert description_embeddings.shape == (num_labels, self.attn_dim), (
                f"description_embeddings shape {description_embeddings.shape} "
                f"!= expected ({num_labels}, {self.attn_dim})"
            )
            with torch.no_grad():
                self.label_queries.copy_(description_embeddings)
            # Still init the projection with Xavier
            nn.init.xavier_uniform_(self.input_projection.weight)
        else:
            self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize parameters per LAAT defaults.

        - Linear projection uses Xavier uniform (PyTorch default for Linear is
          Kaiming uniform; we override to match LAAT's tanh-followed
          initialization, which is Xavier-appropriate).
        - Label queries use Xavier uniform on a 2D parameter.
        """
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.label_queries)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> AxisAttentionOutput:
        """Compute per-label attended representations.

        Args:
            hidden_states: [B, T, d] encoder token outputs.
            attention_mask: [B, T] in {0, 1}. 1 = valid token, 0 = padding.

        Returns:
            AxisAttentionOutput with:
                context: [B, K, d]
                weights: [B, K, T]

        Raises:
            ValueError: If shapes mismatch the configured dimensions.
        """
        B, T, d = hidden_states.shape
        if d != self.hidden_dim:
            raise ValueError(
                f"hidden_states last dim {d} != configured hidden_dim "
                f"{self.hidden_dim}"
            )
        if attention_mask.shape != (B, T):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} != "
                f"hidden_states leading shape ({B}, {T})"
            )

        # Step 1: project encoder outputs.  Z: [B, T, a]
        Z = torch.tanh(self.input_projection(hidden_states))

        # Step 2: compute per-label scores.  scores: [B, T, K]
        # label_queries: [K, a] -> transpose to [a, K] for matmul
        scores = Z @ self.label_queries.t()

        # Step 3: mask out padded positions BEFORE softmax. Setting masked
        # logits to -inf gives them zero weight after softmax. We expand the
        # mask along the K dimension.
        # mask shape: [B, T] -> [B, T, 1] -> broadcast to [B, T, K]
        # Note: very_negative (not -inf) used to avoid NaN if a row is all-masked
        # (which shouldn't happen but is defensive). With float -inf, softmax of
        # all -inf rows yields NaN.
        very_negative = torch.finfo(scores.dtype).min
        mask_expanded = attention_mask.unsqueeze(-1).bool()  # [B, T, 1]
        scores = scores.masked_fill(~mask_expanded, very_negative)

        # Step 4: softmax over the token dimension.  weights: [B, T, K]
        weights_t = torch.softmax(scores, dim=1)

        # Step 5: per-label context = sum over tokens of weight_kt * h_t
        # weights_t: [B, T, K] -> transpose to [B, K, T] then matmul [B, T, d]
        weights = weights_t.transpose(1, 2)  # [B, K, T]
        context = weights @ hidden_states     # [B, K, d]

        return AxisAttentionOutput(context=context, weights=weights)


class MultiAxisLabelAttention(nn.Module):
    """Container for one AxisLabelAttention module per axis.

    Holds attention modules in a ModuleDict keyed by axis name. Calling
    `forward` runs all axes' attention on the same encoder output and returns
    a dict of per-axis outputs.

    This is a convenience wrapper; the heads layer can construct its own
    AxisLabelAttention instances directly if more control is needed.
    """

    def __init__(
        self,
        hidden_dim: int,
        axis_label_counts: dict[str, int],
        attn_dim: int | None = None,
        description_embeddings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Initialize multi-axis attention.

        Args:
            hidden_dim: Encoder hidden dimension d.
            axis_label_counts: Mapping from axis name to number of labels for
                that axis. E.g., {'morphology': 24, 'icd11_stem': 27, ...}.
            attn_dim: Attention projection dimension, shared across axes.
                Defaults to hidden_dim.
            description_embeddings: Optional mapping axis_name -> [K, attn_dim]
                tensor of pre-computed label description embeddings. Axes not
                present in the dict use Xavier init (default).

        Raises:
            ValueError: If axis_label_counts is empty.
        """
        super().__init__()
        if not axis_label_counts:
            raise ValueError("axis_label_counts must be non-empty")

        self.hidden_dim = hidden_dim
        self.axis_names: list[str] = list(axis_label_counts.keys())

        desc_embs = description_embeddings or {}
        self.attentions = nn.ModuleDict(
            {
                axis: AxisLabelAttention(
                    hidden_dim=hidden_dim,
                    num_labels=k,
                    attn_dim=attn_dim,
                    description_embeddings=desc_embs.get(axis),
                )
                for axis, k in axis_label_counts.items()
            }
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, AxisAttentionOutput]:
        """Run per-axis attention.

        Args:
            hidden_states: [B, T, d] encoder outputs.
            attention_mask: [B, T] mask.

        Returns:
            Dict mapping axis name -> AxisAttentionOutput.
        """
        return {
            axis: self.attentions[axis](hidden_states, attention_mask)
            for axis in self.axis_names
        }
