"""MCIS Oncology Coding Engine (OCE) — top-level composition module.

Composes the three building blocks into one nn.Module with a single forward
pass:

    text -> OncologyEncoder        -> hidden_states [B, T, d] + mask [B, T]
         -> MultiAxisLabelAttention -> per-axis context [B, K_axis, d] + weights [B, K_axis, T]
         -> MultiTaskHeads          -> per-axis logits  [B, K_axis]

Design decisions inherited from previous planning:

- Phase control is fully delegated to self.heads (set_phase, set_active_axes).
  OCE itself has no phase-aware state. This keeps `load_state_dict()` clean
  across Phase 1 -> Phase 2.
- Attention weights are always returned in OCEOutput (the multi-axis attention
  always computes them). The encoder's transformer self-attentions are
  optional via the `return_encoder_attentions` flag (defaults False — they're
  only used for diagnostics).
- The attention's per-axis label counts MUST match the heads' per-axis label
  counts. OCE.__init__ enforces this so misconfiguration fails at construction
  time rather than at the first forward pass.
- compute_losses delegates entirely to self.heads.compute_losses. OCE does not
  aggregate losses across axes — that's the trainer's responsibility (so the
  trainer can apply axis-weighting schemes during E5 grid pass).
- from_config builds the full stack from a single YAML-loaded dict, with
  encoder config nested under cfg['encoder'] and head/attention specs under
  cfg['heads'] and cfg['attention'] respectively.

Loss aggregation is explicitly NOT inside this module. The trainer sums the
per-axis losses with whatever weighting scheme is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.encoder import EncoderOutput, OncologyEncoder
from src.models.heads import HeadOutput, MultiTaskHeads
from src.models.label_attention import (
    AxisAttentionOutput,
    MultiAxisLabelAttention,
)


@dataclass
class OCEOutput:
    """Output of OncologyCodingEngine.forward.

    Attributes:
        logits_per_axis: Mapping axis_name -> [B, K_axis] pre-activation logits.
            Single-pick axes use these as pre-softmax; multi-label axes use
            them as pre-sigmoid. The trainer/inference layer applies the
            appropriate activation.
        attention_weights_per_axis: Mapping axis_name -> [B, K_axis, T_eff]
            label-wise attention weights. Used for evidence-span extraction
            in the reviewer dashboard. Always populated.
        n_segments: [B] long tensor of per-record segment counts (from encoder).
            Useful for W&B logging and post-hoc analysis.
        encoder_attentions: Optional list of per-segment transformer
            self-attention tensors (only if return_encoder_attentions=True at
            forward time). Distinct from attention_weights_per_axis (which is
            label-wise attention, not transformer self-attention).
    """

    logits_per_axis: dict[str, torch.Tensor]
    attention_weights_per_axis: dict[str, torch.Tensor]
    n_segments: torch.Tensor
    encoder_attentions: list[torch.Tensor] | None = None


class OncologyCodingEngine(nn.Module):
    """Top-level OCE model: encoder + multi-axis attention + multi-task heads.

    Use OncologyCodingEngine.from_config(cfg) for YAML-driven construction.
    Direct __init__ is for testing or programmatic construction.
    """

    def __init__(
        self,
        encoder: OncologyEncoder,
        single_pick_spec: dict[str, int],
        multi_label_spec: dict[str, int],
        attn_dim: int | None = None,
    ) -> None:
        """Compose encoder + attention + heads.

        Args:
            encoder: A pre-constructed OncologyEncoder (typically from
                OncologyEncoder.from_config(cfg['encoder'])).
            single_pick_spec: Mapping axis -> num_labels for single-pick axes.
                Same dict passed to MultiTaskHeads.
            multi_label_spec: Mapping axis -> num_labels for multi-label axes.
                Same dict passed to MultiTaskHeads.
            attn_dim: Optional attention projection dim, shared across all axes.
                Defaults to encoder.hidden_dim (matches PLM-ICD).

        Raises:
            ValueError: If single_pick_spec and multi_label_spec have any
                overlapping axis names.
        """
        super().__init__()
        self.encoder = encoder

        # Build the union of all axis label counts; this is what attention
        # needs (one attention module per axis, regardless of head kind).
        # Overlap is rejected in MultiTaskHeads but we check here too for an
        # earlier, clearer error message.
        overlap = set(single_pick_spec) & set(multi_label_spec)
        if overlap:
            raise ValueError(
                f"Axis names appear in both single_pick and multi_label "
                f"specs: {sorted(overlap)}"
            )
        axis_label_counts: dict[str, int] = {**single_pick_spec, **multi_label_spec}

        self.attention = MultiAxisLabelAttention(
            hidden_dim=encoder.hidden_dim,
            axis_label_counts=axis_label_counts,
            attn_dim=attn_dim,
        )

        self.heads = MultiTaskHeads(
            hidden_dim=encoder.hidden_dim,
            single_pick_spec=single_pick_spec,
            multi_label_spec=multi_label_spec,
        )

    # -------------------------------------------------------------
    # Forward / loss
    # -------------------------------------------------------------

    def forward(
        self,
        texts: list[str] | list[list[str]],
        return_encoder_attentions: bool = False,
    ) -> OCEOutput:
        """Run the full pipeline on a batch of raw texts.

        Args:
            texts: List of B raw text strings, OR List of B lists of
                pre-segmented strings (when segmentation_mode != 'fixed').
                The encoder auto-detects the format.
            return_encoder_attentions: If True, populate
                OCEOutput.encoder_attentions with per-segment transformer
                self-attentions. Default False (saves memory).

        Returns:
            OCEOutput with per-axis logits, per-axis label-attention weights,
            n_segments, and optional encoder self-attentions.
        """
        encoder_out: EncoderOutput = self.encoder(
            texts, return_attentions=return_encoder_attentions
        )

        attention_outputs: dict[str, AxisAttentionOutput] = self.attention(
            encoder_out.hidden_states, encoder_out.attention_mask
        )

        # Heads consume per-axis context vectors of shape [B, K_axis, d].
        label_contexts: dict[str, torch.Tensor] = {
            axis: out.context for axis, out in attention_outputs.items()
        }
        head_outputs: dict[str, HeadOutput] = self.heads(label_contexts)

        return OCEOutput(
            logits_per_axis={
                axis: out.logits for axis, out in head_outputs.items()
            },
            attention_weights_per_axis={
                axis: out.weights for axis, out in attention_outputs.items()
            },
            n_segments=encoder_out.n_segments,
            encoder_attentions=encoder_out.attentions,
        )

    def compute_losses(
        self,
        outputs: OCEOutput,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, int]]:
        """Compute per-axis losses, respecting head_active flags.

        Delegates entirely to self.heads.compute_losses. Inactive axes are
        excluded from the result dict — the trainer's sum() over the dict
        naturally skips them.

        Args:
            outputs: OCEOutput from forward().
            targets: Mapping axis -> target tensor (LongTensor [B] for
                single-pick, FloatTensor [B, K] for multi-label).

        Returns:
            Mapping axis -> (loss_scalar, n_valid). Inactive axes absent.

        Raises:
            KeyError: If targets is missing any active axis.
        """
        # The heads expect HeadOutput objects, but compute_loss methods only
        # use .logits. We rebuild the lightweight wrapping so the heads layer's
        # contract is preserved without OCEOutput needing to know about HeadOutput.
        head_outputs = {
            axis: HeadOutput(logits=logits, axis=axis)
            for axis, logits in outputs.logits_per_axis.items()
        }
        return self.heads.compute_losses(head_outputs, targets)

    # -------------------------------------------------------------
    # Phase control (delegated to heads)
    # -------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Activate heads for a named training phase.

        Phases:
            - 'icdo3_only': Phase 1 TCGA pretrain (ICD-O-3 axes only)
            - 'all': Phase 2 Baheya fine-tune (every head active)
        """
        self.heads.set_phase(phase)

    def set_active_axes(self, axis_names: set[str]) -> None:
        """Activate explicitly named axes; deactivate all others."""
        self.heads.set_active_axes(axis_names)

    @property
    def active_axes(self) -> list[str]:
        """List of currently-active axis names."""
        return self.heads.active_axes

    # -------------------------------------------------------------
    # Construction from config
    # -------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OncologyCodingEngine":
        """Construct OCE from a YAML-loaded config dict.

        Expected structure:
            cfg = {
                'encoder': {           # passed to OncologyEncoder.from_config
                    'backbone_name': '...',
                    'segment_size': 128,
                    'max_segments': 12,
                    'strict_length': False,
                },
                'heads': {
                    'single_pick': {axis_name: K, ...},
                    'multi_label': {axis_name: K, ...},
                },
                'attention': {
                    'attn_dim': 768,    # optional, defaults to encoder.hidden_dim
                },
            }

        Args:
            cfg: Config dict.

        Returns:
            Constructed OncologyCodingEngine.

        Raises:
            KeyError: If required sections are missing.
            ValueError: If specs are malformed (delegated to underlying
                modules' validators).
        """
        if "encoder" not in cfg:
            raise KeyError("config missing required section: 'encoder'")
        if "heads" not in cfg:
            raise KeyError("config missing required section: 'heads'")

        encoder = OncologyEncoder.from_config(cfg["encoder"])

        heads_cfg = cfg["heads"]
        single_pick = heads_cfg.get("single_pick", {})
        multi_label = heads_cfg.get("multi_label", {})

        attention_cfg = cfg.get("attention", {})
        attn_dim = attention_cfg.get("attn_dim")

        return cls(
            encoder=encoder,
            single_pick_spec=single_pick,
            multi_label_spec=multi_label,
            attn_dim=attn_dim,
        )
