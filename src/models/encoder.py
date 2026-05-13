"""MCIS Oncology Coding Engine (OCE) encoder wrapper.

This module wraps a HuggingFace pretrained encoder (default:
PubMedBERT/BiomedBERT) with PLM-ICD-style segment pooling so that inputs
of any length can be encoded with a uniform downstream interface.

Design decisions (locked in planning sessions):
- Backbone: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
  (PLM-ICD's encoder; same weights as the deprecated PubMedBERT name).
- Segmentation: fixed 128-token segments, no overlap, *always* applied
  (S=1 for short inputs). Single code path for Baheya, TCGA, M2 progress
  notes. No sentence-aware splitting — inherited from PLM-ICD canonical.
- max_segments cap: 12 segments => 1536-token effective ceiling. Covers
  >99% of TCGA non-BRCA. Configurable.
- Overflow: middle-drop (keep head and tail segments). Pathology reports
  are head-and-tail informative; middle-dropping is empirically less
  harmful than tail-dropping. Configurable to strict-raise.
- Special tokens ([CLS]/[SEP]) are kept per segment during training
  (matches PLM-ICD). UI-side filtering of special-token positions for
  evidence-span highlighting is the responsibility of the inference
  post-processor, not this module.
- Freezing: not enabled by default. `freeze_bottom_layers(n)` is provided
  as a regularization helper for E5 grid-pass exploration.
- Phase awareness: NONE at the encoder level. Phase 1 vs Phase 2 head
  activation lives in the heads/loss layer, not here.

Output contract:
    forward(texts: list[str]) -> EncoderOutput with
        hidden_states:  [B, T_eff, hidden_dim]
        attention_mask: [B, T_eff]
        n_segments:     [B]            (per-record segment count, for logging)
        attentions:     None or list[Tensor]  (transformer self-attention,
                                              only if return_attentions=True)

T_eff = max(n_segments_in_batch) * segment_size. Records with fewer
segments are right-padded with zeros and the attention mask reflects this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)

# Locked default backbone. Microsoft renamed the original PubMedBERT repo to
# BiomedBERT; the underlying weights are identical. PLM-ICD (Huang 2022) used
# this exact checkpoint.
DEFAULT_BACKBONE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


@dataclass
class EncoderOutput:
    """Output of OncologyEncoder.forward.

    Attributes:
        hidden_states: float tensor [B, T_eff, hidden_dim]. Per-token contextual
            representations after segment-level encoding and concatenation.
        attention_mask: long tensor [B, T_eff] in {0, 1}. Marks valid (non-pad)
            positions. Includes [CLS] and [SEP] positions per segment.
        n_segments: long tensor [B]. Number of segments actually encoded per
            record (after any overflow handling). Useful for W&B logging and
            per-record analysis.
        attentions: Optional list of transformer self-attention tensors, one
            per segment. Only populated if return_attentions=True at forward
            time. Used at inference for diagnostic purposes; label-wise
            attention (the evidence-span source) is computed downstream.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    n_segments: torch.Tensor
    attentions: list[torch.Tensor] | None = field(default=None)


class OncologyEncoder(nn.Module):
    """PubMedBERT/BiomedBERT encoder with PLM-ICD-style segment pooling.

    See module docstring for design rationale and locked decisions.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        segment_size: int = 128,
        max_segments: int = 12,
        strict_length: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the encoder.

        Args:
            backbone_name: HuggingFace model identifier. Defaults to PubMedBERT.
                Override to swap in PathologyBERT for E4 ablation.
            segment_size: Tokens per segment, including [CLS] and [SEP].
                PLM-ICD uses 128. Setting this to the backbone's max
                (typically 512) reduces to truncate-only behavior for short
                inputs but still segments long inputs.
            max_segments: Hard cap on segments per record. Effective token
                ceiling = max_segments * segment_size. Default 12 (=1536
                tokens) covers >99% of TCGA non-BRCA reports.
            strict_length: If True, raise ValueError when input exceeds
                max_segments * segment_size. If False (default), middle-drop
                segments and log a warning. Use strict=True at training time
                to catch dataset issues; strict=False at deployment for
                tolerance.
            cache_dir: Optional HuggingFace cache directory. None uses default.

        Raises:
            ValueError: If segment_size or max_segments is non-positive, or
                if segment_size exceeds the backbone's position embedding
                limit.
        """
        super().__init__()

        if segment_size <= 0:
            raise ValueError(f"segment_size must be positive, got {segment_size}")
        if max_segments <= 0:
            raise ValueError(f"max_segments must be positive, got {max_segments}")

        self.backbone_name = backbone_name
        self.segment_size = segment_size
        self.max_segments = max_segments
        self.strict_length = strict_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone_name, cache_dir=cache_dir
        )
        # attn_implementation="eager" is required for output_attentions to work
        # on transformers >= 4.36. SDPA (the default) silently ignores the flag.
        # We always use eager so that return_attentions=True works at inference;
        # the speed cost is negligible at our scale (BERT-base, batch ~16).
        self.backbone = AutoModel.from_pretrained(
            backbone_name, cache_dir=cache_dir, attn_implementation="eager"
        )

        # Validate segment_size against backbone capacity.
        backbone_max = self.backbone.config.max_position_embeddings
        if segment_size > backbone_max:
            raise ValueError(
                f"segment_size={segment_size} exceeds backbone max position "
                f"embeddings={backbone_max} for {backbone_name}"
            )

        self.hidden_dim: int = self.backbone.config.hidden_size

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def forward(
        self,
        texts: list[str],
        return_attentions: bool = False,
    ) -> EncoderOutput:
        """Encode a batch of raw text strings.

        Args:
            texts: List of B raw strings. Tokenization happens inside.
            return_attentions: If True, return per-segment self-attention
                tensors in EncoderOutput.attentions. Costs memory; use only
                at inference for diagnostics.

        Returns:
            EncoderOutput. See dataclass docstring for shapes.

        Raises:
            ValueError: If texts is empty, or if strict_length=True and any
                input exceeds the segment cap.
        """
        if len(texts) == 0:
            raise ValueError("texts must be non-empty")

        # Step 1: tokenize each text and segment into chunks of segment_size.
        # Returns a list (one per record) of segment lists (each segment is a
        # dict of input_ids / attention_mask tensors of length segment_size).
        per_record_segments: list[list[BatchEncoding]] = [
            self._segment_one_text(t) for t in texts
        ]

        # Step 2: flatten into a single batch of segments for one encoder call.
        # We track which segments belong to which record so we can reassemble
        # the [B, T_eff, d] output tensor correctly.
        flat_segments, record_index, n_segments = self._flatten_segments(
            per_record_segments
        )

        # Step 3: stack segments into encoder-input tensors.
        input_ids, attention_mask = self._stack_segments(flat_segments)

        # Step 4: run the backbone encoder once over all segments.
        # All segments fit in one forward call regardless of which record
        # they came from; this is just batched BERT.
        device = next(self.backbone.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=return_attentions,
        )
        # outputs.last_hidden_state: [N_total_segments, segment_size, hidden_dim]
        seg_hidden = outputs.last_hidden_state

        # Step 5: scatter segment outputs back into per-record [T_eff, d]
        # arrangement, then pad to a uniform T_eff across the batch.
        hidden_states, full_mask = self._reassemble_per_record(
            seg_hidden=seg_hidden,
            seg_attention_mask=attention_mask,
            record_index=record_index,
            n_segments=n_segments,
            batch_size=len(texts),
        )

        result = EncoderOutput(
            hidden_states=hidden_states,
            attention_mask=full_mask,
            n_segments=torch.tensor(n_segments, dtype=torch.long, device=device),
            attentions=list(outputs.attentions) if return_attentions else None,
        )
        return result

    def freeze_bottom_layers(self, n: int) -> None:
        """Freeze the embedding layer and the first `n` transformer layers.

        Useful as a regularization knob for the E5 grid pass. Pass n=0 to
        freeze only the embeddings; pass n=12 (for BERT-base) to freeze the
        entire encoder (effectively making it a feature extractor).

        Args:
            n: Number of transformer layers to freeze, counting from the
                input side. Must satisfy 0 <= n <= total transformer layers.

        Raises:
            ValueError: If n is negative or exceeds the number of layers.
        """
        n_layers = self.backbone.config.num_hidden_layers
        if n < 0 or n > n_layers:
            raise ValueError(
                f"n must be in [0, {n_layers}] for this backbone, got {n}"
            )

        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False

        for i in range(n):
            for p in self.backbone.encoder.layer[i].parameters():
                p.requires_grad = False

        logger.info(
            "Froze embeddings and first %d/%d transformer layers", n, n_layers
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OncologyEncoder":
        """Construct from a YAML-loaded config dict.

        Expected keys (all optional, defaults applied):
            backbone_name, segment_size, max_segments, strict_length, cache_dir
        """
        return cls(
            backbone_name=cfg.get("backbone_name", DEFAULT_BACKBONE),
            segment_size=cfg.get("segment_size", 128),
            max_segments=cfg.get("max_segments", 12),
            strict_length=cfg.get("strict_length", False),
            cache_dir=cfg.get("cache_dir"),
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _segment_one_text(self, text: str) -> list[BatchEncoding]:
        """Tokenize one text and split into fixed-size segments.

        Each segment includes [CLS] and [SEP] (PLM-ICD style). The HF
        tokenizer's `return_overflowing_tokens` mechanism does this for us
        with stride=0 (no overlap).

        Returns a list of BatchEncoding, each with input_ids and
        attention_mask of length exactly segment_size.

        If the number of segments would exceed max_segments, applies the
        overflow policy (middle-drop or raise).
        """
        encoded = self.tokenizer(
            text,
            max_length=self.segment_size,
            truncation=True,
            return_overflowing_tokens=True,
            stride=0,
            padding="max_length",
            return_tensors="pt",
        )
        # encoded.input_ids: [n_segments, segment_size]
        n_segs = encoded["input_ids"].shape[0]

        if n_segs > self.max_segments:
            n_segs = self._handle_overflow(text, n_segs)
            # Apply middle-drop: keep first half and last half of segments.
            keep_indices = self._middle_drop_indices(n_segs, encoded["input_ids"].shape[0])
            encoded = {
                "input_ids": encoded["input_ids"][keep_indices],
                "attention_mask": encoded["attention_mask"][keep_indices],
            }

        # Convert from batched tensor to list of per-segment dicts for
        # easier handling downstream.
        segments: list[BatchEncoding] = []
        for i in range(encoded["input_ids"].shape[0]):
            segments.append(
                BatchEncoding(
                    {
                        "input_ids": encoded["input_ids"][i],
                        "attention_mask": encoded["attention_mask"][i],
                    }
                )
            )
        return segments

    def _handle_overflow(self, text: str, original_n_segs: int) -> int:
        """Apply overflow policy and return the kept-segment count.

        Args:
            text: The offending input (for logging only).
            original_n_segs: Number of segments the tokenizer produced.

        Returns:
            Number of segments to keep (= max_segments).

        Raises:
            ValueError: If strict_length=True.
        """
        n_kept = self.max_segments
        n_dropped = original_n_segs - n_kept
        text_preview = text[:80].replace("\n", " ")

        if self.strict_length:
            raise ValueError(
                f"Input exceeds max_segments={self.max_segments}: produced "
                f"{original_n_segs} segments. strict_length=True. "
                f"Text preview: {text_preview!r}"
            )

        logger.warning(
            "Input exceeds max_segments cap: produced %d segments, keeping "
            "%d via middle-drop (dropped %d middle segments). Text preview: %r",
            original_n_segs, n_kept, n_dropped, text_preview,
        )
        return n_kept

    @staticmethod
    def _middle_drop_indices(n_kept: int, n_total: int) -> torch.Tensor:
        """Compute indices to keep when middle-dropping.

        Strategy: keep ceil(n_kept/2) from the head and floor(n_kept/2) from
        the tail. This preserves the diagnosis line (top of report) and the
        conclusion (bottom) while sacrificing middle microscopic detail.

        Args:
            n_kept: Number of segments to keep (= max_segments).
            n_total: Total number of segments produced by tokenizer.

        Returns:
            LongTensor of indices into the original segment array.
        """
        if n_kept >= n_total:
            return torch.arange(n_total, dtype=torch.long)

        n_head = (n_kept + 1) // 2
        n_tail = n_kept - n_head
        head = torch.arange(n_head, dtype=torch.long)
        tail = torch.arange(n_total - n_tail, n_total, dtype=torch.long)
        return torch.cat([head, tail])

    @staticmethod
    def _flatten_segments(
        per_record_segments: list[list[BatchEncoding]],
    ) -> tuple[list[BatchEncoding], list[int], list[int]]:
        """Flatten per-record segments into a single batch.

        Returns:
            flat_segments: List of all segments across the batch, in
                record-order then segment-order.
            record_index: For each flat segment, the index of the record it
                came from (length = len(flat_segments)).
            n_segments: Per-record segment count (length = batch size).
        """
        flat: list[BatchEncoding] = []
        record_index: list[int] = []
        n_segments: list[int] = []
        for rec_idx, segs in enumerate(per_record_segments):
            n_segments.append(len(segs))
            for seg in segs:
                flat.append(seg)
                record_index.append(rec_idx)
        return flat, record_index, n_segments

    @staticmethod
    def _stack_segments(
        flat_segments: list[BatchEncoding],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack a list of per-segment encodings into encoder-input tensors.

        Returns:
            input_ids: [N_total_segments, segment_size]
            attention_mask: [N_total_segments, segment_size]
        """
        input_ids = torch.stack([s["input_ids"] for s in flat_segments], dim=0)
        attention_mask = torch.stack(
            [s["attention_mask"] for s in flat_segments], dim=0
        )
        return input_ids, attention_mask

    def _reassemble_per_record(
        self,
        seg_hidden: torch.Tensor,
        seg_attention_mask: torch.Tensor,
        record_index: list[int],
        n_segments: list[int],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reassemble flat segment outputs into per-record [T_eff, d] tensors.

        Args:
            seg_hidden: [N_total_segments, segment_size, hidden_dim]
            seg_attention_mask: [N_total_segments, segment_size]
            record_index: Per-segment record id (len = N_total_segments).
            n_segments: Per-record segment count (len = batch_size).
            batch_size: B.

        Returns:
            hidden_states: [B, T_eff, hidden_dim]
            attention_mask: [B, T_eff]
        """
        max_segs = max(n_segments)
        T_eff = max_segs * self.segment_size
        device = seg_hidden.device
        dtype = seg_hidden.dtype

        hidden_out = torch.zeros(
            (batch_size, T_eff, self.hidden_dim), dtype=dtype, device=device
        )
        mask_out = torch.zeros(
            (batch_size, T_eff), dtype=seg_attention_mask.dtype, device=device
        )

        # Track segment position within each record as we iterate.
        per_record_seg_pos = [0] * batch_size

        for flat_idx, rec_idx in enumerate(record_index):
            seg_pos = per_record_seg_pos[rec_idx]
            start = seg_pos * self.segment_size
            end = start + self.segment_size
            hidden_out[rec_idx, start:end, :] = seg_hidden[flat_idx]
            mask_out[rec_idx, start:end] = seg_attention_mask[flat_idx]
            per_record_seg_pos[rec_idx] += 1

        return hidden_out, mask_out
