"""Multi-task prediction heads for MCIS Oncology Coding Engine.

Per architecture v6 §6.2, M1 OCE has 10 heads across two families:

Single-pick axes (8 heads, masked cross-entropy):
    icdo3_topography        K=8
    icdo3_morphology        K=24
    icdo3_behavior          K=4
    icdo3_grade             K=3   (4 in Phase 1 with TCGA; per-axis vocab)
    icdo3_laterality        K=3   (L/R/B; X-codes go to ICD-11 ext_laterality)
    icd11_stem              K=27
    icd11_ext_laterality    K=6   (WHO ECT single-pick; 0/3055 multi rows)
    icd11_ext_grading       K=5   (ordinal grading axis; 0/3055 multi rows)

Multi-label axes (2 heads, binary cross-entropy):
    icd11_ext_anatomy       K=16  (post-coordinated anatomical extensions)
    icd11_ext_histopath     K=23  (post-coordinated histopathology extensions)

Note: icd11_ext_staging is excluded (zero training examples in M1 biopsy data).

Each head consumes the per-label context from its own AxisLabelAttention
module, so the input shape is [B, K_axis, hidden_dim] — already pooled per
label by attention. The head's job is the final per-label projection to a
logit (single-pick) or independent logits (multi-label).

Phase awareness: each head has a `head_active` flag. When inactive:
    - The head still exists in the model (so checkpoints load cleanly).
    - The head's forward still computes logits (so the same code path runs).
    - The loss function skips this axis entirely (no gradient flows back).
This is the locked design from the planning session — single model class
across phases, head_active toggles per-axis loss participation.

Loss contract:
    - Single-pick: targets are LongTensor [B], values in [0, K) or -1 for null.
      Records with target == -1 are masked from the loss.
    - Multi-label: targets are FloatTensor [B, K], values in {0, 1}.
      No null masking — empty list = all zeros.

The loss functions return a SCALAR per axis. Aggregation across axes (mean,
weighted sum, etc.) is the trainer's responsibility, not the head's.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Sentinel used in single-pick targets to mark "null/missing — mask from loss".
NULL_TARGET_SENTINEL = -1


# -----------------------------------------------------------------------------
# Head primitives
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadOutput:
    """Output of one head's forward call.

    Attributes:
        logits: [B, K_axis]. For single-pick: pre-softmax logits. For
            multi-label: pre-sigmoid logits. The trainer applies the
            appropriate activation when computing predictions.
        axis: name of the axis this head predicts (for downstream routing).
    """

    logits: torch.Tensor
    axis: str


class _AxisHeadBase(nn.Module):
    """Base class for per-axis heads.

    Each head:
      - Owns a Linear layer that projects [B, K_axis, hidden_dim] -> [B, K_axis].
        Conceptually each label gets its own scalar logit from its own context
        vector; we implement this as a single Linear (hidden_dim -> 1) applied
        to every label-context independently.
      - Tracks the axis name and label count.
      - Has a `head_active` boolean flag that controls loss participation.
    """

    def __init__(self, axis: str, num_labels: int, hidden_dim: int) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError(
                f"num_labels for axis {axis!r} must be positive, got {num_labels}"
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        self.axis = axis
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Per-label scalar projection. Each label's context vector
        # [hidden_dim] -> 1 logit. Implemented as Linear(hidden_dim, 1)
        # applied along the last dim.
        self.projection = nn.Linear(hidden_dim, 1)

        # Phase-aware activation flag. True = participate in loss. The flag
        # is a pure Python bool, NOT a registered buffer, so it doesn't
        # round-trip through state_dict (which is what we want — phase is a
        # training-time decision, not a model property).
        self.head_active: bool = True

    def forward(self, label_context: torch.Tensor) -> HeadOutput:
        """Project per-label context vectors into per-label logits.

        Args:
            label_context: [B, K_axis, hidden_dim] from this axis's attention.

        Returns:
            HeadOutput with logits of shape [B, K_axis].

        Raises:
            ValueError: If the input shape doesn't match this head's K_axis or
                hidden_dim.
        """
        B, K, d = label_context.shape
        if K != self.num_labels:
            raise ValueError(
                f"Head {self.axis!r}: expected K={self.num_labels}, got {K}"
            )
        if d != self.hidden_dim:
            raise ValueError(
                f"Head {self.axis!r}: expected hidden_dim={self.hidden_dim}, got {d}"
            )

        # projection: [B, K, d] -> [B, K, 1] -> squeeze -> [B, K]
        logits = self.projection(label_context).squeeze(-1)
        return HeadOutput(logits=logits, axis=self.axis)


class SinglePickHead(_AxisHeadBase):
    """Single-label-per-record axis head.

    Targets are class indices in [0, K) or NULL_TARGET_SENTINEL to mask.
    Loss is masked cross-entropy.
    """

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute masked cross-entropy loss for this axis.

        Args:
            logits: [B, K] pre-softmax logits.
            targets: [B] LongTensor with values in [0, K) or
                NULL_TARGET_SENTINEL (= -1) for records to skip.

        Returns:
            (loss, n_valid):
                loss: scalar tensor (mean over valid records). Returns 0.0 if
                    no valid records (caller should still skip backprop).
                n_valid: number of records that contributed to the loss.

        Raises:
            ValueError: If shapes don't match.
        """
        if targets.dim() != 1:
            raise ValueError(
                f"Head {self.axis!r}: single-pick targets must be 1D, "
                f"got shape {targets.shape}"
            )
        if logits.shape != (targets.shape[0], self.num_labels):
            raise ValueError(
                f"Head {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} mismatch (expected logits=[B, K])"
            )

        valid_mask = targets != NULL_TARGET_SENTINEL
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype), 0

        loss = F.cross_entropy(
            logits[valid_mask],
            targets[valid_mask],
            reduction="mean",
        )
        return loss, n_valid


class MultiLabelHead(_AxisHeadBase):
    """Multi-label axis head.

    Targets are FloatTensor of shape [B, K] with values in {0, 1}.
    Loss is binary cross-entropy with logits, mean-reduced over all
    (record, label) pairs.

    Empty-list targets (all zeros) are valid training signal: they tell the
    model "no positive labels for this axis on this record", which is real
    information for ICD-11 extension axes where most records have no
    positives for a given axis.
    """

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute BCE-with-logits loss for this axis.

        Args:
            logits: [B, K] pre-sigmoid logits.
            targets: [B, K] FloatTensor with values in {0, 1}.

        Returns:
            (loss, n_valid):
                loss: scalar tensor (mean over all B*K positions).
                n_valid: B (every record contributes; multi-label has no null
                    masking).

        Raises:
            ValueError: If shapes don't match.
        """
        if targets.dim() != 2:
            raise ValueError(
                f"Head {self.axis!r}: multi-label targets must be 2D [B, K], "
                f"got shape {targets.shape}"
            )
        if logits.shape != targets.shape:
            raise ValueError(
                f"Head {self.axis!r}: logits {logits.shape} vs targets "
                f"{targets.shape} mismatch (expected matching [B, K])"
            )

        # Cast targets to logits dtype to handle long->float and dtype mismatches.
        targets_f = targets.to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="mean"
        )
        return loss, targets.shape[0]


# -----------------------------------------------------------------------------
# Multi-head container
# -----------------------------------------------------------------------------


class MultiTaskHeads(nn.Module):
    """Container for all M1 prediction heads, with phase-aware activation.

    Constructed from two specs:
        single_pick_spec: dict[axis_name, num_labels]
        multi_label_spec: dict[axis_name, num_labels]

    Provides:
        - forward(label_contexts: dict[axis, [B, K, d]]) -> dict[axis, HeadOutput]
        - compute_losses(logits_per_axis, targets_per_axis) -> per-axis loss dict
        - set_active_axes(axis_names) — activate only the named axes
        - set_phase(phase: 'icdo3_only' | 'all') — convenience preset

    The trainer sums (or weighted-sums) the per-axis losses to get the total.
    Aggregation is NOT inside this module so the trainer can apply its own
    weighting scheme (e.g., upweight rare-axis losses in E5 grid pass).
    """

    # Hard-coded preset: which axes belong to which family.
    # Used by set_phase() to switch all heads in one call.
    PHASE_AXIS_GROUPS = {
        "icdo3_only": {
            "icdo3_topography",
            "icdo3_morphology",
            "icdo3_behavior",
            "icdo3_grade",
            "icdo3_laterality",
        },
        "all": None,  # Sentinel: activate every head present.
    }

    def __init__(
        self,
        hidden_dim: int,
        single_pick_spec: dict[str, int],
        multi_label_spec: dict[str, int],
    ) -> None:
        """Initialize all heads.

        Args:
            hidden_dim: Encoder hidden dim (matches AxisLabelAttention output).
            single_pick_spec: Mapping axis -> K for single-pick axes.
            multi_label_spec: Mapping axis -> K for multi-label axes.

        Raises:
            ValueError: If both specs are empty, or if any axis name appears
                in both specs.
        """
        super().__init__()
        if not single_pick_spec and not multi_label_spec:
            raise ValueError("At least one of single_pick_spec or "
                             "multi_label_spec must be non-empty")

        overlap = set(single_pick_spec) & set(multi_label_spec)
        if overlap:
            raise ValueError(
                f"Axis names appear in both single_pick and multi_label "
                f"specs: {sorted(overlap)}"
            )

        self.hidden_dim = hidden_dim

        # Build heads. Use ModuleDict for named lookup and proper parameter
        # registration. We track which kind each axis is so compute_losses
        # knows which loss function to call.
        self.heads = nn.ModuleDict()
        self._axis_kinds: dict[str, str] = {}  # axis -> 'single' | 'multi'

        for axis, k in single_pick_spec.items():
            self.heads[axis] = SinglePickHead(axis, k, hidden_dim)
            self._axis_kinds[axis] = "single"
        for axis, k in multi_label_spec.items():
            self.heads[axis] = MultiLabelHead(axis, k, hidden_dim)
            self._axis_kinds[axis] = "multi"

    # -------------------------------------------------------------
    # Forward / loss
    # -------------------------------------------------------------

    def forward(
        self,
        label_contexts: dict[str, torch.Tensor],
    ) -> dict[str, HeadOutput]:
        """Run all heads on their corresponding label contexts.

        Args:
            label_contexts: Mapping axis -> [B, K_axis, hidden_dim] tensor.
                Must contain every axis registered in this module.

        Returns:
            Mapping axis -> HeadOutput.

        Raises:
            KeyError: If label_contexts is missing any registered axis.
        """
        missing = set(self.heads.keys()) - set(label_contexts.keys())
        if missing:
            raise KeyError(
                f"label_contexts missing axes: {sorted(missing)}"
            )
        return {
            axis: head(label_contexts[axis])
            for axis, head in self.heads.items()
        }

    def compute_losses(
        self,
        head_outputs: dict[str, HeadOutput],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, int]]:
        """Compute per-axis losses, respecting head_active flags.

        Args:
            head_outputs: Mapping axis -> HeadOutput from forward().
            targets: Mapping axis -> target tensor (shape depends on axis kind).

        Returns:
            Mapping axis -> (loss_scalar, n_valid). Inactive axes are EXCLUDED
            from the result entirely (not present as keys). This way the
            trainer's sum() naturally skips them.

        Raises:
            KeyError: If targets is missing any *active* axis.
        """
        result: dict[str, tuple[torch.Tensor, int]] = {}
        for axis, head in self.heads.items():
            if not head.head_active:
                continue
            if axis not in targets:
                raise KeyError(
                    f"targets missing for active axis {axis!r}"
                )
            out = head_outputs[axis]
            loss, n_valid = head.compute_loss(out.logits, targets[axis])
            result[axis] = (loss, n_valid)
        return result

    # -------------------------------------------------------------
    # Phase / activation control
    # -------------------------------------------------------------

    def set_active_axes(self, axis_names: set[str]) -> None:
        """Activate only the named axes; deactivate all others.

        Args:
            axis_names: Set of axis names to activate. Names not present in
                this module are silently ignored (allows configs to list
                aspirational axes without breaking).

        Raises:
            ValueError: If axis_names is empty (would deactivate the whole
                model — almost certainly a bug).
        """
        if not axis_names:
            raise ValueError(
                "set_active_axes received empty set — would deactivate all "
                "heads. Use set_phase('all') if that's intentional."
            )
        for axis, head in self.heads.items():
            head.head_active = axis in axis_names

    def set_phase(self, phase: str) -> None:
        """Activate heads for a named phase.

        Args:
            phase: One of:
                - 'icdo3_only': activate ICD-O-3 axes only (Phase 1 TCGA pretrain)
                - 'all': activate every head present (Phase 2 Baheya fine-tune)

        Raises:
            ValueError: If phase is not a recognized preset.
        """
        if phase not in self.PHASE_AXIS_GROUPS:
            raise ValueError(
                f"Unknown phase {phase!r}. Valid: "
                f"{sorted(self.PHASE_AXIS_GROUPS.keys())}"
            )
        group = self.PHASE_AXIS_GROUPS[phase]
        if group is None:
            # 'all' preset: activate every head present
            for head in self.heads.values():
                head.head_active = True
        else:
            # Named group: activate only heads in the group that are present
            for axis, head in self.heads.items():
                head.head_active = axis in group

    @property
    def active_axes(self) -> list[str]:
        """List of currently-active axis names."""
        return [axis for axis, head in self.heads.items() if head.head_active]

    @property
    def axis_kinds(self) -> dict[str, str]:
        """Mapping axis name -> 'single' | 'multi'."""
        return dict(self._axis_kinds)
