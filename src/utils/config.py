"""Configuration loading for MCIS — matches configs/B0.yaml schema.

Plain dataclasses, no OmegaConf/Hydra. ``BenchmarkConfig.from_yaml(path)``.
Unknown keys raise ValueError so typos fail loud.

Schema mirrors B0.yaml exactly:
    experiment_name, seed
    data: {parquet, label_vocab, cv_folds, module, text_field, trainable_value, test_value}
    label_cardinalities: {axis -> K}      # frozen K from build-time vocab
    axis_types: {single_pick: [...], multilabel: [...]}
    aux_fields: [...]
    model: {encoder_hf_id, max_seq_length, attention_hidden_dim, ...}
    train: {epochs, micro_batch_size, ..., loss, focal_gamma, ...}
    cv: {enabled, n_folds, seed}
    eval: {rare_code_threshold, report_rare_separately, subgroups, test_only_oov_excluded}
    logging: {project, run_name, output_dir, checkpoint_dir, wandb_enabled, wandb_mode}
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    parquet: str = "data/frozen/m1_model_ready/m1_model_ready.parquet"
    label_vocab: str = "data/frozen/m1_model_ready/label_vocab.json"
    cv_folds: str = "data/frozen/m1_model_ready/cv_folds.csv"
    module: str = "M1"
    text_field: str = "text_section_tagged"  # E1-fmt swaps to text_concatenated
    trainable_value: str = "trainable"
    test_value: str = "test"
    # Phase 1 (TCGA) only: explicit pre-baked val split (e.g. 'pretrain_val').
    # When set AND cv.enabled=False, the no-CV branch uses this split as val
    # instead of slicing 90/10 off trainable. None = legacy 90/10 behavior.
    val_value: str | None = None


@dataclass
class AxisTypesConfig:
    single_pick: list[str] = field(default_factory=list)
    multilabel: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    encoder_hf_id: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    # PLM-ICD segment pooling — encoder.py reads these.
    segment_size: int = 128
    max_segments: int = 12
    strict_length: bool = False
    # Diagnostic only; not enforced by encoder.
    diagnostic_max_tokens: int = 512
    attention_hidden_dim: int = 256
    init_from_checkpoint: str | None = None
    # Selective load flags for init_from_checkpoint (handled by
    # src/utils/checkpoint_loader.load_checkpoint_partial).
    # Defaults: encoder-only transfer (Phase 1 TCGA → Phase 2 Baheya).
    # For stacking on same-vocab base (E5a/E5b on E2 ckpt), set both True.
    init_load_attention: bool = True   # label-wise attention; shape-checked
    init_load_heads: bool = False      # per-axis heads; False = reset for Phase 2
    use_aux_histology_heads: bool = False  # E3 toggle
    aux_loss_weight: float = 0.3
    icd11_two_stage: bool = False  # E6 toggle


@dataclass
class TrainConfig:
    epochs: int = 10
    micro_batch_size: int = 8
    grad_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss: str = "bce"  # bce | focal | apl
    focal_gamma: float = 2.0
    early_stopping_patience: int = 3
    mixed_precision: str = "no"  # no | fp16 | bf16


@dataclass
class CVConfig:
    enabled: bool = True
    n_folds: int = 5
    seed: int = 42


@dataclass
class EvalConfig:
    rare_code_threshold: int = 10
    report_rare_separately: bool = True
    subgroups: list[str] = field(default_factory=list)
    test_only_oov_excluded: bool = True


@dataclass
class LoggingConfig:
    project: str = "MCIS_AI"
    run_name: str | None = None
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    wandb_enabled: bool = True
    wandb_mode: str = "online"  # online | offline | disabled


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    experiment_name: str = "B0"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    label_cardinalities: dict[str, int] = field(default_factory=dict)
    axis_types: AxisTypesConfig = field(default_factory=AxisTypesConfig)
    aux_fields: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    augmentation: dict = field(default_factory=dict)

    # ------------- Convenience accessors -------------

    @property
    def all_axes(self) -> list[str]:
        """All axis names (single-pick + multi-label), preserving order."""
        return list(self.axis_types.single_pick) + list(self.axis_types.multilabel)

    def is_multilabel(self, axis: str) -> bool:
        return axis in self.axis_types.multilabel

    def is_single_pick(self, axis: str) -> bool:
        return axis in self.axis_types.single_pick

    # ------------- Loading -------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        path = Path(path)
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkConfig":
        # Section dispatch: each named section gets its sub-config built first.
        # Top-level keys: experiment_name, seed, label_cardinalities, aux_fields
        sub_specs: dict[str, type] = {
            "data": DataConfig,
            "axis_types": AxisTypesConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "cv": CVConfig,
            "eval": EvalConfig,
            "logging": LoggingConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, sub_cls in sub_specs.items():
            sub_raw = raw.get(key, {}) or {}
            _check_unknown_keys(sub_raw, sub_cls, prefix=f"{key}.")
            kwargs[key] = sub_cls(**sub_raw)

        # Top-level scalars / containers.
        for k in ("experiment_name", "seed"):
            if k in raw:
                kwargs[k] = raw[k]
        if "label_cardinalities" in raw:
            kwargs["label_cardinalities"] = dict(raw["label_cardinalities"] or {})
        if "aux_fields" in raw:
            kwargs["aux_fields"] = list(raw["aux_fields"] or [])

        valid_top = (
            set(sub_specs)
            | {"experiment_name", "seed", "label_cardinalities", "aux_fields", "augmentation"}
        )
        unknown = set(raw) - valid_top
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    # ------------- Validation against the vocab JSON -------------

    def validate_against_vocab(self, vocab_cardinalities: dict[str, int]) -> None:
        """Cross-check label_cardinalities in the YAML against the loaded vocab.

        Raises ValueError on mismatch — protects against running training
        with a stale config that doesn't match the frozen vocab.
        """
        problems: list[str] = []
        for axis in self.all_axes:
            if axis not in self.label_cardinalities:
                continue  # not all configs have to declare K; allow missing
            yaml_k = self.label_cardinalities[axis]
            vocab_k = vocab_cardinalities.get(axis)
            if vocab_k is None:
                problems.append(f"axis {axis!r} declared in YAML but missing from vocab")
            elif vocab_k != yaml_k:
                problems.append(
                    f"axis {axis!r}: YAML K={yaml_k}, vocab K={vocab_k}"
                )
        if problems:
            raise ValueError(
                "label_cardinalities mismatch between config and vocab:\n  "
                + "\n  ".join(problems)
            )


def _check_unknown_keys(raw: dict[str, Any], sub_cls: type, prefix: str = "") -> None:
    valid = {f.name for f in fields(sub_cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(
            f"Unknown config keys under {prefix or '<root>'}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
