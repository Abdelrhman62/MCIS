"""Tests for src.train (Trainer).

We reuse the tiny-BERT pattern from tests/models/test_oce.py to avoid network.
Focus on:
- Optimizer construction (param grouping, weight decay separation)
- Scheduler shape (warmup → decay)
- Loss aggregation (uniform + weighted)
- Checkpoint round-trip preserves active_axes (handoff §2.3)
- Phase auto-config from cfg.data.module
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.data.label_vocab import LabelVocab
from src.data.loaders import BaheyaM1Dataset, collate_m1
from src.models.oce import OncologyCodingEngine
from src.train import (
    Trainer,
    TrainState,
    _linear_warmup_then_decay,
    _resolve_device,
)
from src.utils.config import BenchmarkConfig


# ---------------------------------------------------------------------------
# Tiny-BERT fixtures (reused from test_oce.py pattern)
# ---------------------------------------------------------------------------

_VOCAB_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
    f"tok{i}" for i in range(50)
]


@pytest.fixture(scope="module")
def tiny_vocab_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    p.write_text("\n".join(_VOCAB_TOKENS))
    return p


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_vocab_file):
    return BertTokenizerFast(
        vocab_file=str(tiny_vocab_file),
        do_lower_case=True,
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
        cls_token="[CLS]", mask_token="[MASK]",
    )


@pytest.fixture(scope="module")
def tiny_bert_config():
    return BertConfig(
        vocab_size=len(_VOCAB_TOKENS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )


@pytest.fixture
def patched_encoder_module(monkeypatch, tiny_tokenizer, tiny_bert_config):
    """Patch the encoder's HF imports so OCE.from_config skips the network."""
    import src.models.encoder as enc_mod

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return tiny_tokenizer

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            cfg = tiny_bert_config
            if "attn_implementation" in kwargs:
                cfg = BertConfig(
                    **{**cfg.to_dict(), "_attn_implementation": kwargs["attn_implementation"]}
                )
            return BertModel(cfg)

    monkeypatch.setattr(enc_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(enc_mod, "AutoModel", _FakeAutoModel)


@pytest.fixture
def tiny_oce_for_vocab(patched_encoder_module, tiny_cfg):
    """Build a tiny OCE whose head sizes match the actual vocab K."""
    cfg, vocab = tiny_cfg
    cfg_dict = {
        "encoder": {
            "backbone_name": "fake/tiny",
            "segment_size": 32,
            "max_segments": 2,
            "strict_length": False,
        },
        "heads": {
            "single_pick": {a: vocab[a].num_classes for a in cfg.axis_types.single_pick},
            "multi_label": {a: vocab[a].num_classes for a in cfg.axis_types.multilabel},
        },
        "attention": {"attn_dim": 16},
    }
    return OncologyCodingEngine.from_config(cfg_dict)


@pytest.fixture
def tiny_oce(patched_encoder_module):
    """Build a tiny OCE for trainer tests."""
    cfg_dict = {
        "encoder": {
            "backbone_name": "fake/tiny",
            "segment_size": 32,
            "max_segments": 2,
            "strict_length": False,
        },
        "heads": {
            "single_pick": {"icdo3_morphology": 4, "icdo3_grade": 3},
            "multi_label": {"icd11_ext_anatomy": 3},
        },
        "attention": {"attn_dim": 16},
    }
    return OncologyCodingEngine.from_config(cfg_dict)


# ---------------------------------------------------------------------------
# Helper: tiny dataframe + vocab + cfg
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_df():
    return pd.DataFrame({
        "record_id": [f"R{i}" for i in range(8)],
        "split": ["trainable"] * 6 + ["test"] * 2,
        "fold": [0, 0, 1, 1, 2, 2, -1, -1],
        "batch": [1, 1, 2, 2, 1, 2, 1, 2],
        "duplicate_group_id": [-1] * 8,
        "template_flag": [False] * 8,
        "is_cancer_primary": [True] * 8,
        "text_section_tagged": [f"some text tok{i} for record" for i in range(8)],
        "icdo3_morphology": ["8500", "8520", "8500", "8520", "8500", None, "8500", "8520"],
        "icdo3_grade": ["2", "3", "1", "2", "3", "1", "2", "3"],
        "icd11_ext_anatomy": [["XA1"], [], ["XA1", "XA2"], ["XA2"], [], ["XA1"], [], ["XA2"]],
    })


@pytest.fixture
def tiny_cfg(tiny_df, tmp_path):
    """Build a BenchmarkConfig with paths pointing into tmp_path."""
    parquet_path = tmp_path / "data.parquet"
    tiny_df.to_parquet(parquet_path)

    # Build vocab on trainable, save.
    vocab = LabelVocab.build_from_axis_spec(
        df=tiny_df,
        single_pick=["icdo3_morphology", "icdo3_grade"],
        multi_label=["icd11_ext_anatomy"],
        module="M1",
        dataset="tiny_test",
        built_from_split="trainable",
        seed=42,
    )
    vocab_path = tmp_path / "vocab.json"
    vocab.save_json(vocab_path)

    cfg = BenchmarkConfig.from_dict({
        "experiment_name": "tiny_test",
        "seed": 42,
        "data": {
            "parquet": str(parquet_path),
            "label_vocab": str(vocab_path),
            "cv_folds": str(tmp_path / "missing.csv"),  # triggers fallback to df['fold']
            "module": "M1",
            "text_field": "text_section_tagged",
        },
        "label_cardinalities": vocab.cardinalities(),
        "axis_types": {
            "single_pick": ["icdo3_morphology", "icdo3_grade"],
            "multilabel": ["icd11_ext_anatomy"],
        },
        "aux_fields": [],
        "model": {
            "encoder_hf_id": "fake/tiny",
            "segment_size": 32,
            "max_segments": 2,
            "strict_length": False,
            "diagnostic_max_tokens": 64,
            "attention_hidden_dim": 16,
        },
        "train": {
            "epochs": 2,
            "micro_batch_size": 2,
            "grad_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "warmup_ratio": 0.1,
            "early_stopping_patience": 5,
            "mixed_precision": "no",
        },
        "cv": {"enabled": True, "n_folds": 3, "seed": 42},
        "eval": {"rare_code_threshold": 1, "subgroups": []},
        "logging": {
            "project": "test",
            "wandb_enabled": False,
            "wandb_mode": "disabled",
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "output_dir": str(tmp_path / "out"),
        },
    })
    return cfg, vocab


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_warmup_then_decay_shape(self) -> None:
        params = [torch.nn.Parameter(torch.zeros(1))]
        opt = torch.optim.SGD(params, lr=1.0)
        sched = _linear_warmup_then_decay(opt, num_warmup_steps=10, num_total_steps=100)
        # Step through: at step 0 → ~0, at step 10 → 1.0, at step 100 → 0.0
        lrs = []
        for _ in range(101):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        # Warmup
        assert lrs[0] == 0.0
        assert lrs[5] == pytest.approx(0.5, abs=0.05)
        assert lrs[10] == pytest.approx(1.0, abs=0.01)
        # Decay
        assert lrs[55] == pytest.approx(0.5, abs=0.05)
        assert lrs[100] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Optimizer param grouping
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_decay_groups_separated(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        # Manually build trainer with empty loaders
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        device = torch.device("cpu")
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, device)

        groups = trainer.optimizer.param_groups
        assert len(groups) == 2
        # Decay group has nonzero weight_decay; the other doesn't.
        decays = [g["weight_decay"] for g in groups]
        assert sorted(decays) == [0.0, cfg.train.weight_decay]


# ---------------------------------------------------------------------------
# Loss aggregation (Q1 lock)
# ---------------------------------------------------------------------------


class TestLossAggregation:
    def test_uniform_sum_default(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))

        # Three axis losses, all (loss, n_valid)
        losses = {
            "icdo3_morphology": (torch.tensor(0.5), 4),
            "icdo3_grade":      (torch.tensor(0.3), 4),
            "icd11_ext_anatomy": (torch.tensor(0.2), 4),
        }
        total = trainer._aggregate_losses(losses)
        assert total is not None
        assert total.item() == pytest.approx(1.0)

    def test_weighted_sum(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))
        # Inject weights manually (cfg.train doesn't have loss_weights field by default)
        trainer.loss_weights = {"icdo3_morphology": 2.0, "icd11_ext_anatomy": 0.5}

        losses = {
            "icdo3_morphology": (torch.tensor(0.5), 4),    # × 2.0 = 1.0
            "icdo3_grade":      (torch.tensor(0.3), 4),    # × 1.0 (no weight) = 0.3
            "icd11_ext_anatomy": (torch.tensor(0.2), 4),   # × 0.5 = 0.1
        }
        total = trainer._aggregate_losses(losses)
        assert total.item() == pytest.approx(1.4)

    def test_zero_n_valid_skipped(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))

        losses = {
            "icdo3_morphology": (torch.tensor(0.5), 0),  # all targets null this batch
            "icdo3_grade":      (torch.tensor(0.3), 4),
        }
        total = trainer._aggregate_losses(losses)
        # Only grade contributes
        assert total.item() == pytest.approx(0.3)

    def test_all_zero_returns_none(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))

        losses = {
            "icdo3_morphology": (torch.tensor(0.5), 0),
            "icdo3_grade":      (torch.tensor(0.3), 0),
        }
        total = trainer._aggregate_losses(losses)
        assert total is None


# ---------------------------------------------------------------------------
# Phase auto-config (handoff §2.3 contract)
# ---------------------------------------------------------------------------


class TestPhase:
    def test_module_m1_activates_all(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        cfg.data.module = "M1"
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))
        # 'all' phase activates icdo3_* + icd11_*; the tiny model has all 3.
        assert set(trainer.model.active_axes) == {"icdo3_morphology", "icdo3_grade", "icd11_ext_anatomy"}

    def test_module_tcga_activates_icdo3_only(self, tiny_cfg, tiny_oce):
        cfg, vocab = tiny_cfg
        cfg.data.module = "TCGA_pretrain"
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))
        # icdo3_only → only icdo3_* heads active. icd11_ext_anatomy must be off.
        active = set(trainer.model.active_axes)
        assert "icd11_ext_anatomy" not in active
        assert "icdo3_morphology" in active


# ---------------------------------------------------------------------------
# Checkpoint round-trip (handoff §2.3 / §5.1)
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_load_preserves_active_axes(self, tiny_cfg, tiny_oce, tmp_path):
        from src.eval.metrics import AggregateMetrics

        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))

        # Restrict to a subset of axes (mimics Phase 1)
        trainer.model.set_active_axes({"icdo3_morphology"})
        original_axes = set(trainer.model.active_axes)

        agg = AggregateMetrics(per_axis={}, mean_f1_macro_singlepick=0.5,
                                mean_f1_micro_multilabel=0.0, early_stop_metric=0.5)
        ckpt_path = tmp_path / "best.pt"
        trainer._save_checkpoint(ckpt_path, agg, wandb_run_id="testrun123")
        assert ckpt_path.exists()

        ckpt = torch.load(ckpt_path)
        assert set(ckpt["active_axes"]) == original_axes
        assert ckpt["wandb_run_id"] == "testrun123"
        assert "vocab_sha" in ckpt
        assert "data_sha" in ckpt

    def test_load_restores_active_axes(self, tiny_cfg, tiny_oce, tmp_path):
        from src.eval.metrics import AggregateMetrics

        cfg, vocab = tiny_cfg
        loader = DataLoader([], batch_size=1, collate_fn=lambda b: b)
        trainer = Trainer(cfg, tiny_oce, vocab, loader, loader, torch.device("cpu"))

        trainer.model.set_active_axes({"icdo3_grade"})
        agg = AggregateMetrics(per_axis={}, mean_f1_macro_singlepick=0.0,
                                mean_f1_micro_multilabel=0.0, early_stop_metric=0.0)
        ckpt_path = tmp_path / "ckpt.pt"
        trainer._save_checkpoint(ckpt_path, agg, wandb_run_id=None)

        # Reset model phase to all axes, then load → load should restore subset.
        trainer.model.set_phase("all")
        assert "icdo3_morphology" in trainer.model.active_axes

        ckpt = Trainer.load_checkpoint(ckpt_path, trainer.model)
        assert set(trainer.model.active_axes) == {"icdo3_grade"}


# ---------------------------------------------------------------------------
# Eval pipeline (no train, just forward + metric)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_runs_end_to_end(self, tiny_cfg, tiny_oce_for_vocab, tiny_df):
        cfg, vocab = tiny_cfg

        # Build a real loader on tiny_df trainable rows
        ds = BaheyaM1Dataset(tiny_df[tiny_df["split"] == "trainable"].reset_index(drop=True),
                             vocab, cfg)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_m1, shuffle=False)

        trainer = Trainer(cfg, tiny_oce_for_vocab, vocab, loader, loader, torch.device("cpu"))
        agg = trainer.evaluate(loader, prefix="val")

        # All 3 axes should produce metrics
        assert "icdo3_morphology" in agg.per_axis
        assert "icdo3_grade" in agg.per_axis
        assert "icd11_ext_anatomy" in agg.per_axis

        # early_stop_metric is in [0, 1]
        assert 0.0 <= agg.early_stop_metric <= 1.0
