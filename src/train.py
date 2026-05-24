"""MCIS Trainer — Phase 1 (TCGA pretrain) and Phase 2 (Baheya fine-tune).

Design locks (per chat 2026-05-09):
  Q1 — Loss aggregation: uniform sum across active axes by default; per-axis
       weights configurable via cfg.train.loss_weights for E5 ablation.
  Q2 — Batch composition: cfg.train.{micro_batch_size, grad_accumulation_steps}
       give effective_batch = 8 * 4 = 32. Independent of model.segment_size /
       model.max_segments (per-record token cap is encoder-side, not trainer).
  Q3 — Early stopping: dual-metric average of {mean F1-Macro single-pick,
       mean F1-Micro multi-label}. cfg.train.early_stopping_patience epochs.
  Q4 — W&B granularity: per-step (loss/lr/grad_norm), per-eval (per-axis F1),
       config + repro metadata at run start.
  Q5 — Checkpoints: best.pt + final.pt + meta.json under
       checkpoints/{experiment_name}/. Phase NOT in state_dict — must call
       set_active_axes(...) after load_state_dict() (handoff §2.3).

Phase awareness:
  - Phase 1 (TCGA pretrain): cfg.data.module = 'TCGA_pretrain'. Trainer
    auto-restricts active axes to ICD-O-3 only (TCGA ICD-11 = silver,
    never label per architecture v6 §6.4).
  - Phase 2 (Baheya fine-tune): cfg.data.module = 'M1'. All 10 axes active.
  - To init Phase 2 from Phase 1: set cfg.model.init_from_checkpoint and
    optionally tune cfg.model.{init_load_attention, init_load_heads}.
    Defaults (load encoder + attention, reset heads) match architecture v6
    §4.4: encoder-only transfer, fresh heads sized for Baheya vocab.

Memory model:
  - Dataset is in-memory (BaheyaM1Dataset materializes the whole table).
  - Tokenization happens INSIDE OncologyEncoder.forward (handoff §5.2). The
    DataLoader passes raw List[str] and stacked label tensors via collate_m1.
  - Mixed precision: cfg.train.mixed_precision in {no, fp16, bf16}. M4 uses
    'no'; A100 uses 'bf16' (preferred over fp16 on Ampere+).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.label_vocab import LabelVocab
from src.data.loaders import (
    BaheyaM1Dataset,
    collate_m1,
    load_parquet,
    split_by_fold,
    split_by_value,
    split_trainable_test,
)
from src.eval.metrics import (
    AggregateMetrics,
    aggregate_axis_metrics,
    f1_macro_singlepick,
    f1_multilabel,
)
from src.losses import attach_loss_fns_to_heads, build_loss_fns
from src.models.heads import NULL_TARGET_SENTINEL
from src.models.oce import OCEOutput, OncologyCodingEngine
from src.utils.checkpoint_loader import load_checkpoint_partial
from src.utils.config import BenchmarkConfig
from src.utils.logging import WandbRun, setup_console_logger
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    """Return current git SHA (short) or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _file_sha(path: str | Path) -> str | None:
    """SHA-256 of a file (for reproducibility logging)."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _linear_warmup_then_decay(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0 → 1 then linear decay 1 → 0. PLM-ICD canonical."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_total_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resolve_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


# ---------------------------------------------------------------------------
# Trainer state
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -float("inf")
    best_epoch: int = -1
    epochs_since_improve: int = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """End-to-end training loop for OCE.

    Construct via ``Trainer.from_config(cfg)`` for the full pipeline, or pass
    pre-built model + dataloaders for unit tests.
    """

    def __init__(
        self,
        cfg: BenchmarkConfig,
        model: OncologyCodingEngine,
        vocab: LabelVocab,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        log: Any = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log = log or setup_console_logger("mcis.trainer")

        self.state = TrainState()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler: torch.cuda.amp.GradScaler | None = None
        if cfg.train.mixed_precision == "fp16" and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # Phase: drive active heads from the data module.
        self._set_phase_for_module(cfg.data.module)

        # Loss aggregation weights. None = uniform.
        self.loss_weights: dict[str, float] | None = getattr(
            cfg.train, "loss_weights", None
        )

        # Segmentation config — drives pre-segmentation in _forward().
        self._seg_mode = cfg.model.segmentation_mode
        self._seg_overlap = cfg.model.overlap_tokens
        self._seg_size = cfg.model.segment_size

    # ------------- Construction -------------

    @classmethod
    def from_config(
        cls,
        cfg: BenchmarkConfig,
        fold_idx: int = 0,
        device: str = "auto",
    ) -> "Trainer":
        """Build trainer end-to-end from a YAML-driven config."""
        log = setup_console_logger("mcis.trainer")
        log.info("Building trainer for experiment=%s, fold=%d", cfg.experiment_name, fold_idx)

        set_seed(cfg.seed)

        # 1. Load vocab + parquet.
        log.info("Loading vocab from %s", cfg.data.label_vocab)
        vocab = LabelVocab.load_json(cfg.data.label_vocab)
        cfg.validate_against_vocab(vocab.cardinalities())

        log.info("Loading parquet from %s", cfg.data.parquet)
        df = load_parquet(cfg.data.parquet)

        # 2. Split. Two regimes:
        #   - cv.enabled (Baheya M1): trainable/test 2-way, then fold-wise.
        #   - no CV + data.val_value set (Phase 1 TCGA): pre-baked 3-way
        #     (pretrain_train / pretrain_val / pretrain_test). pretrain_test
        #     is held out, not loaded here.
        #   - no CV + no val_value (legacy): 90/10 slice off trainable.
        if cfg.cv.enabled:
            trainable_df, test_df = split_trainable_test(
                df, cfg.data.trainable_value, cfg.data.test_value
            )
            log.info("Splits: trainable=%d, test=%d", len(trainable_df), len(test_df))
            try:
                folds_df = pd.read_csv(cfg.data.cv_folds)
            except FileNotFoundError:
                folds_df = None
            train_df, val_df = split_by_fold(trainable_df, folds_df, fold_idx)
        elif cfg.data.val_value is not None:
            # Phase 1 TCGA: explicit pre-baked train/val splits.
            train_df = split_by_value(df, cfg.data.trainable_value)
            val_df = split_by_value(df, cfg.data.val_value)
            log.info(
                "Pre-baked split: train(%s)=%d, val(%s)=%d (test split held out)",
                cfg.data.trainable_value, len(train_df),
                cfg.data.val_value, len(val_df),
            )
        else:
            trainable_df, test_df = split_trainable_test(
                df, cfg.data.trainable_value, cfg.data.test_value
            )
            log.info("Splits: trainable=%d, test=%d", len(trainable_df), len(test_df))
            n = len(trainable_df)
            n_val = max(1, int(0.1 * n))
            val_df = trainable_df.iloc[:n_val].reset_index(drop=True)
            train_df = trainable_df.iloc[n_val:].reset_index(drop=True)
        log.info("Fold %d: train=%d, val=%d", fold_idx, len(train_df), len(val_df))
        
        # E5b: inject SEER templates into train fold if augmentation enabled
        aug_cfg = cfg.augmentation if hasattr(cfg, "augmentation") else {}
        if isinstance(aug_cfg, dict) and aug_cfg.get("enabled", False):
            seer_path = aug_cfg.get("seer_parquet", "")
            if seer_path:
                from src.data.loaders import inject_seer_templates
                train_df = inject_seer_templates(train_df, seer_path)
                log.info("SEER augmentation: train size after inject=%d", len(train_df))

        train_ds = BaheyaM1Dataset(train_df, vocab, cfg)
        val_ds = BaheyaM1Dataset(val_df, vocab, cfg)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.micro_batch_size,
            shuffle=True,
            collate_fn=collate_m1,
            num_workers=0,  # in-memory dataset; multiprocess overhead not worth it
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.micro_batch_size,
            shuffle=False,
            collate_fn=collate_m1,
            num_workers=0,
        )

        # 4. Model.
        device_t = _resolve_device(device)
        log.info("Device: %s", device_t)

        model_cfg = cls._build_model_cfg(cfg, vocab)
        model = OncologyCodingEngine.from_config(model_cfg)

        # Optional: init from a pretrain/E2 checkpoint.
        # Defaults (cfg.model.init_load_{attention,heads}): load encoder + attn,
        # reset heads. This is Phase1→Phase2 transfer per arch v6 §4.4.
        # For E5a/E5b/E6 stacking on same-vocab base, override in the YAML:
        #   model:
        #     init_from_checkpoint: checkpoints/E2/fold_0/best.pt
        #     init_load_heads: true
        if cfg.model.init_from_checkpoint:
            report = load_checkpoint_partial(
                model=model,
                ckpt_path=cfg.model.init_from_checkpoint,
                load_encoder=True,
                load_attention=cfg.model.init_load_attention,
                load_heads=cfg.model.init_load_heads,
                map_location="cpu",
                log=log,
            )
            log.info(
                "  init_from_checkpoint: loaded=%d skipped=%d shape_mismatched=%d",
                len(report.loaded_keys),
                len(report.skipped_by_filter),
                len(report.shape_mismatched),
            )

        # Optional: swap per-axis loss strategies (grid A1 class-weighted /
        # A2 focal / focal_weighted). For cfg.train.loss == "bce" (default)
        # build_loss_fns returns {} and attach is a no-op.
        loss_fns = build_loss_fns(cfg, vocab, train_df, log=log)
        attach_loss_fns_to_heads(model, loss_fns, log=log)

        return cls(cfg, model, vocab, train_loader, val_loader, device_t, log)

    @staticmethod
    def _build_model_cfg(cfg: BenchmarkConfig, vocab: LabelVocab) -> dict[str, Any]:
        """Translate BenchmarkConfig + vocab into the OCE.from_config dict."""
        single_pick_spec = {
            axis: vocab[axis].num_classes
            for axis in cfg.axis_types.single_pick
            if axis in vocab.axes
        }
        multi_label_spec = {
            axis: vocab[axis].num_classes
            for axis in cfg.axis_types.multilabel
            if axis in vocab.axes
        }
        return {
            "encoder": {
                "backbone_name": cfg.model.encoder_hf_id,
                "segment_size": cfg.model.segment_size,
                "max_segments": cfg.model.max_segments,
                "strict_length": cfg.model.strict_length,
            },
            "heads": {
                "single_pick": single_pick_spec,
                "multi_label": multi_label_spec,
            },
            "attention": {
                "attn_dim": cfg.model.attention_hidden_dim,
            },
        }

    # ------------- Optimizer / scheduler -------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        # Decay-weight-aware param grouping (BERT-standard: no decay on bias/LayerNorm).
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        decay_params, no_decay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": self.cfg.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=self.cfg.train.learning_rate)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        steps_per_epoch = max(1, len(self.train_loader) // self.cfg.train.grad_accumulation_steps)
        total_steps = steps_per_epoch * self.cfg.train.epochs
        warmup_steps = int(total_steps * self.cfg.train.warmup_ratio)
        return _linear_warmup_then_decay(self.optimizer, warmup_steps, total_steps)

    # ------------- Phase control -------------

    def _set_phase_for_module(self, module: str) -> None:
        if module == "TCGA_pretrain":
            self.model.set_phase("icdo3_only")
            self.log.info("Phase 1 active: ICD-O-3 axes only (TCGA ICD-11 silver, masked)")
        elif module == "M1":
            self.model.set_phase("all")
            self.log.info("Phase 2 active: all 10 axes")
        else:
            self.log.warning("Unknown module %r; leaving phase as-is", module)

    # ------------- Train / eval loop -------------

    def train(self) -> TrainState:
        cfg = self.cfg
        device = self.device

        ckpt_dir = Path(cfg.logging.checkpoint_dir) / cfg.experiment_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        run = WandbRun(
            project=cfg.logging.project,
            run_name=cfg.logging.run_name or f"{cfg.experiment_name}_seed{cfg.seed}",
            config=self._build_repro_config(),
            enabled=cfg.logging.wandb_enabled,
            mode=cfg.logging.wandb_mode,
            tags=[cfg.experiment_name, f"module_{cfg.data.module}", f"seed_{cfg.seed}"],
        )

        with run:
            run.summary.update({"vocab_cardinalities": self.vocab.cardinalities()})

            self.log.info("Starting training: %d epochs, %d steps/epoch (effective batch=%d)",
                          cfg.train.epochs,
                          max(1, len(self.train_loader) // cfg.train.grad_accumulation_steps),
                          cfg.train.micro_batch_size * cfg.train.grad_accumulation_steps)

            for epoch in range(cfg.train.epochs):
                self.state.epoch = epoch
                t0 = time.time()
                train_metrics = self._train_one_epoch(run)
                t_train = time.time() - t0

                t0 = time.time()
                agg = self.evaluate(self.val_loader, prefix="val")
                t_eval = time.time() - t0

                run.log(agg.flatten_for_wandb(prefix="val"), step=self.state.global_step)
                run.log({
                    "epoch": epoch,
                    "epoch_time_train_s": t_train,
                    "epoch_time_eval_s": t_eval,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                }, step=self.state.global_step)

                self.log.info(
                    "epoch=%d  train_loss=%.4f  val_early_stop=%.4f  (mean_macro=%.4f mean_micro=%.4f)  t_train=%.1fs t_eval=%.1fs",
                    epoch, train_metrics["loss_total"],
                    agg.early_stop_metric, agg.mean_f1_macro_singlepick, agg.mean_f1_micro_multilabel,
                    t_train, t_eval,
                )

                # Best-checkpoint logic.
                improved = agg.early_stop_metric > self.state.best_metric
                if improved:
                    self.state.best_metric = agg.early_stop_metric
                    self.state.best_epoch = epoch
                    self.state.epochs_since_improve = 0
                    self._save_checkpoint(ckpt_dir / "best.pt", agg, run.run_id)
                    run.summary["best_metric"] = self.state.best_metric
                    run.summary["best_epoch"] = epoch
                else:
                    self.state.epochs_since_improve += 1

                # Final checkpoint at end of every epoch (resumable).
                self._save_checkpoint(ckpt_dir / "final.pt", agg, run.run_id)

                if self.state.epochs_since_improve >= cfg.train.early_stopping_patience:
                    self.log.info(
                        "Early stop at epoch=%d (best epoch=%d, metric=%.4f)",
                        epoch, self.state.best_epoch, self.state.best_metric,
                    )
                    break

            self._write_meta_json(ckpt_dir / "meta.json", run.run_id)
            run.summary["best_metric"] = self.state.best_metric
            run.summary["best_epoch"] = self.state.best_epoch

        return self.state

    def _train_one_epoch(self, run: WandbRun) -> dict[str, float]:
        cfg = self.cfg
        device = self.device
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        loss_per_axis_sum: dict[str, float] = {}
        n_optim_steps = 0
        accum_steps = cfg.train.grad_accumulation_steps

        for step, batch in enumerate(self.train_loader):
            outputs, losses = self._forward_with_loss(batch)
            total_loss = self._aggregate_losses(losses)

            if total_loss is None:
                # All heads inactive or all targets null this batch — rare but safe.
                continue

            scaled = total_loss / accum_steps
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            loss_sum += float(total_loss.detach().cpu())
            for axis, (axis_loss, _n) in losses.items():
                loss_per_axis_sum[axis] = loss_per_axis_sum.get(axis, 0.0) + float(axis_loss.detach().cpu())

            if (step + 1) % accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.train.max_grad_norm
                )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.state.global_step += 1
                n_optim_steps += 1

                run.log({
                    "train/loss_total": float(total_loss.detach().cpu()),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm),
                }, step=self.state.global_step)

        n = max(1, n_optim_steps)
        per_axis_means = {
            f"loss_{axis}": v / max(1, len(self.train_loader))
            for axis, v in loss_per_axis_sum.items()
        }
        return {"loss_total": loss_sum / max(1, len(self.train_loader)), **per_axis_means}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> AggregateMetrics:
        self.model.eval()
        cfg = self.cfg
        device = self.device

        all_logits: dict[str, list[torch.Tensor]] = {}
        all_targets: dict[str, list[torch.Tensor]] = {}

        for batch in loader:
            outputs = self._forward(batch)
            for axis, logits in outputs.logits_per_axis.items():
                all_logits.setdefault(axis, []).append(logits.detach().cpu())
                target_key = f"labels_{axis}"
                if target_key in batch:
                    all_targets.setdefault(axis, []).append(batch[target_key].detach().cpu())

        per_axis: dict[str, dict[str, float | int]] = {}
        for axis, logits_list in all_logits.items():
            if axis not in all_targets:
                continue
            logits_cat = torch.cat(logits_list, dim=0)
            targets_cat = torch.cat(all_targets[axis], dim=0)
            if axis in self.cfg.axis_types.single_pick:
                num_classes = self.vocab[axis].num_classes
                per_axis[axis] = f1_macro_singlepick(
                    logits_cat, targets_cat, num_classes=num_classes,
                )
            else:
                per_axis[axis] = f1_multilabel(logits_cat, targets_cat)

        return aggregate_axis_metrics(
            per_axis,
            single_pick_axes=list(self.cfg.axis_types.single_pick),
            multi_label_axes=list(self.cfg.axis_types.multilabel),
        )

    # ------------- Forward / loss aggregation -------------

    def _forward(self, batch: dict[str, Any]) -> OCEOutput:
        texts = batch["texts"]
        if self._seg_mode != "fixed":
            from src.data.segmentation import segment_text
            texts = [
                segment_text(
                    t,
                    mode=self._seg_mode,
                    tokenizer=self.model.encoder.tokenizer,
                    segment_size=self._seg_size,
                    overlap_tokens=self._seg_overlap,
                )
                for t in texts
            ]
            # texts is now List[List[str]] — encoder auto-detects pre_segmented
        return self.model(texts)

    def _forward_with_loss(
        self, batch: dict[str, Any]
    ) -> tuple[OCEOutput, dict[str, tuple[torch.Tensor, int]]]:
        outputs = self._forward(batch)
        # Build target dict for active axes only.
        targets: dict[str, torch.Tensor] = {}
        for axis in self.model.active_axes:
            key = f"labels_{axis}"
            if key not in batch:
                continue
            targets[axis] = batch[key].to(self.device)
        losses = self.model.compute_losses(outputs, targets)
        return outputs, losses

    def _aggregate_losses(
        self, losses: dict[str, tuple[torch.Tensor, int]]
    ) -> torch.Tensor | None:
        """Sum per-axis losses, optionally weighted (Q1)."""
        if not losses:
            return None
        terms: list[torch.Tensor] = []
        for axis, (axis_loss, n_valid) in losses.items():
            if n_valid == 0:
                continue
            w = 1.0
            if self.loss_weights and axis in self.loss_weights:
                w = float(self.loss_weights[axis])
            terms.append(w * axis_loss)
        if not terms:
            return None
        return torch.stack(terms).sum()

    # ------------- Checkpoint I/O -------------

    def _save_checkpoint(
        self,
        path: Path,
        last_metrics: AggregateMetrics,
        wandb_run_id: str | None,
    ) -> None:
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "metric_name": "val/early_stop_metric",
            "active_axes": list(self.model.active_axes),  # NOT in state_dict; persist explicitly
            "cfg_dict": self.cfg.to_dict(),
            "vocab_sha": _file_sha(self.cfg.data.label_vocab),
            "data_sha": _file_sha(self.cfg.data.parquet),
            "wandb_run_id": wandb_run_id,
            "git_commit": _git_commit(),
            "saved_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: OncologyCodingEngine,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Load a checkpoint into an existing model.

        Critical (handoff §2.3 / §5.1): ``head_active`` flags are NOT in
        state_dict. Caller MUST call ``model.set_active_axes(active_axes)``
        with the returned set, or the active heads won't match what the
        checkpoint was saved with.
        """
        device_t = _resolve_device(device)
        ckpt = torch.load(path, map_location=device_t)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Restore phase per saved active_axes:
        model.set_active_axes(set(ckpt["active_axes"]))
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

    def _write_meta_json(self, path: Path, wandb_run_id: str | None) -> None:
        meta = {
            "experiment_name": self.cfg.experiment_name,
            "module": self.cfg.data.module,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "metric_name": "val/early_stop_metric",
            "wandb_run_id": wandb_run_id,
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "active_axes": list(self.model.active_axes),
            "vocab_sha": _file_sha(self.cfg.data.label_vocab),
            "data_sha": _file_sha(self.cfg.data.parquet),
            "saved_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(meta, indent=2))

    def _build_repro_config(self) -> dict[str, Any]:
        return {
            **self.cfg.to_dict(),
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "vocab_sha": _file_sha(self.cfg.data.label_vocab),
            "data_sha": _file_sha(self.cfg.data.parquet),
            "device": str(self.device),
            "torch_version": torch.__version__,
        }

# ===========================================================================
# CLI entry point — append to end of src/train.py
# ===========================================================================
#
# Usage:
#     python -m src.train --config configs/B0.yaml
#     python -m src.train --config configs/E1.yaml --fold 2 --device mps
#     python -m src.train --config configs/B0.yaml --smoke      # 1 epoch, no W&B
#
# Per handoff §6.2: ~15-20 lines argparse. Overrides mutate the dataclass
# fields after from_yaml() so any locked-in YAML default can be bumped at the
# CLI for quick experiments (smoke runs, seed sweeps) without forking the YAML.


def _build_arg_parser() -> "argparse.ArgumentParser":
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m src.train",
        description="MCIS trainer (Phase 1 TCGA pretrain / Phase 2 Baheya M1).",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (e.g. configs/B0.yaml).")
    p.add_argument("--fold", type=int, default=0,
                   help="CV fold index (default 0). Ignored if cfg.cv.enabled=False.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="Device override (default auto).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override cfg.seed.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override cfg.train.epochs (smoke: 1).")
    p.add_argument("--experiment-name", default=None,
                   help="Override cfg.experiment_name (avoid ckpt collisions).")
    p.add_argument("--wandb-mode", default=None,
                   choices=["online", "offline", "disabled"],
                   help="Override cfg.logging.wandb_mode.")
    p.add_argument("--smoke", action="store_true",
                   help="Sugar: --epochs 1 --wandb-mode disabled.")
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Override cfg.model.init_from_checkpoint (path to .pt).")
    p.add_argument("--init-load-heads", action="store_true",
                   help="Override cfg.model.init_load_heads=True (stack on same-vocab ckpt).")
    return p


def _apply_cli_overrides(cfg: BenchmarkConfig, args: Any) -> None:
    """Mutate cfg in place per CLI flags. Order: --smoke first, then explicit."""
    if args.smoke:
        cfg.train.epochs = 1
        cfg.logging.wandb_mode = "disabled"
        cfg.logging.wandb_enabled = False
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.wandb_mode is not None:
        cfg.logging.wandb_mode = args.wandb_mode
        cfg.logging.wandb_enabled = args.wandb_mode != "disabled"
    if args.init_from_checkpoint is not None:
        cfg.model.init_from_checkpoint = args.init_from_checkpoint
    if args.init_load_heads:
        cfg.model.init_load_heads = True


def main() -> int:
    import argparse  # noqa: F401  (silences any lint on the parser builder)
    log = setup_console_logger("mcis.train.cli")
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        return 1

    log.info("Loading config: %s", args.config)
    cfg = BenchmarkConfig.from_yaml(args.config)
    _apply_cli_overrides(cfg, args)

    log.info(
        "Resolved run: experiment=%s seed=%d fold=%d epochs=%d wandb=%s device=%s",
        cfg.experiment_name, cfg.seed, args.fold, cfg.train.epochs,
        cfg.logging.wandb_mode, args.device,
    )

    trainer = Trainer.from_config(cfg, fold_idx=args.fold, device=args.device)
    state = trainer.train()

    log.info(
        "Done. best_epoch=%d best_metric=%.4f total_steps=%d",
        state.best_epoch, state.best_metric, state.global_step,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
