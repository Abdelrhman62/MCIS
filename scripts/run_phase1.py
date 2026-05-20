"""Phase 1 runner — TCGA non-BRCA encoder pretrain.

Thin wrapper over ``Trainer.from_config``. Phase 1 is a SINGLE train/val
run (no CV, no fold loop): cfg.cv.enabled=false + data.val_value set, so
the trainer's pre-baked-split branch uses pretrain_train / pretrain_val
directly (pretrain_test held out).

Why a wrapper at all (vs `python -m src.train`):
  1. Encoder-choice sugar: --encoder {pubmedbert,pathologybert} resolves to
     the right Phase 1 YAML so the two E4 anchors aren't fat-fingered.
  2. Prints the exact checkpoint path E2 must consume, so the
     Phase1 → Phase2 handoff is unambiguous (init_from_checkpoint).
  3. Asserts Phase-1 invariants before launch (module=TCGA_pretrain,
     no ICD-11 axes, cv disabled, val_value set) — fail loud, not after
     30 min of GPU.

Usage:
    python -m scripts.run_phase1 --encoder pubmedbert
    python -m scripts.run_phase1 --encoder pathologybert --device cuda
    python -m scripts.run_phase1 --config configs/Phase1_TCGA_pubmedbert.yaml
    python -m scripts.run_phase1 --encoder pubmedbert --smoke   # 1 epoch

Emitted checkpoint (consumed by configs/E2_baheya_from_tcga.yaml):
    checkpoints/<experiment_name>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.train import Trainer, _apply_cli_overrides
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger

_ENCODER_CONFIGS: dict[str, str] = {
    "pubmedbert": "configs/Phase1_TCGA_pubmedbert.yaml",
    "pathologybert": "configs/Phase1_TCGA_pathologybert.yaml",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.run_phase1",
        description="MCIS Phase 1 — TCGA non-BRCA encoder pretrain.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--encoder",
        choices=sorted(_ENCODER_CONFIGS),
        help="Pick a built-in Phase 1 config by encoder.",
    )
    src.add_argument(
        "--config",
        type=Path,
        help="Explicit Phase 1 YAML (overrides --encoder).",
    )
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"],
                   help="Device override (default auto; A100=cuda).")
    p.add_argument("--seed", type=int, default=None,
                   help="Override cfg.seed.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override cfg.train.epochs.")
    p.add_argument("--experiment-name", default=None,
                   help="Override cfg.experiment_name (also moves ckpt dir).")
    p.add_argument("--wandb-mode", default=None,
                   choices=["online", "offline", "disabled"],
                   help="Override cfg.logging.wandb_mode.")
    p.add_argument("--smoke", action="store_true",
                   help="Sugar: --epochs 1 --wandb-mode disabled.")
# Inert for Phase 1 (trains from scratch; YAML has init_from_checkpoint:
    # null). Defined only because the shared _apply_cli_overrides reads them
    # unconditionally. Exposed anyway so an unusual rerun can override.
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Override cfg.model.init_from_checkpoint (rare for Phase 1).")
    p.add_argument("--init-load-heads", action="store_true",
                   help="Override cfg.model.init_load_heads=True (rare for Phase 1).")
    return p

def _assert_phase1_invariants(cfg: BenchmarkConfig, log) -> None:
    """Fail loud before GPU spin-up if the config isn't a valid Phase 1."""
    problems: list[str] = []
    if cfg.data.module != "TCGA_pretrain":
        problems.append(
            f"data.module={cfg.data.module!r}, expected 'TCGA_pretrain' "
            f"(drives Trainer.set_phase('icdo3_only'))"
        )
    if cfg.cv.enabled:
        problems.append("cv.enabled=true; Phase 1 is a single train/val run")
    if cfg.data.val_value in (None, ""):
        problems.append(
            "data.val_value unset; Phase 1 needs an explicit pre-baked val "
            "split (e.g. 'pretrain_val')"
        )
    icd11_axes = [a for a in cfg.all_axes if a.startswith("icd11")]
    if icd11_axes:
        problems.append(
            f"ICD-11 axes present {icd11_axes}; TCGA ICD-11 is silver, "
            f"never supervised in Phase 1 (data card §7, arch v6 §6.4)"
        )
    if cfg.axis_types.multilabel:
        problems.append(
            f"multilabel axes {cfg.axis_types.multilabel}; Phase 1 is "
            f"ICD-O-3 single-pick only"
        )
    if problems:
        for pr in problems:
            log.error("Phase 1 invariant violated: %s", pr)
        raise SystemExit(
            "Aborting: config is not a valid Phase 1 setup "
            f"({len(problems)} problem(s) above)."
        )


def main() -> int:
    log = setup_console_logger("mcis.run_phase1")
    args = _build_arg_parser().parse_args()

    cfg_path = args.config if args.config else Path(_ENCODER_CONFIGS[args.encoder])
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 1

    log.info("Loading Phase 1 config: %s", cfg_path)
    cfg = BenchmarkConfig.from_yaml(cfg_path)
    _apply_cli_overrides(cfg, args)
    _assert_phase1_invariants(cfg, log)

    log.info(
        "Phase 1: experiment=%s encoder=%s seed=%d epochs=%d wandb=%s device=%s",
        cfg.experiment_name, cfg.model.encoder_hf_id, cfg.seed,
        cfg.train.epochs, cfg.logging.wandb_mode, args.device,
    )
    log.info(
        "Splits: train=%s val=%s (test=%s held out)",
        cfg.data.trainable_value, cfg.data.val_value, cfg.data.test_value,
    )

    # Phase 1 = single run. fold_idx=0 is inert because cv.enabled=false
    # and val_value is set (trainer takes the pre-baked-split branch).
    trainer = Trainer.from_config(cfg, fold_idx=0, device=args.device)
    state = trainer.train()

    ckpt = Path(cfg.logging.checkpoint_dir) / cfg.experiment_name / "best.pt"
    log.info(
        "Phase 1 done. best_epoch=%d best_metric=%.4f steps=%d",
        state.best_epoch, state.best_metric, state.global_step,
    )
    log.info("=" * 64)
    log.info("E2 consumes this checkpoint. In configs/E2_baheya_from_tcga.yaml:")
    log.info("  model.init_from_checkpoint: %s", ckpt)
    log.info("  model.init_load_heads: false   # reset for Baheya 10-axis vocab")
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
