#!/usr/bin/env python3
"""run_test_eval.py — Honest held-out test-set evaluation for MCIS.

Loads one or more per-fold best.pt checkpoints, runs inference exclusively on
the held-out test split (split == cfg.data.test_value), and reports:
  - Per-axis F1-Macro (single-pick) and F1-Micro (multi-label)
  - f1_macro_present (only over classes seen in y_true — reveals long-tail performance)
  - Aggregate early-stop metric and mean_macro / mean_micro
  - JSON + Markdown output to results/test_eval/

This script NEVER touches training or validation rows. It is the only
honest evaluation for thesis reporting — CV val numbers are optimistic.

Usage:
    # Single checkpoint (fold 0 only, quick sanity check):
    python scripts/run_test_eval.py \\
        --ckpt checkpoints/E2_phase1_25ep/best.pt \\
        --config configs/E2_baheya_from_tcga.yaml

    # Full 5-fold sweep (averages across all fold checkpoints):
    python scripts/run_test_eval.py \\
        --ckpt-pattern "checkpoints/E2_phase1_25ep_fold{}/best.pt" \\
        --config configs/E2_baheya_from_tcga.yaml \\
        --folds 0 1 2 3 4

    # Single experiment-name folder (flat, not per-fold):
    python scripts/run_test_eval.py \\
        --experiment-dir checkpoints/E2_phase1_25ep \\
        --config configs/E2_baheya_from_tcga.yaml
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.label_vocab import LabelVocab  # noqa: E402
from src.data.loaders import (  # noqa: E402
    BaheyaM1Dataset,
    collate_m1,
    load_parquet,
    split_trainable_test,
)
from src.eval.metrics import (  # noqa: E402
    aggregate_axis_metrics,
    f1_macro_singlepick,
    f1_multilabel,
)
from src.models.oce import OncologyCodingEngine  # noqa: E402
from src.utils.config import BenchmarkConfig  # noqa: E402
from src.utils.logging import setup_console_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _file_sha(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_device(preferred: str) -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model_cfg(cfg: BenchmarkConfig, vocab: LabelVocab) -> dict[str, Any]:
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
        "attention": {"attn_dim": cfg.model.attention_hidden_dim},
    }


def eval_one_checkpoint(
    ckpt_path: Path,
    cfg: BenchmarkConfig,
    vocab: LabelVocab,
    test_df,
    device: torch.device,
    log,
) -> dict[str, Any]:
    """Load one checkpoint, run on test_df, return metric dict."""
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Integrity check
    current_data_sha = _file_sha(cfg.data.parquet)
    ckpt_data_sha = ckpt.get("data_sha")
    if ckpt_data_sha and current_data_sha and ckpt_data_sha != current_data_sha:
        log.warning("DATA SHA MISMATCH: ckpt=%s current=%s — parquet may have changed!",
                    ckpt_data_sha[:12], current_data_sha[:12])
    else:
        log.info("  data_sha: OK (%s)", (current_data_sha or "n/a")[:12])

    log.info("  ckpt best_epoch=%d  best_metric=%.4f",
             ckpt.get("best_epoch", -1), ckpt.get("best_metric", 0.0))

    model_cfg = _build_model_cfg(cfg, vocab)
    model = OncologyCodingEngine.from_config(model_cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        log.warning("  %d missing keys (first 5): %s", len(missing), missing[:5])
    if unexpected:
        log.warning("  %d unexpected keys (first 5): %s", len(unexpected), unexpected[:5])
    model.set_active_axes(set(ckpt["active_axes"]))
    model.to(device)
    model.eval()

    # Build test loader (NEVER touches train/val rows)
    test_ds = BaheyaM1Dataset(test_df, vocab, cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=False,
        collate_fn=collate_m1,
        num_workers=0,
    )
    log.info("  Test set: %d records", len(test_df))

    # Collect logits + targets
    all_logits: dict[str, list[torch.Tensor]] = {}
    all_targets: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["texts"])
            for axis, logits in outputs.logits_per_axis.items():
                all_logits.setdefault(axis, []).append(logits.detach().cpu())
                tkey = f"labels_{axis}"
                if tkey in batch:
                    all_targets.setdefault(axis, []).append(batch[tkey].detach().cpu())

    logits_cat = {a: torch.cat(v, dim=0) for a, v in all_logits.items()}
    targets_cat = {a: torch.cat(v, dim=0) for a, v in all_targets.items()}

    # Single-pick metrics
    sp_axes = list(cfg.axis_types.single_pick)
    sp_metrics: dict[str, dict] = {}
    for axis in sp_axes:
        if axis not in logits_cat:
            continue
        K = vocab[axis].num_classes
        m = f1_macro_singlepick(logits_cat[axis], targets_cat[axis], num_classes=K)
        sp_metrics[axis] = m
        log.info("  [SP] %s: F1-Macro=%.4f  F1-Macro-Present=%.4f  Acc=%.4f  n=%d",
                 axis, m["f1_macro"], m["f1_macro_present"], m["accuracy"], m["n_valid"])

    # Multi-label metrics
    ml_axes = list(cfg.axis_types.multilabel)
    ml_metrics: dict[str, dict] = {}
    for axis in ml_axes:
        if axis not in logits_cat:
            continue
        m = f1_multilabel(logits_cat[axis], targets_cat[axis], threshold=0.5)
        ml_metrics[axis] = m
        log.info("  [ML] %s: F1-Micro=%.4f  F1-Macro=%.4f  n=%d",
                 axis, m["f1_micro"], m["f1_macro"], m["n_records"])

    per_axis_all = dict(sp_metrics)
    per_axis_all.update(ml_metrics)
    agg = aggregate_axis_metrics(per_axis_all, sp_axes, ml_axes)

    log.info("  AGGREGATE: early_stop=%.4f  mean_macro=%.4f  mean_micro=%.4f",
             agg.early_stop_metric, agg.mean_f1_macro_singlepick, agg.mean_f1_micro_multilabel)

    return {
        "ckpt_path": str(ckpt_path),
        "ckpt_best_epoch": int(ckpt.get("best_epoch", -1)),
        "ckpt_best_metric_trainer": float(ckpt.get("best_metric", 0.0)),
        "n_test": int(len(test_df)),
        "single_pick_metrics": sp_metrics,
        "multi_label_metrics": ml_metrics,
        "early_stop_metric": float(agg.early_stop_metric),
        "mean_f1_macro_singlepick": float(agg.mean_f1_macro_singlepick),
        "mean_f1_micro_multilabel": float(agg.mean_f1_micro_multilabel),
    }


def render_markdown(results: list[dict], cfg_path: Path, n_test: int) -> str:
    sp_axes = list(results[0]["single_pick_metrics"].keys()) if results else []
    ml_axes = list(results[0]["multi_label_metrics"].keys()) if results else []

    lines = ["# MCIS Held-Out Test Set Evaluation", ""]
    lines += [
        f"- **Config:** `{cfg_path}`",
        f"- **n_test:** {n_test}",
        f"- **n_checkpoints:** {len(results)}",
        f"- **Generated:** {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Aggregate (mean ± std across checkpoints)",
        "",
    ]

    def _mean_std(key: str) -> tuple[float, float]:
        vals = [r[key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    es_mean, es_std = _mean_std("early_stop_metric")
    macro_mean, macro_std = _mean_std("mean_f1_macro_singlepick")
    micro_mean, micro_std = _mean_std("mean_f1_micro_multilabel")

    lines += [
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
        f"| Early-Stop (mean of means) | {es_mean:.4f} | {es_std:.4f} |",
        f"| Mean F1-Macro (single-pick) | {macro_mean:.4f} | {macro_std:.4f} |",
        f"| Mean F1-Micro (multi-label) | {micro_mean:.4f} | {micro_std:.4f} |",
        "",
        "## Per-Axis (Single-Pick) — F1-Macro & F1-Macro-Present",
        "",
        "| Axis | F1-Macro Mean | F1-Macro Std | F1-Macro-Present Mean | Accuracy Mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis in sp_axes:
        vals_macro = [r["single_pick_metrics"].get(axis, {}).get("f1_macro", 0.0) for r in results]
        vals_present = [r["single_pick_metrics"].get(axis, {}).get("f1_macro_present", 0.0) for r in results]
        vals_acc = [r["single_pick_metrics"].get(axis, {}).get("accuracy", 0.0) for r in results]
        lines.append(
            f"| {axis} | {np.mean(vals_macro):.4f} | {np.std(vals_macro):.4f} | "
            f"{np.mean(vals_present):.4f} | {np.mean(vals_acc):.4f} |"
        )

    lines += [
        "",
        "## Per-Axis (Multi-Label) — F1-Micro & F1-Macro",
        "",
        "| Axis | F1-Micro Mean | F1-Micro Std | F1-Macro Mean |",
        "|---|---:|---:|---:|",
    ]
    for axis in ml_axes:
        vals_micro = [r["multi_label_metrics"].get(axis, {}).get("f1_micro", 0.0) for r in results]
        vals_macro = [r["multi_label_metrics"].get(axis, {}).get("f1_macro", 0.0) for r in results]
        lines.append(
            f"| {axis} | {np.mean(vals_micro):.4f} | {np.std(vals_micro):.4f} | "
            f"{np.mean(vals_macro):.4f} |"
        )

    lines += ["", "## Per-Checkpoint Breakdown", "",
              "| Checkpoint | early_stop | mean_macro | mean_micro |",
              "|---|---:|---:|---:|"]
    for r in results:
        name = Path(r["ckpt_path"]).parent.name
        lines.append(
            f"| {name} | {r['early_stop_metric']:.4f} | "
            f"{r['mean_f1_macro_singlepick']:.4f} | {r['mean_f1_micro_multilabel']:.4f} |"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    ckpt_group = p.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--ckpt", type=Path, nargs="+",
                            help="One or more checkpoint paths to evaluate.")
    ckpt_group.add_argument("--ckpt-pattern", type=str,
                            help="Pattern with {} placeholder for fold index. "
                                 "e.g. 'checkpoints/E2_phase1_25ep_fold{}/best.pt'")
    ckpt_group.add_argument("--experiment-dir", type=Path,
                            help="Single experiment folder (non-folded). "
                                 "e.g. checkpoints/E2_phase1_25ep")

    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Fold indices to use with --ckpt-pattern (default: 0 1 2 3 4)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results" / "test_eval")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_console_logger("mcis.test_eval")

    # Resolve checkpoint list
    if args.ckpt:
        ckpt_paths = [Path(p) for p in args.ckpt]
    elif args.ckpt_pattern:
        ckpt_paths = [Path(args.ckpt_pattern.replace("{}", str(f))) for f in args.folds]
    else:  # --experiment-dir
        ckpt_paths = [args.experiment_dir / "best.pt"]

    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        for m in missing:
            log.error("Checkpoint not found: %s", m)
        return 1

    # Load config + vocab
    cfg = BenchmarkConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    vocab = LabelVocab.load_json(cfg.data.label_vocab)
    cfg.validate_against_vocab(vocab.cardinalities())
    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Load data — extract HELD-OUT TEST SPLIT ONLY
    df = load_parquet(cfg.data.parquet)
    _, test_df = split_trainable_test(df, cfg.data.trainable_value, cfg.data.test_value)
    log.info("Held-out test split: %d records (split == '%s')",
             len(test_df), cfg.data.test_value)

    if len(test_df) == 0:
        log.error("Test split is empty! Check cfg.data.test_value='%s'", cfg.data.test_value)
        return 1

    # Evaluate each checkpoint
    results = []
    for ckpt_path in ckpt_paths:
        result = eval_one_checkpoint(ckpt_path, cfg, vocab, test_df, device, log)
        results.append(result)

    # Aggregate
    es_mean = float(np.mean([r["early_stop_metric"] for r in results]))
    macro_mean = float(np.mean([r["mean_f1_macro_singlepick"] for r in results]))
    micro_mean = float(np.mean([r["mean_f1_micro_multilabel"] for r in results]))
    log.info("=== FINAL TEST RESULTS (%d checkpoints) ===", len(results))
    log.info("  early_stop : %.4f", es_mean)
    log.info("  mean_macro : %.4f", macro_mean)
    log.info("  mean_micro : %.4f", micro_mean)

    # Write outputs
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    exp_name = cfg.experiment_name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_json = args.out_dir / f"{exp_name}_test_{ts}.json"
    out_md = args.out_dir / f"{exp_name}_test_{ts}.md"

    payload = {
        "experiment_name": exp_name,
        "generated_at": ts,
        "n_checkpoints": len(results),
        "n_test": int(len(test_df)),
        "device": str(device),
        "aggregate": {
            "early_stop_metric": es_mean,
            "early_stop_std": float(np.std([r["early_stop_metric"] for r in results])),
            "mean_f1_macro_singlepick": macro_mean,
            "mean_f1_macro_std": float(np.std([r["mean_f1_macro_singlepick"] for r in results])),
            "mean_f1_micro_multilabel": micro_mean,
            "mean_f1_micro_std": float(np.std([r["mean_f1_micro_multilabel"] for r in results])),
        },
        "per_checkpoint": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    md = render_markdown(results, args.config, len(test_df))
    out_md.write_text(md)

    log.info("Wrote JSON: %s", out_json)
    log.info("Wrote MD:   %s", out_md)
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
