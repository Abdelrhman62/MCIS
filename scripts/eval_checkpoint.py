#!/usr/bin/env python3
"""Diagnostic: eval E1 ckpt on fold 0 val, compare vs B1 per-axis.

Loads checkpoints/<exp>/best.pt and re-runs eval on the same fold-0 val
slice the trainer used. For multi-label axes, computes F1 at three
thresholds: (a) production default 0.5, (b) B1's locked global threshold
(matched comparison), (c) E1's own fold-0-best threshold (optimistic).

Outputs JSON + markdown to results/diagnostics/.

Usage:
    python scripts/eval_checkpoint.py
    python scripts/eval_checkpoint.py --ckpt checkpoints/B0/best.pt --config configs/B0.yaml
    python scripts/eval_checkpoint.py --fold 0 --b1-json results/baselines/tfidf_XXX.json
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
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.label_vocab import LabelVocab  # noqa: E402
from src.data.loaders import (  # noqa: E402
    BaheyaM1Dataset,
    collate_m1,
    load_parquet,
    split_by_fold,
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
from torch.utils.data import DataLoader  # noqa: E402

THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


def _find_latest_b1_json() -> Path | None:
    pat = REPO_ROOT / "results" / "baselines"
    if not pat.exists():
        return None
    candidates = sorted(pat.glob("tfidf_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _build_model_cfg(cfg: BenchmarkConfig, vocab: LabelVocab) -> dict[str, Any]:
    """Mirror Trainer._build_model_cfg exactly."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path,
                        default=REPO_ROOT / "checkpoints/B0/best.pt")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/B0.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--b1-json", type=Path, default=None,
                        help="B1 results JSON for comparison (default: auto-discover latest)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log = setup_console_logger("mcis.diagnostics")

    # ---- Load artifacts ----
    if not args.ckpt.exists():
        log.error("Checkpoint not found: %s", args.ckpt)
        return 1
    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        return 1

    log.info("Loading config: %s", args.config)
    cfg = BenchmarkConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    log.info("Loading vocab: %s", cfg.data.label_vocab)
    vocab = LabelVocab.load_json(cfg.data.label_vocab)
    cfg.validate_against_vocab(vocab.cardinalities())

    log.info("Loading checkpoint: %s", args.ckpt)
    device = _resolve_device(args.device)
    log.info("Device: %s", device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # ---- Integrity checks ----
    log.info("=== Integrity checks ===")
    current_data_sha = _file_sha(cfg.data.parquet)
    current_vocab_sha = _file_sha(cfg.data.label_vocab)
    ckpt_data_sha = ckpt.get("data_sha")
    ckpt_vocab_sha = ckpt.get("vocab_sha")

    integrity_ok = True
    if ckpt_data_sha and current_data_sha and ckpt_data_sha != current_data_sha:
        log.error("DATA SHA MISMATCH: ckpt=%s current=%s", ckpt_data_sha[:12], current_data_sha[:12])
        log.error("  Parquet was modified after training. Eval will use NEW data on OLD model.")
        integrity_ok = False
    else:
        log.info("  data_sha match: %s", (current_data_sha or "n/a")[:12])

    if ckpt_vocab_sha and current_vocab_sha and ckpt_vocab_sha != current_vocab_sha:
        log.error("VOCAB SHA MISMATCH: ckpt=%s current=%s", ckpt_vocab_sha[:12], current_vocab_sha[:12])
        integrity_ok = False
    else:
        log.info("  vocab_sha match: %s", (current_vocab_sha or "n/a")[:12])

    log.info("  ckpt epoch: %d  best_epoch: %d  best_metric: %.4f",
             ckpt.get("epoch", -1), ckpt.get("best_epoch", -1),
             ckpt.get("best_metric", 0.0))
    log.info("  ckpt active_axes (%d): %s", len(ckpt.get("active_axes", [])),
             ckpt.get("active_axes", []))

    # Cross-check key cfg fields
    ckpt_cfg = ckpt.get("cfg_dict", {})
    drift_problems: list[str] = []
    for path_tuple in [
        ("data", "parquet"), ("data", "label_vocab"), ("data", "text_field"),
        ("model", "encoder_hf_id"), ("model", "max_segments"),
        ("model", "segment_size"),
    ]:
        a = ckpt_cfg
        b = cfg.to_dict()
        for k in path_tuple:
            a = a.get(k, {}) if isinstance(a, dict) else None
            b = b.get(k, {}) if isinstance(b, dict) else None
        if a != b:
            drift_problems.append(f"{'.'.join(path_tuple)}: ckpt={a!r} current={b!r}")
    if drift_problems:
        log.warning("CONFIG DRIFT vs ckpt:")
        for p in drift_problems:
            log.warning("  %s", p)
    else:
        log.info("  cfg drift: none on critical fields")

    # ---- Build model + load weights ----
    log.info("=== Build model + load weights ===")
    model_cfg = _build_model_cfg(cfg, vocab)
    model = OncologyCodingEngine.from_config(model_cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    log.info("  state_dict loaded. missing=%d unexpected=%d", len(missing), len(unexpected))
    if missing:
        log.warning("  missing keys (first 5): %s", missing[:5])
    if unexpected:
        log.warning("  unexpected keys (first 5): %s", unexpected[:5])
    model.set_active_axes(set(ckpt["active_axes"]))
    model.to(device)
    model.eval()

    # ---- Build fold-0 val loader (mirror Trainer.from_config) ----
    log.info("=== Build fold %d val loader ===", args.fold)
    df = load_parquet(cfg.data.parquet)
    trainable_df, _test_df = split_trainable_test(df, cfg.data.trainable_value, cfg.data.test_value)
    try:
        folds_df = pd.read_csv(cfg.data.cv_folds)
    except FileNotFoundError:
        log.error("cv_folds.csv not found: %s", cfg.data.cv_folds)
        return 1
    train_df, val_df = split_by_fold(trainable_df, folds_df, args.fold)
    log.info("  trainable=%d  fold=%d  train=%d  val=%d",
             len(trainable_df), args.fold, len(train_df), len(val_df))

    val_ds = BaheyaM1Dataset(val_df, vocab, cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.micro_batch_size,
        shuffle=False,
        collate_fn=collate_m1,
        num_workers=0,
    )

    # ---- Forward all val batches; collect logits + targets per axis ----
    log.info("=== Forward val ===")
    all_logits: dict[str, list[torch.Tensor]] = {}
    all_targets: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch["texts"])
            for axis, logits in outputs.logits_per_axis.items():
                all_logits.setdefault(axis, []).append(logits.detach().cpu())
                tkey = f"labels_{axis}"
                if tkey in batch:
                    all_targets.setdefault(axis, []).append(batch[tkey].detach().cpu())

    # Concat
    logits_cat: dict[str, torch.Tensor] = {a: torch.cat(v, dim=0) for a, v in all_logits.items()}
    targets_cat: dict[str, torch.Tensor] = {a: torch.cat(v, dim=0) for a, v in all_targets.items()}

    # ---- Single-pick metrics ----
    log.info("=== Single-pick metrics ===")
    sp_axes = list(cfg.axis_types.single_pick)
    sp_metrics: dict[str, dict[str, float | int]] = {}
    for axis in sp_axes:
        if axis not in logits_cat:
            continue
        K = vocab[axis].num_classes
        m = f1_macro_singlepick(logits_cat[axis], targets_cat[axis], num_classes=K)
        sp_metrics[axis] = m
        log.info("  %s: F1-Macro=%.4f acc=%.4f n_valid=%d",
                 axis, m["f1_macro"], m["accuracy"], m["n_valid"])

    # ---- Multi-label: three threshold strategies ----
    log.info("=== Multi-label metrics ===")
    ml_axes = list(cfg.axis_types.multilabel)

    # Load B1 JSON for matched-threshold + comparison
    b1_path = args.b1_json or _find_latest_b1_json()
    b1_data: dict[str, Any] = {}
    b1_thresholds: dict[str, float] = {}
    if b1_path and b1_path.exists():
        log.info("Loading B1 results: %s", b1_path)
        b1_data = json.loads(b1_path.read_text())
        b1_thresholds = b1_data.get("multi_label_thresholds", {})
    else:
        log.warning("No B1 results JSON found; matched-threshold column will be empty")

    ml_at_05: dict[str, dict[str, float | int]] = {}
    ml_at_b1: dict[str, dict[str, float | int]] = {}
    ml_at_best: dict[str, dict[str, float | int]] = {}
    e1_best_thresholds: dict[str, float] = {}
    e1_sweep: dict[str, dict[str, float]] = {}

    for axis in ml_axes:
        if axis not in logits_cat:
            continue
        L = logits_cat[axis]
        T = targets_cat[axis]

        # (a) production default t=0.5
        ml_at_05[axis] = f1_multilabel(L, T, threshold=0.5)
        # (b) at B1's locked threshold (matched comparison)
        if axis in b1_thresholds:
            t_b1 = float(b1_thresholds[axis])
            ml_at_b1[axis] = f1_multilabel(L, T, threshold=t_b1)
            ml_at_b1[axis]["matched_threshold"] = t_b1
        # (c) E1's own best on fold 0
        sweep: dict[str, float] = {}
        best_t, best_f1 = 0.5, -1.0
        for t in THRESHOLD_GRID:
            m = f1_multilabel(L, T, threshold=t)
            sweep[str(t)] = m["f1_micro"]
            if m["f1_micro"] > best_f1:
                best_f1 = m["f1_micro"]
                best_t = t
        m_best = f1_multilabel(L, T, threshold=best_t)
        m_best["fold0_best_threshold"] = best_t
        ml_at_best[axis] = m_best
        e1_best_thresholds[axis] = best_t
        e1_sweep[axis] = sweep

        log.info("  %s: @0.5 F1-Micro=%.4f  @B1(t=%s) F1-Micro=%.4f  @E1-best(t=%.1f) F1-Micro=%.4f",
                 axis, ml_at_05[axis]["f1_micro"],
                 f"{b1_thresholds.get(axis, 'n/a')}",
                 ml_at_b1.get(axis, {}).get("f1_micro", 0.0),
                 best_t, m_best["f1_micro"])

    # ---- Aggregate: two early-stop numbers ----
    log.info("=== Aggregate early-stop ===")
    per_axis_05 = dict(sp_metrics)
    per_axis_05.update(ml_at_05)
    agg_05 = aggregate_axis_metrics(per_axis_05, sp_axes, ml_axes)

    per_axis_best = dict(sp_metrics)
    per_axis_best.update(ml_at_best)
    agg_best = aggregate_axis_metrics(per_axis_best, sp_axes, ml_axes)

    log.info("  early_stop @ t=0.5      : %.4f  (mean_macro=%.4f mean_micro=%.4f)",
             agg_05.early_stop_metric, agg_05.mean_f1_macro_singlepick,
             agg_05.mean_f1_micro_multilabel)
    log.info("  early_stop @ fold0-best : %.4f  (mean_macro=%.4f mean_micro=%.4f)",
             agg_best.early_stop_metric, agg_best.mean_f1_macro_singlepick,
             agg_best.mean_f1_micro_multilabel)

    # ---- Build report ----
    run_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")

    # B1 per-axis F1 for comparison
    b1_agg = b1_data.get("aggregated", {}) if b1_data else {}
    b1_early_stop_mean = float(np.mean(b1_data["early_stop_per_fold"])) \
        if b1_data.get("early_stop_per_fold") else None
    b0_early_stop = 0.0702  # locked, see decisions doc

    report = {
        "run_at": run_at,
        "ckpt_path": str(args.ckpt),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_best_epoch": int(ckpt.get("best_epoch", -1)),
        "ckpt_best_metric_trainer_view": float(ckpt.get("best_metric", 0.0)),
        "fold": args.fold,
        "device": str(device),
        "integrity": {
            "data_sha_ok": integrity_ok,
            "ckpt_data_sha": ckpt_data_sha,
            "current_data_sha": current_data_sha,
            "ckpt_vocab_sha": ckpt_vocab_sha,
            "current_vocab_sha": current_vocab_sha,
            "cfg_drift": drift_problems,
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
        },
        "n_val": int(len(val_df)),
        "single_pick_axes": sp_axes,
        "multi_label_axes": ml_axes,
        "single_pick_metrics": sp_metrics,
        "multi_label_at_05": ml_at_05,
        "multi_label_at_b1_threshold": ml_at_b1,
        "multi_label_at_e1_best": ml_at_best,
        "e1_best_thresholds": e1_best_thresholds,
        "e1_threshold_sweep": e1_sweep,
        "b1_thresholds": b1_thresholds,
        "early_stop_at_05": float(agg_05.early_stop_metric),
        "mean_macro_at_05": float(agg_05.mean_f1_macro_singlepick),
        "mean_micro_at_05": float(agg_05.mean_f1_micro_multilabel),
        "early_stop_at_fold0_best": float(agg_best.early_stop_metric),
        "mean_macro_at_fold0_best": float(agg_best.mean_f1_macro_singlepick),
        "mean_micro_at_fold0_best": float(agg_best.mean_f1_micro_multilabel),
        "b0_floor": b0_early_stop,
        "b1_floor_mean": b1_early_stop_mean,
        "b1_per_axis_aggregated": b1_agg,
    }

    # ---- Render markdown ----
    md = _render_markdown(report)

    # ---- Write ----
    out_path = args.out or (
        REPO_ROOT / "results" / "diagnostics"
        / f"e1_fold{args.fold}_eval_{run_at.replace(':', '').replace('+0000', 'Z')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    md_path = out_path.with_suffix(".md")
    md_path.write_text(md)
    log.info("Wrote JSON: %s", out_path)
    log.info("Wrote MD:   %s", md_path)
    print()
    print(md)
    return 0


def _render_markdown(r: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# E1 fold-{r['fold']} Diagnostic Eval — vs B0 / B1")
    L.append("")
    L.append(f"- **Run at:** `{r['run_at']}`")
    L.append(f"- **Checkpoint:** `{r['ckpt_path']}`  (epoch {r['ckpt_epoch']}, "
             f"best_epoch={r['ckpt_best_epoch']}, trainer best_metric="
             f"{r['ckpt_best_metric_trainer_view']:.4f})")
    L.append(f"- **n_val:** {r['n_val']}    **device:** {r['device']}")
    L.append("")

    integ = r["integrity"]
    L.append("## Integrity")
    L.append("")
    L.append(f"- data_sha match: {'✅' if integ['data_sha_ok'] else '❌'}")
    L.append(f"- ckpt data_sha:    `{(integ['ckpt_data_sha'] or 'n/a')[:16]}`")
    L.append(f"- current data_sha: `{(integ['current_data_sha'] or 'n/a')[:16]}`")
    L.append(f"- missing state_dict keys: {integ['missing_keys_count']}")
    L.append(f"- unexpected state_dict keys: {integ['unexpected_keys_count']}")
    if integ["cfg_drift"]:
        L.append("- **cfg drift detected**:")
        for p in integ["cfg_drift"]:
            L.append(f"  - {p}")
    else:
        L.append("- cfg drift: none on critical fields")
    L.append("")

    L.append("## Summary (early-stop metric)")
    L.append("")
    L.append("| Baseline | early_stop | Δ vs E1@0.5 | Δ vs E1@best |")
    L.append("|---|---:|---:|---:|")
    e05 = r["early_stop_at_05"]
    ebest = r["early_stop_at_fold0_best"]
    L.append(f"| B0 majority | {r['b0_floor']:.4f} | "
             f"{e05 - r['b0_floor']:+.4f} | {ebest - r['b0_floor']:+.4f} |")
    if r["b1_floor_mean"] is not None:
        L.append(f"| B1 TF-IDF (mean 5-fold) | {r['b1_floor_mean']:.4f} | "
                 f"{e05 - r['b1_floor_mean']:+.4f} | "
                 f"{ebest - r['b1_floor_mean']:+.4f} |")
    L.append(f"| **E1 fold-0 @ t=0.5** | **{e05:.4f}** | — | — |")
    L.append(f"| **E1 fold-0 @ fold0-best t** | **{ebest:.4f}** | — | — |")
    L.append("")
    L.append(f"- mean_macro @0.5     : {r['mean_macro_at_05']:.4f}")
    L.append(f"- mean_micro @0.5     : {r['mean_micro_at_05']:.4f}")
    L.append(f"- mean_macro @best    : {r['mean_macro_at_fold0_best']:.4f}")
    L.append(f"- mean_micro @best    : {r['mean_micro_at_fold0_best']:.4f}")
    L.append("")

    # ---- Single-pick comparison ----
    L.append("## Single-pick axes: E1 vs B1")
    L.append("")
    L.append("| axis | B1 F1-Macro (mean) | E1 F1-Macro | Δ |")
    L.append("|---|---:|---:|---:|")
    b1_agg = r["b1_per_axis_aggregated"]
    for axis in r["single_pick_axes"]:
        em = r["single_pick_metrics"].get(axis, {})
        e_f1 = float(em.get("f1_macro", 0.0))
        b1_f1 = float(b1_agg.get(axis, {}).get("f1_macro_mean", 0.0)) if b1_agg else None
        if b1_f1 is not None:
            delta = e_f1 - b1_f1
            mark = "🟢" if delta >= 0 else "🔴"
            L.append(f"| {axis} | {b1_f1:.4f} | {e_f1:.4f} | {delta:+.4f} {mark} |")
        else:
            L.append(f"| {axis} | n/a | {e_f1:.4f} | — |")
    L.append("")

    # ---- Multi-label comparison ----
    L.append("## Multi-label axes: E1 at three thresholds vs B1")
    L.append("")
    L.append("| axis | B1 t* | B1 F1-Micro | E1@0.5 | E1@B1's t* | E1@fold0-best (t=?) | Δ E1-best vs B1 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for axis in r["multi_label_axes"]:
        b1_t = r["b1_thresholds"].get(axis)
        b1_f1 = float(b1_agg.get(axis, {}).get("f1_micro_mean", 0.0)) if b1_agg else None
        e05 = float(r["multi_label_at_05"].get(axis, {}).get("f1_micro", 0.0))
        e_b1 = float(r["multi_label_at_b1_threshold"].get(axis, {}).get("f1_micro", 0.0)) \
            if axis in r["multi_label_at_b1_threshold"] else None
        e_best_t = r["e1_best_thresholds"].get(axis, 0.5)
        e_best = float(r["multi_label_at_e1_best"].get(axis, {}).get("f1_micro", 0.0))
        delta = (e_best - b1_f1) if b1_f1 is not None else None
        mark = "" if delta is None else ("🟢" if delta >= 0 else "🔴")
        b1_t_str = f"{b1_t}" if b1_t is not None else "n/a"
        b1_f1_str = f"{b1_f1:.4f}" if b1_f1 is not None else "n/a"
        e_b1_str = f"{e_b1:.4f}" if e_b1 is not None else "n/a"
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        L.append(
            f"| {axis} | {b1_t_str} | {b1_f1_str} | {e05:.4f} | "
            f"{e_b1_str} | {e_best:.4f} (t={e_best_t}) | "
            f"{delta_str} {mark} |"
        )
    L.append("")

    # ---- E1 threshold sweep ----
    L.append("### E1 fold-0 threshold sweep (F1-Micro)")
    L.append("")
    grid = THRESHOLD_GRID
    hdr = "| axis | " + " | ".join(f"t={t}" for t in grid) + " |"
    sep = "|---|" + "|".join(["---:"] * len(grid)) + "|"
    L.append(hdr)
    L.append(sep)
    for axis in r["multi_label_axes"]:
        scores = r["e1_threshold_sweep"].get(axis, {})
        cells = [f"{scores.get(str(t), 0.0):.4f}" for t in grid]
        L.append(f"| {axis} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- Interpretation hints ----
    L.append("---")
    L.append("")
    L.append("## Verdict")
    L.append("")
    if r["b1_floor_mean"] is not None:
        b1f = r["b1_floor_mean"]
        if ebest >= b1f + 0.10:
            L.append(f"✅ E1 clears B1 by ≥0.10 absolute at tuned threshold ({ebest:.4f} vs {b1f:.4f}). Proceed.")
        elif ebest >= b1f:
            L.append(f"⚠️  E1 beats B1 at tuned threshold but by <0.10 ({ebest:.4f} vs {b1f:.4f}). PubMedBERT lift small; investigate single-pick axes or train longer.")
        else:
            L.append(f"❌ E1 below B1 even at tuned threshold ({ebest:.4f} vs {b1f:.4f}). Do not proceed to E2. Investigate.")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
