"""
Step 1 — Multi-metric evaluation of one E1 checkpoint.
======================================================
For one (fold, checkpoint) pair, compute per-axis:
  - F1-Macro (full K, the currently-reported metric)
  - F1-Macro-present-only (excludes 0-support classes)
  - F1-Micro
  - Accuracy
For multi-label axes:
  - F1-Micro
  - F1-Macro
  - Accuracy (exact-set match)

Emit JSON + Markdown to results/diagnostics/.

Usage:
  python scripts/step1_multimetric.py --fold 0
  python scripts/step1_multimetric.py --fold 2 --ckpt checkpoints/B0_fold2/best.pt

Defaults to fold 0 + checkpoints/B0/best.pt.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parents[1].name == "MCIS" else Path.cwd()
sys.path.insert(0, str(REPO))

from src.data.label_vocab import LabelVocab
from src.data.loaders import (
    BaheyaM1Dataset, collate_m1, load_parquet, split_by_fold, split_trainable_test,
)
from src.models.oce import OncologyCodingEngine
from src.utils.config import BenchmarkConfig

NULL = -1


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="step1_multimetric")
    p.add_argument("--config", type=Path, default=REPO / "configs" / "B0.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Checkpoint path (default: checkpoints/B0/best.pt for fold 0, "
                        "checkpoints/B0_fold{F}/best.pt otherwise)")
    p.add_argument("--out-dir", type=Path, default=REPO / "results" / "diagnostics")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def _resolve_device(pref: str) -> torch.device:
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
def metrics_single_pick(
    logits: np.ndarray, targets: np.ndarray, codes: list[str]
) -> dict[str, Any]:
    K = len(codes)
    valid = targets != NULL
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"f1_macro": 0.0, "f1_macro_present": 0.0, "f1_micro": 0.0,
                "accuracy": 0.0, "n_valid": 0, "per_class": []}

    y_true = targets[valid]
    preds = logits[valid].argmax(axis=-1)
    support = Counter(y_true.tolist())

    per_class = []
    f1_per_class = np.zeros(K, dtype=np.float64)
    tp_total = fp_total = fn_total = 0
    for k in range(K):
        tp = int(((preds == k) & (y_true == k)).sum())
        fp = int(((preds == k) & (y_true != k)).sum())
        fn = int(((preds != k) & (y_true == k)).sum())
        tp_total += tp; fp_total += fp; fn_total += fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_per_class[k] = f1
        per_class.append({"code": codes[k], "idx": k, "support": support.get(k, 0),
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": prec, "recall": rec, "f1": f1})

    f1_macro = float(f1_per_class.mean())
    classes_present = [k for k in range(K) if support.get(k, 0) > 0]
    f1_macro_present = float(f1_per_class[classes_present].mean()) if classes_present else 0.0
    prec_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    rec_micro  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro) if (prec_micro + rec_micro) else 0.0
    accuracy = float((preds == y_true).mean())

    return {
        "f1_macro": f1_macro,
        "f1_macro_present": f1_macro_present,
        "f1_micro": float(f1_micro),
        "accuracy": accuracy,
        "n_valid": n_valid,
        "n_classes_present": len(classes_present),
        "per_class": per_class,
    }


def metrics_multi_label(
    logits: np.ndarray, targets: np.ndarray, codes: list[str], threshold: float = 0.5
) -> dict[str, Any]:
    N, K = logits.shape
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    y_true = (targets > 0.5).astype(np.int32)

    tp_micro = int(((preds == 1) & (y_true == 1)).sum())
    fp_micro = int(((preds == 1) & (y_true == 0)).sum())
    fn_micro = int(((preds == 0) & (y_true == 1)).sum())
    prec_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) else 0.0
    rec_micro  = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) else 0.0
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro) if (prec_micro + rec_micro) else 0.0

    per_class = []
    f1_per_class = np.zeros(K, dtype=np.float64)
    for k in range(K):
        tp = int(((preds[:, k] == 1) & (y_true[:, k] == 1)).sum())
        fp = int(((preds[:, k] == 1) & (y_true[:, k] == 0)).sum())
        fn = int(((preds[:, k] == 0) & (y_true[:, k] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_per_class[k] = f1
        support = int(y_true[:, k].sum())
        per_class.append({"code": codes[k], "idx": k, "support": support,
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": prec, "recall": rec, "f1": f1})

    f1_macro = float(f1_per_class.mean())
    exact_match = float((preds == y_true).all(axis=-1).mean())
    n_with_any = int((y_true.sum(axis=-1) > 0).sum())

    return {
        "f1_micro": float(f1_micro),
        "f1_macro": f1_macro,
        "accuracy_exact_set": exact_match,
        "n_records": int(N),
        "n_with_any_positive": n_with_any,
        "threshold": threshold,
        "per_class": per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if args.fold == 0:
            ckpt_path = REPO / "checkpoints" / "B0" / "best.pt"
        else:
            ckpt_path = REPO / "checkpoints" / f"B0_fold{args.fold}" / "best.pt"

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return 1

    print(f"Loading: cfg={args.config}, ckpt={ckpt_path}, fold={args.fold}")
    cfg = BenchmarkConfig.from_yaml(args.config)
    vocab = LabelVocab.load_json(cfg.data.label_vocab)
    df = load_parquet(cfg.data.parquet)
    trainable_df, _ = split_trainable_test(df, cfg.data.trainable_value, cfg.data.test_value)
    _, val_df = split_by_fold(trainable_df, None, args.fold)
    print(f"  fold-{args.fold} val rows: {len(val_df)}")

    device = _resolve_device(args.device)
    print(f"  device: {device}")

    val_ds = BaheyaM1Dataset(val_df, vocab, cfg)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.micro_batch_size, shuffle=False,
                            collate_fn=collate_m1, num_workers=0)

    single_pick_spec = {a: vocab[a].num_classes for a in cfg.axis_types.single_pick if a in vocab.axes}
    multi_label_spec = {a: vocab[a].num_classes for a in cfg.axis_types.multilabel if a in vocab.axes}
    model_cfg = {
        "encoder": {"backbone_name": cfg.model.encoder_hf_id,
                    "segment_size": cfg.model.segment_size,
                    "max_segments": cfg.model.max_segments,
                    "strict_length": cfg.model.strict_length},
        "heads": {"single_pick": single_pick_spec, "multi_label": multi_label_spec},
        "attention": {"attn_dim": cfg.model.attention_hidden_dim},
    }
    model = OncologyCodingEngine.from_config(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.set_active_axes(set(ckpt["active_axes"]))
    model.eval()
    print(f"  ckpt epoch={ckpt.get('epoch')} best_epoch={ckpt.get('best_epoch')} "
          f"best_metric={ckpt.get('best_metric'):.4f}")

    # Forward pass
    all_logits: dict[str, list[torch.Tensor]] = {}
    all_targets: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["texts"])
            for axis, logits in out.logits_per_axis.items():
                all_logits.setdefault(axis, []).append(logits.detach().cpu())
                tk = f"labels_{axis}"
                if tk in batch:
                    all_targets.setdefault(axis, []).append(batch[tk].detach().cpu())

    logits_by_axis = {a: torch.cat(v, dim=0).numpy() for a, v in all_logits.items()}
    targets_by_axis = {a: torch.cat(v, dim=0).numpy() for a, v in all_targets.items()}

    # Compute per-axis metrics
    per_axis: dict[str, dict[str, Any]] = {}
    for axis in cfg.axis_types.single_pick:
        if axis in logits_by_axis:
            per_axis[axis] = {"kind": "single_pick",
                              **metrics_single_pick(logits_by_axis[axis],
                                                    targets_by_axis[axis],
                                                    vocab[axis].codes)}
    for axis in cfg.axis_types.multilabel:
        if axis in logits_by_axis:
            per_axis[axis] = {"kind": "multi_label",
                              **metrics_multi_label(logits_by_axis[axis],
                                                    targets_by_axis[axis],
                                                    vocab[axis].codes)}

    # Aggregates
    sp_axes = [a for a in cfg.axis_types.single_pick if a in per_axis]
    ml_axes = [a for a in cfg.axis_types.multilabel if a in per_axis]
    summary = {
        "mean_f1_macro_singlepick": float(np.mean([per_axis[a]["f1_macro"] for a in sp_axes])) if sp_axes else 0.0,
        "mean_f1_macro_present_singlepick": float(np.mean([per_axis[a]["f1_macro_present"] for a in sp_axes])) if sp_axes else 0.0,
        "mean_f1_micro_singlepick": float(np.mean([per_axis[a]["f1_micro"] for a in sp_axes])) if sp_axes else 0.0,
        "mean_accuracy_singlepick": float(np.mean([per_axis[a]["accuracy"] for a in sp_axes])) if sp_axes else 0.0,
        "mean_f1_micro_multilabel": float(np.mean([per_axis[a]["f1_micro"] for a in ml_axes])) if ml_axes else 0.0,
        "mean_f1_macro_multilabel": float(np.mean([per_axis[a]["f1_macro"] for a in ml_axes])) if ml_axes else 0.0,
        "mean_accuracy_exact_set_multilabel": float(np.mean([per_axis[a]["accuracy_exact_set"] for a in ml_axes])) if ml_axes else 0.0,
    }
    summary["early_stop_metric_current"] = (
        summary["mean_f1_macro_singlepick"] + summary["mean_f1_micro_multilabel"]
    ) / 2.0
    summary["early_stop_metric_present_only"] = (
        summary["mean_f1_macro_present_singlepick"] + summary["mean_f1_micro_multilabel"]
    ) / 2.0
    summary["early_stop_metric_accuracy_based"] = (
        summary["mean_accuracy_singlepick"] + summary["mean_accuracy_exact_set_multilabel"]
    ) / 2.0

    # Output
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = args.out_dir / f"step1_multimetric_fold{args.fold}_{ts}"

    payload = {
        "fold": args.fold,
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch"),
        "best_metric_saved": ckpt.get("best_metric"),
        "summary": summary,
        "per_axis": per_axis,
        "saved_at": ts,
    }
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    # Markdown report
    lines: list[str] = []
    lines.append(f"# Step 1 — Multi-Metric Eval (fold {args.fold})")
    lines.append("")
    lines.append(f"**Checkpoint:** `{ckpt_path}`  ")
    lines.append(f"**Best epoch:** {ckpt.get('best_epoch')}  ")
    lines.append(f"**Best metric saved:** {ckpt.get('best_metric'):.4f}  ")
    lines.append(f"**Generated:** {ts}")
    lines.append("")
    lines.append("## Summary across axes")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in summary.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append("## Per-axis (single-pick)")
    lines.append("")
    lines.append("| Axis | K | n_valid | macro | macro_present | micro | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for a in sp_axes:
        m = per_axis[a]
        lines.append(f"| {a} | {len(vocab[a].codes)} | {m['n_valid']} | "
                     f"{m['f1_macro']:.4f} | {m['f1_macro_present']:.4f} | "
                     f"{m['f1_micro']:.4f} | {m['accuracy']:.4f} |")
    lines.append("")
    lines.append("## Per-axis (multi-label)")
    lines.append("")
    lines.append("| Axis | K | N | n_with_any | micro | macro | exact-set acc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for a in ml_axes:
        m = per_axis[a]
        lines.append(f"| {a} | {len(vocab[a].codes)} | {m['n_records']} | {m['n_with_any_positive']} | "
                     f"{m['f1_micro']:.4f} | {m['f1_macro']:.4f} | {m['accuracy_exact_set']:.4f} |")

    base.with_suffix(".md").write_text("\n".join(lines))

    # Console summary
    print()
    for k, v in summary.items():
        print(f"  {k:45s} {v:.4f}")
    print()
    print(f"  Wrote: {base.with_suffix('.json')}")
    print(f"  Wrote: {base.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
