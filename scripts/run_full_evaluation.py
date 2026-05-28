"""Full evaluation script for MCIS thesis — all metrics in one pass.

Usage:
    python scripts/run_full_evaluation.py \
        --checkpoint checkpoints/E7_hierarchical_fold0/best.pt \
        --config configs/MCIS_Best.yaml \
        --vocab data/frozen/m1_model_ready/label_vocab.json \
        --data data/frozen/m1_model_ready/m1_model_ready.parquet \
        --split val_fold --fold 0 \
        --output results/full_eval_E7_fold0.md --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.label_vocab import LabelVocab
from src.data.loaders import BaheyaM1Dataset, collate_m1, load_parquet
from src.eval.metrics import NULL_TARGET_SENTINEL
from src.models.oce import OncologyCodingEngine
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from src.utils.seed import set_seed

log = setup_console_logger("mcis.full_eval")

# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_device(preferred: str) -> torch.device:
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


def _build_model_cfg(cfg: BenchmarkConfig, vocab: LabelVocab) -> dict[str, Any]:
    sp = {a: vocab[a].num_classes for a in cfg.axis_types.single_pick if a in vocab.axes}
    ml = {a: vocab[a].num_classes for a in cfg.axis_types.multilabel if a in vocab.axes}
    return {
        "encoder": {
            "backbone_name": cfg.model.encoder_hf_id,
            "segment_size": cfg.model.segment_size,
            "max_segments": cfg.model.max_segments,
            "strict_length": cfg.model.strict_length,
        },
        "heads": {"single_pick": sp, "multi_label": ml},
        "attention": {"attn_dim": cfg.model.attention_hidden_dim},
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ── per-class F1 helper ─────────────────────────────────────────────────────

def _per_class_prf(preds: np.ndarray, y_true: np.ndarray, K: int):
    """Returns (precision, recall, f1) arrays of length K."""
    prec = np.zeros(K); rec = np.zeros(K); f1 = np.zeros(K)
    for k in range(K):
        tp = int(((preds == k) & (y_true == k)).sum())
        fp = int(((preds == k) & (y_true != k)).sum())
        fn = int(((preds != k) & (y_true == k)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        prec[k] = p; rec[k] = r
        f1[k] = 2 * p * r / (p + r) if (p + r) else 0.0
    return prec, rec, f1


# ── metric functions ─────────────────────────────────────────────────────────

def compute_singlepick_metrics(logits: np.ndarray, targets: np.ndarray, K: int) -> dict:
    valid = targets != NULL_TARGET_SENTINEL
    if valid.sum() == 0:
        return {"f1_macro": 0, "f1_micro": 0, "precision_macro": 0,
                "recall_macro": 0, "f1_macro_present": 0, "accuracy": 0, "n_valid": 0}
    y = targets[valid]; lg = logits[valid]
    preds = lg.argmax(axis=-1)
    prec, rec, f1 = _per_class_prf(preds, y, K)
    present = np.unique(y)
    # micro f1 for single-pick = accuracy
    acc = float((preds == y).mean())
    # micro P/R/F1 via TP pooling
    tp_sum = sum(int(((preds == k) & (y == k)).sum()) for k in range(K))
    fp_sum = sum(int(((preds == k) & (y != k)).sum()) for k in range(K))
    fn_sum = sum(int(((preds != k) & (y == k)).sum()) for k in range(K))
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    return {
        "f1_macro": float(f1.mean()),
        "f1_micro": float(micro_f1),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "f1_macro_present": float(f1[present].mean()) if len(present) else 0.0,
        "accuracy": acc,
        "n_valid": int(valid.sum()),
    }


def compute_multilabel_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    K = logits.shape[1]
    probs = _sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int32)
    yt = (targets > 0.5).astype(np.int32)
    # micro
    tp_m = int(((preds == 1) & (yt == 1)).sum())
    fp_m = int(((preds == 1) & (yt == 0)).sum())
    fn_m = int(((preds == 0) & (yt == 1)).sum())
    mp = tp_m / (tp_m + fp_m) if (tp_m + fp_m) else 0
    mr = tp_m / (tp_m + fn_m) if (tp_m + fn_m) else 0
    micro_f1 = 2 * mp * mr / (mp + mr) if (mp + mr) else 0
    # macro per-class
    pf1 = np.zeros(K); pp = np.zeros(K); pr = np.zeros(K)
    for k in range(K):
        tp = int(((preds[:, k] == 1) & (yt[:, k] == 1)).sum())
        fp = int(((preds[:, k] == 1) & (yt[:, k] == 0)).sum())
        fn = int(((preds[:, k] == 0) & (yt[:, k] == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        pp[k] = p; pr[k] = r
        pf1[k] = 2 * p * r / (p + r) if (p + r) else 0
    return {
        "f1_macro": float(pf1.mean()), "f1_micro": float(micro_f1),
        "precision_macro": float(pp.mean()), "recall_macro": float(pr.mean()),
        "f1_macro_present": float(pf1[pf1 > 0].mean()) if (pf1 > 0).any() else 0.0,
        "accuracy": float(((preds == yt).all(axis=1)).mean()),
        "n_valid": int(logits.shape[0]),
        "n_records": int(logits.shape[0]),
    }


def compute_ranking_metrics(logits: np.ndarray, targets: np.ndarray, K: int) -> dict:
    """P@1, P@5, R@5, Top-3 acc, AUC-ROC for single-pick axes."""
    valid = targets != NULL_TARGET_SENTINEL
    if valid.sum() == 0:
        return {"p_at_1": 0, "p_at_5": 0, "r_at_5": 0, "top3_acc": 0, "auc_roc_macro": 0}
    y = targets[valid]; lg = logits[valid]
    probs = _softmax(lg)
    ranked = np.argsort(-lg, axis=-1)
    n = len(y)
    p1 = float((ranked[:, 0] == y).mean())
    top3 = float(np.any(ranked[:, :3] == y[:, None], axis=1).mean())
    k5 = min(5, K)
    top5_hit = np.any(ranked[:, :k5] == y[:, None], axis=1)
    p5 = float(top5_hit.mean())
    r5 = p5  # for single-pick, recall@5 = precision@5 (one true label)
    # AUC-ROC macro (one-vs-rest)
    auc = float('nan')
    try:
        from sklearn.metrics import roc_auc_score
        classes_present = np.unique(y)
        if len(classes_present) > 1:
            # one-hot encode
            y_onehot = np.zeros((n, K))
            y_onehot[np.arange(n), y] = 1
            y_present = y_onehot[:, classes_present]
            probs_present = probs[:, classes_present]
            # Renormalize subset probabilities
            probs_sum = probs_present.sum(axis=-1, keepdims=True)
            probs_present = np.divide(probs_present, probs_sum, where=probs_sum!=0)
            auc = float(roc_auc_score(y_present, probs_present, average="macro", multi_class="ovr"))
    except Exception:
        auc = 0.0
    return {"p_at_1": p1, "p_at_5": p5, "r_at_5": r5, "top3_acc": top3, "auc_roc_macro": auc}


def compute_ece(logits: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for single-pick axis."""
    valid = targets != NULL_TARGET_SENTINEL
    if valid.sum() == 0:
        return 0.0
    y = targets[valid]; probs = _softmax(logits[valid])
    confs = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == y).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_cohen_kappa(logits: np.ndarray, targets: np.ndarray) -> float:
    """Quadratic weighted Cohen's kappa for ordinal grade axis."""
    valid = targets != NULL_TARGET_SENTINEL
    if valid.sum() < 2:
        return 0.0
    y = targets[valid]; preds = logits[valid].argmax(axis=-1)
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y, preds, weights="quadratic"))
    except Exception:
        return 0.0


def compute_rare_common_f1(logits: np.ndarray, targets: np.ndarray, K: int,
                            rare_indices: set[int]) -> dict:
    """Compute F1-Macro separately for rare vs common codes."""
    valid = targets != NULL_TARGET_SENTINEL
    if valid.sum() == 0:
        return {"rare_f1": 0, "common_f1": 0, "n_rare": 0, "n_common": 0}
    y = targets[valid]; preds = logits[valid].argmax(axis=-1)
    _, _, f1 = _per_class_prf(preds, y, K)
    rare_idx = sorted(rare_indices & set(range(K)))
    common_idx = sorted(set(range(K)) - rare_indices)
    rare_f1 = float(f1[rare_idx].mean()) if rare_idx else 0.0
    common_f1 = float(f1[common_idx].mean()) if common_idx else 0.0
    return {"rare_f1": rare_f1, "common_f1": common_f1,
            "n_rare": len(rare_idx), "n_common": len(common_idx)}


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, cfg, vocab):
    """Collect logits, targets, and metadata for all records."""
    model.eval()
    all_logits: dict[str, list] = defaultdict(list)
    all_targets: dict[str, list] = defaultdict(list)
    all_meta: dict[str, list] = defaultdict(list)

    for batch in loader:
        outputs = model(batch["texts"])
        for axis, logits in outputs.logits_per_axis.items():
            all_logits[axis].append(logits.detach().cpu().numpy())
            tkey = f"labels_{axis}"
            if tkey in batch:
                t = batch[tkey]
                all_targets[axis].append(t.numpy() if isinstance(t, torch.Tensor) else t)
        # metadata
        for key in ["batch", "template_flag", "is_cancer_primary"]:
            if key in batch:
                v = batch[key]
                all_meta[key].append(v.numpy() if isinstance(v, torch.Tensor) else np.array(v))

    out_logits = {a: np.concatenate(v) for a, v in all_logits.items()}
    out_targets = {a: np.concatenate(v) for a, v in all_targets.items()}
    out_meta = {k: np.concatenate(v) for k, v in all_meta.items()}
    return out_logits, out_targets, out_meta


# ── subgroup metrics ─────────────────────────────────────────────────────────

def compute_subgroup_f1(logits_dict, targets_dict, meta, cfg, vocab):
    """Compute F1-Macro per axis under various subgroup splits."""
    rows = []
    n = len(next(iter(meta.values()))) if meta else 0

    # Batch 1 vs 2
    if "batch" in meta:
        for bval in [1, 2]:
            mask = meta["batch"] == bval
            if mask.sum() == 0:
                continue
            for axis in cfg.all_axes:
                if axis not in logits_dict:
                    continue
                lg = logits_dict[axis][mask]
                tg = targets_dict[axis][mask]
                K = vocab[axis].num_classes
                if axis in cfg.axis_types.single_pick:
                    m = compute_singlepick_metrics(lg, tg, K)
                else:
                    m = compute_multilabel_metrics(lg, tg)
                rows.append({"subgroup": f"batch_{bval}", "axis": axis,
                             "f1_macro": m["f1_macro"], "f1_present": m.get("f1_macro_present", m.get("f1_micro", 0.0)), "n": m["n_valid"]})

    # cancer primary vs non
    if "is_cancer_primary" in meta:
        for label, name in [(True, "cancer_primary"), (False, "non_cancer")]:
            mask = meta["is_cancer_primary"] == label
            if mask.sum() == 0:
                continue
            for axis in cfg.all_axes:
                if axis not in logits_dict:
                    continue
                lg = logits_dict[axis][mask]
                tg = targets_dict[axis][mask]
                K = vocab[axis].num_classes
                if axis in cfg.axis_types.single_pick:
                    m = compute_singlepick_metrics(lg, tg, K)
                else:
                    m = compute_multilabel_metrics(lg, tg)
                rows.append({"subgroup": name, "axis": axis,
                             "f1_macro": m["f1_macro"], "f1_present": m.get("f1_macro_present", m.get("f1_micro", 0.0)), "n": m["n_valid"]})

    # template vs non-template
    if "template_flag" in meta:
        for label, name in [(True, "template"), (False, "non_template")]:
            mask = meta["template_flag"] == label
            if mask.sum() == 0:
                continue
            for axis in cfg.all_axes:
                if axis not in logits_dict:
                    continue
                lg = logits_dict[axis][mask]
                tg = targets_dict[axis][mask]
                K = vocab[axis].num_classes
                if axis in cfg.axis_types.single_pick:
                    m = compute_singlepick_metrics(lg, tg, K)
                else:
                    m = compute_multilabel_metrics(lg, tg)
                rows.append({"subgroup": name, "axis": axis,
                             "f1_macro": m["f1_macro"], "f1_present": m.get("f1_macro_present", m.get("f1_micro", 0.0)), "n": m["n_valid"]})

    return rows


# ── report formatter ─────────────────────────────────────────────────────────

def format_report(per_axis, ranking, rare_common, subgroup_rows,
                  ece_dict, kappa, args, n_records, agg_metrics) -> str:
    lines = ["# Full Evaluation Report", "",
             f"**Checkpoint:** `{args.checkpoint}`",
             f"**Config:** `{args.config}`",
             f"**Data:** `{args.data}`",
             f"**Split:** {args.split} (fold {args.fold})",
             f"**Records evaluated:** {n_records}", ""]

    # Summary table
    lines += ["## 1. Per-Axis Metrics", "",
              "| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |",
              "|---|---|---|---|---|---|---|---|"]
    for axis, m in per_axis.items():
        lines.append(f"| {axis} | {m['f1_macro']:.4f} | {m['f1_micro']:.4f} | "
                     f"{m['precision_macro']:.4f} | {m['recall_macro']:.4f} | "
                     f"{m['f1_macro_present']:.4f} | {m['accuracy']:.4f} | {m['n_valid']} |")
    
    # Add official mean
    official_mean = agg_metrics.early_stop_metric
    lines.append(f"| **Official Mean** | **{official_mean:.4f}** | | | | | | |")
    lines.append("")

    # Ranking table
    lines += ["## 2. Ranking Metrics (Single-Pick Only)", "",
              "| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |",
              "|---|---|---|---|---|---|"]
    for axis, m in ranking.items():
        lines.append(f"| {axis} | {m['p_at_1']:.4f} | {m['top3_acc']:.4f} | "
                     f"{m['p_at_5']:.4f} | {m['r_at_5']:.4f} | {m['auc_roc_macro']:.4f} |")
    lines.append("")

    # Rare vs Common
    lines += ["## 3. Rare vs Common Code F1", "",
              "| Axis | Rare F1 | Common F1 | #Rare | #Common |",
              "|---|---|---|---|---|"]
    for axis, m in rare_common.items():
        lines.append(f"| {axis} | {m['rare_f1']:.4f} | {m['common_f1']:.4f} | "
                     f"{m['n_rare']} | {m['n_common']} |")
    lines.append("")

    # Calibration
    lines += ["## 4. Calibration (ECE)", "",
              "| Axis | ECE |", "|---|---|"]
    for axis, ece in ece_dict.items():
        lines.append(f"| {axis} | {ece:.4f} |")
    lines.append("")

    # Cohen's kappa
    if kappa is not None:
        lines += [f"## 5. Grade Ordinal — Cohen's κ (quadratic)", "",
                  f"**κ = {kappa:.4f}**", ""]

    # Subgroup
    # Subgroup F1
    lines += ["## 6. Subgroup-Stratified F1-Macro", "",
              "| Subgroup | Axis | F1-Macro | F1-Present | N |",
              "|---|---|---|---|---|"]
    for row in subgroup_rows:
        lines.append(f"| {row['subgroup']} | {row['axis']} | {row['f1_macro']:.4f} | {row['f1_present']:.4f} | {row['n']} |")
    lines.append("")

    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="MCIS full evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("data/frozen/m1_model_ready/label_vocab.json"))
    parser.add_argument("--data", type=Path, default=Path("data/frozen/m1_model_ready/m1_model_ready.parquet"))
    parser.add_argument("--split", default="test", help="test | val_fold")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/full_eval.md"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-split-filter", action="store_true", help="Evaluate all records in the parquet without split filtering")
    parser.add_argument("--diagnosis-first", action="store_true", help="Reorder segments to place [DIAGNOSIS] at the front")
    args = parser.parse_args()

    for p, n in [(args.checkpoint, "checkpoint"), (args.config, "config"),
                 (args.data, "data"), (args.vocab, "vocab")]:
        if not p.exists():
            log.error("%s not found: %s", n, p)
            return 1

    cfg = BenchmarkConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    vocab = LabelVocab.load_json(args.vocab)

    # Load and filter data
    df = load_parquet(args.data)
    if args.no_split_filter:
        log.info("Skipping split filtering (--no-split-filter), evaluating all %d records.", len(df))
    elif args.split == "test":
        df = df[df["split"] == cfg.data.test_value].copy()
        log.info("Test split: %d records", len(df))
    elif args.split == "val_fold":
        cv_folds = pd.read_csv(cfg.data.cv_folds)
        val_ids = set(cv_folds[cv_folds["fold"] == args.fold]["record_id"].astype(str))
        trainable = df[df["split"] == cfg.data.trainable_value]
        df = trainable[trainable["record_id"].astype(str).isin(val_ids)].copy()
        log.info("Val fold %d: %d records", args.fold, len(df))
    else:
        log.error("Unknown split: %s", args.split)
        return 1

    if args.diagnosis_first:
        from transformers import AutoTokenizer
        from src.data.section_normalizer import reorder_segments_diagnosis_first
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_hf_id)
        seg_size = cfg.model.segment_size
        df["text_section_tagged"] = df["text_section_tagged"].apply(
            lambda t: reorder_segments_diagnosis_first(t, tokenizer, seg_size) if isinstance(t, str) else t
        )
        log.info("Applied diagnosis-first segment reordering (segment_size=%d)", seg_size)

    n_records = len(df)
    if n_records == 0:
        log.error("No records after filtering")
        return 1

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Build & load model
    model_cfg = _build_model_cfg(cfg, vocab)
    model = OncologyCodingEngine.from_config(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    active = set(ckpt.get("active_axes", cfg.all_axes))
    model.set_active_axes(active)
    model = model.to(device)
    log.info("Model loaded. Active axes: %s", sorted(active))

    # Dataset & loader
    ds = BaheyaM1Dataset(df, vocab, cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_m1, num_workers=0)

    # Run inference
    log.info("Running inference on %d records...", n_records)
    logits_dict, targets_dict, meta = run_inference(model, loader, device, cfg, vocab)

    # Compute all metrics
    per_axis = {}
    ranking = {}
    rare_common = {}
    ece_dict = {}
    rare_threshold = cfg.eval.rare_code_threshold

    for axis in cfg.all_axes:
        if axis not in logits_dict:
            continue
        lg = logits_dict[axis]
        tg = targets_dict[axis]
        K = vocab[axis].num_classes

        if axis in cfg.axis_types.single_pick:
            per_axis[axis] = compute_singlepick_metrics(lg, tg, K)
            ranking[axis] = compute_ranking_metrics(lg, tg, K)
            ece_dict[axis] = compute_ece(lg, tg)
            # rare/common
            rare_codes = vocab[axis].rare_codes(rare_threshold)
            c2i = vocab[axis].code_to_idx
            rare_idx = {c2i[c] for c in rare_codes if c in c2i}
            rare_common[axis] = compute_rare_common_f1(lg, tg, K, rare_idx)
        else:
            per_axis[axis] = compute_multilabel_metrics(lg, tg)

    # Grade kappa
    kappa = None
    if "icdo3_grade" in logits_dict:
        kappa = compute_cohen_kappa(logits_dict["icdo3_grade"],
                                     targets_dict["icdo3_grade"])

    # Subgroup
    subgroup_rows = compute_subgroup_f1(logits_dict, targets_dict, meta, cfg, vocab)

    # Compute official aggregate metric (using F1-Micro for ML axes)
    from src.eval.metrics import aggregate_axis_metrics
    agg_metrics = aggregate_axis_metrics(per_axis, list(cfg.axis_types.single_pick), list(cfg.axis_types.multilabel))

    # Flat correct mean (Claude's formula: sum of correct metrics / 10)
    flat_scores = []
    for axis, m in per_axis.items():
        if axis in cfg.axis_types.single_pick:
            flat_scores.append(m["f1_macro"])
        else:
            flat_scores.append(m["f1_micro"])
    flat_mean = float(np.mean(flat_scores)) if flat_scores else 0.0

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = format_report(per_axis, ranking, rare_common, subgroup_rows,
                           ece_dict, kappa, args, n_records, agg_metrics)
    # Inject flat correct mean note into report
    report = report.replace(
        f"| **Official Mean** | **{agg_metrics.early_stop_metric:.4f}**",
        f"| **Official Mean (Agg)** | **{agg_metrics.early_stop_metric:.4f}** | | | | | | |\n| **Flat Correct Mean** | **{flat_mean:.4f}**"
    )

    args.output.write_text(report)
    log.info("Report written to %s", args.output)

    # Print summary
    log.info("Flat Correct Mean (10-axis average): %.4f", flat_mean)
    log.info("Official Aggregated Mean (Mean of SP and ML means): %.4f", agg_metrics.early_stop_metric)
    for axis, m in per_axis.items():
        metric_val = m["f1_macro"] if axis in cfg.axis_types.single_pick else m["f1_micro"]
        metric_name = "F1-Macro" if axis in cfg.axis_types.single_pick else "F1-Micro"
        log.info("  %s: %s=%.4f  Acc=%.4f", axis, metric_name, metric_val, m["accuracy"])
    if kappa is not None:
        log.info("  icdo3_grade Cohen's kappa: %.4f", kappa)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
