"""External validation — inference on TCGA-BRCA using a saved checkpoint.

Evaluates ICD-O-3 axes ONLY. TCGA ICD-11 is silver-standard and never scored.
Now upgraded to include all comprehensive metrics (Ranking, ECE, Subgroups, etc.)

Usage:
    python scripts/run_ext_validation.py \
        --checkpoint checkpoints/MCIS_Best_seed456_fold1/best.pt \
        --config configs/MCIS_Best.yaml \
        --output results/ext_val_MCIS_Best.md \
        --device cuda \
        --diagnosis-first
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.label_vocab import LabelVocab
from src.data.loaders import BaheyaM1Dataset, collate_m1, load_parquet
from src.eval.metrics import NULL_TARGET_SENTINEL
from src.models.oce import OncologyCodingEngine
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from src.utils.seed import set_seed

# Import metric calculators from the full evaluation script
from scripts.run_full_evaluation import (
    compute_singlepick_metrics,
    compute_ranking_metrics,
    compute_rare_common_f1,
    compute_ece,
    compute_cohen_kappa,
    compute_subgroup_f1,
)

log = setup_console_logger("mcis.ext_val")

# ICD-O-3 axes only — never evaluate TCGA ICD-11 (silver labels)
ICDO3_AXES = [
    "icdo3_topography",
    "icdo3_morphology",
    "icdo3_behavior",
    # "icdo3_grade", # Excluded — 0 labels in TCGA-BRCA
    "icdo3_laterality",
    # All ICD-11 axes excluded — silver labels per v6 §11.4
]


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
        "attention": {
            "attn_dim": cfg.model.attention_hidden_dim,
        },
    }


@torch.no_grad()
def run_inference(model, loader, device, temperature=1.0):
    """Collect logits, targets, and metadata for all records."""
    model.eval()
    all_logits: dict[str, list] = defaultdict(list)
    all_targets: dict[str, list] = defaultdict(list)
    all_meta: dict[str, list] = defaultdict(list)

    for batch in loader:
        outputs = model(batch["texts"])
        for axis, logits in outputs.logits_per_axis.items():
            if axis not in ICDO3_AXES:
                continue
            if temperature != 1.0:
                logits = logits / temperature
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


def format_report(per_axis, ranking, rare_common, subgroup_rows,
                  ece_dict, kappa, args, n_records) -> str:
    lines = ["# External Validation Report (TCGA)", "",
             f"**Checkpoint:** `{args.checkpoint}`",
             f"**Config:** `{args.config}`",
             f"**Data:** `{args.data}`",
             f"**Preprocessing:** normalize_sections={args.normalize_sections}, diagnosis_only={args.diagnosis_only}, diagnosis_first={args.diagnosis_first}",
             f"**Records evaluated:** {n_records}", 
             f"**Axes evaluated:** ICD-O-3 only (TCGA ICD-11 excluded — silver labels)", ""]

    # Summary table
    lines += ["## 1. Per-Axis Metrics", "",
              "| Axis | F1-Macro | F1-Micro | Prec-Macro | Rec-Macro | F1-Present | Acc | N |",
              "|---|---|---|---|---|---|---|---|"]
    f1s = []
    for axis in ICDO3_AXES:
        if axis not in per_axis:
            lines.append(f"| {axis} | — | — | — | — | — | — | — |")
            continue
        m = per_axis[axis]
        f1s.append(m["f1_macro"])
        lines.append(f"| {axis} | {m['f1_macro']:.4f} | {m['f1_micro']:.4f} | "
                     f"{m['precision_macro']:.4f} | {m['recall_macro']:.4f} | "
                     f"{m['f1_macro_present']:.4f} | {m['accuracy']:.4f} | {m['n_valid']} |")
    
    if f1s:
        lines.append(f"| **Mean** | **{np.mean(f1s):.4f}** | | | | | | |")
    lines.append("")

    # Ranking table
    lines += ["## 2. Ranking Metrics", "",
              "| Axis | P@1 | Top-3 Acc | P@5 | R@5 | AUC-ROC |",
              "|---|---|---|---|---|---|"]
    for axis in ICDO3_AXES:
        if axis not in ranking:
            continue
        m = ranking[axis]
        lines.append(f"| {axis} | {m['p_at_1']:.4f} | {m['top3_acc']:.4f} | "
                     f"{m['p_at_5']:.4f} | {m['r_at_5']:.4f} | {m['auc_roc_macro']:.4f} |")
    lines.append("")

    # Rare vs Common
    lines += ["## 3. Rare vs Common Code F1", "",
              "| Axis | Rare F1 | Common F1 | #Rare | #Common |",
              "|---|---|---|---|---|"]
    for axis in ICDO3_AXES:
        if axis not in rare_common:
            continue
        m = rare_common[axis]
        lines.append(f"| {axis} | {m['rare_f1']:.4f} | {m['common_f1']:.4f} | "
                     f"{m['n_rare']} | {m['n_common']} |")
    lines.append("")

    # Calibration
    lines += ["## 4. Calibration (ECE)", "",
              "| Axis | ECE |", "|---|---|"]
    for axis in ICDO3_AXES:
        if axis not in ece_dict:
            continue
        ece = ece_dict[axis]
        lines.append(f"| {axis} | {ece:.4f} |")
    lines.append("")

    # Cohen's kappa
    if kappa is not None:
        lines += [f"## 5. Grade Ordinal — Cohen's κ (quadratic)", "",
                  f"**κ = {kappa:.4f}**", ""]

    # Subgroup
    if subgroup_rows:
        lines += ["## 6. Subgroup-Stratified F1-Macro", "",
                  "| Subgroup | Axis | F1-Macro | N |",
                  "|---|---|---|---|"]
        for r in subgroup_rows:
            if r['axis'] in ICDO3_AXES:
                lines.append(f"| {r['subgroup']} | {r['axis']} | {r['f1_macro']:.4f} | {r['n']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_ext_validation",
        description="External validation on TCGA-BRCA with full metrics.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best.pt checkpoint.")
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config used to train the checkpoint.")
    parser.add_argument("--data", type=Path,
                        default=Path("data/frozen/tcga_brca_external_model_ready/tcga_brca_external_model_ready.parquet"),
                        help="TCGA-BRCA parquet path.")
    parser.add_argument("--vocab", type=Path,
                        default=Path("data/frozen/m1_model_ready/label_vocab.json"),
                        help="Baheya label_vocab.json path.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/ext_validation_full.md"),
                        help="Output markdown report path.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalize-sections", action="store_true",
                        help="Re-tag TCGA text to match Baheya section tags.")
    parser.add_argument("--diagnosis-only", action="store_true",
                        help="Extract and evaluate only the diagnosis section.")
    parser.add_argument("--diagnosis-first", action="store_true",
                        help="Reorder segments so diagnosis section appears first.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling factor for logits.")
    args = parser.parse_args()

    # Validate inputs
    for p, name in [(args.checkpoint, "checkpoint"), (args.config, "config"),
                    (args.data, "data"), (args.vocab, "vocab")]:
        if not p.exists():
            log.error("%s not found: %s", name, p)
            return 1

    log.info("Loading config: %s", args.config)
    cfg = BenchmarkConfig.from_yaml(args.config)
    set_seed(cfg.seed)

    log.info("Loading vocab: %s", args.vocab)
    vocab = LabelVocab.load_json(args.vocab)

    log.info("Loading TCGA-BRCA data: %s", args.data)
    df = load_parquet(args.data)
    n_records = len(df)
    log.info("Records: %d", n_records)

    if args.normalize_sections or args.diagnosis_only:
        from src.data.section_normalizer import normalize_sections, extract_diagnosis
        
        def process_text(text: str) -> str:
            if not isinstance(text, str):
                return ""
            if args.normalize_sections:
                text = normalize_sections(text)
            if args.diagnosis_only:
                text = extract_diagnosis(text, fallback_to_full=True)
            return text
            
        df["text_section_tagged"] = df["text_section_tagged"].apply(process_text)
        log.info("Applied text preprocessing: normalize_sections=%s, diagnosis_only=%s",
                 args.normalize_sections, args.diagnosis_only)

    if args.diagnosis_first:
        from transformers import AutoTokenizer
        from src.data.section_normalizer import reorder_segments_diagnosis_first
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_hf_id)
        seg_size = cfg.model.segment_size
        df["text_section_tagged"] = df["text_section_tagged"].apply(
            lambda t: reorder_segments_diagnosis_first(t, tokenizer, seg_size) if isinstance(t, str) else t
        )
        log.info("Applied diagnosis-first segment reordering (segment_size=%d)", seg_size)

    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # Build model from config
    model_cfg = _build_model_cfg(cfg, vocab)
    model = OncologyCodingEngine.from_config(model_cfg)

    # Load checkpoint
    log.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    active_axes = set(ckpt.get("active_axes", list(cfg.axis_types.single_pick) + list(cfg.axis_types.multilabel)))
    model.set_active_axes(active_axes)
    model = model.to(device)
    log.info("Checkpoint loaded. Active axes: %s", sorted(active_axes))

    # Build dataset — use same BaheyaM1Dataset (schema-compatible)
    ds = BaheyaM1Dataset(df, vocab, cfg)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_m1,
        num_workers=0,
    )

    log.info("Running inference on %d records (temperature=%.2f)...", n_records, args.temperature)
    logits_dict, targets_dict, meta = run_inference(model, loader, device, temperature=args.temperature)

    # Compute all metrics for ICD-O-3 axes
    per_axis = {}
    ranking = {}
    rare_common = {}
    ece_dict = {}
    rare_threshold = cfg.eval.rare_code_threshold

    for axis in ICDO3_AXES:
        if axis not in logits_dict or axis not in targets_dict:
            continue
        lg = logits_dict[axis]
        tg = targets_dict[axis]
        K = vocab[axis].num_classes

        # All ICD-O-3 axes are single-pick in this architecture
        per_axis[axis] = compute_singlepick_metrics(lg, tg, K)
        ranking[axis] = compute_ranking_metrics(lg, tg, K)
        ece_dict[axis] = compute_ece(lg, tg)
        
        rare_codes = vocab[axis].rare_codes(rare_threshold)
        c2i = vocab[axis].code_to_idx
        rare_idx = {c2i[c] for c in rare_codes if c in c2i}
        rare_common[axis] = compute_rare_common_f1(lg, tg, K, rare_idx)

    # Grade kappa
    kappa = None
    if "icdo3_grade" in logits_dict and "icdo3_grade" in targets_dict:
        kappa = compute_cohen_kappa(logits_dict["icdo3_grade"], targets_dict["icdo3_grade"])

    # Subgroup
    subgroup_rows = compute_subgroup_f1(logits_dict, targets_dict, meta, cfg, vocab)

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = format_report(per_axis, ranking, rare_common, subgroup_rows,
                           ece_dict, kappa, args, n_records)
    args.output.write_text(report)
    log.info("Report written: %s", args.output)

    # Print summary
    icdo3_present = [a for a in ICDO3_AXES if a in per_axis]
    f1_scores = [per_axis[a]["f1_macro"] for a in icdo3_present]
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    log.info("Mean ICD-O-3 F1-Macro: %.4f", mean_f1)
    for axis in ICDO3_AXES:
        if axis in per_axis:
            log.info("  %s: %.4f", axis, per_axis[axis]["f1_macro"])

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
