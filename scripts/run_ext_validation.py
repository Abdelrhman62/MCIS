"""External validation — inference on TCGA-BRCA using a saved checkpoint.

Evaluates ICD-O-3 axes ONLY. TCGA ICD-11 is silver-standard and never scored.

Usage:
    python -m src.run_ext_validation \
        --checkpoint checkpoints/E2_baheya_from_tcga/best.pt \
        --data data/frozen/tcga_brca_external_model_ready/tcga_brca_external_model_ready.parquet \
        --vocab data/frozen/m1_model_ready/label_vocab.json \
        --config configs/E2_baheya_from_tcga.yaml \
        --output results/ext_val_E2.md \
        --device cuda

    # Ext-1 (E1 checkpoint):
    python -m src.run_ext_validation \
        --checkpoint checkpoints/E1/best.pt \
        --config configs/E1.yaml \
        --output results/ext_val_E1.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.label_vocab import LabelVocab
from src.data.loaders import BaheyaM1Dataset, collate_m1, load_parquet
from src.eval.metrics import aggregate_axis_metrics, f1_macro_singlepick, f1_multilabel
from src.models.oce import OncologyCodingEngine
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from src.utils.seed import set_seed

# ICD-O-3 axes only — never evaluate TCGA ICD-11 (silver labels)
ICDO3_AXES = [
    "icdo3_topography",
    "icdo3_morphology",
    "icdo3_behavior",
    "icdo3_grade",
    "icdo3_laterality",
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
def run_inference(
    model: OncologyCodingEngine,
    loader: DataLoader,
    device: torch.device,
    active_axes: set[str],
    cfg: BenchmarkConfig,
    vocab: LabelVocab,
    temperature: float = 1.0,
) -> dict[str, dict[str, float | int]]:
    """Run inference and compute per-axis metrics. ICD-O-3 only."""
    model.eval()

    all_logits: dict[str, list[torch.Tensor]] = {}
    all_targets: dict[str, list[torch.Tensor]] = {}

    for batch in loader:
        outputs = model(batch["texts"])
        for axis, logits in outputs.logits_per_axis.items():
            if axis not in ICDO3_AXES:
                continue
            if temperature != 1.0:
                logits = logits / temperature
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
        if axis in cfg.axis_types.single_pick:
            num_classes = vocab[axis].num_classes
            per_axis[axis] = f1_macro_singlepick(
                logits_cat, targets_cat, num_classes=num_classes
            )
        else:
            per_axis[axis] = f1_multilabel(logits_cat, targets_cat)

    return per_axis


def format_report(
    per_axis: dict[str, dict[str, float | int]],
    checkpoint_path: str,
    data_path: str,
    n_records: int,
) -> str:
    """Format results as a markdown report."""
    icdo3_present = [a for a in ICDO3_AXES if a in per_axis]
    f1_scores = [per_axis[a]["f1_macro"] for a in icdo3_present]
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    lines = [
        "# External Validation Report",
        "",
        f"**Checkpoint:** `{checkpoint_path}`",
        f"**Data:** `{data_path}`",
        f"**Records:** {n_records}",
        f"**Axes evaluated:** ICD-O-3 only (TCGA ICD-11 excluded — silver labels)",
        "",
        "## Results",
        "",
        f"**Mean ICD-O-3 F1-Macro: {mean_f1:.4f}**",
        "",
        "| Axis | F1-Macro | F1-Macro (present) | N valid | N classes |",
        "|---|---|---|---|---|",
    ]
    for axis in ICDO3_AXES:
        if axis not in per_axis:
            lines.append(f"| {axis} | — | — | — | — |")
            continue
        m = per_axis[axis]
        lines.append(
            f"| {axis} | {m['f1_macro']:.4f} | {m.get('f1_macro_present', 0.0):.4f} "
            f"| {m.get('n_valid', m.get('n_records', 0))} | {m.get('n_classes_present', '—')} |"
        )

    lines += [
        "",
        "## Notes",
        "- ICD-O-3 F1-Macro computed across all K classes (zero-fill for unseen classes).",
        "- Records with NULL_TARGET_SENTINEL (missing label) excluded per axis.",
        "- TCGA ICD-11 codes are silver (mapped from ICD-O-3, not natively coded) — never scored.",
    ]
    return "\n".join(lines)


def main() -> int:
    log = setup_console_logger("mcis.ext_val")

    parser = argparse.ArgumentParser(
        prog="python -m src.run_ext_validation",
        description="External validation on TCGA-BRCA.",
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
                        default=Path("results/ext_validation.md"),
                        help="Output markdown report path.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--normalize-sections", action="store_true",
                        help="Re-tag TCGA text to match Baheya section tags.")
    parser.add_argument("--diagnosis-only", action="store_true",
                        help="Extract and evaluate only the diagnosis section.")
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
    per_axis = run_inference(model, loader, device, active_axes, cfg, vocab, temperature=args.temperature)

    # Compute summary
    icdo3_present = [a for a in ICDO3_AXES if a in per_axis]
    f1_scores = [per_axis[a]["f1_macro"] for a in icdo3_present]
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    log.info("Mean ICD-O-3 F1-Macro: %.4f", mean_f1)
    for axis in ICDO3_AXES:
        if axis in per_axis:
            log.info("  %s: %.4f", axis, per_axis[axis]["f1_macro"])

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = format_report(per_axis, str(args.checkpoint), str(args.data), n_records)
    args.output.write_text(report)
    log.info("Report written: %s", args.output)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
