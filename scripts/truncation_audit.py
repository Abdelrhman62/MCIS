#!/usr/bin/env python3
"""Token length + segment count audit for Baheya M1 and TCGA pretrain.

Architecture v6 §13.2 Phase 0a deliverable. Required before E2 (TCGA pretrain)
to validate the 1536-token (12-segment) ceiling and quantify mid-doc info loss.

What this script measures, per dataset × text field:
    - WordPiece token length distribution (mean / median / p75 / p90 / p95 /
      p99 / max), using the SAME tokenizer the encoder uses (PubMedBERT /
      BiomedBERT base).
    - Truncation rate at three thresholds: 512 (Baheya policy / diagnostic
      soft cap), 1024, 1536 (TCGA hard ceiling = max_segments × segment_size).
    - Segment count distribution at segment_size=128 (the encoder's actual
      tokenization path uses return_overflowing_tokens with stride=0, which
      is what we replicate here — NOT a naive math.ceil(len / 128)).
    - For TCGA: per-cancer-type breakdown so we can see whether overflow
      concentrates in a few cancer types.
    - Records that would trigger middle-drop (>max_segments segments).

Output:
    1. JSON report at --out (default results/truncation_audit/<timestamp>.json)
    2. Markdown summary printed to stdout
    3. Optional --strict mode: exits 2 if any dataset has >5% records exceeding
       the hard ceiling (signal to revisit max_segments).

Why use the encoder's tokenizer call pattern (not a fast approximation):
    - The encoder uses HF `return_overflowing_tokens=True` with stride=0 and
      max_length=segment_size. Each segment includes [CLS]+tokens+[SEP], so
      effective content per segment is 126 tokens, not 128. A naive
      len(tokenize(text)) / 128 underestimates the true segment count by a
      few percent.
    - For honest numbers, mirror the encoder exactly.

Usage:
    # Default: audit both Baheya M1 and TCGA pretrain, both text fields,
    # write to results/truncation_audit/<utc-iso>.json
    python scripts/truncation_audit.py

    # Single dataset
    python scripts/truncation_audit.py --datasets baheya_m1

    # Strict mode (exit 2 if >5% overflow on any dataset/field)
    python scripts/truncation_audit.py --strict

    # Custom paths
    python scripts/truncation_audit.py \\
        --baheya-parquet data/frozen/m1_model_ready/m1_model_ready.parquet \\
        --tcga-parquet data/frozen/tcga_pretrain_model_ready/tcga_pretrain_model_ready.parquet \\
        --out results/truncation_audit/manual.json

Exit codes:
    0  success
    1  setup error (missing parquet, missing column, etc.)
    2  strict-mode warning: at least one dataset has >5% overflow at the
       hard ceiling
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Mirror the encoder defaults so the audit reflects what training will see.
from src.models.encoder import DEFAULT_BACKBONE  # noqa: E402

# Defaults aligned with configs/B0.yaml model.* block.
SEGMENT_SIZE = 128
MAX_SEGMENTS = 12  # 12 * 128 = 1536-token hard ceiling
LENGTH_THRESHOLDS = [512, 1024, 1536]  # report % > each
PERCENTILES = [50, 75, 90, 95, 99]
STRICT_OVERFLOW_THRESHOLD_PCT = 5.0  # % records over 1536 that triggers --strict

# Datasets known to this script. Add new ones here when M2/M3 audits arrive.
KNOWN_DATASETS: dict[str, dict[str, Any]] = {
    "baheya_m1": {
        "default_parquet": "data/frozen/m1_model_ready/m1_model_ready.parquet",
        "text_fields": ["text_section_tagged", "text_concatenated"],
        "stratify_column": None,  # no cancer_type breakdown for Baheya
    },
    "tcga_pretrain": {
        "default_parquet": "data/frozen/tcga_pretrain_model_ready/tcga_pretrain_model_ready.parquet",
        "text_fields": ["text_section_tagged", "text_concatenated"],
        "stratify_column": "cancer_type",  # break out per-cancer overflow rates
    },
}


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("truncation_audit")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Tokenization (mirror encoder's segment construction exactly)
# ---------------------------------------------------------------------------


def count_tokens(tokenizer, text: str) -> int:
    """Total WordPiece token count for one document, NO special tokens.

    This is the 'raw' length used for percentile stats. It is independent of
    segment_size — it answers "how long is this document in WordPiece units".
    """
    if not text or (isinstance(text, float) and np.isnan(text)):
        return 0
    return len(tokenizer(str(text), add_special_tokens=False)["input_ids"])


def count_segments_encoder_exact(tokenizer, text: str, segment_size: int) -> int:
    """Replicate OncologyEncoder._segment_one_text exactly.

    Uses return_overflowing_tokens with stride=0 and max_length=segment_size.
    Each segment carries [CLS]+content+[SEP], so effective content is
    (segment_size - 2) tokens per segment. The HF tokenizer handles this
    accounting internally; we just count the segments it produces.

    Does NOT apply max_segments cap — we want the unbounded segment count so
    overflow rate can be measured.
    """
    if not text or (isinstance(text, float) and np.isnan(text)):
        return 0
    encoded = tokenizer(
        str(text),
        max_length=segment_size,
        truncation=True,
        return_overflowing_tokens=True,
        stride=0,
        padding=False,  # padding inflates segment count zero-rows-wise; skip
        return_tensors=None,
    )
    # input_ids: list of segments, each a list of ints
    return len(encoded["input_ids"])


# ---------------------------------------------------------------------------
# Per-dataset audit
# ---------------------------------------------------------------------------


def audit_text_field(
    df: pd.DataFrame,
    text_field: str,
    tokenizer,
    segment_size: int,
    max_segments: int,
    stratify_column: str | None,
    log: logging.Logger,
) -> dict[str, Any]:
    """Audit one (dataset, text_field) pair. Returns a summary dict."""

    if text_field not in df.columns:
        return {
            "error": f"column {text_field!r} not in DataFrame",
            "n_rows_total": len(df),
        }

    # Drop empty/null text rows — they're not training input.
    texts = df[text_field].dropna().astype(str)
    texts = texts[texts.str.strip() != ""]
    n_rows = len(texts)

    if n_rows == 0:
        return {"error": "no non-empty rows", "n_rows_total": len(df)}

    log.info("  tokenizing %d rows (this is the slow step) ...", n_rows)

    token_lengths = np.zeros(n_rows, dtype=np.int32)
    segment_counts = np.zeros(n_rows, dtype=np.int32)

    # Single pass: tokenize once for length, segment once for count.
    # We avoid double-tokenizing by using a single call with overflow + reading
    # both length and segment-count from the result.
    for i, text in enumerate(texts.tolist()):
        encoded = tokenizer(
            text,
            max_length=segment_size,
            truncation=True,
            return_overflowing_tokens=True,
            stride=0,
            padding=False,
            return_tensors=None,
        )
        segs = encoded["input_ids"]
        segment_counts[i] = len(segs)
        # Raw length = sum of segment lengths minus duplicated specials.
        # Easier and exact: tokenize once more without specials.
        token_lengths[i] = len(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        if (i + 1) % 1000 == 0:
            log.info("    %d / %d", i + 1, n_rows)

    # Stats
    result: dict[str, Any] = {
        "n_rows": int(n_rows),
        "token_length_stats": {
            "mean": float(token_lengths.mean()),
            "min": int(token_lengths.min()),
            "max": int(token_lengths.max()),
            **{
                f"p{p}": int(np.percentile(token_lengths, p))
                for p in PERCENTILES
            },
        },
        "segment_count_stats": {
            "mean": float(segment_counts.mean()),
            "min": int(segment_counts.min()),
            "max": int(segment_counts.max()),
            **{
                f"p{p}": int(np.percentile(segment_counts, p))
                for p in PERCENTILES
            },
        },
        "truncation_rates_pct": {
            str(thr): float((token_lengths > thr).sum() / n_rows * 100.0)
            for thr in LENGTH_THRESHOLDS
        },
        "segment_count_dist": dict(Counter(segment_counts.tolist())),
        "overflow_at_max_segments": {
            "max_segments": int(max_segments),
            "n_overflow_records": int((segment_counts > max_segments).sum()),
            "pct_overflow_records": float(
                (segment_counts > max_segments).sum() / n_rows * 100.0
            ),
            "max_overflow_segments_dropped": int(
                max(0, int(segment_counts.max()) - max_segments)
            ),
        },
    }

    # For overflow records, estimate the mid-doc info loss fraction
    # (= dropped middle segments / total segments) for the worst cases.
    overflow_mask = segment_counts > max_segments
    if overflow_mask.any():
        overflow_segs = segment_counts[overflow_mask]
        info_loss_pct = (overflow_segs - max_segments) / overflow_segs * 100.0
        result["overflow_at_max_segments"]["mean_info_loss_pct"] = float(
            info_loss_pct.mean()
        )
        result["overflow_at_max_segments"]["max_info_loss_pct"] = float(
            info_loss_pct.max()
        )

    # Per-stratum breakdown (TCGA cancer_type)
    if stratify_column is not None and stratify_column in df.columns:
        strat_df = df.loc[texts.index, stratify_column]
        per_stratum: dict[str, dict[str, Any]] = {}
        for stratum_val in sorted(strat_df.dropna().unique().tolist()):
            mask = (strat_df == stratum_val).to_numpy()
            if not mask.any():
                continue
            s_token = token_lengths[mask]
            s_seg = segment_counts[mask]
            per_stratum[str(stratum_val)] = {
                "n_rows": int(mask.sum()),
                "token_length_p95": int(np.percentile(s_token, 95)),
                "segment_count_p95": int(np.percentile(s_seg, 95)),
                "pct_over_1536_tokens": float(
                    (s_token > 1536).sum() / mask.sum() * 100.0
                ),
                "pct_overflow_at_max_segments": float(
                    (s_seg > max_segments).sum() / mask.sum() * 100.0
                ),
            }
        result["per_stratum"] = per_stratum

    return result


def audit_dataset(
    name: str,
    parquet_path: Path,
    spec: dict[str, Any],
    tokenizer,
    segment_size: int,
    max_segments: int,
    log: logging.Logger,
) -> dict[str, Any]:
    if not parquet_path.exists():
        return {"error": f"parquet not found: {parquet_path}"}

    log.info("Reading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  loaded %d rows × %d cols", len(df), len(df.columns))

    out: dict[str, Any] = {
        "parquet": str(parquet_path),
        "n_rows_total": int(len(df)),
        "by_text_field": {},
    }
    for text_field in spec["text_fields"]:
        log.info("  auditing text_field=%s ...", text_field)
        out["by_text_field"][text_field] = audit_text_field(
            df=df,
            text_field=text_field,
            tokenizer=tokenizer,
            segment_size=segment_size,
            max_segments=max_segments,
            stratify_column=spec["stratify_column"],
            log=log,
        )
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    cfg = report["config"]
    lines.append("# MCIS Truncation Audit")
    lines.append("")
    lines.append(f"- **Run at:** `{report['run_at']}`")
    lines.append(f"- **Tokenizer:** `{cfg['tokenizer']}`")
    lines.append(
        f"- **segment_size:** {cfg['segment_size']}  |  "
        f"**max_segments:** {cfg['max_segments']}  "
        f"(hard ceiling = {cfg['segment_size'] * cfg['max_segments']} tokens)"
    )
    lines.append("")

    for ds_name, ds_data in report["datasets"].items():
        lines.append(f"## {ds_name}")
        if "error" in ds_data:
            lines.append(f"- ERROR: {ds_data['error']}")
            continue
        lines.append(f"- Parquet: `{ds_data['parquet']}`")
        lines.append(f"- Rows: {ds_data['n_rows_total']}")
        lines.append("")

        for tf_name, tf_data in ds_data["by_text_field"].items():
            lines.append(f"### {tf_name}")
            if "error" in tf_data:
                lines.append(f"- ERROR: {tf_data['error']}")
                continue
            tl = tf_data["token_length_stats"]
            sc = tf_data["segment_count_stats"]
            tr = tf_data["truncation_rates_pct"]
            ov = tf_data["overflow_at_max_segments"]

            lines.append("**Token length (WordPiece):**")
            lines.append(
                f"- mean={tl['mean']:.1f}  min={tl['min']}  "
                f"p50={tl['p50']}  p75={tl['p75']}  p90={tl['p90']}  "
                f"p95={tl['p95']}  p99={tl['p99']}  max={tl['max']}"
            )
            lines.append("")
            lines.append("**Truncation rates (% rows over threshold):**")
            for thr, pct in tr.items():
                lines.append(f"- > {thr} tokens: {pct:.2f}%")
            lines.append("")
            lines.append(
                f"**Segment count @ segment_size={cfg['segment_size']}:**"
            )
            lines.append(
                f"- mean={sc['mean']:.2f}  min={sc['min']}  "
                f"p50={sc['p50']}  p75={sc['p75']}  p90={sc['p90']}  "
                f"p95={sc['p95']}  p99={sc['p99']}  max={sc['max']}"
            )
            lines.append("")
            lines.append(
                f"**Overflow @ max_segments={ov['max_segments']}:**"
            )
            lines.append(
                f"- {ov['n_overflow_records']} records "
                f"({ov['pct_overflow_records']:.2f}%) exceed cap; "
                f"max dropped segments = {ov['max_overflow_segments_dropped']}"
            )
            if "mean_info_loss_pct" in ov:
                lines.append(
                    f"- on overflow rows: mean info-loss "
                    f"{ov['mean_info_loss_pct']:.1f}%, "
                    f"max {ov['max_info_loss_pct']:.1f}%"
                )
            lines.append("")

            if "per_stratum" in tf_data:
                lines.append("**Per-stratum overflow (top 10 by overflow %):**")
                strata = sorted(
                    tf_data["per_stratum"].items(),
                    key=lambda kv: -kv[1]["pct_overflow_at_max_segments"],
                )
                lines.append("")
                lines.append(
                    "| stratum | n | tok p95 | seg p95 | % >1536 tok | % overflow |"
                )
                lines.append(
                    "|---|---:|---:|---:|---:|---:|"
                )
                for s_val, s_data in strata[:10]:
                    lines.append(
                        f"| {s_val} | {s_data['n_rows']} | "
                        f"{s_data['token_length_p95']} | "
                        f"{s_data['segment_count_p95']} | "
                        f"{s_data['pct_over_1536_tokens']:.1f}% | "
                        f"{s_data['pct_overflow_at_max_segments']:.1f}% |"
                    )
                lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="*",
        default=list(KNOWN_DATASETS.keys()),
        choices=list(KNOWN_DATASETS.keys()),
        help="Which datasets to audit",
    )
    parser.add_argument(
        "--baheya-parquet", type=Path, default=None,
        help="Override Baheya M1 parquet path",
    )
    parser.add_argument(
        "--tcga-parquet", type=Path, default=None,
        help="Override TCGA pretrain parquet path",
    )
    parser.add_argument(
        "--segment-size", type=int, default=SEGMENT_SIZE,
        help=f"Tokens per segment (default {SEGMENT_SIZE}, matches B0.yaml)",
    )
    parser.add_argument(
        "--max-segments", type=int, default=MAX_SEGMENTS,
        help=f"Segment cap (default {MAX_SEGMENTS}, matches B0.yaml)",
    )
    parser.add_argument(
        "--tokenizer", default=DEFAULT_BACKBONE,
        help="HuggingFace tokenizer id (default = encoder backbone)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON path (default results/truncation_audit/<utc>.json)",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Suppress markdown summary to stdout",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=f"Exit 2 if any dataset has > {STRICT_OVERFLOW_THRESHOLD_PCT}% "
             f"records over the hard ceiling",
    )

    args = parser.parse_args()
    log = _setup_logger()

    log.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    run_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "run_at": run_at,
        "config": {
            "tokenizer": args.tokenizer,
            "segment_size": int(args.segment_size),
            "max_segments": int(args.max_segments),
            "length_thresholds": list(LENGTH_THRESHOLDS),
            "percentiles": list(PERCENTILES),
        },
        "datasets": {},
    }

    path_override = {
        "baheya_m1": args.baheya_parquet,
        "tcga_pretrain": args.tcga_parquet,
    }

    for ds_name in args.datasets:
        spec = KNOWN_DATASETS[ds_name]
        parquet = (
            path_override[ds_name]
            if path_override[ds_name] is not None
            else REPO_ROOT / spec["default_parquet"]
        )
        log.info("=== Auditing %s (%s) ===", ds_name, parquet)
        report["datasets"][ds_name] = audit_dataset(
            name=ds_name,
            parquet_path=parquet,
            spec=spec,
            tokenizer=tokenizer,
            segment_size=args.segment_size,
            max_segments=args.max_segments,
            log=log,
        )

    # Write JSON
    out_path = args.out or (
        REPO_ROOT / "results" / "truncation_audit"
        / f"{run_at.replace(':', '').replace('+0000', 'Z')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote JSON report to %s", out_path)

    # Print markdown
    if not args.no_markdown:
        print()
        print(render_markdown(report))

    # Strict-mode check
    if args.strict:
        violators = []
        for ds_name, ds_data in report["datasets"].items():
            if "by_text_field" not in ds_data:
                continue
            for tf_name, tf_data in ds_data["by_text_field"].items():
                if "overflow_at_max_segments" not in tf_data:
                    continue
                pct = tf_data["overflow_at_max_segments"]["pct_overflow_records"]
                if pct > STRICT_OVERFLOW_THRESHOLD_PCT:
                    violators.append(f"{ds_name}/{tf_name}: {pct:.2f}% overflow")
        if violators:
            log.warning(
                "STRICT mode: %d (dataset, text_field) pairs exceed %.1f%% overflow",
                len(violators), STRICT_OVERFLOW_THRESHOLD_PCT,
            )
            for v in violators:
                log.warning("  - %s", v)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
