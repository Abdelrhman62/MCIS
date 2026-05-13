#!/usr/bin/env python3
"""Phase 0.4 — Scan UNK token rate for both candidate encoders on Baheya M1.

Tokenizes every report under both candidate WordPiece tokenizers:
    - PubMedBERT  (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
    - PathologyBERT (tsantos/PathologyBERT)

Both are uncased WordPiece. Reports tokenization stats per record so the team
can decide whether the chosen tokenizer is safe before E1, and flags any
reports above 5% UNK for inspection. Tokenizer-comparison data also feeds
the E4 PathologyBERT smoke ablation.

Tokenization parameters match training-time encoder behaviour:
    - add_special_tokens=True (per OncologyEncoder spec; specials are kept
      per segment in PLM-ICD-style segment pooling)
    - truncation=False (we want true UNK behaviour, not post-truncation)

Two text columns are scanned per the data card §10:
    - text_section_tagged  (architecture §4.4 canonical input)
    - text_concatenated    (legacy format kept for E1-fmt A/B test)

Per-record stats include the PLM-ICD segment count under 128-token segments
(architecture §6.2), so this scan also doubles as a length-distribution
audit for free.

Subgroup breakdowns (per architecture §11.3):
    - batch (1 vs 2)         — catches stylistic drift between batches
    - template_flag           — catches templated vs free-text vocabulary diff

Inputs:
    data/frozen/m1_model_ready.parquet  (default; override with --input)

Outputs:
    results/unk_token_report.csv         — per-record stats
    results/unk_token_summary.md         — human-readable summary

Usage:
    python scripts/unk_token_scan.py
    python scripts/unk_token_scan.py --input /custom/path.parquet
    python scripts/unk_token_scan.py --tokenizers pubmedbert  # subset

Exit codes:
    0  no reports flagged above 5% UNK
    1  setup error (missing file, missing column, etc.)
    3  one or more reports flagged (review needed; not a hard failure)

Note: First run downloads ~600MB of tokenizer files. Cache lives in
~/.cache/huggingface so subsequent runs are instant.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Local import is best-effort — fall back to a basic logger if the project
# logger isn't on the import path (useful for ad-hoc runs from a fresh checkout).
try:
    from src.utils.logging import setup_console_logger  # type: ignore
except ImportError:
    def setup_console_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        return logger

DEFAULT_INPUT = REPO_ROOT / "data" / "frozen" / "m1_model_ready" / "m1_model_ready.parquet"
DEFAULT_REPORT = REPO_ROOT / "results" / "unk_token_report.csv"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "unk_token_summary.md"

# Both encoders are case-insensitive WordPiece (uncased).
CANDIDATE_TOKENIZERS = {
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "pathologybert": "tsantos/PathologyBERT",
}

# Text columns to scan per data card §10
TEXT_COLUMNS = ["text_section_tagged", "text_concatenated"]
PRIMARY_TEXT_COL = "text_section_tagged"   # architecture-canonical

# Acceptance thresholds
UNK_FLAG_THRESHOLD = 0.05    # flag any report with >5% UNK
SEGMENT_SIZE = 128            # PLM-ICD canonical (encoder.py spec §2.1)


def scan_one(
    tokenizer_name: str,
    hf_id: str,
    text_col: str,
    texts: list[str],
    log: logging.Logger,
) -> pd.DataFrame:
    """Tokenize one text column with one tokenizer and return per-record stats."""
    from transformers import AutoTokenizer

    log.info("Tokenizing %d records: tokenizer=%s text_col=%s",
             len(texts), tokenizer_name, text_col)
    tok = AutoTokenizer.from_pretrained(hf_id)
    unk_id = tok.unk_token_id
    if unk_id is None:
        raise RuntimeError(f"Tokenizer {tokenizer_name} has no unk_token_id")

    # Match training-time tokenization (per encoder.py spec): add_special_tokens=True,
    # no truncation so we measure true distribution. Special tokens (CLS/SEP) are
    # not UNK, so they don't pollute the rate.
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        out = tok(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        ids = out["input_ids"]
        n_tot = len(ids)
        n_unk = sum(1 for t in ids if t == unk_id) if n_tot else 0
        unk_rate = (n_unk / n_tot) if n_tot else 0.0
        # Segment count under PLM-ICD: ceil(n_tot / 128). Each segment is its own
        # forward pass through BERT in encoder.py.
        n_segments = -(-n_tot // SEGMENT_SIZE) if n_tot else 0
        rows.append({
            "row_idx": i,
            "n_tokens": n_tot,
            "n_unk": n_unk,
            "unk_rate": unk_rate,
            "n_segments_at_128": n_segments,
        })
    df = pd.DataFrame(rows).set_index("row_idx")
    return df


def report_summary_block(
    summary: list[str],
    tokenizer_name: str,
    hf_id: str,
    text_col: str,
    df: pd.DataFrame,
    record_ids: list[str],
    bookkeeping: pd.DataFrame,
    log: logging.Logger,
) -> int:
    """Append a per-(tokenizer, column) block to the summary. Return n_flagged."""
    n_records = len(df)
    mean_rate = df["unk_rate"].mean()
    median_rate = df["unk_rate"].median()
    p95_rate = df["unk_rate"].quantile(0.95)
    p99_rate = df["unk_rate"].quantile(0.99)
    max_rate = df["unk_rate"].max()
    n_zero = int((df["unk_rate"] == 0).sum())
    n_flagged = int((df["unk_rate"] > UNK_FLAG_THRESHOLD).sum())

    mean_tokens = df["n_tokens"].mean()
    p95_tokens = df["n_tokens"].quantile(0.95)
    p99_tokens = df["n_tokens"].quantile(0.99)
    max_tokens = int(df["n_tokens"].max())
    over_512 = int((df["n_tokens"] > 512).sum())
    over_1536 = int((df["n_tokens"] > SEGMENT_SIZE * 12).sum())

    log.info(
        "  %s / %s: mean=%.4f median=%.4f p95=%.4f max=%.4f flagged=%d "
        "tokens(mean=%.0f p99=%.0f max=%d) over_512=%d over_1536=%d",
        tokenizer_name, text_col,
        mean_rate, median_rate, p95_rate, max_rate, n_flagged,
        mean_tokens, p99_tokens, max_tokens, over_512, over_1536,
    )

    summary.append(f"### {tokenizer_name} — `{text_col}`")
    summary.append("")
    summary.append(f"_Backbone_: `{hf_id}`")
    summary.append("")
    summary.append("**UNK rate distribution**")
    summary.append("")
    summary.append("| stat | value |")
    summary.append("|---|---:|")
    summary.append(f"| mean | {mean_rate:.4%} |")
    summary.append(f"| median | {median_rate:.4%} |")
    summary.append(f"| p95 | {p95_rate:.4%} |")
    summary.append(f"| p99 | {p99_rate:.4%} |")
    summary.append(f"| max | {max_rate:.4%} |")
    summary.append(f"| reports with zero UNK | {n_zero:,} / {n_records:,} ({n_zero/n_records:.1%}) |")
    summary.append(f"| reports flagged (>{UNK_FLAG_THRESHOLD:.0%}) | **{n_flagged}** |")
    summary.append("")

    summary.append("**Token length distribution (informational; truncation is decided in encoder.py)**")
    summary.append("")
    summary.append("| stat | value |")
    summary.append("|---|---:|")
    summary.append(f"| mean tokens | {mean_tokens:.0f} |")
    summary.append(f"| p95 tokens | {p95_tokens:.0f} |")
    summary.append(f"| p99 tokens | {p99_tokens:.0f} |")
    summary.append(f"| max tokens | {max_tokens:,} |")
    summary.append(f"| reports >512 tokens | {over_512:,} ({over_512/n_records:.1%}) |")
    summary.append(f"| reports >1536 tokens (12 × 128 ceiling) | {over_1536:,} |")
    summary.append("")

    # Subgroup breakdown: batch + template_flag
    if "batch" in bookkeeping.columns or "template_flag" in bookkeeping.columns:
        merged = df.copy()
        merged["record_id"] = record_ids
        merged = merged.merge(bookkeeping, on="record_id", how="left")
        summary.append("**Subgroup means (batch / template_flag)**")
        summary.append("")
        summary.append("| subgroup | n | mean UNK rate | mean tokens | n flagged |")
        summary.append("|---|---:|---:|---:|---:|")

        if "batch" in merged.columns:
            for batch in sorted(merged["batch"].dropna().unique()):
                sub = merged[merged["batch"] == batch]
                if len(sub) == 0:
                    continue
                summary.append(
                    f"| batch={int(batch)} | {len(sub):,} | "
                    f"{sub['unk_rate'].mean():.4%} | "
                    f"{sub['n_tokens'].mean():.0f} | "
                    f"{int((sub['unk_rate'] > UNK_FLAG_THRESHOLD).sum())} |"
                )

        if "template_flag" in merged.columns:
            for tflag in [True, False]:
                sub = merged[merged["template_flag"] == tflag]
                if len(sub) == 0:
                    continue
                label = "templated" if tflag else "free-text"
                summary.append(
                    f"| template_flag={tflag} ({label}) | {len(sub):,} | "
                    f"{sub['unk_rate'].mean():.4%} | "
                    f"{sub['n_tokens'].mean():.0f} | "
                    f"{int((sub['unk_rate'] > UNK_FLAG_THRESHOLD).sum())} |"
                )
        summary.append("")

    if n_flagged:
        summary.append("**Top 10 highest UNK-rate reports**")
        summary.append("")
        merged = df.copy()
        merged["record_id"] = record_ids
        top = merged.sort_values("unk_rate", ascending=False).head(10)
        summary.append("| record_id | n_tokens | n_unk | unk_rate |")
        summary.append("|---|---:|---:|---:|")
        for _, row in top.iterrows():
            summary.append(
                f"| `{row['record_id']}` | {int(row['n_tokens'])} | "
                f"{int(row['n_unk'])} | {row['unk_rate']:.4%} |"
            )
        summary.append("")

    return n_flagged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"Parquet to scan (default: {DEFAULT_INPUT})")
    parser.add_argument("--report", default=str(DEFAULT_REPORT),
                        help=f"Per-record CSV output (default: {DEFAULT_REPORT})")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY),
                        help=f"Markdown summary output (default: {DEFAULT_SUMMARY})")
    parser.add_argument("--text-cols", nargs="*", default=TEXT_COLUMNS,
                        help=f"Text columns to scan (default: {TEXT_COLUMNS})")
    parser.add_argument("--tokenizers", nargs="*", default=list(CANDIDATE_TOKENIZERS),
                        help=f"Subset of {sorted(CANDIDATE_TOKENIZERS)}")
    parser.add_argument("--id-col", default="record_id",
                        help="Column to use as record id (default: record_id)")
    args = parser.parse_args()

    log = setup_console_logger("mcis")

    # ----- Load -----
    in_path = Path(args.input)
    if not in_path.exists():
        log.error("Input parquet does not exist: %s", in_path)
        return 1

    log.info("Reading %s", in_path)
    df = pd.read_parquet(in_path)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    if args.id_col not in df.columns:
        log.error("ID column %r not in parquet. Available: %s",
                  args.id_col, list(df.columns))
        return 1

    record_ids = df[args.id_col].astype(str).tolist()

    # Bookkeeping for subgroup analysis (optional — gracefully handle missing cols)
    bookkeeping_cols = [args.id_col]
    for c in ("batch", "template_flag"):
        if c in df.columns:
            bookkeeping_cols.append(c)
        else:
            log.warning("Subgroup column %r not in parquet; skipping that breakdown.", c)
    bookkeeping = df[bookkeeping_cols].rename(columns={args.id_col: "record_id"})

    # Validate text cols
    text_cols_present = [c for c in args.text_cols if c in df.columns]
    text_cols_missing = [c for c in args.text_cols if c not in df.columns]
    for c in text_cols_missing:
        log.warning("Text column %r not in parquet; skipping.", c)
    if not text_cols_present:
        log.error("No requested text columns present. Available: %s",
                  list(df.columns))
        return 1

    # Validate tokenizers
    valid_tokenizers = [t for t in args.tokenizers if t in CANDIDATE_TOKENIZERS]
    invalid = set(args.tokenizers) - set(valid_tokenizers)
    for t in invalid:
        log.warning("Unknown tokenizer %r; skipping.", t)
    if not valid_tokenizers:
        log.error("No valid tokenizers specified. Available: %s",
                  sorted(CANDIDATE_TOKENIZERS))
        return 1

    # ----- Header for outputs -----
    summary: list[str] = []
    summary.append("# UNK Token Scan Summary")
    summary.append("")
    summary.append(f"- **Input:** `{in_path}`")
    summary.append(f"- **Rows scanned:** {len(df):,}")
    summary.append(f"- **Text columns:** {', '.join(text_cols_present)} "
                   f"(primary: `{PRIMARY_TEXT_COL}`)")
    summary.append(f"- **Tokenizers:** {', '.join(valid_tokenizers)}")
    summary.append(f"- **Tokenization parameters:** add_special_tokens=True, "
                   "truncation=False (matches encoder.py training-time behaviour)")
    summary.append(f"- **Flag threshold:** UNK rate > {UNK_FLAG_THRESHOLD:.0%}")
    summary.append(f"- **Segment size for PLM-ICD count:** {SEGMENT_SIZE}")
    summary.append("")
    summary.append("## Note on section tags")
    summary.append("")
    summary.append(
        "`text_section_tagged` contains literal markers like `[REPORT_TYPE]`, "
        "`[DIAGNOSIS]`, `[SPECIMEN]`, `[LATERALITY]`. WordPiece splits these "
        "into `[`, `report`, `_`, `type`, `]` — that is expected and is **not** "
        "UNK. UNK signal here only flags genuine vocabulary coverage gaps."
    )
    summary.append("")

    # Output CSV: stitch all (tokenizer, text_col) results column-wise on record_id
    output_csv = pd.DataFrame({"record_id": record_ids})
    for c in bookkeeping.columns:
        if c != "record_id":
            output_csv[c] = bookkeeping[c].values

    any_flagged = False
    for tokenizer_name in valid_tokenizers:
        hf_id = CANDIDATE_TOKENIZERS[tokenizer_name]
        summary.append(f"## Tokenizer: {tokenizer_name}")
        summary.append("")
        for text_col in text_cols_present:
            texts = df[text_col].fillna("").astype(str).tolist()
            scan_df = scan_one(tokenizer_name, hf_id, text_col, texts, log)
            # Add to wide CSV
            prefix = f"{tokenizer_name}__{text_col}"
            for col in scan_df.columns:
                output_csv[f"{prefix}__{col}"] = scan_df[col].values
            # Summary block
            n_flagged = report_summary_block(
                summary, tokenizer_name, hf_id, text_col,
                scan_df, record_ids, bookkeeping, log,
            )
            if n_flagged:
                any_flagged = True

    # ----- Acceptance footer -----
    summary.append("## Acceptance")
    summary.append("")
    if not any_flagged:
        summary.append(f"✅ **PASS** — no reports above {UNK_FLAG_THRESHOLD:.0%} "
                       "UNK on any (tokenizer, text_col) combination.")
    else:
        summary.append(f"⚠️ **REVIEW** — at least one combination has reports "
                       f"above {UNK_FLAG_THRESHOLD:.0%} UNK. See top-10 lists "
                       "above and inspect those records before E1.")
    summary.append("")

    # ----- Write -----
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    output_csv.to_csv(args.report, index=False)
    log.info("Wrote per-record stats to %s", args.report)

    Path(args.summary).write_text("\n".join(summary))
    log.info("Wrote summary to %s", args.summary)

    return 3 if any_flagged else 0


if __name__ == "__main__":
    sys.exit(main())
