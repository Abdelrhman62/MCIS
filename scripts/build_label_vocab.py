#!/usr/bin/env python3
"""Build (or rebuild) a v3-schema label_vocab.json from a model-ready parquet.

This is the canonical regen tool for label vocabs. Run once per dataset.

Datasets supported out of the box:
    --dataset baheya_m1      → 10 axes (5 ICD-O-3 + ICD-11 stem + 4 ext); icd11_policy=gold_in_baheya
    --dataset tcga_pretrain  → 5 ICD-O-3 axes only; icd11_policy=silver_never_label

For other datasets, use --custom and pass --single-pick / --multi-label lists.

Examples:
    # Regen Baheya M1 vocab from the frozen parquet
    python scripts/build_label_vocab.py \\
        --dataset baheya_m1 \\
        --parquet data/frozen/m1_model_ready/m1_model_ready.parquet \\
        --out data/frozen/m1_model_ready/label_vocab.json

    # Regen TCGA pretrain vocab
    python scripts/build_label_vocab.py \\
        --dataset tcga_pretrain \\
        --parquet data/frozen/tcga_pretrain_model_ready/tcga_pretrain_model_ready.parquet \\
        --out data/frozen/tcga_pretrain_model_ready/label_vocab.json

    # Custom dataset
    python scripts/build_label_vocab.py \\
        --dataset custom \\
        --module M1 --dataset-name my_clinic \\
        --parquet data.parquet --out vocab.json \\
        --single-pick icdo3_morphology icdo3_topography \\
        --multi-label icd11_ext_anatomy

Test-only labels: if a 'split' column is present, the script audits the held-out
split (whatever split values are not the build-from value) and writes any
test-only codes into the JSON's ``test_only_labels`` field.

Exit codes:
    0  success
    1  setup error (missing file / column / etc)
    2  validation warning (test-only codes found — not a failure, just informational)
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.label_vocab import (  # noqa: E402
    LabelVocab,
    _iter_codes,
)


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("build_label_vocab")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        choices=["baheya_m1", "tcga_pretrain", "custom"],
                        help="Pre-defined axis spec, or 'custom' with --single-pick / --multi-label")
    parser.add_argument("--parquet", required=True, type=Path,
                        help="Source model-ready parquet")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSON path")
    parser.add_argument("--built-from-split", default=None,
                        help="Split value to build vocab from (default: dataset-specific)")
    parser.add_argument("--seed", type=int, default=42)

    # Custom mode
    parser.add_argument("--module", default=None,
                        help="(custom only) module name e.g. 'M1'")
    parser.add_argument("--dataset-name", default=None,
                        help="(custom only) dataset name e.g. 'my_clinic'")
    parser.add_argument("--single-pick", nargs="*", default=[],
                        help="(custom only) single-pick axis names")
    parser.add_argument("--multi-label", nargs="*", default=[],
                        help="(custom only) multi-label axis names")
    parser.add_argument("--icd11-policy", default=None,
                        choices=[None, "gold_in_baheya", "silver_never_label"])

    args = parser.parse_args()
    log = _setup_logger()

    if not args.parquet.exists():
        log.error("Parquet not found: %s", args.parquet)
        return 1

    log.info("Reading %s ...", args.parquet)
    df = pd.read_parquet(args.parquet)
    log.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # Build vocab using the appropriate builder.
    if args.dataset == "baheya_m1":
        built_from = args.built_from_split or "trainable"
        vocab = LabelVocab.build_m1(df, built_from_split=built_from, seed=args.seed)
    elif args.dataset == "tcga_pretrain":
        built_from = args.built_from_split or "pretrain_train"
        vocab = LabelVocab.build_tcga_pretrain(df, built_from_split=built_from, seed=args.seed)
    elif args.dataset == "custom":
        if not args.module or not args.dataset_name:
            log.error("--module and --dataset-name are required when --dataset=custom")
            return 1
        if not args.single_pick and not args.multi_label:
            log.error("custom dataset needs at least one --single-pick or --multi-label axis")
            return 1
        built_from = args.built_from_split or "trainable"
        vocab = LabelVocab.build_from_axis_spec(
            df=df,
            single_pick=list(args.single_pick),
            multi_label=list(args.multi_label),
            module=args.module,
            dataset=args.dataset_name,
            built_from_split=built_from,
            seed=args.seed,
            icd11_policy=args.icd11_policy,
        )
    else:
        log.error("unreachable: --dataset=%r", args.dataset)
        return 1

    log.info("Vocab built:")
    for line in vocab.summary().splitlines():
        log.info("  %s", line)

    # Audit test-only codes if there are non-build splits.
    test_only_label_warning = False
    if "split" in df.columns:
        held_out = df[df["split"] != built_from]
        if len(held_out) > 0:
            log.info("Auditing %d held-out rows for test-only codes ...", len(held_out))
            test_only: dict[str, dict[str, object]] = {}
            for axis_name, axis in vocab.axes.items():
                if axis_name not in held_out.columns:
                    continue
                seen_codes: Counter[str] = Counter()
                affected_rows: set[int] = set()
                for ridx, raw in enumerate(held_out[axis_name].tolist()):
                    toks = _iter_codes(raw, multilabel=axis.multilabel, delimiter=axis.delimiter)
                    novel = [t for t in toks if t not in axis.code_to_idx]
                    if novel:
                        for t in novel:
                            seen_codes[t] += 1
                        affected_rows.add(ridx)
                if seen_codes:
                    test_only[axis_name] = {
                        "codes": sorted(seen_codes.keys()),
                        "test_rows_affected": len(affected_rows),
                    }
                    log.warning(
                        "  axis=%s: %d test-only codes (%d rows affected): %s",
                        axis_name, len(seen_codes), len(affected_rows),
                        sorted(seen_codes.keys())[:10],
                    )
            vocab.test_only_labels = test_only
            if test_only:
                test_only_label_warning = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    vocab.save_json(args.out)
    log.info("Wrote vocab to %s", args.out)

    if test_only_label_warning:
        log.warning(
            "Test-only codes detected and recorded in JSON. "
            "These should be excluded from F1 per architecture v6 §11.5."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
