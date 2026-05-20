"""
Step 2 — 5-fold E1 launcher.
============================
Train folds 1, 2, 3, 4 (fold 0 already trained → checkpoints/B0/best.pt).
After each fold, run step1_multimetric to produce per-fold multi-metric JSON.
Finally, aggregate all 5 fold JSONs into a single 5-fold summary.

Each fold trained with current B0.yaml (no config changes — we want a pure
5-fold variance estimate of the current setup).

Per-fold checkpoint dir: checkpoints/B0_fold{N}/
Per-fold step1 output:   results/diagnostics/step1_multimetric_fold{N}_*.json

Usage:
  python scripts/step2_run_5fold.py
  python scripts/step2_run_5fold.py --folds 1 2 3 4
  python scripts/step2_run_5fold.py --aggregate-only   # skip training, just aggregate

Compute: ~23 min/fold on M4 MPS × 4 folds = ~92 min wall.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parents[1].name == "MCIS" else Path.cwd()
DIAG_DIR = REPO / "results" / "diagnostics"
CKPT_DIR = REPO / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="step2_run_5fold")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="Folds to TRAIN (fold 0 assumed done). Default: 1 2 3 4")
    p.add_argument("--config", type=Path, default=REPO / "configs" / "B0.yaml")
    p.add_argument("--device", default="mps", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip training, just aggregate existing step1 outputs")
    p.add_argument("--skip-step1-after-train", action="store_true",
                   help="Skip the per-fold step1 eval (training only)")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    """Run subprocess, stream output."""
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def latest_step1_json(fold: int) -> Path | None:
    """Find the most recent step1_multimetric_fold{N}_*.json for this fold."""
    pattern = str(DIAG_DIR / f"step1_multimetric_fold{fold}_*.json")
    matches = sorted(glob.glob(pattern))
    return Path(matches[-1]) if matches else None


def train_one_fold(fold: int, config: Path, device: str) -> bool:
    """Train fold `fold` via `python -m src.train`. Returns True on success."""
    exp_name = f"B0_fold{fold}"
    cmd = [
        sys.executable, "-m", "src.train",
        "--config", str(config),
        "--fold", str(fold),
        "--device", device,
        "--experiment-name", exp_name,
        # Force a per-fold W&B run name and ckpt dir collision protection.
    ]
    t0 = time.time()
    rc = run(cmd)
    t = time.time() - t0
    print(f"  fold {fold} training: rc={rc} wall={t/60:.1f}min")
    return rc == 0


def step1_one_fold(fold: int, config: Path, device: str) -> bool:
    """Run step1_multimetric.py on the saved best.pt."""
    if fold == 0:
        ckpt = CKPT_DIR / "B0" / "best.pt"
    else:
        ckpt = CKPT_DIR / f"B0_fold{fold}" / "best.pt"
    if not ckpt.exists():
        print(f"  ❌ step1 skipped — ckpt missing: {ckpt}")
        return False
    cmd = [
        sys.executable, "scripts/step1_multimetric.py",
        "--config", str(config),
        "--fold", str(fold),
        "--ckpt", str(ckpt),
        "--device", device,
    ]
    rc = run(cmd)
    return rc == 0


def aggregate_5fold() -> int:
    """Read latest step1 JSON per fold ∈ {0..4}, compute mean/std per metric."""
    print("\n" + "=" * 78)
    print("AGGREGATE 5-fold")
    print("=" * 78)
    fold_payloads: dict[int, dict] = {}
    for fold in range(5):
        p = latest_step1_json(fold)
        if p is None:
            print(f"  fold {fold}: ❌ no step1 JSON found")
            continue
        payload = json.loads(p.read_text())
        fold_payloads[fold] = payload
        print(f"  fold {fold}: {p.name}  best_epoch={payload.get('best_epoch')}")

    if not fold_payloads:
        print("  ❌ no fold data found — aggregation aborted")
        return 1
    if len(fold_payloads) < 5:
        print(f"  ⚠️  only {len(fold_payloads)}/5 folds available — partial aggregation")

    # Aggregate summary block
    summary_keys = list(next(iter(fold_payloads.values()))["summary"].keys())
    agg_summary: dict[str, dict[str, float]] = {}
    for k in summary_keys:
        vals = [fp["summary"].get(k, 0.0) for fp in fold_payloads.values()]
        agg_summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    # Aggregate per-axis F1-macro for single-pick + F1-micro for multi-label
    axis_collect: dict[str, dict[str, list[float]]] = {}
    for fp in fold_payloads.values():
        for axis, m in fp["per_axis"].items():
            slot = axis_collect.setdefault(axis, {})
            if m["kind"] == "single_pick":
                slot.setdefault("f1_macro", []).append(m["f1_macro"])
                slot.setdefault("f1_macro_present", []).append(m["f1_macro_present"])
                slot.setdefault("f1_micro", []).append(m["f1_micro"])
                slot.setdefault("accuracy", []).append(m["accuracy"])
            else:
                slot.setdefault("f1_micro", []).append(m["f1_micro"])
                slot.setdefault("f1_macro", []).append(m["f1_macro"])
                slot.setdefault("accuracy_exact_set", []).append(m["accuracy_exact_set"])

    agg_per_axis: dict[str, dict[str, dict[str, float]]] = {}
    for axis, slots in axis_collect.items():
        agg_per_axis[axis] = {
            metric: {"mean": float(np.mean(vals)),
                     "std": float(np.std(vals)),
                     "values": vals}
            for metric, vals in slots.items()
        }

    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_json = DIAG_DIR / f"step2_5fold_summary_{ts}.json"
    out_md   = DIAG_DIR / f"step2_5fold_summary_{ts}.md"
    out_json.write_text(json.dumps({
        "folds_aggregated": sorted(fold_payloads.keys()),
        "summary": agg_summary,
        "per_axis": agg_per_axis,
        "saved_at": ts,
    }, indent=2))

    # Markdown
    lines: list[str] = []
    lines.append(f"# Step 2 — 5-Fold E1 Summary ({len(fold_payloads)}/5 folds)")
    lines.append("")
    lines.append(f"**Folds aggregated:** {sorted(fold_payloads.keys())}  ")
    lines.append(f"**Generated:** {ts}")
    lines.append("")
    lines.append("## Cross-axis summary (mean ± std across folds)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Per-fold |")
    lines.append("|---|---:|---:|---|")
    for k, v in agg_summary.items():
        per_fold_str = ", ".join(f"{x:.4f}" for x in v["values"])
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} | {per_fold_str} |")
    lines.append("")
    lines.append("## Per-axis (single-pick) F1-Macro across folds")
    lines.append("")
    lines.append("| Axis | macro mean | macro std | macro_present mean | micro mean | accuracy mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for axis, m in agg_per_axis.items():
        if "f1_macro_present" not in m:
            continue
        lines.append(f"| {axis} | {m['f1_macro']['mean']:.4f} | {m['f1_macro']['std']:.4f} | "
                     f"{m['f1_macro_present']['mean']:.4f} | {m['f1_micro']['mean']:.4f} | "
                     f"{m['accuracy']['mean']:.4f} |")
    lines.append("")
    lines.append("## Per-axis (multi-label) across folds")
    lines.append("")
    lines.append("| Axis | micro mean | micro std | macro mean | exact-set acc |")
    lines.append("|---|---:|---:|---:|---:|")
    for axis, m in agg_per_axis.items():
        if "accuracy_exact_set" not in m:
            continue
        lines.append(f"| {axis} | {m['f1_micro']['mean']:.4f} | {m['f1_micro']['std']:.4f} | "
                     f"{m['f1_macro']['mean']:.4f} | {m['accuracy_exact_set']['mean']:.4f} |")

    out_md.write_text("\n".join(lines))
    print(f"\n  Wrote: {out_json}")
    print(f"  Wrote: {out_md}")
    print("\n  Cross-axis summary:")
    for k, v in agg_summary.items():
        print(f"    {k:45s} {v['mean']:.4f} ± {v['std']:.4f}")
    return 0


def main() -> int:
    args = parse_args()

    if args.aggregate_only:
        return aggregate_5fold()

    # Ensure step1 exists for fold 0 (already trained) before kicking off folds 1–4
    if latest_step1_json(0) is None:
        print("[step2] No step1 JSON for fold 0 found. Running it now.")
        ok = step1_one_fold(0, args.config, args.device)
        if not ok:
            print("[step2] ❌ fold-0 step1 failed. Aborting.")
            return 1

    for fold in args.folds:
        print(f"\n{'='*78}\nFOLD {fold} — training\n{'='*78}")
        if not train_one_fold(fold, args.config, args.device):
            print(f"❌ fold {fold} training failed. Continuing with next fold.")
            continue
        if args.skip_step1_after_train:
            continue
        if not step1_one_fold(fold, args.config, args.device):
            print(f"❌ fold {fold} step1 eval failed. Continuing with next fold.")
            continue

    return aggregate_5fold()


if __name__ == "__main__":
    sys.exit(main())
