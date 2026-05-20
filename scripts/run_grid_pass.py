"""Run the E1 grid pass — single-fold training across grid configs.

Per architecture v6 §4.2 / build plan: grid is single-fold each on E1
anchor. Run all grid cells, compare to E1 fold-0 baseline, lock the best
loss config before stacking on E2 (becomes E5a).

Usage:
    # Run a specific subset
    python scripts/run_grid_pass.py --configs A1 A2 A2w --fold 0 --device cuda

    # Run everything in configs/grid/
    python scripts/run_grid_pass.py --all --fold 0 --device cuda

    # Smoke test (1 epoch, no W&B)
    python scripts/run_grid_pass.py --configs A1 --fold 0 --device cuda --smoke

Outputs:
    results/grid_ablation_summary_{utc_ts}.md  — table comparing all cells
    results/grid_ablation_summary_{utc_ts}.json — same data, machine-readable
    checkpoints/<grid_cell_name>/fold_<n>/best.pt — per-cell ckpts

Reads `step1_multimetric_fold0_*.json` files from `results/diagnostics/` for
the E1 baseline comparison. Falls back to "no baseline" if not found.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


GRID_DIR = Path("configs/grid")
RESULTS_DIR = Path("results")
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
CHECKPOINTS_DIR = Path("checkpoints")

# B1 baseline numbers (locked, 5-fold mean from tfidf_2026-05-11T221423Z.md)
B1_BASELINE = {
    "early_stop_macro": 0.6351,
    "icdo3_topography_macro": 0.6659,
    "icdo3_morphology_macro": 0.4035,
    "icdo3_behavior_macro": 0.5625,
    "icdo3_grade_macro": 0.9481,
    "icdo3_laterality_macro": 0.9065,
    "icd11_stem_macro": 0.2901,
    "icd11_ext_laterality_macro": 0.4250,
    "icd11_ext_grading_macro": 0.7222,
    "icd11_ext_anatomy_micro": 0.6282,
    "icd11_ext_histopath_micro": 0.6811,
}

# E1 fold-0 numbers (from MCIS_Pre_E1_Audit_Handoff_2026-05-14)
E1_FOLD0_BASELINE = {
    "early_stop_macro": 0.5305,
    "mean_f1_macro_singlepick": 0.5978,
    "mean_f1_macro_present_singlepick": 0.6546,
    "mean_f1_micro_singlepick": 0.9417,
    "mean_f1_micro_multilabel": 0.4632,
}


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("grid_pass")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        log.addHandler(h)
    return log


def _discover_grid_configs() -> list[Path]:
    return sorted(GRID_DIR.glob("*.yaml"))


def _resolve_grid_configs(names: list[str] | None, run_all: bool) -> list[Path]:
    available = _discover_grid_configs()
    if run_all:
        return available

    selected: list[Path] = []
    missing: list[str] = []
    for name in names or []:
        # Match by basename prefix (e.g. "A1" matches "A1_classweighted.yaml")
        hit = [p for p in available if p.stem.startswith(name)]
        if not hit:
            missing.append(name)
        else:
            selected.extend(hit)
    if missing:
        raise FileNotFoundError(
            f"Grid configs not found: {missing}. "
            f"Available: {[p.stem for p in available]}"
        )
    return selected


def _run_one(
    cfg_path: Path,
    fold: int,
    device: str,
    smoke: bool,
    log: logging.Logger,
) -> dict[str, Any]:
    """Spawn one training run via `python -m src.train`.

    Returns metadata dict on success; raises on failure.
    """
    experiment_name = cfg_path.stem  # e.g. "A1_classweighted"
    log.info("=" * 78)
    log.info("Grid cell: %s  (fold=%d, device=%s)", experiment_name, fold, device)
    log.info("=" * 78)

    cmd = [
        sys.executable, "-m", "src.train",
        "--config", str(cfg_path),
        "--fold", str(fold),
        "--device", device,
        "--experiment-name", experiment_name,
    ]
    if smoke:
        cmd.append("--smoke")

    log.info("$ %s", " ".join(cmd))
    t0 = _dt.datetime.now()
    rc = subprocess.call(cmd)
    wall_min = (_dt.datetime.now() - t0).total_seconds() / 60.0

    if rc != 0:
        raise RuntimeError(
            f"Grid cell {experiment_name} failed with rc={rc}. "
            f"See logs above."
        )

    log.info("  %s training: rc=%d wall=%.1fmin", experiment_name, rc, wall_min)

    # Locate best.pt and load meta
    ckpt_dir = CHECKPOINTS_DIR / experiment_name
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        log.warning("meta.json not found at %s — using placeholder", meta_path)
        meta: dict[str, Any] = {
            "experiment_name": experiment_name,
            "best_metric": None,
            "best_epoch": None,
        }
    else:
        meta = json.loads(meta_path.read_text())

    meta["_grid_cell"] = experiment_name
    meta["_wall_min"] = wall_min
    meta["_config_path"] = str(cfg_path)
    return meta


def _run_step1_eval(
    cfg_path: Path,
    fold: int,
    device: str,
    log: logging.Logger,
) -> dict[str, Any] | None:
    """Run scripts/step1_multimetric.py against the just-trained ckpt.

    Returns the JSON contents on success, None on failure.
    """
    experiment_name = cfg_path.stem
    ckpt_path = CHECKPOINTS_DIR / experiment_name / "best.pt"
    if not ckpt_path.exists():
        log.warning("No best.pt at %s, skipping step1 eval", ckpt_path)
        return None

    cmd = [
        sys.executable, "scripts/step1_multimetric.py",
        "--config", str(cfg_path),
        "--fold", str(fold),
        "--ckpt", str(ckpt_path),
        "--device", device,
    ]
    log.info("$ %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        log.warning("step1 eval rc=%d for %s", rc, experiment_name)
        return None

    # Find the most recent step1 JSON for this experiment+fold
    pattern = f"step1_multimetric_fold{fold}_*.json"
    candidates = sorted(DIAGNOSTICS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    json_path = candidates[-1]
    try:
        return json.loads(json_path.read_text())
    except Exception as e:
        log.warning("Failed parsing %s: %s", json_path, e)
        return None


def _format_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "—"
    d = value - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def _build_markdown_summary(
    cells: list[dict[str, Any]],
    fold: int,
    output_path: Path,
) -> str:
    ts = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    lines = [
        "# Grid Ablation Summary — single-fold E1 anchor",
        "",
        f"**Generated:** {ts}",
        f"**Fold:** {fold}",
        f"**Baselines (locked):**",
        f"- B1 5-fold early_stop_macro: **{B1_BASELINE['early_stop_macro']:.4f}**",
        f"- E1 fold-0 early_stop_macro: **{E1_FOLD0_BASELINE['early_stop_macro']:.4f}**",
        "",
        "## Cross-axis summary",
        "",
        "| Cell | best_epoch | early_stop | Δ vs E1 | Δ vs B1 | macro_present | micro_sp | micro_ml | accuracy_sp | wall_min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        meta = cell["meta"]
        step1 = cell.get("step1") or {}
        early_stop = step1.get("early_stop_metric_current") or meta.get("best_metric")
        macro_present = step1.get("early_stop_metric_present_only")
        micro_sp = step1.get("mean_f1_micro_singlepick")
        micro_ml = step1.get("mean_f1_micro_multilabel")
        acc_sp = step1.get("mean_accuracy_singlepick")
        d_e1 = _format_delta(early_stop, E1_FOLD0_BASELINE["early_stop_macro"])
        d_b1 = _format_delta(early_stop, B1_BASELINE["early_stop_macro"])
        lines.append(
            f"| {cell['name']} "
            f"| {meta.get('best_epoch', '—')} "
            f"| {f'{early_stop:.4f}' if early_stop is not None else '—'} "
            f"| {d_e1} "
            f"| {d_b1} "
            f"| {f'{macro_present:.4f}' if macro_present is not None else '—'} "
            f"| {f'{micro_sp:.4f}' if micro_sp is not None else '—'} "
            f"| {f'{micro_ml:.4f}' if micro_ml is not None else '—'} "
            f"| {f'{acc_sp:.4f}' if acc_sp is not None else '—'} "
            f"| {meta.get('_wall_min', 0):.1f} |"
        )

    # Per-axis comparison (laterality is the canary for class-weighting fixes)
    lines.extend([
        "",
        "## Per-axis F1-Macro (key axes)",
        "",
        "Comparing grid cells against B1 5-fold mean and E1 fold-0:",
        "",
        "| Cell | topo | morph | beh | grade | lat | stem | ext_lat | ext_grade | ext_ana_micro | ext_histo_micro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| B1 (5-fold mean) | "
        f"{B1_BASELINE['icdo3_topography_macro']:.4f} | "
        f"{B1_BASELINE['icdo3_morphology_macro']:.4f} | "
        f"{B1_BASELINE['icdo3_behavior_macro']:.4f} | "
        f"{B1_BASELINE['icdo3_grade_macro']:.4f} | "
        f"**{B1_BASELINE['icdo3_laterality_macro']:.4f}** | "
        f"{B1_BASELINE['icd11_stem_macro']:.4f} | "
        f"**{B1_BASELINE['icd11_ext_laterality_macro']:.4f}** | "
        f"{B1_BASELINE['icd11_ext_grading_macro']:.4f} | "
        f"{B1_BASELINE['icd11_ext_anatomy_micro']:.4f} | "
        f"{B1_BASELINE['icd11_ext_histopath_micro']:.4f} |",
    ])
    for cell in cells:
        step1 = cell.get("step1") or {}
        per_axis = step1.get("per_axis_single_pick", {})
        per_axis_ml = step1.get("per_axis_multi_label", {})

        def fmt(d: dict, axis: str, key: str = "macro") -> str:
            if axis not in d:
                return "—"
            v = d[axis].get(key)
            return f"{v:.4f}" if v is not None else "—"

        lines.append(
            f"| {cell['name']} | "
            f"{fmt(per_axis, 'icdo3_topography')} | "
            f"{fmt(per_axis, 'icdo3_morphology')} | "
            f"{fmt(per_axis, 'icdo3_behavior')} | "
            f"{fmt(per_axis, 'icdo3_grade')} | "
            f"{fmt(per_axis, 'icdo3_laterality')} | "
            f"{fmt(per_axis, 'icd11_stem')} | "
            f"{fmt(per_axis, 'icd11_ext_laterality')} | "
            f"{fmt(per_axis, 'icd11_ext_grading')} | "
            f"{fmt(per_axis_ml, 'icd11_ext_anatomy', 'micro')} | "
            f"{fmt(per_axis_ml, 'icd11_ext_histopath', 'micro')} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Δ vs E1**: positive = grid cell improved over E1 fold-0 (0.5305).",
        "- **Δ vs B1**: positive = grid cell cleared B1 5-fold gate (0.6351).",
        "- The two laterality columns are the canary axes for class-weighting "
          "fixes. B1 wins these in E1 because its sklearn `class_weight='balanced'` "
          "catches rare codes E1 misses with uniform CE.",
        "- Best cell = the one that maximizes early_stop AND clears B1 on "
          "macro early_stop AND lifts laterality macro materially.",
        "- After locking the winner here, copy its loss config to "
          "`configs/E5a_focal_on_e2.yaml` (or equivalent) to stack on E2.",
        "",
    ])
    md = "\n".join(lines)
    output_path.write_text(md)
    return md


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_grid_pass")
    parser.add_argument("--configs", nargs="*",
                        help="Grid config names (e.g. A1 A2 A2w). Prefix match.")
    parser.add_argument("--all", action="store_true",
                        help="Run every YAML under configs/grid/.")
    parser.add_argument("--fold", type=int, default=0,
                        help="CV fold to use (default 0).")
    parser.add_argument("--device", default="cuda",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Device for training (default cuda).")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, W&B disabled. For pipeline sanity.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (assume ckpts already exist); "
                             "run step1 eval + summary only.")
    args = parser.parse_args()

    log = _setup_logger()

    if not args.configs and not args.all:
        log.error("Must specify --configs <names...> or --all")
        return 1

    cfg_paths = _resolve_grid_configs(args.configs, args.all)
    log.info("Selected %d grid cells: %s",
             len(cfg_paths), [p.stem for p in cfg_paths])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    failed: list[str] = []
    for cfg_path in cfg_paths:
        try:
            if args.skip_train:
                meta_path = CHECKPOINTS_DIR / cfg_path.stem / "meta.json"
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
                    "experiment_name": cfg_path.stem,
                    "_wall_min": 0.0,
                }
            else:
                meta = _run_one(cfg_path, args.fold, args.device, args.smoke, log)
            step1 = _run_step1_eval(cfg_path, args.fold, args.device, log)
            cells.append({
                "name": cfg_path.stem,
                "meta": meta,
                "step1": step1,
            })
        except Exception as e:
            log.error("Grid cell %s FAILED: %s", cfg_path.stem, e)
            failed.append(cfg_path.stem)

    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    md_path = RESULTS_DIR / f"grid_ablation_summary_{ts}.md"
    json_path = RESULTS_DIR / f"grid_ablation_summary_{ts}.json"

    md = _build_markdown_summary(cells, args.fold, md_path)
    json_path.write_text(json.dumps({
        "fold": args.fold,
        "cells": cells,
        "failed": failed,
        "baselines": {
            "B1_5fold": B1_BASELINE,
            "E1_fold0": E1_FOLD0_BASELINE,
        },
    }, indent=2, default=str))

    log.info("Wrote: %s", md_path)
    log.info("Wrote: %s", json_path)
    log.info("Done. Cells: %d ok, %d failed.", len(cells), len(failed))
    if failed:
        log.error("Failed cells: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
