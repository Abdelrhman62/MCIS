"""
Stage 1 — Per-class F1 breakdown of E1 fold-0 best checkpoint
==============================================================
Load checkpoints/B0/best.pt, evaluate on fold-0 val, print per-class
precision/recall/F1 + confusion-style support for every axis. Special focus
on laterality axes (icdo3_laterality, icd11_ext_laterality) where the
"regression vs B1" was reported.

Runs ~5 min on M4 MPS (forward-only over 550 rows).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

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

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = REPO / "configs" / "B0.yaml"
CKPT_PATH   = REPO / "checkpoints" / "B0" / "best.pt"
FOLD_IDX    = 0
NULL_TARGET_SENTINEL = -1

SEP = "=" * 78
def banner(s: str) -> None: print(f"\n{SEP}\n{s}\n{SEP}")

# ─────────────────────────────────────────────────────────────────────────────
banner("LOAD")
print(f"  CONFIG: {CONFIG_PATH}")
print(f"  CKPT  : {CKPT_PATH}  exists={CKPT_PATH.exists()}")

cfg = BenchmarkConfig.from_yaml(CONFIG_PATH)
vocab = LabelVocab.load_json(cfg.data.label_vocab)
df = load_parquet(cfg.data.parquet)
trainable_df, _ = split_trainable_test(df, cfg.data.trainable_value, cfg.data.test_value)
_, val_df = split_by_fold(trainable_df, None, FOLD_IDX)
print(f"  fold-{FOLD_IDX} val rows: {len(val_df)}")

# Resolve device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"  device: {device}")

val_ds = BaheyaM1Dataset(val_df, vocab, cfg)
val_loader = DataLoader(val_ds, batch_size=cfg.train.micro_batch_size, shuffle=False,
                        collate_fn=collate_m1, num_workers=0)

# Build model the same way Trainer does
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

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.set_active_axes(set(ckpt["active_axes"]))
model.eval()
print(f"  ckpt epoch={ckpt.get('epoch')} best_epoch={ckpt.get('best_epoch')} "
      f"best_metric={ckpt.get('best_metric'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
banner("FORWARD PASS")
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

logits_by_axis = {a: torch.cat(l, dim=0).numpy() for a, l in all_logits.items()}
targets_by_axis = {a: torch.cat(t, dim=0).numpy() for a, t in all_targets.items()}
print(f"  axes evaluated: {len(logits_by_axis)}")

# ─────────────────────────────────────────────────────────────────────────────
banner("PER-CLASS BREAKDOWN — every single-pick axis")
def per_class_singlepick(axis: str) -> None:
    codes = vocab[axis].codes
    K = len(codes)
    logits = logits_by_axis[axis]
    targets = targets_by_axis[axis]
    valid = targets != NULL_TARGET_SENTINEL
    n_valid = int(valid.sum())
    if n_valid == 0:
        print(f"  {axis}: no valid records")
        return
    y_true = targets[valid]
    preds = logits[valid].argmax(axis=-1)
    support = Counter(y_true.tolist())
    pred_dist = Counter(preds.tolist())

    print(f"\n  AXIS: {axis}  (K={K}, n_valid={n_valid}, accuracy={(preds==y_true).mean():.4f})")
    print(f"    {'code':10s} {'idx':>3s} {'support':>8s} {'pred_n':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}"
          f" {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    per_class_f1 = []
    for k in range(K):
        tp = int(((preds == k) & (y_true == k)).sum())
        fp = int(((preds == k) & (y_true != k)).sum())
        fn = int(((preds != k) & (y_true == k)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_f1.append(f1)
        sup = support.get(k, 0)
        pred_n = pred_dist.get(k, 0)
        print(f"    {codes[k]:10s} {k:3d} {sup:8d} {pred_n:7d} {tp:5d} {fp:5d} {fn:5d}"
              f" {prec:6.3f} {rec:6.3f} {f1:6.3f}")
    macro = float(np.mean(per_class_f1))
    present = [f for k, f in enumerate(per_class_f1) if support.get(k, 0) > 0]
    macro_present = float(np.mean(present)) if present else 0.0
    print(f"    F1-MACRO (full K={K})      : {macro:.4f}   ← reported metric")
    print(f"    F1-MACRO (present-only)    : {macro_present:.4f}   ← excludes 0-support classes")
    print(f"    classes with 0 support     : {[codes[k] for k in range(K) if support.get(k,0)==0]}")

for axis in cfg.axis_types.single_pick:
    if axis in logits_by_axis:
        per_class_singlepick(axis)

# ─────────────────────────────────────────────────────────────────────────────
banner("LATERALITY DEEP-DIVE")
# icdo3_laterality vs icd11_ext_laterality side-by-side
for axis in ("icdo3_laterality", "icd11_ext_laterality"):
    codes = vocab[axis].codes
    logits = logits_by_axis[axis]
    targets = targets_by_axis[axis]
    valid = targets != NULL_TARGET_SENTINEL
    if not valid.any():
        continue
    y_true = targets[valid]
    preds = logits[valid].argmax(axis=-1)
    print(f"\n  {axis} confusion matrix:")
    print(f"    pred\\true   " + " ".join(f"{c:>6s}" for c in codes))
    for ki in range(len(codes)):
        row = [int(((preds == ki) & (y_true == kj)).sum()) for kj in range(len(codes))]
        print(f"    {codes[ki]:10s}  " + " ".join(f"{v:6d}" for v in row))

# ─────────────────────────────────────────────────────────────────────────────
banner("AXIS COVERAGE — null vs valid in fold-0 val")
for axis in cfg.axis_types.single_pick:
    if axis not in targets_by_axis: continue
    t = targets_by_axis[axis]
    n_null = int((t == NULL_TARGET_SENTINEL).sum())
    print(f"  {axis:25s}  valid={len(t)-n_null:4d}  null(masked)={n_null:4d}")

# ─────────────────────────────────────────────────────────────────────────────
banner("DONE")
print("Save log: results/diagnostics/stage1_per_class_$(date).log")
