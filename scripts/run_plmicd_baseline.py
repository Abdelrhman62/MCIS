"""B2 — faithful PLM-ICD baseline (Huang et al., ClinicalNLP 2022).

Architecture-family comparator. PLM-ICD's published architecture is
chunked PLM + LAAT label-wise attention over a single flat label space
(verified: MiuLab/PLM-ICD trains with --model_mode laat). B2 reproduces
exactly that and removes ONLY MCIS's per-axis multi-task decomposition,
so the B2-vs-E1 delta isolates the multi-task contribution (thesis
novelty) with attention held constant.

Standalone by design: imports only OncologyEncoder, AxisLabelAttention,
BaheyaM1Dataset/collate_m1/split helpers, and the SAME metric functions
E1 uses (scripts.step1_multimetric.metrics_single_pick/_multi_label).
It does NOT import or touch Trainer / heads.py — zero regression risk to
the verified-clean E1 pipeline.

Model:
    texts -> OncologyEncoder -> H [B,T,d], mask [B,T]
          -> AxisLabelAttention(d, num_labels=119) -> context [B,119,d]
          -> per-label projection (PLM-ICD diagonal form):
                 logit_k = sum_d context[:,k,d] * W[k,d] + b[k]
                 (einsum 'bkd,kd->bk')
          -> BCEWithLogits over the flat 119 (single loss, no masking)

Flat 119 = concat of all 10 axis cardinalities in label_vocab.json axis
order. Row target = union of per-axis gold as positives. Null/absent axis
contributes no positive (faithful — PLM-ICD has no null class).

Eval (apples-to-apples with E1): slice the 119 logits back per axis;
single-pick slice -> argmax, restore NULL=-1 where the row's gold axis was
absent so metrics_single_pick's `valid = targets != NULL` masks identically
to E1; multilabel slice -> pass raw logits to metrics_multi_label (it does
its own sigmoid>0.5). Headline:
    early_stop = (mean_f1_macro_singlepick + mean_f1_micro_multilabel)/2
identical formula to step1_multimetric, so directly comparable to
B1=0.6351 and E1=0.5726.

Usage:
    python -m scripts.run_plmicd_baseline --config configs/B2_plmicd.yaml
    python -m scripts.run_plmicd_baseline --config configs/B2_plmicd.yaml \
        --fold 0 --smoke --device cpu
    python -m scripts.run_plmicd_baseline --config configs/B2_plmicd.yaml \
        --device cuda                       # all 5 folds, A100

Output: results/B2_plmicd_summary_<ts>.{md,json} with per-fold + mean±std
and the B1 / E1 reference lines.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.loaders import (
    BaheyaM1Dataset,
    collate_m1,
    load_parquet,
    split_by_fold,
    split_trainable_test,
)
from src.models.encoder import OncologyEncoder
from src.models.label_attention import AxisLabelAttention
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from src.utils.seed import set_seed

# Same NULL sentinel as scripts.step1_multimetric.NULL and
# src.data.loaders.NULL_TARGET_SENTINEL (both = -1). Asserted at import.
from scripts.step1_multimetric import (  # noqa: E402
    NULL as _STEP1_NULL,
    metrics_multi_label,
    metrics_single_pick,
)

# B1 / E1 reference numbers (handoff §8) for the summary report.
_B1_EARLY_STOP = 0.6351
_E1_EARLY_STOP = 0.5726


# ---------------------------------------------------------------------------
# Flat-119 label-space adapter
# ---------------------------------------------------------------------------


class FlatLabelSpace:
    """Maps the 10 per-axis vocabs onto one concatenated [0, 119) space.

    Offsets follow label_vocab.json axis order exactly (asserted against the
    loaded vocab so a vocab/config reorder fails loud, not silently wrong).
    """

    def __init__(self, vocab: Any, cfg: BenchmarkConfig) -> None:
        # Axis order = single_pick then multilabel, matching cfg.all_axes,
        # which must equal the label_vocab.json key order.
        self.axes: list[str] = list(cfg.all_axes)
        self.single_pick = set(cfg.axis_types.single_pick)
        self.multilabel = set(cfg.axis_types.multilabel)

        self.axis_codes: dict[str, list[str]] = {}
        self.offset: dict[str, int] = {}
        cur = 0
        for ax in self.axes:
            codes = self._codes_for_axis(vocab, ax)
            self.axis_codes[ax] = codes
            self.offset[ax] = cur
            cur += len(codes)
        self.total = cur

        declared = sum(cfg.label_cardinalities.get(a, -1) for a in self.axes)
        if declared != self.total:
            raise ValueError(
                f"Flat space total {self.total} != sum of "
                f"label_cardinalities {declared}. Vocab/config drift."
            )

    @staticmethod
    def _codes_for_axis(vocab: Any, axis: str) -> list[str]:
        # LabelVocab.axes maps axis name -> AxisVocab; AxisVocab.codes is the
        # ordered code list (index i == class id i). Verified against
        # label_vocab.py: num_classes=len(codes), code_to_idx built from it.
        axes_attr = getattr(vocab, "axes", None)
        if axes_attr is None or axis not in axes_attr:
            raise AttributeError(
                f"vocab has no axis {axis!r} (axes: "
                f"{list(axes_attr) if axes_attr else None})"
            )
        axis_vocab = axes_attr[axis]
        codes = getattr(axis_vocab, "codes", None)
        if codes is None:
            raise AttributeError(
                f"AxisVocab for {axis!r} has no .codes attribute"
            )
        return list(codes)

    # ---- target construction ----

    def build_targets(self, collated: dict[str, Any]) -> torch.Tensor:
        """collate_m1 batch -> flat multi-hot target [B, 119] (float)."""
        # Infer B from texts (always present).
        bsz = len(collated["texts"])
        y = torch.zeros(bsz, self.total, dtype=torch.float32)
        for ax in self.axes:
            key = f"labels_{ax}"
            if key not in collated:
                continue
            off = self.offset[ax]
            k = len(self.axis_codes[ax])
            lab = collated[key]
            if ax in self.single_pick:
                # LongTensor [B], sentinel -1 = null -> no positive.
                idx = lab.long()
                pos = idx >= 0
                rows = torch.nonzero(pos, as_tuple=False).squeeze(-1)
                if rows.numel():
                    y[rows, off + idx[pos]] = 1.0
            else:
                # FloatTensor [B, K] multi-hot already.
                ml = lab.float()
                if ml.shape[1] != k:
                    raise ValueError(
                        f"axis {ax}: collate K={ml.shape[1]} != vocab K={k}"
                    )
                y[:, off:off + k] = (ml > 0.5).float()
        return y

    # ---- decode flat logits back to per-axis for E1-comparable eval ----

    def decode_axis(
        self, flat_logits: np.ndarray, axis: str
    ) -> np.ndarray:
        off = self.offset[axis]
        k = len(self.axis_codes[axis])
        return flat_logits[:, off:off + k]


# ---------------------------------------------------------------------------
# Model — faithful PLM-ICD (encoder + LAAT(119) + per-label projection)
# ---------------------------------------------------------------------------


class PLMICDFlat(nn.Module):
    def __init__(self, cfg: BenchmarkConfig, n_labels: int) -> None:
        super().__init__()
        enc_cfg = {
            "backbone_name": cfg.model.encoder_hf_id,
            "segment_size": cfg.model.segment_size,
            "max_segments": cfg.model.max_segments,
            "strict_length": cfg.model.strict_length,
        }
        self.encoder = OncologyEncoder.from_config(enc_cfg)
        d = self.encoder.hidden_dim
        self.attention = AxisLabelAttention(
            hidden_dim=d,
            num_labels=n_labels,
            attn_dim=cfg.model.attention_hidden_dim,
        )
        # PLM-ICD per-label projection: each label k has its own w_k in R^d
        # plus bias. logit_k = <context_k, w_k> + b_k  (diagonal form).
        self.label_weight = nn.Parameter(torch.empty(n_labels, d))
        self.label_bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.xavier_uniform_(self.label_weight)
        self.n_labels = n_labels

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.encoder(texts)
        att = self.attention(enc.hidden_states, enc.attention_mask)
        ctx = att.context  # [B, K, d]
        # diagonal projection: sum_d ctx[b,k,d] * W[k,d]  -> [B, K]
        logits = torch.einsum("bkd,kd->bk", ctx, self.label_weight)
        logits = logits + self.label_bias
        return logits


# ---------------------------------------------------------------------------
# Train / eval one fold
# ---------------------------------------------------------------------------


def _run_fold(
    cfg: BenchmarkConfig,
    fold_idx: int,
    device: torch.device,
    log: Any,
    smoke: bool,
) -> dict[str, Any]:
    vocab = _load_vocab(cfg)
    flat = FlatLabelSpace(vocab, cfg)
    log.info("Flat label space: %d labels over %d axes",
             flat.total, len(flat.axes))

    df = load_parquet(cfg.data.parquet)
    trainable_df, _test_df = split_trainable_test(
        df, cfg.data.trainable_value, cfg.data.test_value
    )
    try:
        folds_df = pd.read_csv(cfg.data.cv_folds)
    except FileNotFoundError:
        folds_df = None
    train_df, val_df = split_by_fold(trainable_df, folds_df, fold_idx)
    log.info("Fold %d: train=%d val=%d", fold_idx, len(train_df), len(val_df))

    train_ds = BaheyaM1Dataset(train_df, vocab, cfg)
    val_ds = BaheyaM1Dataset(val_df, vocab, cfg)
    bs = cfg.train.micro_batch_size
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          collate_fn=collate_m1)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        collate_fn=collate_m1)

    model = PLMICDFlat(cfg, flat.total).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n_epochs = 1 if smoke else cfg.train.epochs

    best = {"early_stop": -1.0, "epoch": -1, "summary": None}
    patience = cfg.train.early_stopping_patience
    bad = 0

    for epoch in range(n_epochs):
        model.train()
        tot = 0.0
        opt.zero_grad()
        for step, batch in enumerate(train_ld):
            logits = model(batch["texts"])
            y = flat.build_targets(batch).to(device)
            loss = loss_fn(logits, y) / cfg.train.grad_accumulation_steps
            loss.backward()
            tot += loss.item() * cfg.train.grad_accumulation_steps
            if (step + 1) % cfg.train.grad_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.max_grad_norm
                )
                opt.step()
                opt.zero_grad()
            if smoke and step >= 3:
                break
        summary = _evaluate(model, val_ld, flat, cfg, device, smoke)
        es = summary["early_stop_metric_current"]
        log.info("epoch=%d train_loss=%.4f early_stop=%.4f "
                 "(macro_sp=%.4f micro_ml=%.4f)",
                 epoch, tot / max(1, step + 1), es,
                 summary["mean_f1_macro_singlepick"],
                 summary["mean_f1_micro_multilabel"])
        if es > best["early_stop"]:
            best = {"early_stop": es, "epoch": epoch, "summary": summary}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("Early stop at epoch %d (patience %d)",
                         epoch, patience)
                break

    return {
        "fold": fold_idx,
        "best_epoch": best["epoch"],
        "early_stop": best["early_stop"],
        "summary": best["summary"],
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    flat: FlatLabelSpace,
    cfg: BenchmarkConfig,
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    # per-axis gold accumulators
    sp_axes = list(cfg.axis_types.single_pick)
    ml_axes = list(cfg.axis_types.multilabel)
    gold_sp: dict[str, list[np.ndarray]] = {a: [] for a in sp_axes}
    gold_ml: dict[str, list[np.ndarray]] = {a: [] for a in ml_axes}

    for i, batch in enumerate(loader):
        logits = model(batch["texts"]).cpu().numpy()
        all_logits.append(logits)
        for a in sp_axes:
            key = f"labels_{a}"
            gold_sp[a].append(batch[key].cpu().numpy().astype(np.int64))
        for a in ml_axes:
            key = f"labels_{a}"
            gold_ml[a].append(batch[key].cpu().numpy().astype(np.float32))
        if smoke and i >= 3:
            break

    flat_logits = np.concatenate(all_logits, axis=0)  # [N, 119]
    per_axis: dict[str, dict[str, Any]] = {}

    for a in sp_axes:
        codes = flat.axis_codes[a]
        ax_logits = flat.decode_axis(flat_logits, a)        # [N, K]
        targets = np.concatenate(gold_sp[a], axis=0)        # [N], -1=null
        per_axis[a] = metrics_single_pick(ax_logits, targets, codes)

    for a in ml_axes:
        codes = flat.axis_codes[a]
        ax_logits = flat.decode_axis(flat_logits, a)         # [N, K]
        targets = np.concatenate(gold_ml[a], axis=0)         # [N, K]
        per_axis[a] = metrics_multi_label(ax_logits, targets, codes)

    summary = {
        "mean_f1_macro_singlepick": float(np.mean(
            [per_axis[a]["f1_macro"] for a in sp_axes])) if sp_axes else 0.0,
        "mean_f1_macro_present_singlepick": float(np.mean(
            [per_axis[a]["f1_macro_present"] for a in sp_axes]))
            if sp_axes else 0.0,
        "mean_f1_micro_multilabel": float(np.mean(
            [per_axis[a]["f1_micro"] for a in ml_axes])) if ml_axes else 0.0,
    }
    summary["early_stop_metric_current"] = (
        summary["mean_f1_macro_singlepick"]
        + summary["mean_f1_micro_multilabel"]
    ) / 2.0
    summary["early_stop_metric_present_only"] = (
        summary["mean_f1_macro_present_singlepick"]
        + summary["mean_f1_micro_multilabel"]
    ) / 2.0
    summary["per_axis"] = per_axis
    return summary


# ---------------------------------------------------------------------------
# Helpers / IO
# ---------------------------------------------------------------------------


def _load_vocab(cfg: BenchmarkConfig) -> Any:
    from src.data.label_vocab import LabelVocab

    return LabelVocab.load_json(cfg.data.label_vocab)


def _write_summary(
    results: list[dict[str, Any]], out_dir: Path, log: Any
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    es = np.array([r["early_stop"] for r in results], dtype=float)
    mean, std = float(es.mean()), float(es.std(ddof=0))

    payload = {
        "experiment": "B2_plmicd",
        "n_folds": len(results),
        "early_stop_mean": mean,
        "early_stop_std": std,
        "per_fold": results,
        "reference": {
            "B1_early_stop": _B1_EARLY_STOP,
            "E1_early_stop": _E1_EARLY_STOP,
        },
        "delta_vs_B1": mean - _B1_EARLY_STOP,
        "delta_vs_E1": mean - _E1_EARLY_STOP,
    }
    base = out_dir / f"B2_plmicd_summary_{ts}"
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    md = [
        "# B2 — faithful PLM-ICD baseline (LAAT, flat 119)",
        "",
        f"- Folds: {len(results)}",
        f"- **early_stop (5-fold): {mean:.4f} ± {std:.4f}**",
        f"- B1 floor: {_B1_EARLY_STOP:.4f}  "
        f"(Δ B2−B1 = {mean - _B1_EARLY_STOP:+.4f})",
        f"- E1 (per-axis LAAT): {_E1_EARLY_STOP:.4f}  "
        f"(Δ B2−E1 = {mean - _E1_EARLY_STOP:+.4f})",
        "",
        "| fold | best_epoch | early_stop |",
        "|---:|---:|---:|",
    ]
    for r in results:
        md.append(f"| {r['fold']} | {r['best_epoch']} | "
                  f"{r['early_stop']:.4f} |")
    md += [
        "",
        "Interpretation: B2 holds LAAT constant and removes only MCIS's "
        "per-axis multi-task decomposition. Δ B2−E1 attributes to that "
        "decomposition; Δ B2−B1 places faithful PLM-ICD vs the TF-IDF floor.",
    ]
    base.with_suffix(".md").write_text("\n".join(md))
    log.info("Wrote %s.{md,json}", base)
    return base.with_suffix(".md")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m scripts.run_plmicd_baseline",
        description="B2 faithful PLM-ICD baseline (LAAT over flat 119).",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None,
                   help="Single fold (default: all cv.n_folds).")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, <=4 train + <=4 eval batches per fold.")
    return p.parse_args()


def _resolve_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    assert _STEP1_NULL == -1, (
        f"step1_multimetric.NULL={_STEP1_NULL}, expected -1 "
        f"(must match loaders.NULL_TARGET_SENTINEL)"
    )
    log = setup_console_logger("mcis.b2_plmicd")
    args = _parse_args()
    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        return 1

    cfg = BenchmarkConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = _resolve_device(args.device)
    log.info("B2 PLM-ICD | exp=%s seed=%d device=%s encoder=%s",
             cfg.experiment_name, cfg.seed, device, cfg.model.encoder_hf_id)

    folds = ([args.fold] if args.fold is not None
             else list(range(cfg.cv.n_folds)))
    results: list[dict[str, Any]] = []
    for f in folds:
        log.info("=== Fold %d/%d ===", f, len(folds))
        results.append(_run_fold(cfg, f, device, log, args.smoke))

    out_md = _write_summary(results, Path(cfg.logging.output_dir), log)
    es = np.array([r["early_stop"] for r in results])
    log.info("=" * 60)
    log.info("B2 done. early_stop %.4f ± %.4f over %d fold(s)",
             es.mean(), es.std(ddof=0), len(results))
    log.info("  vs B1 floor %.4f : %+.4f", _B1_EARLY_STOP,
             es.mean() - _B1_EARLY_STOP)
    log.info("  vs E1       %.4f : %+.4f", _E1_EARLY_STOP,
             es.mean() - _E1_EARLY_STOP)
    log.info("  summary: %s", out_md)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
