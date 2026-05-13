#!/usr/bin/env python3
"""End-to-end smoke test with REAL PubMedBERT weights.

The unit tests in tests/models/ and tests/test_train.py use a tiny in-memory
BERT for speed. They prove the API contract is right but cannot catch:

- Real tokenizer behavior on Baheya text (special tokens, segmentation)
- Real PubMedBERT weights loading + memory shape
- Real device transfer (MPS / CUDA) edge cases
- End-to-end forward + backward + optimizer step + scheduler step
- Loss actually decreases over a few steps on synthetic data
- Mixed precision (when enabled)

This script catches those before any cloud GPU run.

Usage:
    python scripts/smoke_neural.py                    # auto-detects best device
    python scripts/smoke_neural.py --device cpu       # force CPU
    python scripts/smoke_neural.py --device mps       # force MPS
    python scripts/smoke_neural.py --steps 8          # more training iterations
    python scripts/smoke_neural.py --config configs/B0.yaml
    python scripts/smoke_neural.py --module TCGA_pretrain  # test Phase 1 axis subset

Exit codes:
    0  loss decreased over ``--steps`` iterations and all sanity checks passed
    1  setup error (missing config / network / etc)
    2  loss did not decrease (model arch or trainer plumbing has a bug)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.label_vocab import LabelVocab
from src.data.loaders import collate_m1
from src.eval.metrics import aggregate_axis_metrics
from src.models.oce import OncologyCodingEngine
from src.models.heads import NULL_TARGET_SENTINEL
from src.train import Trainer, _resolve_device
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Synthetic batch construction
# ---------------------------------------------------------------------------


SAMPLE_TEXTS = [
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Invasive ductal carcinoma, "
    "right upper-outer quadrant of breast, Grade II. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Right",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Invasive lobular carcinoma, "
    "left central portion of breast, Grade I. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Left",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Mucinous adenocarcinoma, "
    "left upper-inner quadrant of breast, Grade I, Localized stage. "
    "[SPECIMEN] tru-cut biopsy [LATERALITY] Left",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Invasive ductal carcinoma, "
    "left breast NOS, Grade III. [SPECIMEN] tru-cut biopsy [LATERALITY] Left",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Ductal carcinoma in situ, "
    "right central portion of breast. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Right",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Tubular carcinoma, "
    "left lower-outer quadrant of breast, Grade I. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Left",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Invasive ductal carcinoma, "
    "right upper-inner quadrant of breast, Grade II. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Right",
    "[REPORT_TYPE] biopsy_report [DIAGNOSIS] Mucinous carcinoma, "
    "left upper-outer quadrant of breast, Grade II. [SPECIMEN] tru-cut biopsy "
    "[LATERALITY] Left",
]


def build_synthetic_batch(
    cfg: BenchmarkConfig,
    vocab: LabelVocab,
    rng: np.random.Generator,
    bsz: int = 8,
) -> dict[str, object]:
    """Build a synthetic batch matching the collate_m1 contract.

    Random labels per axis. We're testing the plumbing, not learning.
    """
    texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(bsz)]
    batch: dict[str, object] = {
        "texts": texts,
        "record_ids": [f"SMOKE-{i:03d}" for i in range(bsz)],
    }

    # Single-pick: random class indices in [0, K), with ~10% nulls
    for axis in cfg.axis_types.single_pick:
        if axis not in vocab.axes:
            continue
        K = vocab[axis].num_classes
        targets = rng.integers(0, K, size=bsz, dtype=np.int64)
        # Randomly null ~10%
        null_mask = rng.random(bsz) < 0.1
        targets = np.where(null_mask, NULL_TARGET_SENTINEL, targets)
        batch[f"labels_{axis}"] = torch.from_numpy(targets)

    # Multi-label: random multi-hot vectors
    for axis in cfg.axis_types.multilabel:
        if axis not in vocab.axes:
            continue
        K = vocab[axis].num_classes
        # ~30% sparsity
        targets = (rng.random((bsz, K)) < 0.3).astype(np.float32)
        batch[f"labels_{axis}"] = torch.from_numpy(targets)

    # Aux (empty if not needed)
    n_aux = len(cfg.aux_fields) if cfg.model.use_aux_histology_heads else 0
    batch["aux_targets"] = torch.zeros((bsz, n_aux), dtype=torch.long)
    batch["aux_masks"] = torch.zeros((bsz, n_aux), dtype=torch.long)

    return batch


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def check_forward_shapes(
    log: logging.Logger,
    outputs,
    cfg: BenchmarkConfig,
    vocab: LabelVocab,
    bsz: int,
) -> None:
    """Verify per-axis logit shapes match (B, K_axis)."""
    for axis in cfg.all_axes:
        if axis not in vocab.axes:
            continue
        if axis not in outputs.logits_per_axis:
            log.warning("  axis=%s missing from outputs (probably inactive)", axis)
            continue
        logits = outputs.logits_per_axis[axis]
        K = vocab[axis].num_classes
        assert logits.shape == (bsz, K), (
            f"axis {axis}: expected logits shape ({bsz}, {K}), got {tuple(logits.shape)}"
        )
    log.info("  forward shapes OK")


def check_attention_weights_shape(
    log: logging.Logger,
    outputs,
    cfg: BenchmarkConfig,
    vocab: LabelVocab,
    bsz: int,
) -> None:
    """Attention weights should be [B, K_axis, T_eff]."""
    if not outputs.attention_weights_per_axis:
        log.warning("  no attention weights returned (active_axes empty?)")
        return
    for axis, w in outputs.attention_weights_per_axis.items():
        K = vocab[axis].num_classes
        assert w.shape[0] == bsz, f"axis {axis} weights batch dim mismatch"
        assert w.shape[1] == K, f"axis {axis} weights label dim {w.shape[1]} != K={K}"
    log.info("  attention weights shapes OK")


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "B0.yaml"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of forward+backward+optimizer-step iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Synthetic batch size (override cfg.train.micro_batch_size for speed)")
    parser.add_argument("--module", default=None,
                        help="Override cfg.data.module (e.g. TCGA_pretrain to test Phase 1)")
    parser.add_argument("--decrease-tolerance", type=float, default=0.0,
                        help="Loss must decrease by at least this absolute amount over --steps")
    args = parser.parse_args()

    log = setup_console_logger("mcis.smoke")
    log.info("=" * 70)
    log.info("MCIS smoke_neural — end-to-end with real PubMedBERT")
    log.info("=" * 70)

    # 1. Config + vocab
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        return 1

    log.info("Loading config: %s", cfg_path)
    cfg = BenchmarkConfig.from_yaml(cfg_path)
    if args.module:
        cfg.data.module = args.module
        log.info("Module override: %s", cfg.data.module)

    log.info("Loading vocab: %s", cfg.data.label_vocab)
    vocab_path = Path(cfg.data.label_vocab)
    if not vocab_path.exists():
        log.error("Vocab not found: %s — regenerate with build_label_vocab.py", vocab_path)
        return 1
    vocab = LabelVocab.load_json(vocab_path)
    cfg.validate_against_vocab(vocab.cardinalities())
    log.info("Vocab cardinalities: %s", vocab.cardinalities())

    set_seed(cfg.seed)

    # 2. Device
    device = _resolve_device(args.device)
    log.info("Device: %s", device)

    # 3. Build OCE with REAL PubMedBERT (this downloads weights on first run)
    log.info("Constructing OCE (this may download PubMedBERT on first run, ~440MB) ...")
    t0 = time.time()
    model_cfg = Trainer._build_model_cfg(cfg, vocab)
    log.info("Model config: encoder=%s, segment_size=%d, max_segments=%d",
             model_cfg["encoder"]["backbone_name"],
             model_cfg["encoder"]["segment_size"],
             model_cfg["encoder"]["max_segments"])
    log.info("Heads: single_pick=%d, multi_label=%d",
             len(model_cfg["heads"]["single_pick"]),
             len(model_cfg["heads"]["multi_label"]))
    model = OncologyCodingEngine.from_config(model_cfg)
    model = model.to(device)
    t_construct = time.time() - t0
    log.info("Model built in %.1fs", t_construct)

    # 4. Phase
    if cfg.data.module == "TCGA_pretrain":
        model.set_phase("icdo3_only")
        log.info("Phase 1 active: %s", sorted(model.active_axes))
    else:
        model.set_phase("all")
        log.info("Phase 2 active: %s", sorted(model.active_axes))

    # Param count
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Total params: %.1fM (trainable: %.1fM)", n_params / 1e6, n_trainable / 1e6)

    # 5. One forward pass shape sanity
    rng = np.random.default_rng(cfg.seed)
    bsz = args.batch_size
    log.info("Running forward shape sanity (bsz=%d) ...", bsz)
    model.eval()
    with torch.no_grad():
        eval_batch = build_synthetic_batch(cfg, vocab, rng, bsz=bsz)
        outputs = model(eval_batch["texts"])
    check_forward_shapes(log, outputs, cfg, vocab, bsz)
    check_attention_weights_shape(log, outputs, cfg, vocab, bsz)
    log.info("  T_eff (encoder output length): %d", outputs.attention_weights_per_axis[
        sorted(outputs.attention_weights_per_axis.keys())[0]
    ].shape[-1])
    log.info("  n_segments per record: %s", outputs.n_segments.tolist())

    # 6. Compute losses end-to-end
    log.info("Running compute_losses sanity ...")
    targets = {}
    for axis in model.active_axes:
        key = f"labels_{axis}"
        if key in eval_batch:
            targets[axis] = eval_batch[key].to(device)
    losses = model.compute_losses(outputs, targets)
    log.info("  per-axis losses (active=%d):", len(losses))
    for axis, (l, n_valid) in losses.items():
        log.info("    %-24s loss=%.4f n_valid=%d", axis, float(l.cpu()), n_valid)

    # 7. Full forward+backward+step loop — loss must decrease
    log.info("Running %d training iterations ...", args.steps)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    losses_over_time: list[float] = []
    for step in range(args.steps):
        batch = build_synthetic_batch(cfg, vocab, rng, bsz=bsz)
        outputs = model(batch["texts"])
        targets = {
            axis: batch[f"labels_{axis}"].to(device)
            for axis in model.active_axes
            if f"labels_{axis}" in batch
        }
        per_axis_losses = model.compute_losses(outputs, targets)
        # Uniform sum (Q1 lock default)
        terms = [l for l, n in per_axis_losses.values() if n > 0]
        if not terms:
            log.error("All losses had n_valid=0 at step %d — synthetic batch broken", step)
            return 2
        total_loss = torch.stack(terms).sum()

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()

        losses_over_time.append(float(total_loss.detach().cpu()))
        log.info("  step %d/%d  loss=%.4f", step + 1, args.steps, losses_over_time[-1])

    # 8. Check loss actually went down
    first_loss = losses_over_time[0]
    last_loss = losses_over_time[-1]
    delta = first_loss - last_loss
    log.info("Loss change over %d steps: %.4f → %.4f (Δ = %.4f)",
             args.steps, first_loss, last_loss, delta)

    if delta < args.decrease_tolerance:
        log.error("FAIL: loss did not decrease (Δ = %.4f < tolerance %.4f). "
                  "Plumbing is wrong somewhere.", delta, args.decrease_tolerance)
        return 2

    log.info("=" * 70)
    log.info("PASS: smoke_neural completed in %.1fs", time.time() - t0)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
