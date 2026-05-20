"""
Pre-E1 Audit — Tokenizer Sanity + Adjacent Integrity Checks
============================================================
Run from MCIS repo root:  python scripts/pre_e1_audit.py

Goal: Before re-training E1 to investigate laterality regression,
confirm the entire data → tokenizer → encoder pipeline is sound.

Checks (each independent — partial failures still print useful info):
  H1. Tokenizer sanity         — does [LATERALITY] Left/Right tokenize distinctly?
  H2. Dataset field            — does BaheyaM1Dataset return text_section_tagged?
  H3. Truncation sanity        — does [LATERALITY] survive in real samples?
  H4. Label vocab integrity    — do label IDs in parquet match label_vocab.json?
  H5. Fold-0 split integrity   — any train/val row overlap?
  H6. Mask logic               — null axis label → mask=0 in __getitem__?

Output: prints a verdict block at the end. Copy entire stdout to chat.
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parents[1].name == "MCIS" else Path.cwd()
sys.path.insert(0, str(REPO))

# ─────────────────────────────────────────────────────────────────────────────
# Config — adjust if paths differ
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH       = REPO / "configs" / "B0.yaml"
PARQUET_PATH      = REPO / "data" / "frozen" / "m1_model_ready" / "m1_model_ready.parquet"
CV_FOLDS_PATH     = REPO / "data" / "frozen" / "m1_model_ready" / "cv_folds.csv"
LABEL_VOCAB_PATH  = REPO / "data" / "frozen" / "m1_model_ready" / "label_vocab.json"

# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 78
verdicts: dict[str, str] = {}

def banner(name: str) -> None:
    print(f"\n{SEP}\n{name}\n{SEP}")

def ok(check: str, msg: str = "") -> None:
    verdicts[check] = "PASS"
    print(f"  ✅ {check}: PASS  {msg}")

def fail(check: str, msg: str) -> None:
    verdicts[check] = "FAIL"
    print(f"  ❌ {check}: FAIL  {msg}")

def warn(check: str, msg: str) -> None:
    verdicts[check] = "WARN"
    print(f"  ⚠️  {check}: WARN  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Preamble: load shared resources once
# ─────────────────────────────────────────────────────────────────────────────
banner("PREAMBLE — load shared resources")

import yaml
import pandas as pd

print(f"  REPO root        : {REPO}")
print(f"  CONFIG_PATH      : {CONFIG_PATH}  exists={CONFIG_PATH.exists()}")
print(f"  PARQUET_PATH     : {PARQUET_PATH}  exists={PARQUET_PATH.exists()}")
print(f"  CV_FOLDS_PATH    : {CV_FOLDS_PATH}  exists={CV_FOLDS_PATH.exists()}")
print(f"  LABEL_VOCAB_PATH : {LABEL_VOCAB_PATH}  exists={LABEL_VOCAB_PATH.exists()}")

cfg = yaml.safe_load(CONFIG_PATH.read_text())
print(f"\n  cfg keys: {list(cfg.keys())}")

# Config key resolution per real B0.yaml (confirmed 2026-05-13)
model_section = cfg.get("model", {})
data_section = cfg.get("data", {})
ENCODER_NAME = model_section.get("encoder_hf_id", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
TEXT_FIELD = data_section.get("text_field", "text_section_tagged")
SEG_SIZE    = model_section.get("segment_size", 128)
MAX_SEGS    = model_section.get("max_segments", 20)

print(f"\n  Resolved ENCODER_NAME : {ENCODER_NAME}")
print(f"  Resolved TEXT_FIELD   : {TEXT_FIELD}")
print(f"  Resolved SEG_SIZE     : {SEG_SIZE}")
print(f"  Resolved MAX_SEGS     : {MAX_SEGS}")
print(f"  Effective max tokens  : {SEG_SIZE * MAX_SEGS}")

df = pd.read_parquet(PARQUET_PATH)
folds = pd.read_csv(CV_FOLDS_PATH)
label_vocab_raw = json.loads(LABEL_VOCAB_PATH.read_text())
# v3 schema: top-level "axes" key is {axis: [codes]}
vocab_axes = label_vocab_raw["axes"]
axis_meta = label_vocab_raw.get("axis_meta", {})
print(f"\n  parquet rows         : {len(df)}")
print(f"  parquet cols         : {len(df.columns)}")
print(f"  cv_folds rows        : {len(folds)}")
print(f"  cv_folds cols        : {list(folds.columns)}")
print(f"  label_vocab axes     : {list(vocab_axes.keys())}")

# Confirmed from data_card §10: parquet has 'record_id' as primary key.
fold_id_col = "record_id" if ("record_id" in folds.columns and "record_id" in df.columns) else None
if fold_id_col is None:
    # Fallback discovery
    for cand in ("record_id", "patient_id", "id", "row_id", "uid"):
        if cand in folds.columns and cand in df.columns:
            fold_id_col = cand
            break
print(f"  fold/df join key     : {fold_id_col!r}")


# ─────────────────────────────────────────────────────────────────────────────
# H1. Tokenizer sanity — the headline test
# ─────────────────────────────────────────────────────────────────────────────
banner("H1. TOKENIZER SANITY — does [LATERALITY] Left vs Right differ?")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    print(f"  Loaded tokenizer: {tok.__class__.__name__}")
    print(f"  do_lower_case   : {getattr(tok, 'do_lower_case', 'n/a')}")
    print(f"  vocab_size      : {tok.vocab_size}")
    print(f"  model_max_length: {tok.model_max_length}")

    # 1a. Isolated [LATERALITY] phrases
    print("\n  1a. Isolated tokenization of [LATERALITY] phrases:")
    phrases = [
        "[LATERALITY] Left",
        "[LATERALITY] Right",
        "[LATERALITY] Bilateral",
        "[LATERALITY] Lt.",
        "[LATERALITY] Rt.",
        "[LATERALITY] L",        # bare-code form, if any rows use this
        "[LATERALITY] R",
        "[LATERALITY] B",
    ]
    isolated_tokens: dict[str, list[str]] = {}
    isolated_ids: dict[str, list[int]] = {}
    for p in phrases:
        ids = tok(p, add_special_tokens=False)["input_ids"]
        pieces = tok.convert_ids_to_tokens(ids)
        isolated_tokens[p] = pieces
        isolated_ids[p] = ids
        print(f"    {p!r:30s} n={len(ids):2d}  pieces={pieces}")
        print(f"    {'':30s}        ids={ids}")

    # 1b. Check all pairs differ
    print("\n  1b. Pairwise distinguishability matrix (number of differing token IDs):")
    key_pairs = [
        ("[LATERALITY] Left", "[LATERALITY] Right"),
        ("[LATERALITY] Lt.",  "[LATERALITY] Rt."),
        ("[LATERALITY] L",    "[LATERALITY] R"),
        ("[LATERALITY] Left", "[LATERALITY] Bilateral"),
        ("[LATERALITY] Right","[LATERALITY] Bilateral"),
    ]
    h1_pass = True
    for a, b in key_pairs:
        sa, sb = set(isolated_ids[a]), set(isolated_ids[b])
        diff = sa.symmetric_difference(sb)
        flag = "✅" if diff else "❌"
        print(f"    {flag} {a!r:30s} vs {b!r:30s}  diff_ids={sorted(diff)}")
        if not diff:
            h1_pass = False
    if h1_pass:
        ok("H1a-pairs-distinguishable", "every pair of laterality forms tokenizes distinctly")
    else:
        fail("H1a-pairs-distinguishable", "see ❌ lines above — at least one pair collides")

    # 1c. Bracket survival
    sample_p = "[LATERALITY] Left"
    sample_pieces = isolated_tokens[sample_p]
    has_bracket = any("[" in t for t in sample_pieces)
    has_lateral = any("lateral" in t.lower() for t in sample_pieces)
    print(f"\n  1c. Bracket / 'lateral' subword survival in {sample_p!r}:")
    print(f"      brackets present in tokens : {has_bracket}")
    print(f"      'lateral' substring present: {has_lateral}")
    if has_lateral:
        ok("H1b-lateral-subword-survives", "")
    else:
        fail("H1b-lateral-subword-survives", "tokenizer mangles 'lateral' away — encoder can't see the tag")

    # 1d. Token cost of one [LATERALITY] tag (matters for truncation budget)
    n_tok_one_tag = len(isolated_ids["[LATERALITY] Left"])
    print(f"\n  1d. Token cost of one [LATERALITY] Left tag: {n_tok_one_tag}")
    if n_tok_one_tag > 12:
        warn("H1c-tag-token-cost", f"{n_tok_one_tag} tokens per tag — heavy")
    else:
        ok("H1c-tag-token-cost", f"{n_tok_one_tag} tokens — acceptable")

except Exception:
    fail("H1-tokenizer-load", "exception below")
    traceback.print_exc()
    tok = None


# ─────────────────────────────────────────────────────────────────────────────
# H2. Dataset returns the right text field
# ─────────────────────────────────────────────────────────────────────────────
banner("H2. DATASET FIELD — BaheyaM1Dataset returns text_section_tagged?")
ds_sample = None
ds = None
try:
    from src.data.loaders import BaheyaM1Dataset
    print(f"  Imported: src.data.loaders.BaheyaM1Dataset")
    import inspect
    sig = inspect.signature(BaheyaM1Dataset.__init__)
    print(f"  __init__ signature: {sig}")

    # Real signature: (df, vocab: LabelVocab, cfg: BenchmarkConfig, include_aux=None)
    from src.data.label_vocab import LabelVocab
    vocab = LabelVocab.load_json(LABEL_VOCAB_PATH)
    print(f"  Loaded LabelVocab: {len(vocab.axes)} axes")

    # BenchmarkConfig — import path may differ
    BenchmarkConfig = None
    for modpath in ("src.utils.config", "src.config", "src.benchmark_config"):
        try:
            mod = __import__(modpath, fromlist=["BenchmarkConfig"])
            if hasattr(mod, "BenchmarkConfig"):
                BenchmarkConfig = getattr(mod, "BenchmarkConfig")
                print(f"  Imported BenchmarkConfig from {modpath}")
                break
        except Exception as e:
            print(f"  import {modpath} failed: {type(e).__name__}: {e}")

    if BenchmarkConfig is None:
        warn("H2-config-import", "could not locate BenchmarkConfig — H2 + H7 skipped")
    else:
        cfg_obj = BenchmarkConfig.from_yaml(CONFIG_PATH)
        print(f"  Loaded BenchmarkConfig from {CONFIG_PATH.name}")

        # Filter df to fold-0 val (trainable + fold==0)
        fold0_val_df = df[(df["split"] == "trainable") & (df["fold"] == 0)].reset_index(drop=True)
        print(f"  fold-0 val slice: {len(fold0_val_df)} rows")

        ds = BaheyaM1Dataset(df=fold0_val_df, vocab=vocab, cfg=cfg_obj)
        print(f"  ✅ Dataset constructed. len = {len(ds)}")

        ds_sample = ds[0]
        sample_type = type(ds_sample).__name__
        print(f"  __getitem__(0) type: {sample_type}")
        if isinstance(ds_sample, dict):
            print(f"  __getitem__(0) keys: {list(ds_sample.keys())}")
            for k, v in ds_sample.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, str):
                    print(f"    {k}: str len={len(v)}")
                elif isinstance(v, (list, tuple)):
                    print(f"    {k}: {type(v).__name__} len={len(v)}")
                else:
                    print(f"    {k}: {type(v).__name__} = {v!r}"[:120])
        elif hasattr(ds_sample, "_fields"):  # NamedTuple
            print(f"  NamedTuple fields: {ds_sample._fields}")
        else:
            print(f"  raw repr: {ds_sample!r}"[:300])

        # Try to find/decode text
        text_out = None
        if isinstance(ds_sample, dict):
            for k in ("text", "raw_text", "input_text", TEXT_FIELD, "text_section_tagged"):
                if k in ds_sample and isinstance(ds_sample[k], str):
                    text_out = ds_sample[k]
                    print(f"  Found raw text at key {k!r}")
                    break
            if text_out is None and "input_ids" in ds_sample and tok is not None:
                iids = ds_sample["input_ids"]
                flat = iids.flatten().tolist() if hasattr(iids, "flatten") else list(iids)
                # Filter out PAD/CLS/SEP repetition noise for preview
                text_out = tok.decode(flat[:300], skip_special_tokens=False)
                print(f"  Decoded from input_ids (first 300 tok)")

        if text_out:
            preview = text_out[:500].replace("\n", " ⏎ ")
            print(f"  text preview: {preview!r}")
            has_tag = "[laterality]" in text_out.lower() or "[diagnosis]" in text_out.lower()
            if has_tag:
                ok("H2-text-tagged", "section tag present in dataset output")
            else:
                fail("H2-text-tagged", "no section tag in dataset text → wrong field?")
        else:
            warn("H2-text-key", "no text/input_ids key found — inspect sample keys above")

except ImportError as e:
    fail("H2-dataset-import", f"{e}")
    traceback.print_exc()
except Exception:
    fail("H2-dataset", "exception below")
    traceback.print_exc()

# Always do a raw parquet-based text inspection so H3 has a baseline even if H2 fails
print("\n  Parquet-direct text preview (independent of dataset class):")
if TEXT_FIELD in df.columns:
    raw0 = df[TEXT_FIELD].iloc[0]
    raw1 = df[TEXT_FIELD].iloc[1]
    print(f"    row 0 ({TEXT_FIELD}): {raw0[:300]!r}")
    print(f"    row 1 ({TEXT_FIELD}): {raw1[:300]!r}")
else:
    print(f"    ❌ column {TEXT_FIELD!r} not in parquet")


# ─────────────────────────────────────────────────────────────────────────────
# H3. Truncation — does [LATERALITY] survive after tokenization in real rows?
# ─────────────────────────────────────────────────────────────────────────────
banner("H3. TRUNCATION — [LATERALITY] survives in real fold-0 samples?")
try:
    if tok is None:
        warn("H3-truncation", "tokenizer not available, skipped")
    elif TEXT_FIELD not in df.columns:
        warn("H3-truncation", f"text field {TEXT_FIELD!r} not in parquet columns")
    elif fold_id_col is None:
        warn("H3-truncation", "no fold join key")
    elif "fold" not in folds.columns:
        warn("H3-truncation", "no 'fold' column in cv_folds.csv")
    else:
        val_ids = set(folds.loc[folds["fold"] == 0, fold_id_col].tolist())
        val_df = df[df[fold_id_col].isin(val_ids)].copy()
        print(f"  evaluating {len(val_df)} fold-0 val rows")

        max_eff_tokens = SEG_SIZE * MAX_SEGS

        # 3a. Show one real example per laterality value (if column exists)
        print("\n  3a. Real [LATERALITY] text snippets per laterality value:")
        if "icdo3_laterality" in val_df.columns:
            for lat_val in ["L", "R", "B"]:
                sub = val_df[val_df["icdo3_laterality"] == lat_val].head(1)
                if not sub.empty:
                    text = sub[TEXT_FIELD].iloc[0]
                    # Find [LATERALITY] line
                    idx = text.find("[LATERALITY]")
                    if idx >= 0:
                        snippet = text[idx:idx + 80].replace("\n", " ⏎ ")
                        print(f"    label={lat_val}  '{snippet}'")
                    else:
                        print(f"    label={lat_val}  NO [LATERALITY] tag found in text!")
                else:
                    print(f"    label={lat_val}  no fold-0 val rows with this label")

        # 3b. Survival statistics
        print("\n  3b. Truncation-survival breakdown:")
        survival_counter = Counter()
        truncated_examples = []
        no_tag_examples = []
        tok_lens: list[int] = []
        for _, row in val_df.iterrows():
            text = row[TEXT_FIELD]
            if not isinstance(text, str):
                survival_counter["non_string"] += 1
                continue
            if "[LATERALITY]" not in text:
                survival_counter["no_tag_in_text"] += 1
                if len(no_tag_examples) < 2:
                    no_tag_examples.append(text[:200])
                continue
            ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
            n_tok = len(ids)
            tok_lens.append(n_tok)
            if n_tok > max_eff_tokens:
                survival_counter["exceeds_budget"] += 1
                if len(truncated_examples) < 2:
                    truncated_examples.append((n_tok, text[-200:]))
            else:
                survival_counter["fits_in_budget"] += 1
            ids_trunc = ids[:max_eff_tokens]
            pieces = tok.convert_ids_to_tokens(ids_trunc)
            if any("lateral" in p.lower() for p in pieces):
                survival_counter["lateral_subword_in_window"] += 1
            else:
                survival_counter["lateral_subword_DROPPED"] += 1

        for k, v in survival_counter.items():
            print(f"    {k:35s} {v}")
        if tok_lens:
            import statistics
            print(f"\n    Token-length stats (val rows w/ tag): "
                  f"mean={statistics.mean(tok_lens):.0f} "
                  f"p50={statistics.median(tok_lens):.0f} "
                  f"p95={sorted(tok_lens)[int(0.95*len(tok_lens))]} "
                  f"max={max(tok_lens)} "
                  f"budget={max_eff_tokens}")
        if no_tag_examples:
            print(f"\n  Sample rows with NO [LATERALITY] tag:")
            for ex in no_tag_examples:
                print(f"    {ex!r}")
        if truncated_examples:
            print(f"\n  Sample rows that exceed token budget (tail shown):")
            for n_tok, tail in truncated_examples:
                print(f"    n_tok={n_tok}  tail={tail!r}")

        dropped = survival_counter["lateral_subword_DROPPED"]
        if dropped == 0:
            ok("H3-laterality-survives", "every fold-0 val row that has [LATERALITY] keeps 'lateral' subword in encoder window")
        else:
            fail("H3-laterality-survives", f"{dropped} rows lose 'lateral' subword inside encoder window")

except Exception:
    fail("H3-truncation", "exception below")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# H4. Label vocab — parquet label columns consistent with label_vocab.json?
# ─────────────────────────────────────────────────────────────────────────────
banner("H4. LABEL VOCAB — parquet ↔ label_vocab.json consistency")
try:
    # Use the project's canonical normalizer to handle ndarray/list/scalar/null uniformly.
    from src.data.label_vocab import _iter_codes, _normalize_code

    axes_to_check = [
        "icdo3_topography", "icdo3_morphology", "icdo3_behavior", "icdo3_grade",
        "icdo3_laterality", "icd11_stem", "icd11_ext_laterality", "icd11_ext_grading",
        "icd11_ext_anatomy", "icd11_ext_histopath",
    ]
    all_clean = True
    test_only = label_vocab_raw.get("test_only_labels", {})
    trainable_only = df[df["split"] == "trainable"] if "split" in df.columns else df

    for axis in axes_to_check:
        if axis not in df.columns:
            print(f"  {axis:25s} ❌ column missing in parquet")
            all_clean = False
            continue
        if axis not in vocab_axes:
            print(f"  {axis:25s} ❌ axis missing in label_vocab.axes")
            all_clean = False
            continue

        meta = axis_meta.get(axis, {})
        is_multi = meta.get("type") == "multi_label"
        delim = meta.get("delimiter", "|")

        vocab_codes_norm = {_normalize_code(c) for c in vocab_axes[axis]}

        seen: set[str] = set()
        for v in trainable_only[axis]:
            for tok_code in _iter_codes(v, multilabel=is_multi, delimiter=delim):
                seen.add(tok_code)

        extra_in_data = seen - vocab_codes_norm
        missing_in_data = vocab_codes_norm - seen
        K = len(vocab_codes_norm)
        n_extra = len(extra_in_data)
        flag = "✅" if n_extra == 0 else "❌"
        type_str = "multi " if is_multi else "single"
        print(f"  {axis:25s} {type_str} K={K:3d}  seen_trainable={len(seen):3d}  "
              f"not_in_vocab={n_extra}  not_seen_in_data={len(missing_in_data)}  {flag}")
        if n_extra > 0:
            print(f"    extras: {sorted(extra_in_data)[:10]}")
            all_clean = False

    # Test-only check: rows in test split with codes flagged as test_only
    print(f"\n  Test-only labels declared in vocab:")
    for axis, info in test_only.items():
        print(f"    {axis}: codes={info.get('codes')}  test_rows_affected={info.get('test_rows_affected')}")

    if all_clean:
        ok("H4-vocab-consistent", "every trainable-split code is in label_vocab")
    else:
        fail("H4-vocab-consistent", "see per-axis lines above")

except Exception:
    fail("H4-vocab", "exception below")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# H5. Fold-0 split integrity
# ─────────────────────────────────────────────────────────────────────────────
banner("H5. FOLD-0 SPLIT INTEGRITY — sizes match handoff, no overlap")
try:
    # Per data_card §10, parquet has 'split' and 'fold' columns.
    # cv_folds.csv is auxiliary (record_id → fold for trainable only).
    if "fold" not in df.columns or "split" not in df.columns:
        warn("H5-parquet-cols", "parquet missing 'fold' or 'split' column — check build_m1_model_ready.py")
    else:
        train_rows = df[df["split"] == "trainable"]
        test_rows = df[df["split"] == "test"]
        print(f"  parquet trainable n   : {len(train_rows)}  (data_card: 2759)")
        print(f"  parquet test n        : {len(test_rows)}   (data_card: 296)")

        f0_val = train_rows[train_rows["fold"] == 0]
        f0_train = train_rows[train_rows["fold"].isin([1, 2, 3, 4])]
        print(f"  fold-0 val n          : {len(f0_val)}   (data_card: 550)")
        print(f"  fold-0 train n        : {len(f0_train)} (handoff: 2209)")

        # Overlap check by record_id
        val_ids_set = set(f0_val["record_id"])
        train_ids_set = set(f0_train["record_id"])
        test_ids_set = set(test_rows["record_id"])
        overlap_tv = val_ids_set & train_ids_set
        overlap_test = (val_ids_set | train_ids_set) & test_ids_set
        if overlap_tv:
            fail("H5-overlap-train-val", f"{len(overlap_tv)} ids in both")
        else:
            ok("H5-overlap-train-val", "")
        if overlap_test:
            fail("H5-overlap-trainable-test", f"{len(overlap_test)} ids in both")
        else:
            ok("H5-overlap-trainable-test", "")

        # Cross-check cv_folds.csv agrees with parquet
        if fold_id_col and "fold" in folds.columns:
            disagreements = 0
            csv_map = dict(zip(folds[fold_id_col], folds["fold"]))
            for _, r in train_rows.iterrows():
                csv_fold = csv_map.get(r["record_id"])
                if csv_fold is not None and csv_fold != r["fold"]:
                    disagreements += 1
            if disagreements == 0:
                ok("H5-cv_folds-csv-matches-parquet", "")
            else:
                fail("H5-cv_folds-csv-matches-parquet", f"{disagreements} record_ids disagree")

        expected = {"val": 550, "train": 2209, "test": 296, "trainable_total": 2759}
        actual = {
            "val": len(f0_val),
            "train": len(f0_train),
            "test": len(test_rows),
            "trainable_total": len(train_rows),
        }
        if actual == expected:
            ok("H5-sizes-match-handoff", "")
        else:
            warn("H5-sizes-match-handoff", f"actual={actual}  expected={expected}")

        # Duplicate-group leakage check (data_card §8 acceptance bar)
        if "duplicate_group_id" in df.columns:
            train_groups = set(f0_train[f0_train["duplicate_group_id"] >= 0]["duplicate_group_id"])
            val_groups = set(f0_val[f0_val["duplicate_group_id"] >= 0]["duplicate_group_id"])
            grp_overlap = train_groups & val_groups
            if grp_overlap:
                fail("H5-no-dup-group-leakage", f"{len(grp_overlap)} duplicate-groups span train+val")
            else:
                ok("H5-no-dup-group-leakage", "")

except Exception:
    fail("H5-fold-split", "exception below")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# H6. Mask logic — null axis label → mask=0 in dataset __getitem__
# ─────────────────────────────────────────────────────────────────────────────
banner("H6. MASK LOGIC — null axis → mask=0 in __getitem__")
try:
    if ds_sample is None or not isinstance(ds_sample, dict):
        warn("H6-mask", "no ds_sample dict — skipped")
    else:
        mask_keys = [k for k in ds_sample.keys() if "mask" in k.lower()]
        print(f"  mask-related keys: {mask_keys}")
        if not mask_keys:
            warn("H6-mask-keys", "no mask keys — masking might happen in collator/loss; check src/eval/metrics.py or training loop")
        else:
            for k in mask_keys:
                v = ds_sample[k]
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} value={v.tolist() if v.numel() < 50 else '...'}")
                else:
                    print(f"    {k}: {v}")
            ok("H6-mask-present", "mask keys exist; verify semantics in H7")

except Exception:
    fail("H6-mask", "exception below")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# H7. Per-axis label decode — pick a row with known L vs known R, decode through
#     dataset, confirm the integer label and the one-hot vector match the source
# ─────────────────────────────────────────────────────────────────────────────
banner("H7. LABEL ROUND-TRIP — does dataset encode laterality L/R correctly?")
try:
    if ds is None:
        warn("H7-label-roundtrip", "no dataset available — skipped")
    elif "icdo3_laterality" not in fold0_val_df.columns:
        warn("H7-label-roundtrip", "no icdo3_laterality col in slice")
    else:
        # Per vocab, icdo3_laterality codes ordering: ['L', 'R', 'B']
        # So index 0=L, 1=R, 2=B
        lat_codes = vocab_axes["icdo3_laterality"]
        print(f"  vocab axes ordering for icdo3_laterality: {lat_codes}")
        expected_idx = {c: i for i, c in enumerate(lat_codes)}
        print(f"  expected: L→{expected_idx.get('L')}, R→{expected_idx.get('R')}, B→{expected_idx.get('B')}")

        # Pick first row of each laterality value
        for label_val in ["L", "R", "B"]:
            cand = fold0_val_df[fold0_val_df["icdo3_laterality"] == label_val]
            if cand.empty:
                print(f"  label={label_val}: no rows in fold-0 val")
                continue
            # Use the positional index in fold0_val_df (since ds is built on it w/ reset_index)
            pos_idx = cand.index[0]
            row = fold0_val_df.iloc[pos_idx]
            sample = ds[pos_idx]
            print(f"\n  row {pos_idx} (record_id={row['record_id']}, parquet_laterality={label_val!r}):")

            # Try common label key patterns
            label_keys = [k for k in sample.keys() if "icdo3_laterality" in k or k == "icdo3_laterality"]
            print(f"    keys matching icdo3_laterality: {label_keys}")
            for k in label_keys:
                v = sample[k]
                if hasattr(v, "item") and getattr(v, "numel", lambda: 1)() == 1:
                    print(f"    {k}: scalar = {v.item()}")
                elif hasattr(v, "shape"):
                    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} value={v.tolist()}")
                else:
                    print(f"    {k}: {v}")

            # Also do icd11_ext_laterality for comparison
            ext_keys = [k for k in sample.keys() if "icd11_ext_laterality" in k]
            for k in ext_keys:
                v = sample[k]
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} value={v.tolist()}")
                else:
                    print(f"    {k}: {v}")

            # Raw value from parquet for sanity
            print(f"    parquet icd11_ext_laterality raw: {row.get('icd11_ext_laterality')!r}")

        ok("H7-label-roundtrip-printed", "see per-row decode above; eyeball that L→0, R→1, B→2")

except Exception:
    fail("H7-label-roundtrip", "exception below")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
banner("FINAL VERDICT")
n_pass = sum(1 for v in verdicts.values() if v == "PASS")
n_fail = sum(1 for v in verdicts.values() if v == "FAIL")
n_warn = sum(1 for v in verdicts.values() if v == "WARN")
print(f"  PASS: {n_pass}   FAIL: {n_fail}   WARN: {n_warn}\n")
for check, status in verdicts.items():
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}[status]
    print(f"  {icon} {status:4s}  {check}")

if n_fail:
    print("\n  ❌ One or more critical checks failed. DO NOT proceed to E1 retrain until resolved.")
    sys.exit(1)
elif n_warn:
    print("\n  ⚠️  Soft warnings only. Review each before proceeding.")
    sys.exit(0)
else:
    print("\n  ✅ All checks clean. Pipeline basis is sound — proceed with next-step diagnosis.")
    sys.exit(0)
