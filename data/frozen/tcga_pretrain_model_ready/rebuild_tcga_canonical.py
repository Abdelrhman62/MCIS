"""
rebuild_tcga_canonical.py

Reads the previously-built TCGA pretrain + TCGA-BRCA parquets and rewrites them
in the canonical unified schema (per MCIS_Dataset_Unification_Handoff §2):

  - icdo3_laterality: X-codes -> L/R/B (raw preserved in icdo3_laterality_raw)
  - icdo3_grade: float -> string ('1'/'2'/'3'/'4'/None)
  - icd11_ext_{anatomy,histopath,laterality,grading}: parsed from
    icd11_full_postcoord into list[str] (multi-label; empty list = no positives)
  - Bookkeeping columns added: fold, batch, template_flag, is_cancer_primary,
    coding_source, data_quality_flag
  - Aux columns added (None for TCGA): nuclear_grade, tubular_score,
    mitosis_score, lvi, dcis_in_specimen

Vocab rebuild: icdo3_laterality keyed on L/R/B (was XK8G/XK9K/XK9J).
Other axes retained from existing vocab (codes unchanged, only encoding flipped).

Outputs:
  /home/claude/data/frozen/tcga_pretrain_model_ready/
    tcga_pretrain_model_ready.parquet
    label_vocab.json
    splits.csv
    data_card.md
    CHECKSUMS.txt
  /home/claude/data/frozen/tcga_brca_external_model_ready/
    tcga_brca_external_model_ready.parquet
    data_card.md
    CHECKSUMS.txt
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path('/home/claude/tcga_src/TCGA model ready')
OUT_PRE = Path('/home/claude/data/frozen/tcga_pretrain_model_ready')
OUT_BRCA = Path('/home/claude/data/frozen/tcga_brca_external_model_ready')
BAHEYA_PARQUET = Path('/mnt/user-data/uploads/m1_model_ready.parquet')

OUT_PRE.mkdir(parents=True, exist_ok=True)
OUT_BRCA.mkdir(parents=True, exist_ok=True)

# Canonical column order (per handoff §2.1)
CANONICAL_COLS = [
    # Bookkeeping
    'record_id', 'patient_id', 'dataset', 'split', 'fold', 'duplicate_group_id', 'batch',
    'template_flag', 'is_cancer_primary', 'coding_source', 'data_quality_flag', 'cancer_type',
    'text_length_words', 'text_length_bert_tokens',
    # Model input
    'text_section_tagged', 'text_concatenated',
    # ICD-O-3 single-pick
    'icdo3_topography', 'icdo3_morphology', 'icdo3_behavior', 'icdo3_grade', 'icdo3_laterality',
    # ICD-11 single-pick
    'icd11_stem',
    # ICD-11 multi-label
    'icd11_ext_anatomy', 'icd11_ext_histopath', 'icd11_ext_laterality', 'icd11_ext_grading',
    # Raw bookkeeping
    'icdo3_grade_raw', 'icdo3_laterality_raw',
    # Aux (Baheya-only; None for TCGA)
    'nuclear_grade', 'tubular_score', 'mitosis_score', 'lvi', 'dcis_in_specimen',
    # Reference (TCGA-only)
    'icd11_full_postcoord',
]

# ---------- Normalization helpers ----------

LAT_X_TO_LRB = {'XK8G': 'L', 'XK9K': 'R', 'XK9J': 'B'}


def norm_lat_lrb(x):
    """X-code -> L/R/B. None for missing or unknown."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return LAT_X_TO_LRB.get(s, None)


def norm_grade_str(x):
    """float/int/str -> '1'|'2'|'3'|'4'|None. Anything else -> None."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        i = int(float(x))
    except (TypeError, ValueError):
        return None
    if i in (1, 2, 3, 4):
        return str(i)
    return None


def parse_postcoord_axes(s):
    """
    Parse 'STEM&XAxxxx&XHxxxx&XKxxxx&XSxxxx&XSyyyy' into a dict of axis -> list[str].
    Stem (first &-separated part) is dropped here (already captured in icd11_stem column).
    """
    out = {'XA': [], 'XH': [], 'XK': [], 'XS': []}
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return out
    parts = str(s).split('&')
    for p in parts[1:]:  # skip stem
        p = p.strip()
        if not p:
            continue
        pfx = p[:2]
        if pfx in out:
            out[pfx].append(p)
    return out


# ---------- Apply canonical-schema transform ----------

def transform(df_in: pd.DataFrame, *, batch_default=-1) -> pd.DataFrame:
    df = df_in.copy()
    n = len(df)

    # Preserve raw values BEFORE rewriting
    df['icdo3_laterality_raw'] = df['icdo3_laterality'].copy()
    # icdo3_grade_raw already exists from previous build

    # Normalize laterality to L/R/B
    df['icdo3_laterality'] = df['icdo3_laterality_raw'].apply(norm_lat_lrb)

    # Normalize grade to string
    df['icdo3_grade'] = df['icdo3_grade'].apply(norm_grade_str)

    # Parse postcoord -> 4 multi-label axes
    parsed = df['icd11_full_postcoord'].apply(parse_postcoord_axes)
    df['icd11_ext_anatomy']    = parsed.apply(lambda d: d['XA'])
    df['icd11_ext_histopath']  = parsed.apply(lambda d: d['XH'])
    df['icd11_ext_laterality'] = parsed.apply(lambda d: d['XK'])
    df['icd11_ext_grading']    = parsed.apply(lambda d: d['XS'])

    # Bookkeeping defaults
    df['fold'] = -1  # TCGA has no CV folds (single 70/15/15 split)
    df['batch'] = batch_default
    df['template_flag'] = False
    df['is_cancer_primary'] = True  # TCGA is all cancer reports
    df['coding_source'] = 'gdc_curated'
    df['data_quality_flag'] = None

    # Aux fields (Baheya-only) -> None
    for c in ['nuclear_grade', 'tubular_score', 'mitosis_score', 'lvi', 'dcis_in_specimen']:
        df[c] = None

    # Reorder columns
    out = df[[c for c in CANONICAL_COLS if c in df.columns]].copy()

    # Sanity dtype enforcement
    for col in ['record_id', 'patient_id', 'dataset', 'split', 'duplicate_group_id',
                'coding_source', 'cancer_type', 'text_section_tagged', 'text_concatenated',
                'icdo3_topography', 'icdo3_morphology', 'icdo3_behavior',
                'icdo3_grade', 'icdo3_laterality', 'icd11_stem',
                'icdo3_grade_raw', 'icdo3_laterality_raw',
                'lvi', 'dcis_in_specimen', 'data_quality_flag', 'icd11_full_postcoord']:
        if col in out.columns:
            out[col] = out[col].astype('object')

    out['fold'] = out['fold'].astype('int64')
    out['batch'] = out['batch'].astype('int64')
    out['template_flag'] = out['template_flag'].astype('bool')
    out['is_cancer_primary'] = out['is_cancer_primary'].astype('bool')
    out['text_length_words'] = out['text_length_words'].astype('int64')
    out['text_length_bert_tokens'] = out['text_length_bert_tokens'].astype('int64')
    return out


# ---------- Load + transform ----------

print('Loading source parquets...')
pre_in = pd.read_parquet(SRC / 'tcga_pretrain_model_ready.parquet')
brca_in = pd.read_parquet(SRC / 'tcga_brca_external_model_ready.parquet')

# BRCA's icdo3_grade column is dtype=object but all None; transform handles it
print(f'  pretrain: {pre_in.shape}, brca: {brca_in.shape}')

print('Transforming pretrain...')
pre = transform(pre_in)
print('Transforming BRCA...')
brca = transform(brca_in)

print(f'  pretrain canonical: {pre.shape}')
print(f'  brca canonical:     {brca.shape}')


# ---------- Verification (must pass before write) ----------

def verify_canonical(df, name):
    issues = []
    missing = set(CANONICAL_COLS) - set(df.columns)
    if missing:
        issues.append(f'missing cols: {sorted(missing)}')

    # Laterality must be L/R/B/None
    bad_lat = set(df['icdo3_laterality'].dropna().unique()) - {'L', 'R', 'B'}
    if bad_lat:
        issues.append(f'icdo3_laterality has bad values: {bad_lat}')

    # Grade must be '1'/'2'/'3'/'4'/None (string)
    bad_grade = set(df['icdo3_grade'].dropna().astype(str).unique()) - {'1', '2', '3', '4'}
    if bad_grade:
        issues.append(f'icdo3_grade has bad values: {bad_grade}')

    # Multi-label axes must be lists
    for col in ['icd11_ext_anatomy', 'icd11_ext_histopath', 'icd11_ext_laterality', 'icd11_ext_grading']:
        sample_types = df[col].apply(type).value_counts()
        if list not in sample_types.index:
            issues.append(f'{col} not list-typed; types: {sample_types.to_dict()}')

    # XK extension codes must be valid X-codes (XK8G/XK9K/XK9J — none lost in conversion)
    flat_xk = [c for lst in df['icd11_ext_laterality'] for c in lst]
    bad_xk = set(flat_xk) - {'XK8G', 'XK9K', 'XK9J'}
    if bad_xk:
        issues.append(f'icd11_ext_laterality unexpected codes: {bad_xk}')

    if issues:
        print(f'  ❌ {name}: {len(issues)} issue(s)')
        for i in issues:
            print(f'     - {i}')
        return False
    print(f'  ✓ {name}: schema verified')
    return True


print('\nSchema verification:')
ok_pre = verify_canonical(pre, 'pretrain')
ok_brca = verify_canonical(brca, 'brca')
assert ok_pre and ok_brca, 'Schema verification failed'

# Cross-check: laterality round-trip preserves data
n_xcode_pre = pre_in['icdo3_laterality'].notna().sum()
n_lrb_pre = pre['icdo3_laterality'].notna().sum()
assert n_xcode_pre == n_lrb_pre, f'Laterality lost rows: {n_xcode_pre} -> {n_lrb_pre}'
n_xcode_ext_pre = sum(len(lst) for lst in pre['icd11_ext_laterality'])
assert n_xcode_pre == n_xcode_ext_pre, f'XK code count mismatch: orig={n_xcode_pre}, ext={n_xcode_ext_pre}'

n_xcode_brca = brca_in['icdo3_laterality'].notna().sum()
n_lrb_brca = brca['icdo3_laterality'].notna().sum()
assert n_xcode_brca == n_lrb_brca
n_xcode_ext_brca = sum(len(lst) for lst in brca['icd11_ext_laterality'])
assert n_xcode_brca == n_xcode_ext_brca

print(f'  ✓ Laterality round-trip preserved ({n_xcode_pre} pre, {n_xcode_brca} brca)')


# ---------- Vocab rebuild from pretrain_train ----------

print('\nRebuilding vocab from pretrain_train...')
train = pre[pre['split'] == 'pretrain_train']

def build_axis_vocab(series):
    counts = series.dropna().value_counts()
    return {code: i for i, code in enumerate(counts.index.tolist())}

new_vocab = {
    'version': 2,  # bumped from previous v1
    'dataset': 'tcga_pretrain',
    'built_from_split': 'pretrain_train',
    'axes': {
        'icdo3_topography': build_axis_vocab(train['icdo3_topography']),
        'icdo3_morphology': build_axis_vocab(train['icdo3_morphology']),
        'icdo3_behavior':   build_axis_vocab(train['icdo3_behavior']),
        'icdo3_grade':      build_axis_vocab(train['icdo3_grade']),
        'icdo3_laterality': build_axis_vocab(train['icdo3_laterality']),
    },
    'axis_meta': {
        'icdo3_topography': {'type': 'single_pick', 'null_policy': 'mask_in_loss'},
        'icdo3_morphology': {'type': 'single_pick', 'null_policy': 'mask_in_loss'},
        'icdo3_behavior':   {'type': 'single_pick', 'null_policy': 'mask_in_loss'},
        'icdo3_grade':      {'type': 'single_pick', 'null_policy': 'mask_in_loss', 'encoding': 'string'},
        'icdo3_laterality': {'type': 'single_pick', 'null_policy': 'mask_in_loss',
                             'encoding': 'L_R_B', 'mapping_from_raw': LAT_X_TO_LRB},
    },
    'icd11_policy': 'silver_never_label',
    'seed': 42,
    'built_at': datetime.now(timezone.utc).isoformat(),
    'changelog_v1_to_v2': [
        'icdo3_laterality re-keyed from XK8G/XK9K/XK9J to L/R/B (Baheya scheme)',
        'icdo3_grade encoding changed from int to string',
        'ICD-11 extension axes added to row schema as list[str]; not present in vocab (multi-label, no head built here)',
    ],
}

print('  vocab cardinalities:')
for axis, codes in new_vocab['axes'].items():
    print(f"    {axis}: K={len(codes)}")


# ---------- Write pretrain artifact ----------

print('\nWriting tcga_pretrain artifact...')
pre_path = OUT_PRE / 'tcga_pretrain_model_ready.parquet'

# Use pyarrow schema with explicit list<string> for extension axes
pa_schema = pa.schema([
    pa.field('record_id', pa.string()),
    pa.field('patient_id', pa.string()),
    pa.field('dataset', pa.string()),
    pa.field('split', pa.string()),
    pa.field('fold', pa.int64()),
    pa.field('duplicate_group_id', pa.string()),
    pa.field('batch', pa.int64()),
    pa.field('template_flag', pa.bool_()),
    pa.field('is_cancer_primary', pa.bool_()),
    pa.field('coding_source', pa.string()),
    pa.field('data_quality_flag', pa.string()),
    pa.field('cancer_type', pa.string()),
    pa.field('text_length_words', pa.int64()),
    pa.field('text_length_bert_tokens', pa.int64()),
    pa.field('text_section_tagged', pa.string()),
    pa.field('text_concatenated', pa.string()),
    pa.field('icdo3_topography', pa.string()),
    pa.field('icdo3_morphology', pa.string()),
    pa.field('icdo3_behavior', pa.string()),
    pa.field('icdo3_grade', pa.string()),
    pa.field('icdo3_laterality', pa.string()),
    pa.field('icd11_stem', pa.string()),
    pa.field('icd11_ext_anatomy', pa.list_(pa.string())),
    pa.field('icd11_ext_histopath', pa.list_(pa.string())),
    pa.field('icd11_ext_laterality', pa.list_(pa.string())),
    pa.field('icd11_ext_grading', pa.list_(pa.string())),
    pa.field('icdo3_grade_raw', pa.string()),
    pa.field('icdo3_laterality_raw', pa.string()),
    pa.field('nuclear_grade', pa.int64()),
    pa.field('tubular_score', pa.int64()),
    pa.field('mitosis_score', pa.int64()),
    pa.field('lvi', pa.string()),
    pa.field('dcis_in_specimen', pa.string()),
    pa.field('icd11_full_postcoord', pa.string()),
])

# Reorder pre to match schema exactly
pre_ordered = pre[[f.name for f in pa_schema]]
table_pre = pa.Table.from_pandas(pre_ordered, schema=pa_schema, preserve_index=False)
pq.write_table(table_pre, pre_path, compression='snappy')
print(f'  wrote {pre_path} ({pre_path.stat().st_size / 1e6:.2f} MB)')

# Vocab + splits
with open(OUT_PRE / 'label_vocab.json', 'w') as f:
    json.dump(new_vocab, f, indent=2)
print(f'  wrote {OUT_PRE / "label_vocab.json"}')

# splits.csv (regenerate from final df, not just copy — guaranteed in sync)
splits_df = pre[['record_id', 'split']].copy()
splits_df.to_csv(OUT_PRE / 'splits.csv', index=False)
print(f'  wrote {OUT_PRE / "splits.csv"} ({len(splits_df)} rows)')


# ---------- Write BRCA artifact ----------

print('\nWriting tcga_brca_external artifact...')
brca_path = OUT_BRCA / 'tcga_brca_external_model_ready.parquet'
brca_ordered = brca[[f.name for f in pa_schema]]
table_brca = pa.Table.from_pandas(brca_ordered, schema=pa_schema, preserve_index=False)
pq.write_table(table_brca, brca_path, compression='snappy')
print(f'  wrote {brca_path} ({brca_path.stat().st_size / 1e6:.2f} MB)')


# ---------- Baheya-vocab coverage analysis for BRCA data card §4.2 ----------

print('\nComputing Baheya-vocab coverage for BRCA (data card §4.2)...')
bah = pd.read_parquet(BAHEYA_PARQUET)

bah_topo = set(bah['icdo3_topography'].dropna().unique())
bah_morph = set(bah['icdo3_morphology'].dropna().unique())
bah_lat = set(bah['icdo3_laterality'].dropna().unique())
bah_grade = set(bah['icdo3_grade'].dropna().unique())
bah_behavior = set(bah['icdo3_behavior'].dropna().unique())

brca_topo = set(brca['icdo3_topography'].dropna().unique())
brca_morph = set(brca['icdo3_morphology'].dropna().unique())
brca_lat = set(brca['icdo3_laterality'].dropna().unique())
brca_grade = set(brca['icdo3_grade'].dropna().unique())
brca_behavior = set(brca['icdo3_behavior'].dropna().unique())

baheya_coverage = {
    'icdo3_topography': {
        'baheya_K': len(bah_topo),
        'brca_codes': sorted(brca_topo),
        'in_baheya_vocab': sorted(brca_topo & bah_topo),
        'oov_under_baheya': sorted(brca_topo - bah_topo),
        'rows_total': int(brca['icdo3_topography'].notna().sum()),
        'rows_in_vocab': int(brca['icdo3_topography'].isin(bah_topo).sum()),
    },
    'icdo3_morphology': {
        'baheya_K': len(bah_morph),
        'brca_unique_K': len(brca_morph),
        'in_baheya_vocab': sorted(brca_morph & bah_morph),
        'oov_under_baheya': sorted(brca_morph - bah_morph),
        'rows_total': int(brca['icdo3_morphology'].notna().sum()),
        'rows_in_vocab': int(brca['icdo3_morphology'].isin(bah_morph).sum()),
    },
    'icdo3_laterality': {
        'baheya_K': len(bah_lat),
        'brca_unique': sorted(brca_lat),
        'in_baheya_vocab': sorted(brca_lat & bah_lat),
        'oov_under_baheya': sorted(brca_lat - bah_lat),
        'rows_total': int(brca['icdo3_laterality'].notna().sum()),
        'rows_in_vocab': int(brca['icdo3_laterality'].isin(bah_lat).sum()),
    },
    'icdo3_behavior': {
        'baheya_K': len(bah_behavior),
        'brca_unique': sorted(brca_behavior),
        'in_baheya_vocab': sorted(brca_behavior & bah_behavior),
        'oov_under_baheya': sorted(brca_behavior - bah_behavior),
        'rows_total': int(brca['icdo3_behavior'].notna().sum()),
        'rows_in_vocab': int(brca['icdo3_behavior'].isin(bah_behavior).sum()),
    },
    'icdo3_grade': {
        'baheya_K': len(bah_grade),
        'brca_grade_coverage_pct': float(brca['icdo3_grade'].notna().mean() * 100),
        'note': 'BRCA has 0% grade populated; coverage analysis moot',
    },
}

print('  Baheya-vocab coverage on BRCA:')
for axis, info in baheya_coverage.items():
    print(f"    {axis}: {info}")

# Save coverage analysis as JSON for reference
with open(OUT_BRCA / 'baheya_vocab_coverage.json', 'w') as f:
    json.dump(baheya_coverage, f, indent=2)


# ---------- Data cards ----------

def axis_summary(df, axis):
    notna = df[axis].notna().sum()
    return f"{notna}/{len(df)} ({100*notna/len(df):.1f}%)"

def multilabel_summary(df, axis):
    counts = df[axis].apply(len)
    n_pos = (counts > 0).sum()
    return f"{n_pos}/{len(df)} ({100*n_pos/len(df):.1f}%) have ≥1 positive; max per row={counts.max()}"

print('\nWriting data cards...')

pretrain_card = f"""# TCGA Pretrain — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `tcga_pretrain_model_ready/`
- **Built at:** {new_vocab['built_at']}
- **Vocab version:** v2 (canonical schema; supersedes v1)
- **Built from:** `pretrain_train` split

## 2. Row counts
- **Total rows:** {len(pre):,}
- **Splits:** {pre['split'].value_counts().to_dict()}
- **Cancer types (top 10):** {pre['cancer_type'].value_counts().head(10).to_dict()}

## 3. Schema (canonical, 34 cols; matches Baheya M1 + SEER + TCGA-BRCA)
See `MCIS_Dataset_Unification_Handoff.md` §2.1 for full column list.

## 4. Label coverage
| Axis | Coverage | Type |
|---|---|---|
| `icdo3_topography` | {axis_summary(pre, 'icdo3_topography')} | single-pick |
| `icdo3_morphology` | {axis_summary(pre, 'icdo3_morphology')} | single-pick |
| `icdo3_behavior` | {axis_summary(pre, 'icdo3_behavior')} | single-pick |
| `icdo3_grade` | {axis_summary(pre, 'icdo3_grade')} | single-pick (string) |
| `icdo3_laterality` | {axis_summary(pre, 'icdo3_laterality')} | single-pick (L/R/B) |
| `icd11_stem` | {axis_summary(pre, 'icd11_stem')} | silver, never label |
| `icd11_ext_anatomy` | {multilabel_summary(pre, 'icd11_ext_anatomy')} | multi-label list[str] |
| `icd11_ext_histopath` | {multilabel_summary(pre, 'icd11_ext_histopath')} | multi-label list[str] |
| `icd11_ext_laterality` | {multilabel_summary(pre, 'icd11_ext_laterality')} | multi-label list[str] |
| `icd11_ext_grading` | {multilabel_summary(pre, 'icd11_ext_grading')} | multi-label list[str] |

## 5. Vocab (built from `pretrain_train` only)
| Axis | K | First codes |
|---|---:|---|
| `icdo3_topography` | {len(new_vocab['axes']['icdo3_topography'])} | {list(new_vocab['axes']['icdo3_topography'].keys())[:5]} |
| `icdo3_morphology` | {len(new_vocab['axes']['icdo3_morphology'])} | {list(new_vocab['axes']['icdo3_morphology'].keys())[:5]} |
| `icdo3_behavior` | {len(new_vocab['axes']['icdo3_behavior'])} | {list(new_vocab['axes']['icdo3_behavior'].keys())} |
| `icdo3_grade` | {len(new_vocab['axes']['icdo3_grade'])} | {list(new_vocab['axes']['icdo3_grade'].keys())} (string-encoded) |
| `icdo3_laterality` | {len(new_vocab['axes']['icdo3_laterality'])} | {list(new_vocab['axes']['icdo3_laterality'].keys())} |

ICD-11 extension axes are **not** part of the pretrain vocab (silver-only, never used as supervision).
A head built from this artifact has no ICD-11 component.

## 6. Encoding decisions
- **`icdo3_laterality`:** L/R/B single-pick. Original X-codes preserved in `icdo3_laterality_raw`. Mapping: `XK8G→L, XK9K→R, XK9J→B`.
- **`icdo3_grade`:** string `'1'`–`'4'`. Null = mask in loss (not treated as Grade-Unknown class).
- **ICD-11 extension axes:** parsed from `icd11_full_postcoord` into `list[str]`. Empty list = no positives. Stored for schema parity only; not used in pretraining.
- **Splits:** 70/15/15 iterative-stratified (cancer_type × morphology × behavior), group-aware on `duplicate_group_id`, seed=42. Identical to v1.

## 7. Hard rules
- `icd11_*` columns are silver. Never used as supervision. Never used in eval.
- The `text_section_tagged` and `text_concatenated` columns are auditable — no label-derived fields injected.
- Vocab built from `pretrain_train` only. OOV codes in `pretrain_val`/`pretrain_test` are excluded from F1 (footnoted, not silently dropped).

## 8. Changelog v1 → v2
- icdo3_laterality re-keyed: X-codes → L/R/B (matches Baheya).
- icdo3_grade encoding: int → string.
- ICD-11 extension axes added as list[str] columns: anatomy, histopath, laterality, grading.
- Bookkeeping cols added: fold (=-1 for TCGA), batch (=-1), template_flag (=False), is_cancer_primary (=True), coding_source (='gdc_curated'), data_quality_flag (=None).
- Aux cols added (=None): nuclear_grade, tubular_score, mitosis_score, lvi, dcis_in_specimen.
- icdo3_laterality_raw added.
- icd11_full_postcoord retained (TCGA-only reference).

## 9. Files
- `tcga_pretrain_model_ready.parquet` — model-ready data
- `label_vocab.json` — frozen vocab (v2)
- `splits.csv` — record_id → split mapping
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts
"""

(OUT_PRE / 'data_card.md').write_text(pretrain_card)
print(f'  wrote {OUT_PRE / "data_card.md"}')

brca_card = f"""# TCGA-BRCA External — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `tcga_brca_external_model_ready/`
- **Built at:** {new_vocab['built_at']}
- **Purpose:** External validation only. Never used in training.

## 2. Row counts
- **Total rows:** {len(brca):,}
- **Split:** all `brca_external` (single label; v1 internal `brca_train`/`brca_val`/`brca_test` discarded)

## 3. Schema (canonical, 34 cols)
Matches `tcga_pretrain_model_ready/`, `m1_model_ready/`, and `seer_rare_augmentation_model_ready/`. See handoff §2.1.

## 4. Label coverage

### 4.1 Native value distributions
| Axis | Coverage | Notes |
|---|---|---|
| `icdo3_topography` | {axis_summary(brca, 'icdo3_topography')} | {len(brca_topo)} unique codes |
| `icdo3_morphology` | {axis_summary(brca, 'icdo3_morphology')} | {len(brca_morph)} unique codes |
| `icdo3_behavior` | {axis_summary(brca, 'icdo3_behavior')} | values: {sorted(brca_behavior)} |
| `icdo3_grade` | {axis_summary(brca, 'icdo3_grade')} | TCGA-BRCA has no curated grade |
| `icdo3_laterality` | {axis_summary(brca, 'icdo3_laterality')} | values: {sorted(brca_lat)} |
| `icd11_ext_anatomy` | {multilabel_summary(brca, 'icd11_ext_anatomy')} | parsed from postcoord |
| `icd11_ext_histopath` | {multilabel_summary(brca, 'icd11_ext_histopath')} | parsed from postcoord |
| `icd11_ext_laterality` | {multilabel_summary(brca, 'icd11_ext_laterality')} | parsed from postcoord |
| `icd11_ext_grading` | {multilabel_summary(brca, 'icd11_ext_grading')} | parsed from postcoord |

### 4.2 Baheya-vocab coverage (Ext-1 evaluation)
This BRCA artifact is evaluated against the **Baheya M1 vocab** (Ext-1) and against the **TCGA-pretrain vocab** (Ext-2). Coverage under Baheya M1:

#### `icdo3_topography` (Baheya K={len(bah_topo)}, BRCA has {len(brca_topo)} unique)
- **In Baheya vocab:** {sorted(brca_topo & bah_topo)} ({baheya_coverage['icdo3_topography']['rows_in_vocab']:,}/{baheya_coverage['icdo3_topography']['rows_total']:,} rows = {100*baheya_coverage['icdo3_topography']['rows_in_vocab']/baheya_coverage['icdo3_topography']['rows_total']:.1f}% of populated rows)
- **OOV under Baheya:** {sorted(brca_topo - bah_topo)}

#### `icdo3_morphology` (Baheya K={len(bah_morph)}, BRCA has {len(brca_morph)} unique)
- **In Baheya vocab ({len(brca_morph & bah_morph)} codes):** {sorted(brca_morph & bah_morph)}
- **OOV under Baheya ({len(brca_morph - bah_morph)} codes):** {sorted(brca_morph - bah_morph)}
- **Row-level:** {baheya_coverage['icdo3_morphology']['rows_in_vocab']:,}/{baheya_coverage['icdo3_morphology']['rows_total']:,} rows in vocab = {100*baheya_coverage['icdo3_morphology']['rows_in_vocab']/baheya_coverage['icdo3_morphology']['rows_total']:.1f}%

#### `icdo3_laterality` (after L/R/B normalization)
- **BRCA values:** {sorted(brca_lat)}
- **Baheya vocab covers:** {sorted(brca_lat & bah_lat)} → 100% coverage

#### `icdo3_behavior`
- **BRCA values:** {sorted(brca_behavior)}; **Baheya values:** {sorted(bah_behavior)}
- **OOV:** {sorted(brca_behavior - bah_behavior)}

#### `icdo3_grade`
- BRCA has 0% grade populated. Coverage analysis is moot.

## 5. Encoding decisions
Same as TCGA pretrain (see that data card §6).

## 6. Hard rules
- This is **external validation only**. No training. No fold structure.
- Eval uses two protocols:
  - **Ext-1:** Baheya M1 vocab (this artifact's labels intersected with Baheya vocab).
  - **Ext-2:** TCGA-pretrain vocab.
- ICD-11 columns are silver; never used in eval (per architecture v6 §2.4).

## 7. Files
- `tcga_brca_external_model_ready.parquet` — eval data
- `baheya_vocab_coverage.json` — Ext-1 coverage details
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts
"""

(OUT_BRCA / 'data_card.md').write_text(brca_card)
print(f'  wrote {OUT_BRCA / "data_card.md"}')


# ---------- Checksums ----------

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def write_checksums(out_dir):
    files = sorted([p for p in out_dir.iterdir() if p.is_file() and p.name != 'CHECKSUMS.txt'])
    lines = []
    for p in files:
        h = sha256_file(p)
        lines.append(f'{h}  {p.name}')
    (out_dir / 'CHECKSUMS.txt').write_text('\n'.join(lines) + '\n')

write_checksums(OUT_PRE)
write_checksums(OUT_BRCA)
print('\nChecksums written.')


# ---------- Final summary ----------

print('\n' + '=' * 70)
print('REBUILD COMPLETE')
print('=' * 70)
for d in [OUT_PRE, OUT_BRCA]:
    print(f'\n{d}:')
    for p in sorted(d.iterdir()):
        if p.is_file():
            print(f'  {p.name:<45} {p.stat().st_size:>12,} B')
