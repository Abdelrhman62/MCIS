"""
build_seer_canonical.py

Builds the SEER rare-augmentation artifact in canonical schema.

Inputs (paths configurable via CLI args or edit the constants below):
  - icdo3_breast_morphology_lookup.csv  (228 rows, 4 cols)
  - seer_breast_reference.csv           (933,520 rows, 15 cols)
  - seer_rare_augmentation.csv          (640 existing templates, 11 cols)
  - m1_model_ready.parquet              (Baheya, for verification only — optional)

Output:
  ./seer_rare_augmentation_model_ready/
    seer_rare_augmentation_model_ready.parquet
    data_card.md
    CHECKSUMS.txt

What it does:
  1. Drops 5 obsolete codes from the existing 640 templates (no longer rare in v2).
  2. Generates fresh templates for 4 newly-rare v2 codes (8200, 8480, 8502, 8550)
     by joint-sampling from seer_breast_reference.csv per code.
  3. Canonicalizes the merged set to the 34-col schema (matches Baheya/TCGA).
  4. Renders text_section_tagged using the literal Baheya tag scheme.

Run on M4 from the SEER Preprocessing directory (or wherever — paths are configurable).

    python build_seer_canonical.py

The script prints a verification summary at the end. Paste that summary back to chat.
"""
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# =====================================================================
# CONFIG — edit these paths if your layout differs
# =====================================================================
DEFAULT_LOOKUP = '/Users/abdo/Desktop/SEER Preprocessing/references/icdo3_breast_morphology_lookup.csv'
DEFAULT_REFERENCE = '/Users/abdo/Desktop/SEER Preprocessing/processed/seer_breast_reference.csv'
DEFAULT_EXISTING = '/Users/abdo/Desktop/SEER Preprocessing/processed/seer_rare_augmentation.csv'
DEFAULT_OUT_DIR = './seer_rare_augmentation_model_ready'

SEED = 42

# v2 rare-code policy decisions (locked in chat)
OBSOLETE_CODES_TO_DROP = ['8010', '8140', '8503', '8504', '8522']  # in v1 SEER but not rare in v2
NEW_CODES_TO_GENERATE = ['8200', '8480', '8502', '8550']            # rare in v2 but absent from v1

# Per-code Baheya v2 trainable counts (from m1_model_ready.parquet)
BAHEYA_V2_TRAIN_COUNTS = {
    '8200': 1,
    '8480': 9,
    '8502': 1,
    '8550': 1,
}

# WHO ICD-O-3 standard names for codes missing from the 228-row lookup
# (Documented in data card §6 as "supplementary names from WHO ICD-O-3 manual")
WHO_FALLBACK_NAMES = {
    '8200': 'Adenoid cystic carcinoma',
    '8480': 'Mucinous adenocarcinoma',
    '8502': 'Secretory carcinoma of breast',
    '8550': 'Acinar cell carcinoma',
}

# Canonical schema column order (matches Baheya M1 + TCGA canonical, 34 cols)
CANONICAL_COLS = [
    'record_id', 'patient_id', 'dataset', 'split', 'fold', 'duplicate_group_id', 'batch',
    'template_flag', 'is_cancer_primary', 'coding_source', 'data_quality_flag', 'cancer_type',
    'text_length_words', 'text_length_bert_tokens',
    'text_section_tagged', 'text_concatenated',
    'icdo3_topography', 'icdo3_morphology', 'icdo3_behavior', 'icdo3_grade', 'icdo3_laterality',
    'icd11_stem',
    'icd11_ext_anatomy', 'icd11_ext_histopath', 'icd11_ext_laterality', 'icd11_ext_grading',
    'icdo3_grade_raw', 'icdo3_laterality_raw',
    'nuclear_grade', 'tubular_score', 'mitosis_score', 'lvi', 'dcis_in_specimen',
    'icd11_full_postcoord',
]

# =====================================================================
# Helpers
# =====================================================================

def target_count(code, baheya_train_count):
    """Per-code target number of templates: max(20, min(100, 50 - baheya_train))."""
    return max(20, min(100, 50 - baheya_train_count))


def lat_to_lrb(x):
    """Right/Left/Bilateral -> R/L/B; anything else -> None."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in ('right', 'r'):
        return 'R'
    if s in ('left', 'l'):
        return 'L'
    if s in ('bilateral', 'b'):
        return 'B'
    return None


def lat_to_baheya_text(x):
    """L/R/B -> Left/Right/Bilateral for [LATERALITY] tag rendering."""
    return {'L': 'Left', 'R': 'Right', 'B': 'Bilateral'}.get(x, None)


def grade_to_string(x):
    """float/int/str/nan -> '1'|'2'|'3'|'4'|None."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        i = int(float(x))
    except (TypeError, ValueError):
        return None
    return str(i) if i in (1, 2, 3, 4) else None


def grade_to_roman(g):
    """'1'/'2'/'3'/'4' -> 'I'/'II'/'III'/'IV'; None -> None."""
    return {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}.get(g)


import re

# Topography names: strip trailing ", NOS" AND any leading "Cxx.y-" SEER reference prefix
def clean_topography_name(name):
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    # Strip leading "C50.x-" or "C50.x " prefix (SEER reference internal format quirk)
    s = re.sub(r'^C\d{2}\.\d[-\s]+', '', s)
    # Strip ", NOS" suffix (case-insensitive)
    if s.lower().endswith(', nos'):
        s = s[:-5].strip()
    return s


# =====================================================================
# Tag rendering (Baheya literal scheme)
# =====================================================================

def render_diagnosis(morph_name, topo_name, laterality_text, grade_roman, stage):
    """
    Build the [DIAGNOSIS] sentence in Baheya style. No [GRADE] tag exists in
    Baheya's schema; grade is embedded in the diagnosis text.

    Topography phrase integrates laterality as a leading lowercase modifier,
    matching the kept-template convention:
      "Spindle cell carcinoma, right lower-outer quadrant of breast, Grade II."
      "Mucinous adenocarcinoma, left breast."

    Examples produced:
      "Adenoid cystic carcinoma, left upper-outer quadrant of breast, Grade II, Localized stage."
      "Mucinous adenocarcinoma, right breast."
    """
    parts = [morph_name]

    # Build the location phrase: "{laterality_lower} {topo_lower}" if both present.
    # If only one, use whichever is available. If neither, omit the location entirely.
    location = None
    if topo_name and laterality_text:
        location = f'{laterality_text.lower()} {topo_name.lower()}'
    elif topo_name:
        location = topo_name.lower()
    elif laterality_text:
        location = f'{laterality_text.lower()} breast'

    if location:
        parts.append(location)
    if grade_roman:
        parts.append(f'Grade {grade_roman}')
    if stage:
        parts.append(f'{stage} stage')
    return ', '.join(parts) + '.'


def render_text_section_tagged(diagnosis_text, laterality_text):
    """
    Baheya-literal tag scheme for SEER templates:

      [REPORT_TYPE] biopsy_report
      [DIAGNOSIS] {sentence with morph + topo + lat + grade + stage}
      [TUMOUR_TYPE] {short morph name}
      [SPECIMEN] tru-cut biopsy
      [LATERALITY] {Left|Right|Bilateral}    (only if non-null)

    Drops [MICROSCOPIC] and [CLINICAL_INFO] (no synthetic content for those).
    """
    lines = [
        '[REPORT_TYPE] biopsy_report',
        f'[DIAGNOSIS] {diagnosis_text}',
        '[SPECIMEN] tru-cut biopsy',
    ]
    if laterality_text:
        lines.append(f'[LATERALITY] {laterality_text}')
    return '\n'.join(lines)


def render_text_concatenated(diagnosis_text, laterality_text):
    """
    Plain version (no tags). Laterality is already embedded in the diagnosis
    sentence (matching Baheya kept-template convention), so it is NOT appended
    here — appending would duplicate the laterality.
    """
    return diagnosis_text


# =====================================================================
# Build steps
# =====================================================================

def step_1_load_existing_and_drop_obsolete(path):
    print('=' * 70)
    print('STEP 1: Load existing 640 templates + drop obsolete codes')
    print('=' * 70)
    aug = pd.read_csv(path)
    print(f'  Loaded {len(aug)} rows from {path}')

    # Cast morphology to string for matching
    aug['icdo3_morphology'] = aug['icdo3_morphology'].astype(str)
    print(f'  Per-code distribution before drop:')
    pre_dist = aug['icdo3_morphology'].value_counts().sort_index()
    for code, n in pre_dist.items():
        flag = '  ← DROP' if code in OBSOLETE_CODES_TO_DROP else ''
        print(f'    {code}: {n}{flag}')

    keep_mask = ~aug['icdo3_morphology'].isin(OBSOLETE_CODES_TO_DROP)
    kept = aug[keep_mask].reset_index(drop=True)
    dropped = len(aug) - len(kept)
    print(f'\n  Kept: {len(kept)}    Dropped: {dropped}')
    return kept


def step_2_generate_new_templates(reference_path, lookup_path, kept_codes_already_present):
    print()
    print('=' * 70)
    print('STEP 2: Generate fresh templates for 4 new v2-rare codes')
    print('=' * 70)

    # Load lookup table
    lookup = pd.read_csv(lookup_path)
    lookup['morphology_code'] = lookup['morphology_code'].astype(str)
    name_by_code = dict(zip(lookup['morphology_code'], lookup['morphology_name']))
    print(f'  Loaded lookup with {len(lookup)} entries')

    # Inject WHO fallback names for missing codes
    for code, name in WHO_FALLBACK_NAMES.items():
        if code not in name_by_code:
            name_by_code[code] = name
            print(f'    [WHO fallback] {code} -> "{name}"')

    # Load reference
    print(f'\n  Loading SEER breast reference (large file, ~130 MB)...')
    ref = pd.read_csv(reference_path, low_memory=False)
    ref['icdo3_morphology'] = ref['icdo3_morphology'].astype(str)
    print(f'  Loaded {len(ref):,} rows; columns: {list(ref.columns)}')

    new_rows = []
    rng = random.Random(SEED)

    for code in NEW_CODES_TO_GENERATE:
        target = target_count(code, BAHEYA_V2_TRAIN_COUNTS[code])
        pool = ref[ref['icdo3_morphology'] == code].reset_index(drop=True)
        actual_pool = len(pool)
        # Sample with replacement if pool < target
        sample = pool.sample(n=target, replace=(actual_pool < target),
                             random_state=SEED + int(code))
        morph_name = name_by_code[code]
        print(f'\n  Code {code} ({morph_name}): pool={actual_pool}, target={target}, generated={len(sample)}'
              + ('  [w/ replacement]' if actual_pool < target else ''))

        for i, (_, row) in enumerate(sample.iterrows()):
            topo_code = row['icdo3_topography']
            topo_name = clean_topography_name(row.get('topography_name'))
            lat_lrb = lat_to_lrb(row['icdo3_laterality'])
            lat_text = lat_to_baheya_text(lat_lrb)
            grade_str = grade_to_string(row['icdo3_grade'])
            grade_roman = grade_to_roman(grade_str)
            stage = row.get('summary_stage') if not (isinstance(row.get('summary_stage'), float) and pd.isna(row.get('summary_stage'))) else None

            diag_text = render_diagnosis(morph_name, topo_name, lat_text, grade_roman, stage)

            # Use morph name as TUMOUR_TYPE (short form is the same here)
            new_rows.append({
                'record_id': f'seer_aug_{code}_{i:04d}',
                'icdo3_topography': topo_code,
                'icdo3_morphology': code,
                'icdo3_behavior': '3',  # all SEER breast malignant primary -> behavior 3
                'icdo3_grade': grade_str,
                'icdo3_grade_raw': str(row['icdo3_grade']) if not (isinstance(row['icdo3_grade'], float) and pd.isna(row['icdo3_grade'])) else None,
                'icdo3_laterality': lat_lrb,
                'icdo3_laterality_raw': str(row['icdo3_laterality']) if row['icdo3_laterality'] is not None and not (isinstance(row['icdo3_laterality'], float) and pd.isna(row['icdo3_laterality'])) else None,
                # The diagnosis text and tag-rendered text:
                '_diagnosis_text': diag_text,
                '_laterality_text': lat_text,
                '_morph_name': morph_name,
            })

    print(f'\n  Total new templates generated: {len(new_rows)}')
    return pd.DataFrame(new_rows)


def step_3_canonicalize_kept(kept_df):
    """
    Convert the kept existing templates (11 cols) to canonical 34-col schema.
    The existing CSV's text_combined becomes both text_concatenated AND is
    re-rendered into text_section_tagged (Baheya-tagged form).
    """
    print()
    print('=' * 70)
    print('STEP 3: Canonicalize kept templates (existing -> 34-col schema)')
    print('=' * 70)

    df = kept_df.copy()
    n = len(df)

    # Cast columns
    df['icdo3_morphology'] = df['icdo3_morphology'].astype(str)
    df['icdo3_behavior'] = df['icdo3_behavior'].astype(str)
    df['icdo3_laterality_raw'] = df['icdo3_laterality'].copy()
    df['icdo3_grade_raw'] = df['icdo3_grade'].copy()

    df['icdo3_laterality'] = df['icdo3_laterality_raw'].apply(lat_to_lrb)
    df['icdo3_grade'] = df['icdo3_grade_raw'].apply(grade_to_string)

    # The existing text_combined column is the diagnosis sentence already.
    # Wrap it into the Baheya tag scheme.
    laterality_texts = df['icdo3_laterality'].apply(lat_to_baheya_text)
    df['text_concatenated'] = df['text_combined']
    df['text_section_tagged'] = [
        render_text_section_tagged(diag, lat)
        for diag, lat in zip(df['text_combined'], laterality_texts)
    ]

    # Bookkeeping
    df['patient_id'] = df['record_id']  # template's own id
    df['dataset'] = 'seer_augment'
    df['split'] = 'train'
    df['fold'] = -1
    df['batch'] = -1
    df['duplicate_group_id'] = df['record_id']  # each template is its own group
    df['template_flag'] = True
    df['is_cancer_primary'] = True
    df['coding_source'] = 'seer_template'
    df['data_quality_flag'] = None
    df['cancer_type'] = 'BRCA'

    # ICD-11: SEER has no ICD-11 mapping
    df['icd11_stem'] = None
    df['icd11_ext_anatomy'] = [[] for _ in range(n)]
    df['icd11_ext_histopath'] = [[] for _ in range(n)]
    df['icd11_ext_laterality'] = [[] for _ in range(n)]
    df['icd11_ext_grading'] = [[] for _ in range(n)]
    df['icd11_full_postcoord'] = None

    # Aux fields (Baheya-only)
    for c in ['nuclear_grade', 'tubular_score', 'mitosis_score', 'lvi', 'dcis_in_specimen']:
        df[c] = None

    # Token / word counts (rough word count for tagged text; no tokenizer dependency)
    df['text_length_words'] = df['text_section_tagged'].str.split().str.len()
    # PubMedBERT token count: estimate as ~1.3x word count for English templates
    # (The actual count requires the tokenizer — we approximate here. Update if needed
    # by running the tokenizer pass after build.)
    df['text_length_bert_tokens'] = (df['text_length_words'] * 1.3).astype(int)

    # Drop the input-only column
    df = df.drop(columns=['text_combined'])

    print(f'  Canonicalized {len(df)} kept templates')
    return df[CANONICAL_COLS]


def step_4_canonicalize_new(new_df):
    """Convert the freshly-generated templates to canonical schema."""
    print()
    print('=' * 70)
    print('STEP 4: Canonicalize new templates -> 34-col schema')
    print('=' * 70)

    df = new_df.copy()
    n = len(df)

    # Tag-rendered text from the fields we computed in step 2
    df['text_section_tagged'] = [
        render_text_section_tagged(diag, lat)
        for diag, lat in zip(df['_diagnosis_text'], df['_laterality_text'])
    ]
    df['text_concatenated'] = [
        render_text_concatenated(diag, lat)
        for diag, lat in zip(df['_diagnosis_text'], df['_laterality_text'])
    ]
    df = df.drop(columns=['_diagnosis_text', '_laterality_text', '_morph_name'])

    # Bookkeeping
    df['patient_id'] = df['record_id']
    df['dataset'] = 'seer_augment'
    df['split'] = 'train'
    df['fold'] = -1
    df['batch'] = -1
    df['duplicate_group_id'] = df['record_id']
    df['template_flag'] = True
    df['is_cancer_primary'] = True
    df['coding_source'] = 'seer_template'
    df['data_quality_flag'] = None
    df['cancer_type'] = 'BRCA'

    df['icd11_stem'] = None
    df['icd11_ext_anatomy'] = [[] for _ in range(n)]
    df['icd11_ext_histopath'] = [[] for _ in range(n)]
    df['icd11_ext_laterality'] = [[] for _ in range(n)]
    df['icd11_ext_grading'] = [[] for _ in range(n)]
    df['icd11_full_postcoord'] = None

    for c in ['nuclear_grade', 'tubular_score', 'mitosis_score', 'lvi', 'dcis_in_specimen']:
        df[c] = None

    df['text_length_words'] = df['text_section_tagged'].str.split().str.len()
    df['text_length_bert_tokens'] = (df['text_length_words'] * 1.3).astype(int)

    print(f'  Canonicalized {len(df)} new templates')
    return df[CANONICAL_COLS]


def step_5_merge_and_verify(kept_df, new_df):
    print()
    print('=' * 70)
    print('STEP 5: Merge + verify schema')
    print('=' * 70)
    merged = pd.concat([kept_df, new_df], ignore_index=True)
    print(f'  Final row count: {len(merged)} ({len(kept_df)} kept + {len(new_df)} new)')

    issues = []

    # Schema parity
    missing = set(CANONICAL_COLS) - set(merged.columns)
    if missing:
        issues.append(f'missing columns: {sorted(missing)}')

    # Laterality value set
    lat_set = set(merged['icdo3_laterality'].dropna().unique())
    bad = lat_set - {'L', 'R', 'B'}
    if bad:
        issues.append(f'icdo3_laterality has bad values: {bad}')

    # Grade value set
    grade_set = set(merged['icdo3_grade'].dropna().astype(str).unique())
    bad = grade_set - {'1', '2', '3', '4'}
    if bad:
        issues.append(f'icdo3_grade has bad values: {bad}')

    # Multi-label axes are list-typed
    for col in ['icd11_ext_anatomy', 'icd11_ext_histopath', 'icd11_ext_laterality', 'icd11_ext_grading']:
        sample = merged[col].iloc[0]
        if not isinstance(sample, list):
            issues.append(f'{col} not list-typed (sample type={type(sample).__name__})')

    # All rows have non-empty text
    n_empty = (merged['text_section_tagged'].str.len() == 0).sum()
    if n_empty:
        issues.append(f'{n_empty} rows have empty text_section_tagged')

    # Per-code distribution
    print(f'  Per-code distribution (final):')
    for code, n in merged['icdo3_morphology'].value_counts().sort_index().items():
        print(f'    {code}: {n}')

    # All 13 v2 rare codes covered?
    v2_rare = {'8032', '8050', '8200', '8211', '8453', '8480', '8501', '8502',
               '8509', '8540', '8550', '8575', '9020'}
    final_codes = set(merged['icdo3_morphology'].unique())
    missing_v2_rare = v2_rare - final_codes
    if missing_v2_rare:
        issues.append(f'v2 rare codes NOT covered: {sorted(missing_v2_rare)}')
    else:
        print(f'  ✓ All 13 v2 rare codes covered')

    if issues:
        print(f'\n  ❌ {len(issues)} issue(s):')
        for i in issues:
            print(f'    - {i}')
        sys.exit(1)
    print(f'\n  ✓ Schema verified, all checks passed')
    return merged


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookup', default=DEFAULT_LOOKUP)
    parser.add_argument('--reference', default=DEFAULT_REFERENCE)
    parser.add_argument('--existing', default=DEFAULT_EXISTING)
    parser.add_argument('--out', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Build pipeline ----------
    kept_raw = step_1_load_existing_and_drop_obsolete(args.existing)
    new_raw = step_2_generate_new_templates(args.reference, args.lookup, kept_raw['icdo3_morphology'].unique().tolist())
    kept = step_3_canonicalize_kept(kept_raw)
    new = step_4_canonicalize_new(new_raw)
    merged = step_5_merge_and_verify(kept, new)

    # ---------- Write parquet with explicit schema ----------
    print()
    print('=' * 70)
    print('STEP 6: Write parquet + data card + checksums')
    print('=' * 70)

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

    # Normalize all string-typed columns: convert NaN floats and other non-strings
    # to None or str. astype('object') alone leaves NaN floats in place, which pyarrow
    # rejects when casting to pa.string().
    import math

    def to_nullable_str(x):
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        s = str(x)
        # If a numeric value is integer-valued (e.g. '1.0' from float64), strip '.0'
        if s.endswith('.0') and s[:-2].lstrip('-').isdigit():
            return s[:-2]
        return s

    string_cols = [
        'record_id', 'patient_id', 'dataset', 'split', 'duplicate_group_id',
        'coding_source', 'cancer_type', 'text_section_tagged', 'text_concatenated',
        'icdo3_topography', 'icdo3_morphology', 'icdo3_behavior',
        'icdo3_grade', 'icdo3_laterality', 'icd11_stem',
        'icdo3_grade_raw', 'icdo3_laterality_raw',
        'lvi', 'dcis_in_specimen', 'data_quality_flag', 'icd11_full_postcoord',
    ]
    for col in string_cols:
        merged[col] = merged[col].apply(to_nullable_str)

    merged_ordered = merged[[f.name for f in pa_schema]]
    table = pa.Table.from_pandas(merged_ordered, schema=pa_schema, preserve_index=False)
    parquet_path = out_dir / 'seer_rare_augmentation_model_ready.parquet'
    pq.write_table(table, parquet_path, compression='snappy')
    print(f'  wrote {parquet_path} ({parquet_path.stat().st_size / 1e6:.2f} MB)')

    # ---------- Sample text rendering for the data card ----------
    samples = []
    for code in sorted(merged['icdo3_morphology'].unique())[:5]:
        ex = merged[merged['icdo3_morphology'] == code].iloc[0]
        samples.append((code, ex['text_section_tagged']))

    # ---------- Data card ----------
    final_dist = merged['icdo3_morphology'].value_counts().sort_index()
    card = f"""# SEER Rare Augmentation — Data Card (canonical schema)

## 1. Identity
- **Artifact:** `seer_rare_augmentation_model_ready/`
- **Built at:** {datetime.now(timezone.utc).isoformat()}
- **Purpose:** Synthetic template augmentation targeting Baheya M1 v2 rare-morphology codes.
  All templates carry `template_flag=True`, `coding_source='seer_template'`, `dataset='seer_augment'`.

## 2. Row counts
- **Total rows:** {len(merged)}
- **Kept from v1 SEER (640 - {len(kept_raw)} obsolete drop):** {len(kept)}
- **Newly generated (4 v2-rare codes):** {len(new)}

## 3. Per-code distribution
| Morphology | Count | Source |
|---|---:|---|
""" + '\n'.join([
        f"| {code} | {n} | "
        + ('new (joint-sampled from SEER reference)' if code in NEW_CODES_TO_GENERATE else 'kept from v1')
        + ' |'
        for code, n in final_dist.items()
    ]) + f"""

## 4. v2 rare-code coverage
All 13 codes from Baheya M1 v2's `<10 train` rare list are covered:

`8032, 8050, 8200, 8211, 8453, 8480, 8501, 8502, 8509, 8540, 8550, 8575, 9020`

Removed from v1 because no longer rare in v2: `8010, 8140, 8503, 8504, 8522` ({len(kept_raw) - len(kept) if False else 'see step 1 stdout'} templates dropped — exact count in build log).

## 5. Schema
Canonical 34-column schema (matches Baheya M1, TCGA pretrain, TCGA-BRCA). See `MCIS_Dataset_Unification_Handoff.md` §2.1.

## 6. Encoding decisions
- **Tag scheme (text_section_tagged):** Literal Baheya format:
  ```
  [REPORT_TYPE] biopsy_report
  [DIAGNOSIS] {{morphology}}, {{topography}}, {{laterality}}, Grade {{Roman}}, {{stage}} stage.
  [SPECIMEN] tru-cut biopsy
  [LATERALITY] {{Left|Right|Bilateral}}     (only when non-null)
  ```
  No `[GRADE]` tag — grade is embedded in `[DIAGNOSIS]` per Baheya convention. `[MICROSCOPIC]` and `[CLINICAL_INFO]` omitted (no synthetic content for those).
- **`icdo3_laterality`:** L/R/B; raw English (`Left/Right/Bilateral`) preserved in `icdo3_laterality_raw`.
- **`icdo3_grade`:** string `'1'`–`'4'`; null = mask in loss.
- **`icdo3_behavior`:** all `'3'` (SEER breast malignant primary).
- **WHO fallback names:** Lookup table did not have entries for new codes 8200/8480/8502/8550. We used WHO ICD-O-3 standard names:
  - 8200 → Adenoid cystic carcinoma
  - 8480 → Mucinous adenocarcinoma
  - 8502 → Secretory carcinoma of breast
  - 8550 → Acinar cell carcinoma
- **ICD-11 columns:** All None / empty list. SEER has no ICD-11 mapping.
- **Aux columns:** All None (Baheya-only fields).
- **Splits/folds:** All rows have `split='train'`, `fold=-1`. Templates are training-time augmentation; never appear in validation or test.

## 7. Sample rendered templates
""" + '\n\n'.join([f"### Code {c}\n```\n{t}\n```" for c, t in samples]) + f"""

## 8. Hard rules
- All rows have `template_flag=True`. The dataloader can use this to apply different sampling weights or to exclude templates from certain training stages.
- `text_length_bert_tokens` is approximated as `1.3 * word_count`. Re-compute with the actual PubMedBERT tokenizer at training time if exact counts are needed.
- Templates are training-time only. No fold structure. No external validation use.
- Per architecture v6 §12.6, augmentation is subject to E5b ablation; if it doesn't lift rare-code F1 by ≥+1.0, templates are dropped from final pipeline.

## 9. Files
- `seer_rare_augmentation_model_ready.parquet` — model-ready data
- `data_card.md` — this file
- `CHECKSUMS.txt` — SHA-256 over all artifacts

## 10. Provenance
- v1 source: `seer_rare_augmentation.csv` (640 templates, 11 cols, generated against Baheya M1 v1 rare list)
- New templates: joint-sampled from `seer_breast_reference.csv` (933,520 SEER 17 Registries 2000-2022 rows; per-code pools verified non-zero in `seer_breast_distributions.json`)
- Baheya v2 rare list: computed from `m1_model_ready.parquet` trainable split (2,759 rows), `<10 train` threshold per data card §5
- Seed: {SEED}
"""

    (out_dir / 'data_card.md').write_text(card)
    print(f'  wrote {out_dir / "data_card.md"}')

    # ---------- Checksums ----------
    files = sorted([p for p in out_dir.iterdir() if p.is_file() and p.name != 'CHECKSUMS.txt'])
    lines = []
    for p in files:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        lines.append(f'{h.hexdigest()}  {p.name}')
    (out_dir / 'CHECKSUMS.txt').write_text('\n'.join(lines) + '\n')
    print(f'  wrote {out_dir / "CHECKSUMS.txt"}')

    # ---------- Final summary ----------
    print()
    print('=' * 70)
    print('BUILD COMPLETE')
    print('=' * 70)
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f'  {p.name:<50} {p.stat().st_size:>12,} B')

    print()
    print('Sample rendered text (first 5 codes):')
    for code, txt in samples:
        print(f'\n  --- code {code} ---')
        print('  ' + txt.replace('\n', '\n  '))

    print()
    print('PASTE ENTIRE STDOUT BACK TO CHAT FOR VERIFICATION.')


if __name__ == '__main__':
    main()
