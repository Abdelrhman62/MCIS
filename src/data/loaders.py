"""PyTorch Datasets for MCIS Baheya M1 (and TCGA pretrain).

Design (post-Phase-0, parquet-native):
- ``BaheyaM1Dataset`` materializes the entire encoded view in memory at
  construction time. The dataset is small (3,055 rows). No streaming, no
  per-worker re-tokenization. PLM-ICD-style segment tokenization happens
  *inside* OncologyEncoder.forward() — this loader passes raw text strings.
- Multi-label cells in parquet are ``list[str]`` (per data card §10).
  Legacy pipe-separated strings also work via ``AxisVocab.encode_multi``.
- Aux histology fields (E3 toggle) are masked-loss targets: every row
  carries a (target, mask) pair where mask=0 means "no label".
- No ``column_map`` indirection. Parquet column names == axis names ==
  config keys. v3 schema lock.

Key contract change vs old loader:
- Constructor takes ``BenchmarkConfig`` (or just an axis spec) + ``LabelVocab``.
  No more `text_column` / individual flags scattered.
- DataLoader yields raw text strings in collate output (encoder tokenizes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.label_vocab import LabelVocab, _is_null
from src.utils.config import BenchmarkConfig


def _coerce_group_id(val: Any) -> int:
    """duplicate_group_id → stable int.

    Baheya group ids are integers; TCGA group ids are hex strings
    (data card §6). This column is bookkeeping only (split-time group
    awareness; never used in loss/training), so the exact value is
    irrelevant — only a stable, collisional-safe int is required so
    collate's LongTensor cast doesn't crash. int-like parses directly;
    anything else is hashed deterministically (md5, seed-free) so the
    same group string always maps to the same id within and across runs.
    """
    if _is_null(val):
        return -1
    try:
        return int(val)
    except (ValueError, TypeError):
        import hashlib
        h = hashlib.md5(str(val).encode("utf-8")).hexdigest()
        # 60-bit slice: fits int64 (LongTensor), astronomically low collision
        return int(h[:15], 16)

# ---------------------------------------------------------------------------
# Aux histology encoders
# ---------------------------------------------------------------------------

# Nottingham subscores: integer 1/2/3 → class index 0/1/2.
_NOTTINGHAM_VALUES: dict[int, int] = {1: 0, 2: 1, 3: 2}

# Binary fields: case-insensitive presence/absence words.
_BINARY_PRESENT: set[str] = {"PRESENT", "POSITIVE", "YES"}
_BINARY_ABSENT: set[str] = {"ABSENT", "NEGATIVE", "NO"}

_NOTTINGHAM_FIELDS: set[str] = {"nuclear_grade", "tubular_score", "mitosis_score"}
_BINARY_FIELDS: set[str] = {"lvi", "dcis_in_specimen"}


def _encode_nottingham(v: Any) -> tuple[int, int]:
    """Return (class_idx, mask). mask=0 if missing or out-of-range."""
    if _is_null(v):
        return 0, 0
    try:
        i = int(float(v))
    except (TypeError, ValueError):
        return 0, 0
    if i not in _NOTTINGHAM_VALUES:
        return 0, 0
    return _NOTTINGHAM_VALUES[i], 1


def _encode_binary(v: Any) -> tuple[int, int]:
    """Encode 'Absent'/'Present' (or yes/no, positive/negative) to (0/1, mask)."""
    if _is_null(v):
        return 0, 0
    s = str(v).strip().upper()
    if s in _BINARY_PRESENT:
        return 1, 1
    if s in _BINARY_ABSENT:
        return 0, 1
    return 0, 0


def _encode_aux_value(field_name: str, v: Any) -> tuple[int, int]:
    """Dispatch by field type. Default = mask (0, 0)."""
    if field_name in _NOTTINGHAM_FIELDS:
        return _encode_nottingham(v)
    if field_name in _BINARY_FIELDS:
        return _encode_binary(v)
    return 0, 0


# ---------------------------------------------------------------------------
# Per-row record
# ---------------------------------------------------------------------------


@dataclass
class _M1Row:
    record_id: str
    text: str
    # Single-pick axis → int class index, or NULL_TARGET_SENTINEL (=-1) for null.
    single_labels: dict[str, int]
    # Multi-label axis → multi-hot float32 vector.
    multi_labels: dict[str, np.ndarray]
    # Aux histology (when include_aux=True).
    aux_targets: np.ndarray  # int32 [n_aux]
    aux_masks: np.ndarray    # int32 [n_aux]
    # Bookkeeping for stratified eval.
    batch: int
    duplicate_group_id: int
    template_flag: bool
    is_cancer_primary: bool
    fold: int


NULL_TARGET_SENTINEL = -1  # mirrors src.models.heads.NULL_TARGET_SENTINEL


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BaheyaM1Dataset(Dataset):
    """In-memory dataset for Baheya M1 (or any v3-canonical-schema parquet).

    Args:
        df: DataFrame slice (already filtered to the desired split/fold).
        vocab: Fitted LabelVocab — must match cfg.axis_types and cfg.label_cardinalities.
        cfg: BenchmarkConfig — drives text_field, axis selection, aux toggle.
        include_aux: If True, encode aux histology fields (cfg.aux_fields).
            Default False; enabled by E3 / cfg.model.use_aux_histology_heads.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        vocab: LabelVocab,
        cfg: BenchmarkConfig,
        include_aux: bool | None = None,
    ) -> None:
        self.vocab = vocab
        self.cfg = cfg

        # Aux toggle: explicit override > config setting.
        if include_aux is None:
            include_aux = cfg.model.use_aux_histology_heads
        self.include_aux = include_aux

        text_field = cfg.data.text_field
        if text_field not in df.columns:
            raise KeyError(
                f"text_field {text_field!r} not in DataFrame. "
                f"Available text-like columns: "
                f"{[c for c in df.columns if 'text' in c.lower()]}"
            )

        # Validate axes are all present.
        for axis in cfg.all_axes:
            if axis not in df.columns:
                raise KeyError(
                    f"Axis column {axis!r} (declared in cfg.axis_types) "
                    f"not in DataFrame."
                )
            if axis not in vocab.axes:
                raise KeyError(
                    f"Axis {axis!r} declared in cfg.axis_types but not in vocab."
                )

        # Validate vocab vs config cardinalities (defensive).
        cfg.validate_against_vocab(vocab.cardinalities())

        # Validate aux fields when needed.
        if self.include_aux:
            missing = [f for f in cfg.aux_fields if f not in df.columns]
            if missing:
                raise KeyError(
                    f"Aux fields {missing} declared but missing from DataFrame."
                )

        self._aux_field_names: list[str] = list(cfg.aux_fields) if self.include_aux else []
        self._rows: list[_M1Row] = []
        self._materialize(df)

    def _materialize(self, df: pd.DataFrame) -> None:
        text_field = self.cfg.data.text_field
        single_axes = list(self.cfg.axis_types.single_pick)
        multi_axes = list(self.cfg.axis_types.multilabel)

        for i, (_, row) in enumerate(df.iterrows()):
            text = row[text_field]
            text = "" if _is_null(text) else str(text)
            
            # Automatically apply section normalizer to raw TCGA text
            if "[FULL_REPORT_TEXT]" in text:
                from src.data.section_normalizer import normalize_sections
                text = normalize_sections(text)

            # Single-pick targets.
            singles: dict[str, int] = {}
            for axis in single_axes:
                axis_vocab = self.vocab[axis]
                idx = axis_vocab.encode(row[axis])
                singles[axis] = NULL_TARGET_SENTINEL if idx is None else idx

            # Multi-label targets.
            multis: dict[str, np.ndarray] = {}
            for axis in multi_axes:
                axis_vocab = self.vocab[axis]
                multis[axis] = axis_vocab.encode_multi(row[axis])

            # Aux.
            n_aux = len(self._aux_field_names)
            if n_aux > 0:
                aux_t = np.zeros(n_aux, dtype=np.int32)
                aux_m = np.zeros(n_aux, dtype=np.int32)
                for j, fname in enumerate(self._aux_field_names):
                    t, m = _encode_aux_value(fname, row.get(fname))
                    aux_t[j] = t
                    aux_m[j] = m
            else:
                aux_t = np.zeros(0, dtype=np.int32)
                aux_m = np.zeros(0, dtype=np.int32)

            self._rows.append(
                _M1Row(
                    record_id=str(row.get("record_id", f"row_{i}")),
                    text=text,
                    single_labels=singles,
                    multi_labels=multis,
                    aux_targets=aux_t,
                    aux_masks=aux_m,
                    batch=int(row["batch"]) if "batch" in row and not _is_null(row["batch"]) else -1,
                    duplicate_group_id=_coerce_group_id(row["duplicate_group_id"]) if "duplicate_group_id" in row else -1,
                    template_flag=bool(row["template_flag"]) if "template_flag" in row and not _is_null(row["template_flag"]) else False,
                    is_cancer_primary=bool(row["is_cancer_primary"]) if "is_cancer_primary" in row and not _is_null(row["is_cancer_primary"]) else True,
                    fold=int(row["fold"]) if "fold" in row and not _is_null(row["fold"]) else -1,
                )
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[idx]
        return {
            "record_id": r.record_id,
            "text": r.text,
            "single_labels": dict(r.single_labels),
            "multi_labels": {k: v.copy() for k, v in r.multi_labels.items()},
            "aux_targets": r.aux_targets,
            "aux_masks": r.aux_masks,
            "batch": r.batch,
            "duplicate_group_id": r.duplicate_group_id,
            "template_flag": r.template_flag,
            "is_cancer_primary": r.is_cancer_primary,
            "fold": r.fold,
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_m1(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate raw-text + label batch.

    Encoder takes raw texts (PLM-ICD-style segmentation happens in encoder.forward),
    so the collate output keeps `texts` as a list[str], not pre-tokenized tensors.

    Returns:
        {
            'texts':              list[str], len=B,
            'record_ids':         list[str],
            'labels_<axis>':      LongTensor [B] for single, FloatTensor [B, K] for multi,
            'aux_targets':        LongTensor [B, n_aux] (or [B, 0] if disabled),
            'aux_masks':          LongTensor [B, n_aux],
            'batch':              LongTensor [B],
            'template_flag':      BoolTensor [B],
            'is_cancer_primary':  BoolTensor [B],
            'duplicate_group_id': LongTensor [B],
            'fold':               LongTensor [B],
        }
    """
    bsz = len(batch)
    out: dict[str, Any] = {
        "texts": [item["text"] for item in batch],
        "record_ids": [item["record_id"] for item in batch],
    }

    # Single-pick axes → LongTensor [B] with sentinel for nulls.
    if batch[0]["single_labels"]:
        for axis_name in batch[0]["single_labels"]:
            arr = np.array(
                [item["single_labels"][axis_name] for item in batch],
                dtype=np.int64,
            )
            out[f"labels_{axis_name}"] = torch.from_numpy(arr)

    # Multi-label axes → FloatTensor [B, K].
    if batch[0]["multi_labels"]:
        for axis_name in batch[0]["multi_labels"]:
            arr = np.stack([item["multi_labels"][axis_name] for item in batch], axis=0)
            out[f"labels_{axis_name}"] = torch.from_numpy(arr).float()

    # Aux.
    aux_t = np.stack([item["aux_targets"] for item in batch], axis=0).astype(np.int64)
    aux_m = np.stack([item["aux_masks"] for item in batch], axis=0).astype(np.int64)
    out["aux_targets"] = torch.from_numpy(aux_t)
    out["aux_masks"] = torch.from_numpy(aux_m)

    # Bookkeeping.
    out["batch"] = torch.tensor([item["batch"] for item in batch], dtype=torch.long)
    out["duplicate_group_id"] = torch.tensor(
        [item["duplicate_group_id"] for item in batch], dtype=torch.long
    )
    out["template_flag"] = torch.tensor(
        [item["template_flag"] for item in batch], dtype=torch.bool
    )
    out["is_cancer_primary"] = torch.tensor(
        [item["is_cancer_primary"] for item in batch], dtype=torch.bool
    )
    out["fold"] = torch.tensor([item["fold"] for item in batch], dtype=torch.long)

    return out


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def load_parquet(path: str) -> pd.DataFrame:
    """Read a v3-canonical model-ready parquet."""
    return pd.read_parquet(path)

def inject_seer_templates(
    train_df: "pd.DataFrame",
    seer_parquet: str,
) -> "pd.DataFrame":
    """Concatenate SEER rare-code augmentation templates into the train fold.

    SEER rows already have template_flag=True and match M1 schema exactly.
    Appended to train only — val/test folds are never augmented.
    """
    seer_df = load_parquet(seer_parquet)
    seer_df = seer_df[seer_df["template_flag"] == True].copy()
    combined = pd.concat([train_df, seer_df], ignore_index=True)
    return combined

def split_trainable_test(
    df: pd.DataFrame,
    trainable_value: str = "trainable",
    test_value: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (trainable_pool, held-out test) DataFrames per ``split`` column."""
    if "split" not in df.columns:
        raise KeyError("DataFrame has no 'split' column")
    tr = df[df["split"] == trainable_value].reset_index(drop=True)
    te = df[df["split"] == test_value].reset_index(drop=True)
    return tr, te


def split_by_value(
    df: pd.DataFrame,
    split_value: str,
    split_column: str = "split",
) -> pd.DataFrame:
    """Return rows where df[split_column] == split_value (index reset).

    Phase 1 (TCGA) uses pre-baked 3-way splits (pretrain_train/val/test);
    split_trainable_test only does the Baheya 2-way (trainable/test). This
    pulls one named split without touching that path. Raises if the value
    is absent so a typo'd config fails loud, not silently empty.
    """
    if split_column not in df.columns:
        raise KeyError(f"DataFrame has no {split_column!r} column")
    sub = df[df[split_column] == split_value].reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            f"No rows with {split_column}=={split_value!r}. "
            f"Available: {sorted(df[split_column].dropna().unique().tolist())}"
        )
    return sub


def split_by_fold(
    trainable_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    fold_idx: int,
    record_id_col: str = "record_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Given folds map (record_id → fold), return (train, val) for one fold.

    The trainable_df may already carry a 'fold' column (M1 parquet does, per
    data card §10). If so we use it directly. Otherwise we merge from folds_df.
    """
    if "fold" in trainable_df.columns and folds_df is None:
        fold_assign = trainable_df["fold"]
    else:
        if record_id_col not in trainable_df.columns:
            raise KeyError(f"{record_id_col!r} missing in trainable_df")
        if "fold" not in folds_df.columns:
            raise KeyError("folds_df must have a 'fold' column")
        fold_map = dict(zip(folds_df[record_id_col], folds_df["fold"], strict=False))
        fold_assign = trainable_df[record_id_col].map(fold_map)
        if fold_assign.isna().any():
            missing = trainable_df[fold_assign.isna()][record_id_col].tolist()[:5]
            raise ValueError(
                f"{fold_assign.isna().sum()} trainable records missing from folds_df. "
                f"First few: {missing}. Did you regenerate folds after a data update?"
            )

    val_mask = fold_assign == fold_idx
    val = trainable_df[val_mask].reset_index(drop=True)
    train = trainable_df[~val_mask].reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Subgroup mask derivation (for B0.yaml eval.subgroups)
# ---------------------------------------------------------------------------


def derive_subgroup_masks(
    df: pd.DataFrame,
    subgroup_names: list[str],
    nos_topography_code: str = "C50.9",
) -> dict[str, dict[str, np.ndarray]]:
    """Compute per-subgroup boolean masks for stratified eval reporting.

    Architecture v6 §11.3: subgroup-stratified F1 is reported per category.
    B0.yaml lists ``[batch, template_flag, is_cancer_primary, nos_topography_derived]``.
    Most are direct bool/categorical columns; ``nos_topography_derived`` is the
    derived predicate ``df['icdo3_topography'] == nos_topography_code`` (default
    C50.9 = NOS, per Baheya data card §5).

    Returns:
        Mapping subgroup_category → {value_label → bool array of len(df)}.
        For 'batch': {'1': mask, '2': mask}; for bool fields: {'True': mask, 'False': mask}.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    n = len(df)

    for sg in subgroup_names:
        if sg == "nos_topography_derived":
            if "icdo3_topography" not in df.columns:
                continue
            mask = (df["icdo3_topography"] == nos_topography_code).to_numpy()
            out[sg] = {
                f"is_{nos_topography_code}": mask,
                f"not_{nos_topography_code}": ~mask,
            }
        elif sg in df.columns:
            col = df[sg]
            if col.dtype == bool or set(col.dropna().unique()) <= {True, False}:
                m = col.fillna(False).astype(bool).to_numpy()
                out[sg] = {"True": m, "False": ~m}
            else:
                # Categorical (e.g. batch=1/2)
                groups: dict[str, np.ndarray] = {}
                for val in sorted(col.dropna().unique().tolist()):
                    groups[str(val)] = (col == val).to_numpy()
                out[sg] = groups
        # else: silently skip unknown subgroups

    return out
