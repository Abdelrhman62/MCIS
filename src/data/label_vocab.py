"""Label vocabulary for MCIS — v3 schema.

A LabelVocab is a deterministic, frozen mapping between human-readable code
strings (e.g. ``"8500"`` for IDC, ``"2C61.0"`` for breast cancer NOS) and the
integer indices the model emits.

Schema v3 (current; supersedes v1 and v2):
    {
      "version": 3,
      "module": "M1" | "TCGA_pretrain" | ...,
      "dataset": "baheya_m1" | "tcga_pretrain" | ...,
      "built_from_split": "trainable" | "pretrain_train" | ...,
      "built_at": ISO-8601 UTC,
      "seed": int,
      "ordering": "frequency-descending",
      "axes": {axis_name: [code1, code2, ...]},   # index = position
      "axis_meta": {
        axis_name: {
          "type": "single_pick" | "multi_label",
          "null_policy": "mask_in_loss" | "empty_list_means_none",
          "delimiter": "|",                        # multi-label only
          ... optional fields ...
        }
      },
      "axes_counts": {axis_name: {code: count}},   # for rare-code analysis
      "test_only_labels": {
        axis_name: {"codes": [...], "test_rows_affected": N}
      },
      "icd11_policy": "gold_in_baheya" | "silver_never_label" | None
    }

Design choices:
- Vocab is built from the **trainable** split only (or `pretrain_train` for
  TCGA). Held-out test split must not influence label space.
- Codes appear in **frequency-descending** order, with alphabetical tiebreak
  for determinism.
- Multilabel axes (icd11 extension columns) accept either:
  * `list[str]` (parquet-native, post-build_m1_model_ready)
  * pipe-separated string (legacy CSV format)
  Both are normalized to a list of codes during build/encode.
- For closed-set axes, no UNK token: unknown codes at encode time return None
  (single-pick) or are silently dropped (multi-label).

Persistence: ``vocab.save_json(path)`` / ``LabelVocab.load_json(path)``.
Plain JSON, human-readable, no pickle.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VOCAB_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Helpers (exported — loaders.py imports _is_null)
# ---------------------------------------------------------------------------


def _is_null(v: Any) -> bool:
    """True for None, NaN, empty string, empty list, and the literal 'nan'."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str):
        s = v.strip()
        return s == "" or s.lower() == "nan"
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == 0
    return False


def _normalize_code(v: Any) -> str:
    """Normalize a code to canonical string form.

    - Floats from pandas (8500.0 because column has NaNs) → '8500'
    - Float-shaped strings ('8500.0') → '8500'
    - Real codes with meaningful decimals ('2C61.0', 'C50.9') survive unchanged.
    - Stripped + uppercased.
    """
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return str(v)
    s = str(v).strip()
    if "." in s:
        head, _, tail = s.partition(".")
        if head.isdigit() and tail and all(ch == "0" for ch in tail):
            s = head
    return s.upper()


def _iter_codes(value: Any, multilabel: bool, delimiter: str = "|") -> list[str]:
    """Extract list of code strings from a cell value.

    Handles all input shapes:
        list[str]       → as-is
        "A|B|C"         → split on delimiter (multilabel only)
        "A"             → [A]
        None / NaN / [] → []
    Always normalized + uppercased.
    """
    if _is_null(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalize_code(t) for t in value if not _is_null(t)]
    if multilabel:
        return [
            _normalize_code(t)
            for t in str(value).split(delimiter)
            if t.strip()
        ]
    return [_normalize_code(value)]


# ---------------------------------------------------------------------------
# AxisVocab
# ---------------------------------------------------------------------------


@dataclass
class AxisVocab:
    """Vocabulary for a single label axis.

    Attributes:
        name: Logical axis name (matches parquet column, e.g. 'icdo3_morphology').
        codes: Frequency-descending ordered list. Index = class index.
        multilabel: True if axis is multi-label (BCE-trained).
        delimiter: Separator for legacy pipe-separated cells.
        counts: Frequency counts on build split.
        meta: Free-form metadata (encoding notes, etc.).
    """

    name: str
    codes: list[str]
    multilabel: bool = False
    delimiter: str = "|"
    counts: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.codes)

    @property
    def code_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.codes)}

    @property
    def column(self) -> str:
        """Parquet column name = axis name in v3 (no column_map indirection)."""
        return self.name

    def encode(self, value: Any) -> int | None:
        """Single-label encode. None if null/unknown.

        Raises if axis is multilabel.
        """
        if self.multilabel:
            raise ValueError(f"Axis {self.name!r} is multilabel; use encode_multi")
        if _is_null(value):
            return None
        codes = _iter_codes(value, multilabel=False)
        if not codes:
            return None
        return self.code_to_idx.get(codes[0])

    def encode_multi(self, value: Any) -> np.ndarray:
        """Multilabel encode → multi-hot float32 vector of length num_classes.

        Null/empty → all zeros. Unknown codes silently dropped.
        Accepts list[str], "A|B|C", or single str.
        """
        vec = np.zeros(self.num_classes, dtype=np.float32)
        if _is_null(value):
            return vec
        tokens = _iter_codes(value, multilabel=self.multilabel, delimiter=self.delimiter)
        for tok in tokens:
            idx = self.code_to_idx.get(tok)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def decode(self, idx: int) -> str:
        return self.codes[idx]

    def rare_codes(self, threshold: int) -> list[str]:
        """Codes with build-time count < threshold."""
        return [c for c in self.codes if self.counts.get(c, 0) < threshold]

    def to_dict(self) -> dict[str, Any]:
        """Subset returned for embedding inside a LabelVocab JSON.

        The full v3 JSON splits codes/counts/meta across top-level keys; this
        method returns the per-axis pieces for callers that want one axis only.
        """
        return {
            "codes": list(self.codes),
            "counts": dict(self.counts),
            "type": "multi_label" if self.multilabel else "single_pick",
            "delimiter": self.delimiter,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        codes: list[str],
        multilabel: bool,
        counts: dict[str, int] | None = None,
        delimiter: str = "|",
        meta: dict[str, Any] | None = None,
    ) -> "AxisVocab":
        return cls(
            name=name,
            codes=list(codes),
            multilabel=multilabel,
            delimiter=delimiter,
            counts=dict(counts or {}),
            meta=dict(meta or {}),
        )

    @classmethod
    def build(
        cls,
        df: pd.DataFrame,
        name: str,
        column: str | None = None,
        multilabel: bool = False,
        delimiter: str = "|",
        min_count: int = 1,
        meta: dict[str, Any] | None = None,
    ) -> "AxisVocab":
        """Build axis vocab from a DataFrame slice.

        Codes sorted by descending frequency, alphabetical tiebreak.
        """
        col = column if column is not None else name
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not in DataFrame for axis {name!r}")

        counter: Counter[str] = Counter()
        for raw in df[col].tolist():
            for tok in _iter_codes(raw, multilabel=multilabel, delimiter=delimiter):
                counter[tok] += 1

        filtered = [(c, n) for c, n in counter.items() if n >= min_count]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        codes = [c for c, _ in filtered]
        counts = dict(filtered)

        return cls(
            name=name,
            codes=codes,
            multilabel=multilabel,
            delimiter=delimiter,
            counts=counts,
            meta=meta or {},
        )


# ---------------------------------------------------------------------------
# LabelVocab (multi-axis container)
# ---------------------------------------------------------------------------


@dataclass
class LabelVocab:
    """Container for all label axes used by a module.

    Attributes:
        module: e.g. 'M1', 'TCGA_pretrain'.
        dataset: e.g. 'baheya_m1', 'tcga_pretrain'.
        axes: Dict of axis_name → AxisVocab.
        built_from_split: Source split label.
        built_at: ISO-8601 UTC timestamp.
        seed: Build seed.
        test_only_labels: Per-axis codes seen in test/val but absent from build split.
        icd11_policy: 'gold_in_baheya', 'silver_never_label', or None.
    """

    module: str
    dataset: str
    axes: dict[str, AxisVocab] = field(default_factory=dict)
    built_from_split: str = ""
    built_at: str = ""
    seed: int = 42
    test_only_labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    icd11_policy: str | None = None

    def __getitem__(self, axis_name: str) -> AxisVocab:
        return self.axes[axis_name]

    def __contains__(self, axis_name: str) -> bool:
        return axis_name in self.axes

    @property
    def axis_names(self) -> list[str]:
        return list(self.axes.keys())

    def cardinalities(self) -> dict[str, int]:
        return {n: a.num_classes for n, a in self.axes.items()}

    def summary(self) -> str:
        lines = [f"LabelVocab[{self.module}/{self.dataset}]:"]
        for name, axis in self.axes.items():
            mode = "multi" if axis.multilabel else "single"
            lines.append(
                f"  {name:24s}  K={axis.num_classes:>3d}  ({mode})"
            )
        return "\n".join(lines)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        axes_codes: dict[str, list[str]] = {}
        axis_meta: dict[str, dict[str, Any]] = {}
        axes_counts: dict[str, dict[str, int]] = {}
        for name, axis in self.axes.items():
            axes_codes[name] = list(axis.codes)
            axes_counts[name] = dict(axis.counts)
            meta_entry: dict[str, Any] = {
                "type": "multi_label" if axis.multilabel else "single_pick",
            }
            if axis.multilabel:
                meta_entry["delimiter"] = axis.delimiter
                meta_entry["null_policy"] = "empty_list_means_none"
            else:
                meta_entry["null_policy"] = "mask_in_loss"
            meta_entry.update(axis.meta)
            axis_meta[name] = meta_entry

        out: dict[str, Any] = {
            "version": VOCAB_SCHEMA_VERSION,
            "module": self.module,
            "dataset": self.dataset,
            "built_from_split": self.built_from_split,
            "built_at": self.built_at,
            "seed": self.seed,
            "ordering": "frequency-descending",
            "axes": axes_codes,
            "axis_meta": axis_meta,
            "axes_counts": axes_counts,
        }
        if self.test_only_labels:
            out["test_only_labels"] = dict(self.test_only_labels)
        if self.icd11_policy is not None:
            out["icd11_policy"] = self.icd11_policy
        return out

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))

    @classmethod
    def load_json(cls, path: str | Path) -> "LabelVocab":
        d = json.loads(Path(path).read_text())
        version = d.get("version", 1)
        if version != VOCAB_SCHEMA_VERSION:
            raise ValueError(
                f"Vocab schema mismatch: file version={version}, "
                f"expected {VOCAB_SCHEMA_VERSION}. "
                f"Regenerate with scripts/build_label_vocab.py."
            )

        axes_codes = d["axes"]
        axis_meta = d.get("axis_meta", {})
        axes_counts = d.get("axes_counts", {})

        axes: dict[str, AxisVocab] = {}
        for name, codes in axes_codes.items():
            meta = dict(axis_meta.get(name, {}))
            multilabel = meta.pop("type", "single_pick") == "multi_label"
            delimiter = meta.pop("delimiter", "|")
            meta.pop("null_policy", None)
            axes[name] = AxisVocab.from_dict(
                name=name,
                codes=codes,
                multilabel=multilabel,
                counts=axes_counts.get(name, {}),
                delimiter=delimiter,
                meta=meta,
            )

        return cls(
            module=d.get("module", ""),
            dataset=d.get("dataset", ""),
            axes=axes,
            built_from_split=d.get("built_from_split", ""),
            built_at=d.get("built_at", ""),
            seed=int(d.get("seed", 42)),
            test_only_labels=dict(d.get("test_only_labels", {})),
            icd11_policy=d.get("icd11_policy"),
        )

    # -- builders ---------------------------------------------------------

    @classmethod
    def build_from_axis_spec(
        cls,
        df: pd.DataFrame,
        single_pick: list[str],
        multi_label: list[str],
        module: str,
        dataset: str,
        built_from_split: str = "trainable",
        seed: int = 42,
        split_column: str = "split",
        icd11_policy: str | None = None,
        axis_meta_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> "LabelVocab":
        """Generic builder: takes axis lists + dataset metadata.

        Filters df to ``built_from_split`` if that column exists.
        """
        if split_column in df.columns:
            build_df = df[df[split_column] == built_from_split].copy()
            if build_df.empty:
                raise ValueError(
                    f"No rows with {split_column}=={built_from_split!r}. "
                    f"Available: {sorted(df[split_column].dropna().unique().tolist())}"
                )
        else:
            build_df = df.copy()

        overlap = set(single_pick) & set(multi_label)
        if overlap:
            raise ValueError(
                f"Axis names appear in both single_pick and multi_label: {sorted(overlap)}"
            )

        axes: dict[str, AxisVocab] = {}
        overrides = axis_meta_overrides or {}

        for axis in single_pick:
            axes[axis] = AxisVocab.build(
                build_df, name=axis, multilabel=False,
                meta=overrides.get(axis, {}),
            )
        for axis in multi_label:
            axes[axis] = AxisVocab.build(
                build_df, name=axis, multilabel=True,
                meta=overrides.get(axis, {}),
            )

        built_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()

        return cls(
            module=module,
            dataset=dataset,
            axes=axes,
            built_from_split=built_from_split,
            built_at=built_at,
            seed=seed,
            icd11_policy=icd11_policy,
        )

    @classmethod
    def build_m1(
        cls,
        df: pd.DataFrame,
        built_from_split: str = "trainable",
        seed: int = 42,
    ) -> "LabelVocab":
        """Convenience builder for Baheya M1.

        Axes per architecture v6 §6.2:
          single-pick (8): 5 ICD-O-3 axes + icd11_stem + icd11_ext_laterality
                           + icd11_ext_grading
          multi-label (2): icd11_ext_anatomy + icd11_ext_histopath
        Excluded: icd11_ext_staging (zero examples in M1 biopsy data).
        WHO ECT axis rules: ext_laterality and ext_grading are mutually
        exclusive (single-pick); confirmed by Baheya audit (0/3055 multi rows).
        """
        single_pick = [
            "icdo3_topography",
            "icdo3_morphology",
            "icdo3_behavior",
            "icdo3_grade",
            "icdo3_laterality",
            "icd11_stem",
            # Per WHO ICD-11 ECT axis rules + data audit (Baheya v3.1: 0/3055
            # multi-code rows): laterality and grading extensions are mutually
            # exclusive single-pick axes, not multi-label. Architecture v6 §6.2.
            "icd11_ext_laterality",
            "icd11_ext_grading",
        ]
        multi_label = [
            "icd11_ext_anatomy",
            "icd11_ext_histopath",
        ]
        return cls.build_from_axis_spec(
            df=df,
            single_pick=single_pick,
            multi_label=multi_label,
            module="M1",
            dataset="baheya_m1",
            built_from_split=built_from_split,
            seed=seed,
            icd11_policy="gold_in_baheya",
        )

    @classmethod
    def build_tcga_pretrain(
        cls,
        df: pd.DataFrame,
        built_from_split: str = "pretrain_train",
        seed: int = 42,
    ) -> "LabelVocab":
        """Builder for TCGA pretrain (silver ICD-11; never used as supervision).

        Per architecture v6 §6.4 + TCGA data card §7: ICD-11 columns are silver
        in TCGA. Vocab includes only ICD-O-3 axes; Phase 1 head trains only on
        these. ``icd11_policy='silver_never_label'`` records the decision.
        """
        single_pick = [
            "icdo3_topography",
            "icdo3_morphology",
            "icdo3_behavior",
            "icdo3_grade",
            "icdo3_laterality",
        ]
        multi_label: list[str] = []
        return cls.build_from_axis_spec(
            df=df,
            single_pick=single_pick,
            multi_label=multi_label,
            module="TCGA_pretrain",
            dataset="tcga_pretrain",
            built_from_split=built_from_split,
            seed=seed,
            icd11_policy="silver_never_label",
        )


# ---------------------------------------------------------------------------
# Test-only / OOV audit
# ---------------------------------------------------------------------------


def aggregate_unknown_codes(
    df: pd.DataFrame, vocab: LabelVocab
) -> dict[str, list[tuple[str, int]]]:
    """Return per-axis list of (code, count) for codes in df not in vocab.

    Used to audit the test split: any code in test but not in build split is
    reported here so it can be excluded from F1 transparently.
    """
    out: dict[str, list[tuple[str, int]]] = {}
    for name, axis in vocab.axes.items():
        if name not in df.columns:
            continue
        seen: Counter[str] = Counter()
        for raw in df[name].tolist():
            for tok in _iter_codes(raw, multilabel=axis.multilabel, delimiter=axis.delimiter):
                if tok not in axis.code_to_idx:
                    seen[tok] += 1
        out[name] = sorted(seen.items(), key=lambda x: -x[1])
    return out
