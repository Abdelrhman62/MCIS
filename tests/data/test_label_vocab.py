"""Unit tests for src.data.label_vocab (v3 schema).

Run: pytest tests/data/test_label_vocab.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.label_vocab import (
    VOCAB_SCHEMA_VERSION,
    AxisVocab,
    LabelVocab,
    _is_null,
    _iter_codes,
    _normalize_code,
    aggregate_unknown_codes,
)


# ---------------------------------------------------------------------------
# Helpers: _is_null
# ---------------------------------------------------------------------------


class TestIsNull:
    def test_none(self) -> None:
        assert _is_null(None)

    def test_nan_float(self) -> None:
        assert _is_null(math.nan)

    def test_empty_string(self) -> None:
        assert _is_null("")
        assert _is_null("   ")

    def test_string_nan(self) -> None:
        assert _is_null("nan")
        assert _is_null("NaN")
        assert _is_null("NAN")

    def test_empty_list_is_null(self) -> None:
        # multi-label parquet stores empty list = no labels
        assert _is_null([])
        assert _is_null(())
        assert _is_null(np.array([]))

    def test_real_values_not_null(self) -> None:
        assert not _is_null(0)
        assert not _is_null(0.0)
        assert not _is_null("0")
        assert not _is_null("8500")
        assert not _is_null(False)
        assert not _is_null(["XA1", "XA2"])


# ---------------------------------------------------------------------------
# Helpers: _normalize_code
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_pure_int_float_strips_dot_zero(self) -> None:
        assert _normalize_code(8500.0) == "8500"
        assert _normalize_code(3.0) == "3"

    def test_string_with_dot_zero_strips(self) -> None:
        assert _normalize_code("8500.0") == "8500"
        assert _normalize_code("8500.00") == "8500"

    def test_real_icd_codes_survive(self) -> None:
        # ICD-11 stems with meaningful decimals
        assert _normalize_code("2C61.0") == "2C61.0"
        assert _normalize_code("2C61.1") == "2C61.1"
        # ICD-O-3 topography
        assert _normalize_code("C50.2") == "C50.2"
        assert _normalize_code("C50.9") == "C50.9"

    def test_case_normalization(self) -> None:
        assert _normalize_code("xh1yz3") == "XH1YZ3"
        assert _normalize_code("2c61.0") == "2C61.0"

    def test_strip_whitespace(self) -> None:
        assert _normalize_code("  XH1YZ3  ") == "XH1YZ3"


# ---------------------------------------------------------------------------
# Helpers: _iter_codes (new in v3)
# ---------------------------------------------------------------------------


class TestIterCodes:
    def test_list_input_passthrough(self) -> None:
        assert _iter_codes(["XA1", "XA2"], multilabel=True) == ["XA1", "XA2"]

    def test_pipe_string_split_when_multilabel(self) -> None:
        assert _iter_codes("XA1|XA2|XA3", multilabel=True) == ["XA1", "XA2", "XA3"]

    def test_pipe_string_NOT_split_when_single(self) -> None:
        # Single-label axis: don't split, just normalize the whole string.
        # In practice single-pick fields don't contain pipes; this is defensive.
        assert _iter_codes("8500", multilabel=False) == ["8500"]

    def test_null_returns_empty(self) -> None:
        assert _iter_codes(None, multilabel=True) == []
        assert _iter_codes([], multilabel=True) == []
        assert _iter_codes("", multilabel=False) == []
        assert _iter_codes(math.nan, multilabel=False) == []

    def test_normalizes_in_list(self) -> None:
        assert _iter_codes([8500.0, "8520.0"], multilabel=True) == ["8500", "8520"]


# ---------------------------------------------------------------------------
# AxisVocab.build
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Toy DataFrame: morphology (single), icd11_ext_anatomy (multi as list)."""
    return pd.DataFrame(
        {
            "icdo3_morphology": [8500.0, 8500.0, 8500.0, 8520.0, 8470.0, None, "nan", 8500.0],
            "icd11_stem": ["2C61.0", "2C61.0", "2C61.1", "2C61.0", "2C61.0", "2C61.1", "2E65", None],
            "icd11_ext_anatomy": [
                ["XA3LS6"],
                ["XA2Q54"],
                ["XA3LS6", "XA2Q54"],
                None,
                ["XA3LS6"],
                ["XA3LS6"],
                ["XA2Q54"],
                [],
            ],
        }
    )


class TestAxisVocabBuild:
    def test_frequency_descending(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icdo3_morphology")
        # 8500 appears 4 times → index 0
        assert v.codes[0] == "8500"
        assert v.code_to_idx["8500"] == 0

    def test_alphabetical_tiebreak(self) -> None:
        df = pd.DataFrame({"c": ["b", "a", "c", "b", "a", "c"]})
        v = AxisVocab.build(df, "c", "c")
        # All three appear twice → alphabetical
        assert v.codes == ["A", "B", "C"]

    def test_skip_nulls(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icdo3_morphology")
        assert "nan" not in v.codes
        assert "" not in v.codes
        assert "None" not in v.codes

    def test_multilabel_list_input(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        # XA3LS6: rows 0, 2, 4, 5 = 4 occurrences
        # XA2Q54: rows 1, 2, 6 = 3 occurrences
        assert v.codes[0] == "XA3LS6"
        assert v.code_to_idx["XA2Q54"] == 1
        assert v.counts["XA3LS6"] == 4

    def test_min_count_filters(self) -> None:
        df = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 2 + ["c"] * 1})
        v = AxisVocab.build(df, "c", "c", min_count=3)
        assert v.codes == ["A"]

    def test_column_defaults_to_name(self, small_df: pd.DataFrame) -> None:
        # column param omitted → uses name as column
        v = AxisVocab.build(small_df, "icdo3_morphology")
        assert v.column == "icdo3_morphology"

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError, match="Column 'xxx'"):
            AxisVocab.build(df, "xxx")


# ---------------------------------------------------------------------------
# AxisVocab.encode / encode_multi
# ---------------------------------------------------------------------------


class TestAxisVocabEncode:
    def test_encode_single(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icdo3_morphology")
        assert v.encode(8500.0) == 0
        assert v.encode("8500") == 0
        assert v.encode("8500.0") == 0
        assert v.encode(None) is None
        assert v.encode(math.nan) is None
        assert v.encode("") is None

    def test_encode_unknown_returns_none(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icdo3_morphology")
        assert v.encode(9999) is None

    def test_encode_multi_list_returns_multihot(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        out = v.encode_multi(["XA3LS6", "XA2Q54"])
        assert out.shape == (v.num_classes,)
        assert out[v.code_to_idx["XA3LS6"]] == 1.0
        assert out[v.code_to_idx["XA2Q54"]] == 1.0
        assert out.sum() == 2.0

    def test_encode_multi_pipe_str_still_works(self, small_df: pd.DataFrame) -> None:
        # Backward compat: legacy pipe-separated string also works
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        out = v.encode_multi("XA3LS6|XA2Q54")
        assert out.sum() == 2.0

    def test_encode_multi_null_returns_zeros(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        for null_val in [None, [], math.nan, ""]:
            out = v.encode_multi(null_val)
            assert out.sum() == 0.0

    def test_encode_multi_unknown_silently_dropped(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        out = v.encode_multi(["XA3LS6", "UNSEEN"])
        assert out.sum() == 1.0

    def test_encode_raises_on_multilabel_axis(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icd11_ext_anatomy", multilabel=True)
        with pytest.raises(ValueError, match="multilabel"):
            v.encode("XA3LS6")

    def test_decode_round_trip(self, small_df: pd.DataFrame) -> None:
        v = AxisVocab.build(small_df, "icdo3_morphology")
        for c in v.codes:
            assert v.decode(v.encode(c)) == c

    def test_rare_codes(self) -> None:
        df = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 3 + ["c"] * 1})
        v = AxisVocab.build(df, "c", "c")
        assert v.rare_codes(5) == ["B", "C"]
        assert v.rare_codes(2) == ["C"]
        assert v.rare_codes(1) == []


# ---------------------------------------------------------------------------
# LabelVocab.build_m1
# ---------------------------------------------------------------------------


@pytest.fixture
def m1_like_df() -> pd.DataFrame:
    """Tiny DataFrame matching M1 schema enough for build_m1.

    All 10 axes present, list-typed multi-label cols.
    """
    return pd.DataFrame(
        {
            "split": ["trainable"] * 6 + ["test"] * 2,
            "icdo3_morphology": [8500.0, 8500.0, 8520.0, 8500.0, 8470.0, 8500.0, 8500.0, 8503.0],
            "icdo3_topography": ["C50.2"] * 6 + ["C50.2", "C50.4"],
            "icdo3_behavior": ["3"] * 8,
            "icdo3_grade": ["2", "2", "2", "3", "1", "2", "2", "3"],
            "icdo3_laterality": ["L", "R", "L", "L", "R", "L", "R", "L"],
            "icd11_stem": ["2C61.0", "2C61.0", "2C61.1", "2C61.0", "2C61.0", "2C61.0", "2C61.0", "2C61.1"],
            "icd11_ext_anatomy": [["XA3LS6"], [], ["XA2Q54"], ["XA3LS6"], [], ["XA3LS6"], ["XA3LS6"], ["XA2Q54"]],
            "icd11_ext_histopath": [[], ["XH1YZ3"], ["XH1YZ3", "XH4TA4"], [], ["XH1YZ3"], [], [], ["XH8UE4"]],
            "icd11_ext_laterality": [["XK8G"], ["XK8H"], ["XK8G"], ["XK8G"], ["XK8H"], ["XK8G"], ["XK8G"], ["XK8G"]],
            "icd11_ext_grading": [["XS58"]] * 8,
        }
    )


class TestLabelVocabBuildM1:
    def test_basic_build(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        assert v.module == "M1"
        assert v.dataset == "baheya_m1"
        assert v.icd11_policy == "gold_in_baheya"
        # Trainable-only build (6 rows): morphology codes {8500, 8520, 8470} → K=3
        assert v["icdo3_morphology"].num_classes == 3
        assert v["icd11_stem"].num_classes == 2
        # Multi-label axes (BCE): post-coordinated anatomy + histopath
        assert v["icd11_ext_anatomy"].multilabel
        assert v["icd11_ext_histopath"].multilabel
        # Single-pick axes (masked CE): WHO ECT axis rules — laterality and
        # grading are mutually exclusive. Arch v6 §6.2.
        assert not v["icd11_ext_laterality"].multilabel
        assert not v["icd11_ext_grading"].multilabel
        assert "icd11_ext_staging" not in v.axes  # excluded

    def test_full_axis_set(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        # All 10 axes from architecture v6 §6.2
        expected = {
            "icdo3_topography", "icdo3_morphology", "icdo3_behavior",
            "icdo3_grade", "icdo3_laterality",
            "icd11_stem",
            "icd11_ext_anatomy", "icd11_ext_histopath",
            "icd11_ext_laterality", "icd11_ext_grading",
        }
        assert set(v.axes.keys()) == expected

    def test_summary_runs(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        s = v.summary()
        assert "icdo3_morphology" in s
        assert "K=" in s


class TestLabelVocabBuildTcga:
    def test_tcga_excludes_icd11(self, m1_like_df: pd.DataFrame) -> None:
        # Reuse same df with split renamed
        df = m1_like_df.copy()
        df["split"] = df["split"].map({"trainable": "pretrain_train", "test": "pretrain_test"})
        v = LabelVocab.build_tcga_pretrain(df)
        assert v.module == "TCGA_pretrain"
        assert v.icd11_policy == "silver_never_label"
        # TCGA vocab has only 5 ICD-O-3 axes; no ICD-11
        for axis in ("icd11_stem", "icd11_ext_anatomy", "icd11_ext_histopath"):
            assert axis not in v.axes
        assert "icdo3_morphology" in v.axes


# ---------------------------------------------------------------------------
# Persistence (v3 schema)
# ---------------------------------------------------------------------------


class TestLabelVocabPersistence:
    def test_save_load_round_trip(self, m1_like_df: pd.DataFrame, tmp_path) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        p = tmp_path / "vocab.json"
        v.save_json(p)
        v2 = LabelVocab.load_json(p)
        assert v.cardinalities() == v2.cardinalities()
        assert v.module == v2.module
        assert v.dataset == v2.dataset
        assert v.icd11_policy == v2.icd11_policy
        for axis_name in v.axes:
            assert v[axis_name].codes == v2[axis_name].codes
            assert v[axis_name].multilabel == v2[axis_name].multilabel

    def test_v3_format_keys(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        d = v.to_dict()
        assert d["version"] == VOCAB_SCHEMA_VERSION == 3
        assert d["ordering"] == "frequency-descending"
        assert "axes" in d
        assert "axis_meta" in d
        assert "axes_counts" in d
        # axes is dict of axis -> list (NOT dict of axis -> dict-of-fields)
        assert isinstance(d["axes"]["icdo3_morphology"], list)
        # axis_meta has type field
        assert d["axis_meta"]["icdo3_morphology"]["type"] == "single_pick"
        assert d["axis_meta"]["icd11_ext_anatomy"]["type"] == "multi_label"

    def test_load_rejects_old_version(self, tmp_path) -> None:
        old = tmp_path / "v1_vocab.json"
        old.write_text('{"version": 1, "module": "M1", "axes": {}}')
        with pytest.raises(ValueError, match="Vocab schema mismatch"):
            LabelVocab.load_json(old)


# ---------------------------------------------------------------------------
# Test-only / OOV audit
# ---------------------------------------------------------------------------


class TestAggregateUnknownCodes:
    def test_finds_unknowns(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        test_df = m1_like_df[m1_like_df["split"] == "test"]
        out = aggregate_unknown_codes(test_df, v)
        # morphology 8503 in test but not trainable → unknown
        morph_unk = dict(out["icdo3_morphology"])
        assert "8503" in morph_unk

    def test_known_codes_not_reported(self, m1_like_df: pd.DataFrame) -> None:
        v = LabelVocab.build_m1(m1_like_df)
        test_df = m1_like_df[m1_like_df["split"] == "test"]
        out = aggregate_unknown_codes(test_df, v)
        stem_unk = dict(out["icd11_stem"])
        assert "2C61.0" not in stem_unk
        assert "2C61.1" not in stem_unk
