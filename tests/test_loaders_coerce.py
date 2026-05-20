"""Guards loaders.py Task-4 edits.

_coerce_group_id: Baheya int path byte-identical; TCGA hex path stable int64.
split_by_value: returns the named split; raises loud on typo / missing col.
"""
import pandas as pd
import pytest

from src.data.loaders import _coerce_group_id, split_by_value


# ---- _coerce_group_id ----

def test_int_path_unchanged():
    assert _coerce_group_id(42) == 42
    assert _coerce_group_id("42") == 42  # int-like string still int path


def test_null_sentinel():
    assert _coerce_group_id(None) == -1


def test_hex_string_stable_int64():
    h = _coerce_group_id("7e74cc46e44fdba7")  # real TCGA group id form
    assert isinstance(h, int)
    assert 0 <= h < 2**63  # LongTensor-safe
    assert h == _coerce_group_id("7e74cc46e44fdba7")  # deterministic


def test_distinct_hex_distinct_ids():
    assert _coerce_group_id("abc123") != _coerce_group_id("def456")


# ---- split_by_value ----

def _df():
    return pd.DataFrame(
        {"split": ["pretrain_train", "pretrain_train", "pretrain_val",
                   "pretrain_test"],
         "record_id": [1, 2, 3, 4]}
    )


def test_split_by_value_pulls_named_split():
    out = split_by_value(_df(), "pretrain_train")
    assert len(out) == 2
    assert list(out["record_id"]) == [1, 2]
    assert list(out.index) == [0, 1]  # reset


def test_split_by_value_val():
    assert len(split_by_value(_df(), "pretrain_val")) == 1


def test_split_by_value_missing_value_raises():
    with pytest.raises(ValueError, match="No rows with split=="):
        split_by_value(_df(), "typo_split")


def test_split_by_value_missing_column_raises():
    with pytest.raises(KeyError, match="no 'split' column"):
        split_by_value(pd.DataFrame({"x": [1]}), "pretrain_train")
