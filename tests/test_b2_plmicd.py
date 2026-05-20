"""Guards B2's flat-119 adapter: target build + per-axis decode.

No torch model / encoder needed — tests FlatLabelSpace logic in isolation
with a stub vocab. The faithful-PLM-ICD claim hinges on this mapping being
exactly right (axis order, single-pick one-hot placement, multilabel slice,
null = no positive), so it is unit-locked.
"""
import numpy as np
import torch

from scripts.run_plmicd_baseline import FlatLabelSpace


class _StubVocab:
    """Minimal vocab exposing axes -> code list, like LabelVocab."""

    def __init__(self, axes: dict[str, list[str]]):
        self.axes = axes


class _StubCfg:
    """Just the attrs FlatLabelSpace reads off BenchmarkConfig."""

    class _AT:
        single_pick = ["a_sp", "b_sp"]
        multilabel = ["c_ml"]

    def __init__(self):
        self.axis_types = _StubCfg._AT()
        self.label_cardinalities = {"a_sp": 2, "b_sp": 3, "c_ml": 2}

    @property
    def all_axes(self):
        return self.axis_types.single_pick + self.axis_types.multilabel


def _space():
    vocab = _StubVocab({
        "a_sp": ["a0", "a1"],          # K=2
        "b_sp": ["b0", "b1", "b2"],    # K=3
        "c_ml": ["c0", "c1"],          # K=2
    })
    return FlatLabelSpace(vocab, _StubCfg())


def test_offsets_and_total():
    s = _space()
    assert s.total == 7
    assert s.offset == {"a_sp": 0, "b_sp": 2, "c_ml": 5}


def test_total_mismatch_raises():
    bad_vocab = _StubVocab({"a_sp": ["a0"], "b_sp": ["b0", "b1", "b2"],
                            "c_ml": ["c0", "c1"]})  # a_sp K=1, declared 2
    import pytest
    with pytest.raises(ValueError, match="Vocab/config drift"):
        FlatLabelSpace(bad_vocab, _StubCfg())


def test_build_targets_singlepick_and_null():
    s = _space()
    collated = {
        "texts": ["x", "y"],
        # row0: a_sp=1, b_sp=null(-1) ; row1: a_sp=null, b_sp=2
        "labels_a_sp": torch.tensor([1, -1], dtype=torch.long),
        "labels_b_sp": torch.tensor([-1, 2], dtype=torch.long),
        "labels_c_ml": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    }
    y = s.build_targets(collated).numpy()
    assert y.shape == (2, 7)
    # row0: a_sp idx1 -> flat col 0+1=1 ; b_sp null -> no positive in [2:5]
    assert y[0, 1] == 1.0
    assert y[0, 0] == 0.0
    assert y[0, 2:5].sum() == 0.0
    # row0 c_ml: c0 positive -> flat col 5
    assert y[0, 5] == 1.0 and y[0, 6] == 0.0
    # row1: a_sp null -> [0:2] all zero ; b_sp idx2 -> flat col 2+2=4
    assert y[1, 0:2].sum() == 0.0
    assert y[1, 4] == 1.0
    # row1 c_ml: no positive
    assert y[1, 5:7].sum() == 0.0


def test_decode_axis_slices_correctly():
    s = _space()
    flat = np.arange(2 * 7, dtype=float).reshape(2, 7)  # [[0..6],[7..13]]
    a = s.decode_axis(flat, "a_sp")
    b = s.decode_axis(flat, "b_sp")
    c = s.decode_axis(flat, "c_ml")
    assert a.shape == (2, 2) and np.allclose(a[0], [0, 1])
    assert b.shape == (2, 3) and np.allclose(b[0], [2, 3, 4])
    assert c.shape == (2, 2) and np.allclose(c[1], [12, 13])


def test_multilabel_threshold_binarized():
    s = _space()
    collated = {
        "texts": ["x"],
        "labels_a_sp": torch.tensor([0], dtype=torch.long),
        "labels_b_sp": torch.tensor([0], dtype=torch.long),
        "labels_c_ml": torch.tensor([[0.9, 0.3]]),  # >0.5 -> 1, else 0
    }
    y = s.build_targets(collated).numpy()
    assert y[0, 5] == 1.0 and y[0, 6] == 0.0
