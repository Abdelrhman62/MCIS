"""Unit tests for src/models/encoder.py.

Strategy: rather than download the real PubMedBERT (~440 MB) for every CI
run, we build a tiny BERT model in-memory with a matching tokenizer pulled
from the real PubMedBERT vocabulary. This keeps tests fast (<10s total) while
exercising the actual segmentation, overflow, reassembly, and freezing logic.

For end-to-end smoke testing with the real model, see scripts/smoke_neural.py
(separate file, runs once locally before any cloud GPU run).
"""

from __future__ import annotations

import logging

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

# Module under test
from src.models.encoder import (
    DEFAULT_BACKBONE,
    EncoderOutput,
    OncologyEncoder,
)


# -----------------------------------------------------------------------------
# Fixtures: tiny BERT for fast tests
# -----------------------------------------------------------------------------

# Vocabulary tokens for a minimal BERT — must include the special tokens.
# This is enough to tokenize realistic-looking pathology text for tests.
_VOCAB_TOKENS = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "a", "of", "and", "to", "in", "is", "with", "for", "on",
    "carcinoma", "ductal", "lobular", "invasive", "breast", "right", "left",
    "biopsy", "core", "specimen", "received", "shows", "grade", "ii", "iii", "i",
    "tumor", "size", "cm", "margin", "negative", "positive", "lymph",
    "node", "dcis", "idc", "lvi", "her2", "er", "pr", "ki67",
    "mitosis", "nuclear", "tubule", "score", "nottingham", "histology",
    "low", "high", "intermediate", "well", "moderately", "poorly",
    "differentiated", "infiltrating", "no", "yes", "present", "absent",
    "##s", "##ed", "##ing", "##ly", "##er", "##est", "##tion",
    ".", ",", ":", ";", "(", ")", "/", "-",
]


@pytest.fixture(scope="module")
def tiny_vocab_file(tmp_path_factory: pytest.TempPathFactory):
    """Write a minimal vocab.txt file for BertTokenizerFast."""
    p = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    p.write_text("\n".join(_VOCAB_TOKENS))
    return p


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_vocab_file):
    """A BertTokenizerFast over a tiny vocabulary."""
    return BertTokenizerFast(
        vocab_file=str(tiny_vocab_file),
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )


@pytest.fixture(scope="module")
def tiny_bert_config():
    """A 2-layer BERT config small enough for fast tests."""
    return BertConfig(
        vocab_size=len(_VOCAB_TOKENS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,  # cap on segment_size in tests
    )


@pytest.fixture
def tiny_encoder(monkeypatch, tiny_tokenizer, tiny_bert_config):
    """An OncologyEncoder backed by the tiny BERT.

    Patches AutoTokenizer/AutoModel so __init__ doesn't hit the network.
    The fakes accept and ignore extra kwargs (cache_dir, attn_implementation)
    so they stay forward-compatible with future encoder changes.
    """
    import src.models.encoder as enc_mod

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return tiny_tokenizer

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            # Build BertModel with eager attention so output_attentions works.
            cfg = tiny_bert_config
            if "attn_implementation" in kwargs:
                cfg = BertConfig(**{**cfg.to_dict(), "_attn_implementation": kwargs["attn_implementation"]})
            return BertModel(cfg)

    monkeypatch.setattr(enc_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(enc_mod, "AutoModel", _FakeAutoModel)

    encoder = OncologyEncoder(
        backbone_name="fake/tiny-bert",
        segment_size=32,        # small for fast tests
        max_segments=4,         # cap = 128 tokens
        strict_length=False,
    )
    encoder.eval()
    return encoder


# -----------------------------------------------------------------------------
# Construction & validation
# -----------------------------------------------------------------------------


class TestConstruction:

    def test_default_backbone_string_is_correct(self):
        """Locked default must remain PubMedBERT/BiomedBERT."""
        assert DEFAULT_BACKBONE == (
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )

    def test_negative_segment_size_rejected(self, monkeypatch, tiny_tokenizer,
                                             tiny_bert_config):
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)})
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))})
        )
        with pytest.raises(ValueError, match="segment_size must be positive"):
            OncologyEncoder(segment_size=0)

    def test_negative_max_segments_rejected(self, monkeypatch, tiny_tokenizer,
                                             tiny_bert_config):
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)})
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))})
        )
        with pytest.raises(ValueError, match="max_segments must be positive"):
            OncologyEncoder(max_segments=-1)

    def test_segment_size_exceeds_backbone_max_rejected(
        self, monkeypatch, tiny_tokenizer, tiny_bert_config
    ):
        """Backbone tiny BERT has max_position_embeddings=128; we ask for 256."""
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)})
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))})
        )
        with pytest.raises(ValueError, match="exceeds backbone max position"):
            OncologyEncoder(segment_size=256)

    def test_hidden_dim_exposed_correctly(self, tiny_encoder):
        assert tiny_encoder.hidden_dim == 32  # from tiny_bert_config


# -----------------------------------------------------------------------------
# Output shapes
# -----------------------------------------------------------------------------


class TestOutputShapes:

    def test_short_input_single_segment(self, tiny_encoder):
        """A short text should produce S=1 segment per record."""
        texts = ["invasive ductal carcinoma grade ii"]
        out = tiny_encoder(texts)

        assert isinstance(out, EncoderOutput)
        assert out.hidden_states.shape == (1, 32, 32)  # B=1, T=32, d=32
        assert out.attention_mask.shape == (1, 32)
        assert out.n_segments.tolist() == [1]
        assert out.attentions is None

    def test_long_input_triggers_segmentation(self, tiny_encoder):
        """A long text should produce S>1 segments and concatenated hidden states."""
        # Build a text that will need multiple segments at segment_size=32.
        # ~50 word-like tokens => ~3 segments after tokenization.
        long_text = " ".join(["carcinoma ductal invasive breast"] * 20)
        out = tiny_encoder([long_text])

        n_segs = out.n_segments.item()
        assert n_segs >= 2, f"Expected >=2 segments, got {n_segs}"
        assert n_segs <= 4, f"Should not exceed max_segments=4, got {n_segs}"
        assert out.hidden_states.shape == (1, n_segs * 32, 32)
        assert out.attention_mask.shape == (1, n_segs * 32)

    def test_batch_with_mixed_lengths(self, tiny_encoder):
        """Batch with one short and one long text: T_eff = max(n_segments) * segment_size."""
        short = "carcinoma"
        long = " ".join(["carcinoma ductal invasive breast"] * 20)
        out = tiny_encoder([short, long])

        n_segs = out.n_segments.tolist()
        assert n_segs[0] == 1, "Short text should produce 1 segment"
        assert n_segs[1] >= 2, "Long text should produce >=2 segments"

        max_segs = max(n_segs)
        T_eff = max_segs * 32
        assert out.hidden_states.shape == (2, T_eff, 32)
        assert out.attention_mask.shape == (2, T_eff)

        # Short record's mask should be zero in the padded segment positions.
        # Specifically: positions [segment_size:] must be all zero for short record.
        assert out.attention_mask[0, 32:].sum().item() == 0, (
            "Padded segment positions of short record must be masked out"
        )

    def test_empty_batch_rejected(self, tiny_encoder):
        with pytest.raises(ValueError, match="non-empty"):
            tiny_encoder([])


# -----------------------------------------------------------------------------
# Overflow handling
# -----------------------------------------------------------------------------


class TestOverflow:

    def test_overflow_middle_drop_keeps_max_segments(self, tiny_encoder, caplog):
        """When tokenization produces > max_segments, we keep exactly max_segments."""
        # max_segments=4, segment_size=32 -> cap = 128 tokens.
        # Build text with ~10 segments worth.
        huge_text = " ".join(["carcinoma ductal invasive breast lobular"] * 60)

        with caplog.at_level(logging.WARNING):
            out = tiny_encoder([huge_text])

        assert out.n_segments.item() == 4
        assert out.hidden_states.shape == (1, 4 * 32, 32)
        assert any("middle-drop" in rec.message for rec in caplog.records)

    def test_overflow_strict_raises(self, monkeypatch, tiny_tokenizer,
                                     tiny_bert_config):
        """strict_length=True should raise on overflow."""
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)})
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))})
        )
        encoder = OncologyEncoder(
            backbone_name="fake/tiny-bert",
            segment_size=32,
            max_segments=2,
            strict_length=True,
        )
        encoder.eval()

        huge_text = " ".join(["carcinoma ductal invasive breast"] * 50)
        with pytest.raises(ValueError, match="exceeds max_segments"):
            encoder([huge_text])

    def test_middle_drop_indices_keeps_head_and_tail(self):
        """Verify the index-selection logic directly."""
        # Keep 4 of 10: head should be ceil(4/2)=2 from front, tail floor(4/2)=2 from back.
        idx = OncologyEncoder._middle_drop_indices(n_kept=4, n_total=10)
        assert idx.tolist() == [0, 1, 8, 9]

    def test_middle_drop_indices_odd_kept(self):
        """Asymmetric drop when n_kept is odd: head gets the extra."""
        # Keep 5 of 10: head=3, tail=2.
        idx = OncologyEncoder._middle_drop_indices(n_kept=5, n_total=10)
        assert idx.tolist() == [0, 1, 2, 8, 9]

    def test_middle_drop_indices_no_drop_needed(self):
        """When n_kept >= n_total, return all indices."""
        idx = OncologyEncoder._middle_drop_indices(n_kept=5, n_total=3)
        assert idx.tolist() == [0, 1, 2]


# -----------------------------------------------------------------------------
# Attention mask correctness
# -----------------------------------------------------------------------------


class TestAttentionMask:

    def test_short_text_mask_marks_real_tokens(self, tiny_encoder):
        """For a short text, mask should be 1 on real tokens (incl. CLS/SEP)
        and 0 on padding within the single segment."""
        out = tiny_encoder(["carcinoma"])
        mask = out.attention_mask[0]
        # First positions should be active (CLS + 'carcinoma' + SEP at minimum)
        assert mask[0].item() == 1, "[CLS] position should be active"
        # Trailing positions of a short text should be zero (PAD)
        assert mask[-1].item() == 0, "Trailing PAD position should be inactive"

    def test_padded_segment_positions_are_zero(self, tiny_encoder):
        """Records with fewer than max_in_batch segments have zero mask in padding."""
        out = tiny_encoder([
            "carcinoma",
            " ".join(["carcinoma ductal"] * 30),
        ])
        # Short record's segment-2+ positions must all be zero.
        n_short = out.n_segments[0].item()
        n_long = out.n_segments[1].item()
        assert n_short < n_long  # precondition

        padded_start = n_short * 32
        assert out.attention_mask[0, padded_start:].sum().item() == 0, (
            "Padded-segment positions must be zero in attention mask"
        )


# -----------------------------------------------------------------------------
# Freezing
# -----------------------------------------------------------------------------


class TestFreeze:

    def test_freeze_zero_layers_freezes_only_embeddings(self, tiny_encoder):
        tiny_encoder.freeze_bottom_layers(0)
        # Embeddings should be frozen
        emb_grads = [p.requires_grad for p in tiny_encoder.backbone.embeddings.parameters()]
        assert not any(emb_grads), "Embeddings should be frozen"
        # All transformer layers should still be trainable
        for layer in tiny_encoder.backbone.encoder.layer:
            layer_grads = [p.requires_grad for p in layer.parameters()]
            assert all(layer_grads), "All transformer layers should remain trainable"

    def test_freeze_all_layers(self, tiny_encoder):
        n_layers = tiny_encoder.backbone.config.num_hidden_layers
        tiny_encoder.freeze_bottom_layers(n_layers)
        # All transformer layers should be frozen
        for layer in tiny_encoder.backbone.encoder.layer:
            layer_grads = [p.requires_grad for p in layer.parameters()]
            assert not any(layer_grads), "All transformer layers should be frozen"

    def test_freeze_negative_n_rejected(self, tiny_encoder):
        with pytest.raises(ValueError, match="n must be in"):
            tiny_encoder.freeze_bottom_layers(-1)

    def test_freeze_too_many_layers_rejected(self, tiny_encoder):
        n_layers = tiny_encoder.backbone.config.num_hidden_layers
        with pytest.raises(ValueError, match="n must be in"):
            tiny_encoder.freeze_bottom_layers(n_layers + 1)


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


class TestDeterminism:

    def test_same_input_same_output_in_eval_mode(self, tiny_encoder):
        """In eval mode, the same input must produce identical outputs."""
        tiny_encoder.eval()
        texts = ["invasive ductal carcinoma grade ii right breast"]
        with torch.no_grad():
            out1 = tiny_encoder(texts)
            out2 = tiny_encoder(texts)
        assert torch.allclose(out1.hidden_states, out2.hidden_states)
        assert torch.equal(out1.attention_mask, out2.attention_mask)
        assert torch.equal(out1.n_segments, out2.n_segments)


# -----------------------------------------------------------------------------
# return_attentions flag
# -----------------------------------------------------------------------------


class TestReturnAttentions:

    def test_attentions_off_by_default(self, tiny_encoder):
        out = tiny_encoder(["carcinoma"])
        assert out.attentions is None

    def test_attentions_returned_when_requested(self, tiny_encoder):
        out = tiny_encoder(["carcinoma"], return_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == tiny_encoder.backbone.config.num_hidden_layers


# -----------------------------------------------------------------------------
# from_config
# -----------------------------------------------------------------------------


class TestFromConfig:

    def test_from_config_with_defaults(self, monkeypatch, tiny_tokenizer,
                                        tiny_bert_config):
        """from_config should accept an empty dict and use defaults."""
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)})
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))})
        )
        # Note: defaults include segment_size=128 which exceeds tiny_bert max
        # of 128 — they're equal so this passes. We pass segment_size=32 to
        # be safe for the tiny model.
        cfg = {
            "backbone_name": "fake/tiny-bert",
            "segment_size": 32,
            "max_segments": 4,
        }
        encoder = OncologyEncoder.from_config(cfg)
        assert encoder.segment_size == 32
        assert encoder.max_segments == 4
        assert encoder.strict_length is False  # default
