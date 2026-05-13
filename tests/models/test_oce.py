"""Unit tests for src/models/oce.py.

Uses the same tiny-BERT fixture pattern as test_encoder.py to avoid network
downloads. End-to-end forward and gradient propagation are verified.
"""

from __future__ import annotations

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from src.models.encoder import OncologyEncoder
from src.models.heads import NULL_TARGET_SENTINEL
from src.models.oce import OCEOutput, OncologyCodingEngine


# -----------------------------------------------------------------------------
# Fixtures (mirrors test_encoder.py's tiny-BERT setup)
# -----------------------------------------------------------------------------

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
    p = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    p.write_text("\n".join(_VOCAB_TOKENS))
    return p


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_vocab_file):
    return BertTokenizerFast(
        vocab_file=str(tiny_vocab_file),
        do_lower_case=True,
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
        cls_token="[CLS]", mask_token="[MASK]",
    )


@pytest.fixture(scope="module")
def tiny_bert_config():
    return BertConfig(
        vocab_size=len(_VOCAB_TOKENS),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_encoder(monkeypatch, tiny_tokenizer, tiny_bert_config):
    """Tiny OncologyEncoder for OCE tests (no network)."""
    import src.models.encoder as enc_mod

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return tiny_tokenizer

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            cfg = tiny_bert_config
            if "attn_implementation" in kwargs:
                cfg = BertConfig(
                    **{**cfg.to_dict(), "_attn_implementation": kwargs["attn_implementation"]}
                )
            return BertModel(cfg)

    monkeypatch.setattr(enc_mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(enc_mod, "AutoModel", _FakeAutoModel)

    encoder = OncologyEncoder(
        backbone_name="fake/tiny-bert",
        segment_size=32,
        max_segments=4,
        strict_length=False,
    )
    encoder.eval()
    return encoder


# Reusable specs for the OCE tests — small enough for fast tests but match the
# *shape* of the real M1 architecture (single-pick + multi-label).
SINGLE_PICK_SPEC_SMALL = {
    "icdo3_topography": 4,
    "icdo3_morphology": 6,
    "icd11_stem": 5,
}
MULTI_LABEL_SPEC_SMALL = {
    "icd11_ext_anatomy": 3,
    "icd11_ext_histopath": 4,
}


@pytest.fixture
def oce(tiny_encoder):
    """A small OCE instance using the tiny encoder."""
    model = OncologyCodingEngine(
        encoder=tiny_encoder,
        single_pick_spec=SINGLE_PICK_SPEC_SMALL,
        multi_label_spec=MULTI_LABEL_SPEC_SMALL,
    )
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


class TestConstruction:

    def test_basic_construction(self, oce):
        # Heads count: 3 single-pick + 2 multi-label = 5 total
        assert len(oce.heads.heads) == 5
        # Attention should match: one module per axis
        assert len(oce.attention.attentions) == 5
        # All five axes appear in both attention and heads
        assert set(oce.attention.attentions.keys()) == set(oce.heads.heads.keys())

    def test_axis_overlap_rejected(self, tiny_encoder):
        """Same axis name in both single_pick and multi_label specs is invalid."""
        with pytest.raises(ValueError, match="appear in both"):
            OncologyCodingEngine(
                encoder=tiny_encoder,
                single_pick_spec={"axis_x": 3},
                multi_label_spec={"axis_x": 5},
            )

    def test_attention_uses_encoder_hidden_dim(self, oce, tiny_encoder):
        """Attention modules should be sized to match encoder.hidden_dim."""
        assert oce.attention.hidden_dim == tiny_encoder.hidden_dim
        for axis_attn in oce.attention.attentions.values():
            assert axis_attn.hidden_dim == tiny_encoder.hidden_dim

    def test_heads_use_encoder_hidden_dim(self, oce, tiny_encoder):
        for head in oce.heads.heads.values():
            assert head.hidden_dim == tiny_encoder.hidden_dim

    def test_attention_label_counts_match_heads(self, oce):
        """Critical correctness invariant: each axis's attention K must equal
        its head's K, otherwise the head would receive context tensors of
        wrong shape."""
        for axis in oce.heads.heads:
            head_k = oce.heads.heads[axis].num_labels
            attn_k = oce.attention.attentions[axis].num_labels
            assert head_k == attn_k, (
                f"Axis {axis}: head K={head_k} but attention K={attn_k}"
            )

    def test_custom_attn_dim(self, tiny_encoder):
        """attn_dim should be passed through to all axis-attention modules."""
        oce = OncologyCodingEngine(
            encoder=tiny_encoder,
            single_pick_spec={"a": 3},
            multi_label_spec={"b": 4},
            attn_dim=64,
        )
        for axis_attn in oce.attention.attentions.values():
            assert axis_attn.attn_dim == 64


# -----------------------------------------------------------------------------
# Forward pass
# -----------------------------------------------------------------------------


class TestForward:

    def test_forward_returns_oce_output(self, oce):
        out = oce(["invasive ductal carcinoma grade ii"])
        assert isinstance(out, OCEOutput)

    def test_forward_logits_shapes(self, oce):
        """Each axis's logits must be [B, K_axis]."""
        out = oce(["carcinoma ductal", "lobular invasive"])
        B = 2
        for axis, logits in out.logits_per_axis.items():
            expected_K = (
                SINGLE_PICK_SPEC_SMALL.get(axis)
                or MULTI_LABEL_SPEC_SMALL.get(axis)
            )
            assert logits.shape == (B, expected_K), (
                f"Axis {axis}: expected ({B}, {expected_K}), got {logits.shape}"
            )

    def test_forward_attention_weights_shapes(self, oce):
        """Attention weights for each axis must be [B, K_axis, T_eff]."""
        texts = ["carcinoma", "lobular invasive ductal carcinoma grade ii"]
        out = oce(texts)
        B = len(texts)
        max_segs = out.n_segments.max().item()
        T_eff = max_segs * oce.encoder.segment_size

        for axis, weights in out.attention_weights_per_axis.items():
            expected_K = (
                SINGLE_PICK_SPEC_SMALL.get(axis)
                or MULTI_LABEL_SPEC_SMALL.get(axis)
            )
            assert weights.shape == (B, expected_K, T_eff), (
                f"Axis {axis}: expected ({B}, {expected_K}, {T_eff}), "
                f"got {weights.shape}"
            )

    def test_forward_n_segments_populated(self, oce):
        out = oce(["short", "much longer text to force segmentation " * 10])
        assert out.n_segments.shape == (2,)
        assert out.n_segments[0].item() == 1
        assert out.n_segments[1].item() >= 1

    def test_encoder_attentions_off_by_default(self, oce):
        out = oce(["carcinoma"])
        assert out.encoder_attentions is None

    def test_encoder_attentions_returned_when_requested(self, oce):
        out = oce(["carcinoma"], return_encoder_attentions=True)
        assert out.encoder_attentions is not None
        assert len(out.encoder_attentions) == oce.encoder.backbone.config.num_hidden_layers

    def test_forward_all_axes_present(self, oce):
        """Every axis from both specs must appear in logits and weights."""
        out = oce(["carcinoma"])
        expected = set(SINGLE_PICK_SPEC_SMALL) | set(MULTI_LABEL_SPEC_SMALL)
        assert set(out.logits_per_axis.keys()) == expected
        assert set(out.attention_weights_per_axis.keys()) == expected


# -----------------------------------------------------------------------------
# Loss computation
# -----------------------------------------------------------------------------


class TestComputeLosses:

    def _make_targets(self, B: int = 2) -> dict:
        return {
            # Single-pick: LongTensor [B] with class indices in [0, K)
            "icdo3_topography": torch.zeros(B, dtype=torch.long),
            "icdo3_morphology": torch.zeros(B, dtype=torch.long),
            "icd11_stem": torch.zeros(B, dtype=torch.long),
            # Multi-label: FloatTensor [B, K] with values in {0, 1}
            "icd11_ext_anatomy": torch.zeros(B, 3, dtype=torch.float),
            "icd11_ext_histopath": torch.zeros(B, 4, dtype=torch.float),
        }

    def test_compute_losses_all_active(self, oce):
        """With all heads active, every axis appears in losses dict."""
        out = oce(["carcinoma", "lobular"])
        losses = oce.compute_losses(out, self._make_targets(B=2))
        assert set(losses.keys()) == set(SINGLE_PICK_SPEC_SMALL) | set(MULTI_LABEL_SPEC_SMALL)
        for axis, (loss, n_valid) in losses.items():
            assert loss.item() >= 0
            assert n_valid > 0

    def test_compute_losses_with_null_targets(self, oce):
        """Single-pick with NULL targets should mask those records from loss."""
        out = oce(["carcinoma", "lobular"])
        targets = self._make_targets(B=2)
        # Mark record 0 as null for icdo3_morphology
        targets["icdo3_morphology"] = torch.tensor([NULL_TARGET_SENTINEL, 1])
        losses = oce.compute_losses(out, targets)
        loss_morph, n_valid_morph = losses["icdo3_morphology"]
        assert n_valid_morph == 1, "Only one valid record after null masking"

    def test_compute_losses_phase_icdo3_only(self, oce):
        """When phase = icdo3_only, ICD-11 axes must NOT appear in losses."""
        oce.set_phase("icdo3_only")
        out = oce(["carcinoma"])
        # Provide only icdo3 targets (icd11 axes are inactive so missing OK).
        targets = {
            "icdo3_topography": torch.zeros(1, dtype=torch.long),
            "icdo3_morphology": torch.zeros(1, dtype=torch.long),
        }
        losses = oce.compute_losses(out, targets)
        assert "icd11_stem" not in losses
        assert "icd11_ext_anatomy" not in losses
        assert "icd11_ext_histopath" not in losses


# -----------------------------------------------------------------------------
# Gradient flow (proves the full stack is connected)
# -----------------------------------------------------------------------------


class TestGradientFlow:

    def test_backward_updates_encoder_attention_and_heads(self, oce):
        """Critical end-to-end test: a forward + loss + backward must produce
        nonzero gradients in the encoder, attention, AND heads. If any of
        these has zero gradient, the model isn't actually trainable end-to-end."""
        oce.train()
        out = oce(["carcinoma ductal grade ii"])
        targets = {
            "icdo3_topography": torch.zeros(1, dtype=torch.long),
            "icdo3_morphology": torch.zeros(1, dtype=torch.long),
            "icd11_stem": torch.zeros(1, dtype=torch.long),
            "icd11_ext_anatomy": torch.zeros(1, 3, dtype=torch.float),
            "icd11_ext_histopath": torch.zeros(1, 4, dtype=torch.float),
        }
        losses = oce.compute_losses(out, targets)
        total = sum(loss for loss, _ in losses.values())
        total.backward()

        # Encoder: at least one parameter has nonzero grad
        encoder_grad_present = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in oce.encoder.parameters()
        )
        assert encoder_grad_present, "No gradient reached the encoder"

        # Attention: at least one parameter per axis has nonzero grad
        for axis, axis_attn in oce.attention.attentions.items():
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in axis_attn.parameters()
            ), f"No gradient reached attention for axis {axis}"

        # Heads: every active head has nonzero projection grad
        for axis, head in oce.heads.heads.items():
            if head.head_active:
                grad = head.projection.weight.grad
                assert grad is not None and grad.abs().sum() > 0, (
                    f"No gradient reached head for axis {axis}"
                )

    def test_inactive_heads_get_zero_gradient_after_backward(self, oce):
        """In phase 'icdo3_only', the ICD-11 heads must NOT receive gradients."""
        oce.train()
        oce.set_phase("icdo3_only")

        out = oce(["carcinoma"])
        targets = {
            "icdo3_topography": torch.zeros(1, dtype=torch.long),
            "icdo3_morphology": torch.zeros(1, dtype=torch.long),
        }
        losses = oce.compute_losses(out, targets)
        total = sum(loss for loss, _ in losses.values())
        total.backward()

        # icd11_stem head must have no gradient (or zero gradient)
        stem_grad = oce.heads.heads["icd11_stem"].projection.weight.grad
        assert stem_grad is None or stem_grad.abs().sum() == 0, (
            "Inactive icd11_stem head received gradient — head_active broken"
        )
        # Same for ICD-11 extension heads
        for axis in ["icd11_ext_anatomy", "icd11_ext_histopath"]:
            grad = oce.heads.heads[axis].projection.weight.grad
            assert grad is None or grad.abs().sum() == 0, (
                f"Inactive {axis} head received gradient"
            )


# -----------------------------------------------------------------------------
# Phase delegation
# -----------------------------------------------------------------------------


class TestPhaseControl:

    def test_set_phase_delegates_to_heads(self, oce):
        oce.set_phase("icdo3_only")
        assert "icd11_stem" not in oce.active_axes
        assert "icdo3_morphology" in oce.active_axes

    def test_set_active_axes_delegates(self, oce):
        oce.set_active_axes({"icdo3_morphology"})
        assert oce.active_axes == ["icdo3_morphology"]

    def test_phase_not_in_state_dict(self, oce):
        """head_active must NOT round-trip through state_dict.
        After load_state_dict, phase must be set explicitly by the trainer."""
        oce.set_phase("icdo3_only")
        sd = oce.state_dict()
        # No state dict key should encode head_active
        assert not any("head_active" in k for k in sd.keys())


# -----------------------------------------------------------------------------
# from_config
# -----------------------------------------------------------------------------


class TestFromConfig:

    def test_from_config_full_stack(self, monkeypatch, tiny_tokenizer,
                                     tiny_bert_config):
        """Build OCE from a YAML-style config dict end-to-end."""
        import src.models.encoder as enc_mod

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return tiny_tokenizer

        class _FakeAutoModel:
            @staticmethod
            def from_pretrained(name, **kwargs):
                cfg = tiny_bert_config
                if "attn_implementation" in kwargs:
                    cfg = BertConfig(**{
                        **cfg.to_dict(),
                        "_attn_implementation": kwargs["attn_implementation"],
                    })
                return BertModel(cfg)

        monkeypatch.setattr(enc_mod, "AutoTokenizer", _FakeAutoTokenizer)
        monkeypatch.setattr(enc_mod, "AutoModel", _FakeAutoModel)

        cfg = {
            "encoder": {
                "backbone_name": "fake/tiny-bert",
                "segment_size": 32,
                "max_segments": 4,
                "strict_length": False,
            },
            "heads": {
                "single_pick": {"icdo3_topography": 4, "icdo3_morphology": 6},
                "multi_label": {"icd11_ext_anatomy": 3},
            },
            "attention": {"attn_dim": 32},
        }
        model = OncologyCodingEngine.from_config(cfg)
        assert len(model.heads.heads) == 3
        assert model.attention.attentions["icdo3_topography"].attn_dim == 32

    def test_from_config_missing_encoder_raises(self):
        with pytest.raises(KeyError, match="encoder"):
            OncologyCodingEngine.from_config({"heads": {}})

    def test_from_config_missing_heads_raises(self, monkeypatch, tiny_tokenizer,
                                                tiny_bert_config):
        import src.models.encoder as enc_mod
        monkeypatch.setattr(
            enc_mod, "AutoTokenizer",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: tiny_tokenizer)}),
        )
        monkeypatch.setattr(
            enc_mod, "AutoModel",
            type("F", (), {"from_pretrained": staticmethod(lambda *a, **k: BertModel(tiny_bert_config))}),
        )
        with pytest.raises(KeyError, match="heads"):
            OncologyCodingEngine.from_config({
                "encoder": {"backbone_name": "fake/tiny", "segment_size": 32, "max_segments": 4}
            })


# -----------------------------------------------------------------------------
# Checkpoint round-trip (load_state_dict between phases)
# -----------------------------------------------------------------------------


class TestCheckpointRoundTrip:

    def test_state_dict_roundtrip_preserves_weights(self, oce, tiny_encoder):
        """Save state_dict, instantiate fresh OCE, load — weights must match."""
        # Snapshot trained-ish weights (just take whatever is initialized)
        sd = oce.state_dict()

        # Build a fresh OCE with same specs
        fresh = OncologyCodingEngine(
            encoder=tiny_encoder,
            single_pick_spec=SINGLE_PICK_SPEC_SMALL,
            multi_label_spec=MULTI_LABEL_SPEC_SMALL,
        )
        # Fresh has different random init; load original's state_dict
        missing, unexpected = fresh.load_state_dict(sd, strict=True)
        # Standard load_state_dict returns lists; for strict=True these are empty
        assert list(missing) == []
        assert list(unexpected) == []

        # Verify a sample parameter matches
        original_w = oce.heads.heads["icdo3_morphology"].projection.weight.detach()
        loaded_w = fresh.heads.heads["icdo3_morphology"].projection.weight.detach()
        assert torch.equal(original_w, loaded_w)

    def test_phase_must_be_set_after_load(self, oce, tiny_encoder):
        """The CRITICAL gotcha: head_active is NOT in state_dict.
        After load_state_dict, the trainer must explicitly set the phase.
        """
        oce.set_phase("icdo3_only")
        sd = oce.state_dict()

        # Fresh model defaults to all-active
        fresh = OncologyCodingEngine(
            encoder=tiny_encoder,
            single_pick_spec=SINGLE_PICK_SPEC_SMALL,
            multi_label_spec=MULTI_LABEL_SPEC_SMALL,
        )
        fresh.load_state_dict(sd, strict=True)

        # IMPORTANT: After load_state_dict, fresh is still all-active because
        # head_active was never serialized. This is the contract.
        assert "icd11_stem" in fresh.active_axes, (
            "After load_state_dict, fresh model should still have all heads "
            "active — phase is a training-time decision, not a checkpoint property"
        )
        # Trainer must explicitly call set_phase after loading.
        fresh.set_phase("icdo3_only")
        assert "icd11_stem" not in fresh.active_axes
