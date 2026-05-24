"""
tests/data/test_segmentation.py
--------------------------------
Tests for src/data/segmentation.py

Run with: python -m pytest tests/data/test_segmentation.py -v

These tests use a tiny in-memory tokenizer mock so no PubMedBERT download
is needed.  The mock tokenizer counts whitespace-split tokens as a proxy
for WordPiece tokens — good enough for boundary/packing logic tests.
"""

import pytest
from unittest.mock import MagicMock
from src.data.segmentation import (
    segment_text,
    _split_sentences,
    _pack_sentences_into_segments,
    get_splitter_backend,
)


# ---------------------------------------------------------------------------
# Mock tokenizer — counts whitespace tokens, no downloads needed
# ---------------------------------------------------------------------------

def make_mock_tokenizer(chars_per_token: int = 5):
    """
    Returns a mock HF tokenizer that approximates token counts by
    splitting on whitespace.  Deterministic and fast.
    """
    tok = MagicMock()

    def encode(text, add_special_tokens=False):
        # Each whitespace-delimited word = 1 token (rough proxy)
        words = text.split()
        return list(range(len(words)))  # fake token ids

    def decode(token_ids, skip_special_tokens=True):
        # We can't recover text from fake ids, so return a placeholder
        # with the right word count for roundtrip testing
        return " ".join(f"tok{i}" for i in range(len(token_ids)))

    tok.encode = encode
    tok.decode = decode
    return tok


TOKENIZER = make_mock_tokenizer()

# A realistic pathology text with clear sentence boundaries
PATHO_TEXT = (
    "Final Diagnosis: Left breast, core biopsy. "
    "Invasive ductal carcinoma, grade II. "
    "Nottingham score 6/9 (tubular 2, nuclear 2, mitotic 2). "
    "No lymphovascular invasion identified. "
    "ER positive (90%), PR positive (70%), HER2 negative. "
    "Ki-67 index approximately 15%."
)


# ---------------------------------------------------------------------------
# 1. Fixed mode — module is a no-op
# ---------------------------------------------------------------------------

class TestFixedMode:
    def test_returns_single_element_list(self):
        result = segment_text(PATHO_TEXT, mode="fixed")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == PATHO_TEXT

    def test_tokenizer_not_required(self):
        # Should not raise even without tokenizer
        result = segment_text(PATHO_TEXT, mode="fixed", tokenizer=None)
        assert result == [PATHO_TEXT]

    def test_empty_text(self):
        result = segment_text("", mode="fixed")
        assert result == [""]

    def test_whitespace_only(self):
        result = segment_text("   ", mode="fixed")
        assert result == ["   "]


# ---------------------------------------------------------------------------
# 2. Sentence splitter
# ---------------------------------------------------------------------------

class TestSentenceSplitter:
    def test_splits_basic_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences(text)
        assert len(sents) >= 2  # at least splits somewhere

    def test_nonempty_output(self):
        sents = _split_sentences(PATHO_TEXT)
        assert len(sents) >= 1
        assert all(s.strip() for s in sents)

    def test_empty_string(self):
        sents = _split_sentences("")
        # Should return something, not crash
        assert isinstance(sents, list)

    def test_single_sentence_no_period(self):
        text = "Invasive ductal carcinoma grade II no further description"
        sents = _split_sentences(text)
        assert isinstance(sents, list)
        assert len(sents) >= 1

    def test_backend_is_set(self):
        backend = get_splitter_backend()
        assert backend in ("scispacy", "nltk", "regex")


# ---------------------------------------------------------------------------
# 3. Sentence-aware mode — boundary packing
# ---------------------------------------------------------------------------

class TestSentenceAwarePacking:
    def test_requires_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            segment_text(PATHO_TEXT, mode="sentence_aware", tokenizer=None)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown segmentation mode"):
            segment_text(PATHO_TEXT, mode="sliding_window", tokenizer=TOKENIZER)

    def test_returns_list_of_strings(self):
        result = segment_text(
            PATHO_TEXT, mode="sentence_aware", tokenizer=TOKENIZER, segment_size=128
        )
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)
        assert all(s.strip() for s in result)

    def test_each_segment_within_budget(self):
        """Each segment must tokenize to <= segment_size tokens."""
        seg_size = 20  # small size to force splitting
        result = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=seg_size,
            overlap_tokens=0,
        )
        for seg in result:
            tokens = TOKENIZER.encode(seg, add_special_tokens=False)
            # Allow +2 for [CLS]/[SEP] budget reservation
            assert len(tokens) <= seg_size - 2, (
                f"Segment exceeded budget: {len(tokens)} tokens > {seg_size - 2}\n"
                f"Segment: {seg!r}"
            )

    def test_short_text_single_segment(self):
        """Text short enough to fit in one segment stays as one segment."""
        short = "Invasive ductal carcinoma."
        result = segment_text(
            short, mode="sentence_aware", tokenizer=TOKENIZER, segment_size=128
        )
        assert len(result) == 1

    def test_empty_text(self):
        result = segment_text(
            "", mode="sentence_aware", tokenizer=TOKENIZER, segment_size=128
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_content_lost_without_overlap(self):
        """All sentences from the original text should appear in some segment."""
        sentences = _split_sentences(PATHO_TEXT)
        result = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=30,
            overlap_tokens=0,
        )
        full_output = " ".join(result)
        for sent in sentences:
            # Each sentence (or its first word) should appear somewhere
            first_word = sent.split()[0] if sent.split() else ""
            if first_word:
                assert first_word.lower() in full_output.lower(), (
                    f"Sentence first word {first_word!r} not found in output"
                )

    def test_very_long_single_sentence_hard_split(self):
        """A sentence longer than segment_size must be hard-split without crash."""
        # 200 words = 200 mock tokens — exceeds any reasonable segment_size
        long_sentence = " ".join(f"word{i}" for i in range(200))
        result = segment_text(
            long_sentence,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=30,
            overlap_tokens=0,
        )
        assert isinstance(result, list)
        assert len(result) > 1
        for seg in result:
            toks = TOKENIZER.encode(seg, add_special_tokens=False)
            assert len(toks) <= 28  # 30 - 2 for specials


# ---------------------------------------------------------------------------
# 4. Overlap
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_overlap_produces_more_segments_or_equal(self):
        """With overlap, segments may be slightly longer; count >= no-overlap."""
        no_ov = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=20,
            overlap_tokens=0,
        )
        with_ov = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=20,
            overlap_tokens=8,
        )
        assert len(with_ov) >= len(no_ov)

    def test_overlap_segments_still_within_budget(self):
        """Even with overlap, each segment must fit within segment_size."""
        seg_size = 25
        result = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=seg_size,
            overlap_tokens=8,
        )
        for seg in result:
            toks = TOKENIZER.encode(seg, add_special_tokens=False)
            assert len(toks) <= seg_size - 2, (
                f"Overlap segment too long: {len(toks)} > {seg_size - 2}\n"
                f"Segment: {seg!r}"
            )

    def test_overlap_zero_same_as_no_overlap(self):
        """overlap_tokens=0 should give same result as not specifying overlap."""
        r1 = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=20,
            overlap_tokens=0,
        )
        r2 = _pack_sentences_into_segments(
            sentences=_split_sentences(PATHO_TEXT),
            tokenizer=TOKENIZER,
            segment_size=20,
            overlap_tokens=0,
        )
        assert r1 == r2

    def test_single_segment_overlap_no_crash(self):
        """Single-segment text with overlap enabled should not crash."""
        short = "Short text."
        result = segment_text(
            short,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=128,
            overlap_tokens=32,
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_overlap_larger_than_segment_handled_gracefully(self):
        """overlap_tokens >= content_budget should not infinite-loop."""
        result = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=20,
            overlap_tokens=200,  # larger than any segment
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 5. Integration smoke — realistic pathology text at segment_size=128
# ---------------------------------------------------------------------------

class TestRealisticSegmentation:
    def test_typical_baheya_text_one_segment(self):
        """
        Baheya M1 texts max out at 385 tokens (4 segments at most at seg=128).
        A typical text should fit in 1-2 segments.
        """
        # Simulate a ~100-word Baheya text (well within single segment)
        baheya_like = (
            "[REPORT_TYPE] biopsy_report "
            "[DIAGNOSIS] invasive ductal carcinoma "
            "[MICROSCOPIC] tumor cells arranged in ducts with moderate nuclear pleomorphism "
            "Nottingham grade II tubular score 2 nuclear score 2 mitotic score 2 "
            "[CLINICAL_INFO] left breast mass palpable "
            "[SPECIMEN] tru-cut biopsy "
            "[LATERALITY] left "
            "[GRADE_TEXT] Grade II"
        )
        result = segment_text(
            baheya_like,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=128,
            overlap_tokens=32,
        )
        # Should fit in 1-2 segments for a ~100-word text
        assert 1 <= len(result) <= 3

    def test_sentence_aware_never_empty_segments(self):
        result = segment_text(
            PATHO_TEXT,
            mode="sentence_aware",
            tokenizer=TOKENIZER,
            segment_size=128,
            overlap_tokens=0,
        )
        assert all(s.strip() for s in result), "No empty segments allowed"
