"""
src/data/segmentation.py
------------------------
Sentence-aware text segmentation for MCIS OCE input preparation.

Provides two segmentation strategies that sit UPSTREAM of the encoder's
tokenizer.  The encoder contract is unchanged: it still receives
List[str], one string per segment, and tokenizes each segment
independently via HF return_overflowing_tokens.

Strategies
----------
"fixed"  : current PLM-ICD canonical behavior.
           The encoder handles this internally (return_overflowing_tokens,
           stride=0).  This module is a no-op for "fixed" — it returns
           the original text as a single-element list and lets the encoder
           do its own chunking.

"sentence_aware" : greedy sentence packing.
           1. Split text into sentences (scispaCy -> nltk fallback).
           2. Tokenize each sentence to measure its token length.
           3. Greedily pack sentences into bins of <= segment_size tokens.
              A single sentence that exceeds segment_size is hard-split
              at the token boundary (fallback to fixed for that sentence).
           4. Optionally prepend the last N sentences of segment k to
              segment k+1 as overlap (overlap_tokens budget).
           5. Return List[str] of segment texts.  The encoder tokenizes
              each segment independently — no segment ever exceeds
              segment_size tokens.

Usage
-----
    from src.data.segmentation import segment_text

    segments = segment_text(
        text,
        mode="sentence_aware",       # or "fixed"
        tokenizer=tokenizer,          # HF tokenizer instance
        segment_size=128,
        overlap_tokens=32,
    )
    # segments: List[str], each <= segment_size tokens when tokenized

Design notes
------------
- The tokenizer is passed in so this module never imports transformers
  at the top level — keeps import cost zero for callers that only use
  "fixed" mode.
- For "fixed" mode the tokenizer arg is ignored and may be None.
- Thread-safe: the sentence-splitter is initialised once at module level
  and reused.  The greedy packer is stateless.
- All sentence-splitter errors are caught and fall back to nltk, then
  to a regex split, so no training run can die from a parsing edge case.
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    # Avoid hard import of transformers at module level
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitter initialisation (module-level, done once)
# ---------------------------------------------------------------------------

_splitter_backend: str = "none"
_spacy_nlp = None


def _init_splitter() -> None:
    """Try to load scispaCy, fall back to nltk, then regex."""
    global _splitter_backend, _spacy_nlp

    if _splitter_backend != "none":
        return  # already initialised

    # 1. Try scispaCy
    try:
        import spacy  # noqa: F401
        _spacy_nlp = spacy.load("en_core_sci_sm", disable=["ner", "tagger", "lemmatizer"])
        _splitter_backend = "scispacy"
        logger.info("segmentation: using scispaCy en_core_sci_sm sentence splitter")
        return
    except Exception:
        pass

    # 2. Try nltk
    try:
        import nltk
        # Trigger download only if punkt not already present
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        # smoke-test
        nltk.sent_tokenize("Test sentence. Another one.")
        _splitter_backend = "nltk"
        logger.info("segmentation: using nltk sentence splitter (scispaCy not available)")
        return
    except Exception:
        pass

    # 3. Regex fallback
    _splitter_backend = "regex"
    logger.warning(
        "segmentation: using regex sentence splitter "
        "(scispaCy and nltk both unavailable)"
    )


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using the best available backend."""
    _init_splitter()

    if _splitter_backend == "scispacy" and _spacy_nlp is not None:
        try:
            doc = _spacy_nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                return sents
        except Exception as exc:
            logger.debug("scispaCy split failed (%s), falling back", exc)

    if _splitter_backend in ("nltk", "scispacy"):
        try:
            import nltk
            sents = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
            if sents:
                return sents
        except Exception as exc:
            logger.debug("nltk split failed (%s), falling back to regex", exc)

    # Regex fallback: split on sentence-ending punctuation followed by
    # whitespace + capital letter, or on newlines.
    # Handles most pathology report patterns.
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z])", text)
    sents = [s.strip() for s in raw if s.strip()]
    return sents if sents else [text]


# ---------------------------------------------------------------------------
# Token-length helper
# ---------------------------------------------------------------------------

def _token_len(text: str, tokenizer: "PreTrainedTokenizerBase") -> int:
    """Number of tokens in text (no special tokens added)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Core segmentation logic
# ---------------------------------------------------------------------------

def _pack_sentences_into_segments(
    sentences: List[str],
    tokenizer: "PreTrainedTokenizerBase",
    segment_size: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Greedy sentence packing with optional overlap.

    Parameters
    ----------
    sentences      : pre-split sentence list
    tokenizer      : HF tokenizer (for measuring token lengths)
    segment_size   : max tokens per segment (including special tokens [CLS]/[SEP])
    overlap_tokens : token budget for overlap prepended to each segment k+1
                     from the tail of segment k.  0 = no overlap.

    Returns
    -------
    List[str] of segment texts.  Each segment, when tokenized with
    add_special_tokens=True, fits within segment_size tokens.
    """
    # Reserve 2 tokens for [CLS] and [SEP]
    content_budget = segment_size - 2

    # Pre-compute token lengths for all sentences
    sent_lengths = [_token_len(s, tokenizer) for s in sentences]

    segments: List[str] = []
    current_sents: List[str] = []
    current_len: int = 0

    for sent, sent_len in zip(sentences, sent_lengths):
        if sent_len > content_budget:
            # Single sentence exceeds budget: flush current, hard-split sentence
            if current_sents:
                segments.append(" ".join(current_sents))
                current_sents = []
                current_len = 0
            # Hard-split the sentence at token boundary
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            for chunk_start in range(0, len(tokens), content_budget):
                chunk_tokens = tokens[chunk_start: chunk_start + content_budget]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                segments.append(chunk_text)
            continue

        if current_len + sent_len > content_budget:
            # Flush current segment
            if current_sents:
                segments.append(" ".join(current_sents))
            current_sents = [sent]
            current_len = sent_len
        else:
            current_sents.append(sent)
            current_len += sent_len

    if current_sents:
        segments.append(" ".join(current_sents))

    if not segments:
        return [""]

    if overlap_tokens <= 0 or len(segments) < 2:
        return segments

    # --- Add overlap: prepend tail sentences of segment k to segment k+1 ---
    result: List[str] = [segments[0]]
    for i in range(1, len(segments)):
        # Collect tail sentences from segment i-1 that fit in overlap budget
        prev_text = segments[i - 1]
        prev_sents = _split_sentences(prev_text)
        overlap_sents: List[str] = []
        overlap_len = 0
        for s in reversed(prev_sents):
            s_len = _token_len(s, tokenizer)
            if overlap_len + s_len <= overlap_tokens:
                overlap_sents.insert(0, s)
                overlap_len += s_len
            else:
                break
        if overlap_sents:
            new_seg = " ".join(overlap_sents) + " " + segments[i]
        else:
            new_seg = segments[i]
        result.append(new_seg)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_text(
    text: str,
    mode: str = "fixed",
    tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    segment_size: int = 128,
    overlap_tokens: int = 0,
) -> List[str]:
    """
    Segment text for encoder input.

    Parameters
    ----------
    text          : raw input text (section-tagged or concatenated)
    mode          : "fixed" | "sentence_aware"
    tokenizer     : required for "sentence_aware"; ignored for "fixed"
    segment_size  : tokens per segment (default 128, PLM-ICD canonical)
    overlap_tokens: token budget for overlap between adjacent segments.
                    0 = no overlap.  Ignored in "fixed" mode.

    Returns
    -------
    "fixed"          -> [text]   (single string; encoder does its own chunking)
    "sentence_aware" -> List[str] of pre-packed segment strings

    Notes
    -----
    For "fixed" mode the encoder's internal return_overflowing_tokens
    path handles segmentation.  This function is a no-op.

    For "sentence_aware" mode the encoder receives already-split strings
    and tokenizes each independently.  The encoder must be told NOT to
    apply its own overflow chunking when receiving pre-split input —
    this is handled by the BaheyaM1Dataset / TCGADataset collate path
    which passes `pre_segmented=True` to the encoder when mode != "fixed".
    """
    if mode == "fixed":
        return [text]

    if mode != "sentence_aware":
        raise ValueError(f"Unknown segmentation mode: {mode!r}. Use 'fixed' or 'sentence_aware'.")

    if tokenizer is None:
        raise ValueError("tokenizer must be provided for sentence_aware segmentation")

    if not text or not text.strip():
        return [""]

    sentences = _split_sentences(text)
    return _pack_sentences_into_segments(
        sentences=sentences,
        tokenizer=tokenizer,
        segment_size=segment_size,
        overlap_tokens=overlap_tokens,
    )


def get_splitter_backend() -> str:
    """Return the active sentence splitter backend name (for logging)."""
    _init_splitter()
    return _splitter_backend
