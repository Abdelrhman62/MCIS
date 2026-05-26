"""Section normalizer for cross-institution pathology report format alignment.

Converts unstructured pathology report text (e.g. TCGA surgical pathology)
into the section-tagged format used by Baheya training data.

Baheya format uses explicit section tags:
    [REPORT_TYPE] biopsy_report
    [DIAGNOSIS] Invasive duct carcinoma, grade II...
    [MICROSCOPIC] Invasive duct carcinoma grade II with...
    [CLINICAL_INFO] Lt. breast UQ suspicious lesion...

TCGA format wraps everything in a single blob:
    [REPORT_TYPE] surgical_pathology
    [FULL_REPORT_TEXT] FINAL DIAGNOSIS: ... GROSS DESCRIPTION: ...

This module bridges that gap by detecting section boundaries in
unstructured text using regex patterns derived from a comprehensive
audit of 1,025 TCGA-BRCA reports (2026-05-25).

Usage:
    from src.data.section_normalizer import normalize_sections, extract_diagnosis

    # Full normalization: re-tag sections to match Baheya format
    normalized = normalize_sections(raw_tcga_text)

    # Diagnosis-only: extract just the diagnosis section
    diagnosis = extract_diagnosis(raw_tcga_text)
"""
from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section detection patterns (from TCGA-BRCA audit, 2026-05-25)
# ---------------------------------------------------------------------------
# Ordered by specificity — more specific patterns first to avoid partial matches.
# Each tuple: (compiled_regex, baheya_tag)

_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # DIAGNOSIS variants (348 DIAGNOSIS + 194 FINAL DIAGNOSIS + 71 CLINICAL DIAGNOSIS)
    (re.compile(r'(?i)(?:^|\n)\s*(?:FINAL\s+)?DIAGNOSIS\s*:', re.MULTILINE), '[DIAGNOSIS]'),
    (re.compile(r'(?i)(?:^|\n)\s*CLINICAL\s+DIAGNOSIS\s*:', re.MULTILINE), '[DIAGNOSIS]'),
    # OCR-corrupted variants found in TCGA (FINAL UIAGNOSIS, DIABNOSIS)
    (re.compile(r'(?i)(?:^|\n)\s*(?:FINAL\s+)?[UD]IA[BG]NOSIS\s*:', re.MULTILINE), '[DIAGNOSIS]'),

    # MICROSCOPIC variants (147 MICROSCOPIC DESCRIPTION + 22 MICROSCOPIC EXAMINATION)
    (re.compile(r'(?i)(?:^|\n)\s*MICROSCOPIC\s+(?:DESCRIPTION|EXAMINATION|FINDINGS)\s*:', re.MULTILINE), '[MICROSCOPIC]'),
    (re.compile(r'(?i)(?:^|\n)\s*HISTOLOGIC\s+(?:DESCRIPTION|EXAMINATION|FINDINGS)\s*:', re.MULTILINE), '[MICROSCOPIC]'),

    # GROSS variants (345 GROSS DESCRIPTION + 58 GROSS EXAMINATION + 2 GROSS FINDINGS)
    (re.compile(r'(?i)(?:^|\n)\s*(?:GROSS|MACROSCOPIC)\s+(?:DESCRIPTION|EXAMINATION|FINDINGS)\s*:', re.MULTILINE), '[GROSS]'),

    # CLINICAL INFO variants (255 CLINICAL HISTORY + 8 HISTORY + 25 PATIENT HISTORY)
    (re.compile(r'(?i)(?:^|\n)\s*(?:CLINICAL|PATIENT)\s+(?:HISTORY|INFO|INFORMATION|DATA)\s*:', re.MULTILINE), '[CLINICAL_INFO]'),
    (re.compile(r'(?i)(?:^|\n)\s*HISTORY\s*:', re.MULTILINE), '[CLINICAL_INFO]'),
    # OCR-corrupted variants (PAIIENI MISTOKY, FATIENI MISTURT)
    (re.compile(r'(?i)(?:^|\n)\s*[A-Z]+\s+MIST[A-Z]+\s*:', re.MULTILINE), '[CLINICAL_INFO]'),

    # SPECIMEN
    (re.compile(r'(?i)(?:^|\n)\s*SPECIMENS?\s*(?:SUBMITTED|RECEIVED)?\s*:', re.MULTILINE), '[SPECIMEN]'),

    # SYNOPTIC / STAGING (51 PATHOLOGIC STAGING)
    (re.compile(r'(?i)(?:^|\n)\s*(?:PATHOLOGIC(?:AL)?\s+)?STAGING\s*:', re.MULTILINE), '[STAGING]'),

    # COMMENT / NOTE (215 COMMENT + 219 NOTE)
    (re.compile(r'(?i)(?:^|\n)\s*COMMENT\s*:', re.MULTILINE), '[COMMENT]'),
    (re.compile(r'(?i)(?:^|\n)\s*(?:CLINICAL\s+)?NOTE\s*:', re.MULTILINE), '[COMMENT]'),

    # IMPRESSION (2 records)
    (re.compile(r'(?i)(?:^|\n)\s*(?:FINAL\s+)?IMPRESSION\s*:', re.MULTILINE), '[DIAGNOSIS]'),
]

# Pattern to strip TCGA wrapper tags
_TCGA_WRAPPER = re.compile(
    r'\[REPORT_TYPE\]\s*surgical_pathology\s*\n?'
    r'|\[FULL_REPORT_TEXT\]\s*',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_sections(text: str) -> str:
    """Convert unstructured pathology text to Baheya-style section tags.

    Detects section boundaries using regex patterns and inserts Baheya-
    compatible tags ([DIAGNOSIS], [MICROSCOPIC], [GROSS], [CLINICAL_INFO]).

    If the input already contains Baheya tags (e.g. [DIAGNOSIS]), it is
    returned unchanged.

    Parameters
    ----------
    text : str
        Raw pathology report text (may contain TCGA wrapper tags).

    Returns
    -------
    str
        Text with Baheya-style section tags inserted.
    """
    if not text or not text.strip():
        return text

    # Skip if already in Baheya format (has [DIAGNOSIS] etc.)
    if re.search(r'\[(DIAGNOSIS|MICROSCOPIC|CLINICAL_INFO)\]', text):
        return text

    # Strip TCGA wrapper tags
    clean = _TCGA_WRAPPER.sub('', text).strip()

    # Track which sections we find and their positions
    sections: list[tuple[int, int, str, str]] = []  # (match_start, match_end, tag, remaining_text)

    for pattern, tag in _SECTION_PATTERNS:
        for match in pattern.finditer(clean):
            sections.append((match.start(), match.end(), tag, match.group()))

    if not sections:
        # No sections detected — wrap as [DIAGNOSIS] (best effort)
        return f"[REPORT_TYPE] surgical_pathology\n[DIAGNOSIS] {clean}"

    # Sort by position in text
    sections.sort(key=lambda x: x[0])

    # Rebuild text with tags
    parts = []
    parts.append("[REPORT_TYPE] surgical_pathology")

    # Text before first section (if any)
    if sections[0][0] > 0:
        pre = clean[:sections[0][0]].strip()
        if pre:
            parts.append(f"[CLINICAL_INFO] {pre}")

    for i, (start, end, tag, _original) in enumerate(sections):
        # Get text from after this header to before the next header
        if i + 1 < len(sections):
            section_text = clean[end:sections[i + 1][0]].strip()
        else:
            section_text = clean[end:].strip()

        if section_text:
            parts.append(f"{tag} {section_text}")

    return "\n".join(parts)


def extract_diagnosis(text: str, fallback_to_full: bool = True) -> str:
    """Extract only the diagnosis section from a pathology report.

    This is useful for cross-institution evaluation where TCGA reports
    are ~465 words but Baheya reports are ~61 words. Extracting just
    the diagnosis makes TCGA inputs match Baheya in length and focus.

    Parameters
    ----------
    text : str
        Raw or normalized pathology report text.
    fallback_to_full : bool
        If True and no diagnosis section is found, return the full text.
        If False, return empty string.

    Returns
    -------
    str
        The diagnosis section text, or full text if no diagnosis found
        and fallback_to_full is True.
    """
    if not text or not text.strip():
        return text

    # Strip TCGA wrapper tags
    clean = _TCGA_WRAPPER.sub('', text).strip()

    # Try to find diagnosis section
    # Pattern: DIAGNOSIS: ... until next section header or end
    diagnosis_pattern = re.compile(
        r'(?i)(?:FINAL\s+)?DIAGNOSIS\s*:\s*'
        r'(.*?)'
        r'(?=(?:GROSS|MICROSCOPIC|MACROSCOPIC|CLINICAL|PATIENT|SPECIMEN|COMMENT|NOTE|STAGING|PATHOLOGIC|IMPRESSION)\s*:|$)',
        re.DOTALL,
    )

    match = diagnosis_pattern.search(clean)
    if match:
        diagnosis = match.group(1).strip()
        if diagnosis:
            return f"[REPORT_TYPE] surgical_pathology\n[DIAGNOSIS] {diagnosis}"

    # Also try Baheya-format [DIAGNOSIS] tag
    baheya_match = re.search(r'\[DIAGNOSIS\]\s*(.*?)(?=\[(?:MICROSCOPIC|GROSS|CLINICAL_INFO|SPECIMEN|COMMENT)\]|$)', text, re.DOTALL)
    if baheya_match:
        diagnosis = baheya_match.group(1).strip()
        if diagnosis:
            return f"[REPORT_TYPE] surgical_pathology\n[DIAGNOSIS] {diagnosis}"

    if fallback_to_full:
        logger.debug("No diagnosis section found, falling back to full text")
        return text
    return ""



def reorder_segments_diagnosis_first(
    text: str,
    tokenizer,
    segment_size: int = 128,
) -> str:
    """Reorder 128-token segments so [DIAGNOSIS] section appears first.

    For reports where diagnosis appears after specimen descriptions,
    this ensures label-wise attention sees the diagnosis in segment 1.
    Falls back to original order if no [DIAGNOSIS] tag found.

    Parameters
    ----------
    text : str
        Pathology report text (raw or normalized).
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer instance (must match the model's tokenizer).
    segment_size : int
        Token count per segment (must match encoder config, default 128).

    Returns
    -------
    str
        Text with segments reordered so diagnosis appears first,
        or original text if single-segment or no diagnosis found.
    """
    if not text or not text.strip():
        return text

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= segment_size:
        return text  # Single segment — no reordering needed

    # Split into segments
    segments = []
    for i in range(0, len(tokens), segment_size):
        seg_tokens = tokens[i:i + segment_size]
        seg_text = tokenizer.decode(seg_tokens)
        segments.append(seg_text)

    # Find segment containing [DIAGNOSIS] or "final diagnosis"
    diag_idx = None
    for i, seg in enumerate(segments):
        lower = seg.lower()
        if '[diagnosis]' in lower or 'final diagnosis' in lower:
            diag_idx = i
            break

    if diag_idx is None or diag_idx == 0:
        return text  # Already first or not found

    # Reorder: diagnosis segment first, then rest in original order
    reordered = [segments[diag_idx]] + [s for i, s in enumerate(segments) if i != diag_idx]
    return ' '.join(reordered)


def get_normalization_stats(texts: list[str]) -> dict[str, int]:
    """Compute normalization statistics for a batch of texts.

    Returns counts of how many texts have each detected section type.
    Useful for logging/debugging.
    """
    stats: dict[str, int] = {
        "total": len(texts),
        "already_baheya_format": 0,
        "has_diagnosis": 0,
        "has_microscopic": 0,
        "has_gross": 0,
        "has_clinical_info": 0,
        "no_sections_detected": 0,
    }

    for text in texts:
        if re.search(r'\[(DIAGNOSIS|MICROSCOPIC|CLINICAL_INFO)\]', text):
            stats["already_baheya_format"] += 1
            continue

        clean = _TCGA_WRAPPER.sub('', text).strip()
        found_any = False
        for pattern, tag in _SECTION_PATTERNS:
            if pattern.search(clean):
                found_any = True
                if tag == '[DIAGNOSIS]':
                    stats["has_diagnosis"] += 1
                elif tag == '[MICROSCOPIC]':
                    stats["has_microscopic"] += 1
                elif tag == '[GROSS]':
                    stats["has_gross"] += 1
                elif tag == '[CLINICAL_INFO]':
                    stats["has_clinical_info"] += 1
                break  # count each text once per tag

        if not found_any:
            stats["no_sections_detected"] += 1

    return stats
