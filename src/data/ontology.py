"""ICD-O-3 ontology definitions and hierarchical distance matrices.

Provides structured knowledge about ICD-O-3 code relationships for
ontology-aware loss functions (E7 experiment).

The key insight: when a model predicts C50.4 (upper outer quadrant) but
the ground truth is C50.9 (NOS / unspecified), that's a LESS severe error
than predicting C50.1 (central) when the truth is C50.4. The distance
matrix encodes these relationships.

Usage:
    from src.data.ontology import build_topography_distance_matrix

    vocab_labels = ["C50.0", "C50.1", "C50.2", ...]
    dist_matrix = build_topography_distance_matrix(vocab_labels)
    # dist_matrix[i][j] = semantic distance between label i and label j
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# ICD-O-3 Topography: C50.x Breast sub-sites
# ---------------------------------------------------------------------------
# WHO ICD-O-3 hierarchy for breast (C50):
#   C50.0  Nipple
#   C50.1  Central portion of breast
#   C50.2  Upper inner quadrant
#   C50.3  Lower inner quadrant
#   C50.4  Upper outer quadrant
#   C50.5  Lower outer quadrant
#   C50.6  Axillary tail
#   C50.8  Overlapping lesion of breast
#   C50.9  Breast, NOS (not otherwise specified)
#
# Semantic groupings:
#   - Quadrant codes: C50.2, C50.3, C50.4, C50.5 (anatomically specific)
#   - Non-quadrant specific: C50.0, C50.1, C50.6 (site-specific but not quadrant)
#   - Catch-all: C50.8 (overlapping), C50.9 (NOS = unspecified)

# Distance tiers between topography codes:
_TOPO_DISTANCE_SAME = 0.0          # Same code
_TOPO_DISTANCE_SPECIFIC_TO_NOS = 0.2   # C50.4 → C50.9: specific → unspecified (mild error)
_TOPO_DISTANCE_NOS_TO_SPECIFIC = 0.4   # C50.9 → C50.4: unspecified → specific (moderate)
_TOPO_DISTANCE_ADJACENT_QUAD = 0.5     # C50.2 → C50.4: adjacent quadrants
_TOPO_DISTANCE_OPPOSITE_QUAD = 0.7     # C50.2 → C50.5: opposite quadrants
_TOPO_DISTANCE_QUAD_TO_NONQUAD = 0.8   # C50.4 → C50.0: quadrant → nipple/central
_TOPO_DISTANCE_MAX = 1.0               # Completely unrelated (fallback)

# Quadrant adjacency: which quadrants are neighbors
_ADJACENT_QUADS = {
    "C50.2": {"C50.4", "C50.3"},  # upper inner ↔ upper outer, lower inner
    "C50.3": {"C50.2", "C50.5"},  # lower inner ↔ upper inner, lower outer
    "C50.4": {"C50.2", "C50.5"},  # upper outer ↔ upper inner, lower outer
    "C50.5": {"C50.3", "C50.4"},  # lower outer ↔ lower inner, upper outer
}

_QUADRANT_CODES = {"C50.2", "C50.3", "C50.4", "C50.5"}
_NOS_CODES = {"C50.8", "C50.9"}  # catch-all / unspecified


def _topography_distance(code_a: str, code_b: str) -> float:
    """Compute semantic distance between two ICD-O-3 topography codes."""
    if code_a == code_b:
        return _TOPO_DISTANCE_SAME

    a_is_nos = code_a in _NOS_CODES
    b_is_nos = code_b in _NOS_CODES
    a_is_quad = code_a in _QUADRANT_CODES
    b_is_quad = code_b in _QUADRANT_CODES

    # Specific → NOS (mild: the model predicted something specific, truth is unspecified)
    if not a_is_nos and b_is_nos:
        return _TOPO_DISTANCE_SPECIFIC_TO_NOS

    # NOS → Specific (moderate: the model was vague, truth is specific)
    if a_is_nos and not b_is_nos:
        return _TOPO_DISTANCE_NOS_TO_SPECIFIC

    # Both quadrant codes: check adjacency
    if a_is_quad and b_is_quad:
        if code_b in _ADJACENT_QUADS.get(code_a, set()):
            return _TOPO_DISTANCE_ADJACENT_QUAD
        return _TOPO_DISTANCE_OPPOSITE_QUAD

    # Quadrant ↔ non-quadrant specific (nipple, central, axillary tail)
    if (a_is_quad and not b_is_quad) or (not a_is_quad and b_is_quad):
        return _TOPO_DISTANCE_QUAD_TO_NONQUAD

    # Both NOS codes (C50.8 ↔ C50.9)
    if a_is_nos and b_is_nos:
        return _TOPO_DISTANCE_SPECIFIC_TO_NOS  # very close

    return _TOPO_DISTANCE_MAX


def build_topography_distance_matrix(
    vocab_labels: list[str],
) -> np.ndarray:
    """Build a distance matrix for ICD-O-3 topography codes.

    Parameters
    ----------
    vocab_labels : list[str]
        Ordered list of topography codes as they appear in the label vocab.
        Example: ["C50.4", "C50.9", "C50.2", "C50.1", ...]

    Returns
    -------
    np.ndarray
        Float32 matrix of shape [K, K] where entry [i, j] is the semantic
        distance between vocab_labels[i] and vocab_labels[j].
        0.0 = identical, 1.0 = maximally different.
    """
    k = len(vocab_labels)
    matrix = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            matrix[i, j] = _topography_distance(vocab_labels[i], vocab_labels[j])
    return matrix


def build_soft_targets(
    hard_target: int,
    distance_matrix: np.ndarray,
    smoothing: float = 0.3,
) -> np.ndarray:
    """Convert a hard target index to soft targets using the distance matrix.

    Parameters
    ----------
    hard_target : int
        Index of the true class.
    distance_matrix : np.ndarray
        [K, K] distance matrix.
    smoothing : float
        Controls how much probability mass to redistribute. 0.0 = hard targets,
        1.0 = fully distance-weighted.

    Returns
    -------
    np.ndarray
        Float32 vector of shape [K] summing to 1.0.
    """
    k = distance_matrix.shape[0]
    distances = distance_matrix[hard_target]  # [K]

    # Convert distances to similarity (closer = more similar)
    # similarity = 1 - distance, then normalize
    similarity = 1.0 - distances  # [K]
    similarity[hard_target] = 0.0  # exclude self from redistribution

    # Redistribute smoothing fraction of probability mass by similarity
    if similarity.sum() > 0:
        soft = similarity / similarity.sum() * smoothing
    else:
        soft = np.zeros(k, dtype=np.float32)

    soft[hard_target] = 1.0 - smoothing
    return soft.astype(np.float32)


# ---------------------------------------------------------------------------
# ICD-O-3 Morphology hierarchy
# ---------------------------------------------------------------------------
# ICD-O-3 morphology codes follow a 4-digit scheme where the first 3 digits
# define the histological family and the 4th digit is a variant. Codes
# sharing a 3-digit prefix are clinically related (e.g., 8500 IDC NOS and
# 8501 comedocarcinoma are both ductal carcinomas).
#
# For breast cancer (Baheya M1 vocab, 24 classes), the families are:
#
#   Ductal carcinomas (850x):
#       8500  Infiltrating duct carcinoma, NOS
#       8501  Comedocarcinoma, NOS
#       8502  Secretory carcinoma of breast
#       8503  Intraductal papillary adenocarcinoma with invasion
#       8504  Intracystic carcinoma, NOS
#       8507  Invasive micropapillary carcinoma of breast
#       8509  Solid papillary carcinoma (in situ / invasive)
#
#   Lobular carcinomas (852x):
#       8520  Lobular carcinoma, NOS
#       8522  Infiltrating duct and lobular carcinoma (mixed)
#
#   Mucinous / cystic (847x–848x):
#       8470  Mucinous cystadenocarcinoma, NOS
#       8480  Mucinous adenocarcinoma
#       8453  Intraductal papillary-mucinous carcinoma, invasive
#
#   Adenocarcinoma family (814x–821x):
#       8140  Adenocarcinoma, NOS
#       8200  Adenoid cystic carcinoma
#       8201  Cribriform carcinoma, NOS
#       8211  Tubular adenocarcinoma
#
#   Special breast types:
#       8510  Medullary carcinoma, NOS
#       8540  Paget disease of breast
#       8550  Acinar cell carcinoma
#       8575  Metaplastic carcinoma, NOS
#
#   Non-specific / other:
#       8010  Carcinoma, NOS
#       8032  Spindle cell carcinoma, NOS
#       8050  Papillary carcinoma, NOS
#       9020  Phyllodes tumor, malignant
#
# Distance tiers reflect ICD-O-3 hierarchical structure:
#   - Same 4-digit code:        0.0 (identity)
#   - Same 3-digit family:      0.2 (very close — ductal variants)
#   - Related families:         0.5 (e.g., ductal ↔ lobular, both carcinomas)
#   - Unrelated families:       0.8 (e.g., ductal ↔ phyllodes)
#   - NOS ↔ specific:           0.3 (8010 NOS is a parent of most carcinomas)

# Morphology family groupings by 3-digit ICD-O-3 prefix
_MORPH_FAMILIES: dict[str, set[str]] = {
    "ductal":       {"8500", "8501", "8502", "8503", "8504", "8507", "8509"},
    "lobular":      {"8520", "8522"},
    "mucinous":     {"8470", "8480", "8453"},
    "adenoca":      {"8140", "8200", "8201", "8211"},
    "special":      {"8510", "8540", "8550", "8575"},
    "other":        {"8010", "8032", "8050", "9020"},
}

# Code-to-family lookup (built from _MORPH_FAMILIES)
_CODE_TO_FAMILY: dict[str, str] = {}
for _fam, _codes in _MORPH_FAMILIES.items():
    for _c in _codes:
        _CODE_TO_FAMILY[_c] = _fam

# Which families are "related" (histologically close enough that confusion
# is less severe). Based on carcinoma subtype relationships:
#   - ductal ↔ lobular (8522 is literally "mixed ductal & lobular")
#   - ductal ↔ mucinous (mucinous is a ductal variant in some classifications)
#   - ductal ↔ adenoca (IDC is an adenocarcinoma)
#   - adenoca ↔ mucinous (mucinous adenocarcinoma)
_RELATED_FAMILIES: set[frozenset[str]] = {
    frozenset({"ductal", "lobular"}),
    frozenset({"ductal", "mucinous"}),
    frozenset({"ductal", "adenoca"}),
    frozenset({"adenoca", "mucinous"}),
    frozenset({"ductal", "special"}),   # medullary/metaplastic are ductal variants
    frozenset({"lobular", "adenoca"}),
}

# NOS codes: generic labels that are parents of more specific codes
_MORPH_NOS_CODES = {"8010"}  # "Carcinoma, NOS" — generic parent

# Distance tier constants
_MORPH_DISTANCE_SAME = 0.0
_MORPH_DISTANCE_SAME_FAMILY = 0.2
_MORPH_DISTANCE_NOS_TO_SPECIFIC = 0.3
_MORPH_DISTANCE_RELATED_FAMILY = 0.5
_MORPH_DISTANCE_UNRELATED = 0.8
_MORPH_DISTANCE_MAX = 1.0


def _morphology_distance(code_a: str, code_b: str) -> float:
    """Compute semantic distance between two ICD-O-3 morphology codes.

    Distance tiers follow the ICD-O-3 hierarchy:
        0.0  Same code
        0.2  Same 3-digit family (e.g., 8500 ↔ 8501, both ductal)
        0.3  NOS ↔ specific carcinoma (8010 is a parent of most codes)
        0.5  Related families (e.g., ductal ↔ lobular)
        0.8  Unrelated families (e.g., ductal ↔ phyllodes)
        1.0  Completely unknown relationship (fallback)
    """
    if code_a == code_b:
        return _MORPH_DISTANCE_SAME

    fam_a = _CODE_TO_FAMILY.get(code_a)
    fam_b = _CODE_TO_FAMILY.get(code_b)

    # If either code is unknown, max distance
    if fam_a is None or fam_b is None:
        return _MORPH_DISTANCE_MAX

    # NOS code (8010) to any specific carcinoma
    if code_a in _MORPH_NOS_CODES or code_b in _MORPH_NOS_CODES:
        # 9020 (phyllodes) is NOT a carcinoma, so NOS → phyllodes is far
        if code_a == "9020" or code_b == "9020":
            return _MORPH_DISTANCE_UNRELATED
        return _MORPH_DISTANCE_NOS_TO_SPECIFIC

    # Same family
    if fam_a == fam_b:
        return _MORPH_DISTANCE_SAME_FAMILY

    # Related families
    if frozenset({fam_a, fam_b}) in _RELATED_FAMILIES:
        return _MORPH_DISTANCE_RELATED_FAMILY

    return _MORPH_DISTANCE_UNRELATED


def build_morphology_distance_matrix(
    vocab_labels: list[str],
) -> np.ndarray:
    """Build a distance matrix for ICD-O-3 morphology codes.

    Parameters
    ----------
    vocab_labels : list[str]
        Ordered list of morphology codes as they appear in the label vocab.
        Example: ["8500", "8520", "8510", ...]

    Returns
    -------
    np.ndarray
        Float32 matrix of shape [K, K] where entry [i, j] is the semantic
        distance between vocab_labels[i] and vocab_labels[j].
        0.0 = identical, 1.0 = maximally different.
    """
    k = len(vocab_labels)
    matrix = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            matrix[i, j] = _morphology_distance(vocab_labels[i], vocab_labels[j])
    return matrix


# ---------------------------------------------------------------------------
# Convenience: build all available distance matrices
# ---------------------------------------------------------------------------

def build_distance_matrices(
    axis_vocabs: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Build distance matrices for all axes that have ontology support.

    Currently supports topography and morphology.

    Parameters
    ----------
    axis_vocabs : dict[str, list[str]]
        Mapping axis_name → ordered list of label strings.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping axis_name → [K, K] distance matrix.
        Only includes axes with ontology support.
    """
    matrices = {}
    if "icdo3_topography" in axis_vocabs:
        matrices["icdo3_topography"] = build_topography_distance_matrix(
            axis_vocabs["icdo3_topography"]
        )
    if "icdo3_morphology" in axis_vocabs:
        matrices["icdo3_morphology"] = build_morphology_distance_matrix(
            axis_vocabs["icdo3_morphology"]
        )
    return matrices
