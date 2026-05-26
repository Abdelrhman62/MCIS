"""Official WHO descriptions for all MCIS label codes.

Used by E9 (description-initialized label queries) to replace Xavier random
initialization of label attention query vectors with PubMedBERT embeddings
of each label's textual description.

Sources:
    - ICD-O-3: WHO International Classification of Diseases for Oncology, 3rd Ed.
    - ICD-11: WHO ICD-11 for Mortality and Morbidity Statistics (2024)
    - ICD-11 Extension Codes: WHO ICD-11 Extension Code System

Each entry maps a label code string (as it appears in label_vocab.json) to
a short clinical description suitable for embedding via PubMedBERT.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# ICD-O-3 Topography (C50.x breast sub-sites)
# ---------------------------------------------------------------------------
ICDO3_TOPOGRAPHY: dict[str, str] = {
    "C50.0": "Nipple of breast",
    "C50.1": "Central portion of breast",
    "C50.2": "Upper inner quadrant of breast",
    "C50.3": "Lower inner quadrant of breast",
    "C50.4": "Upper outer quadrant of breast",
    "C50.5": "Lower outer quadrant of breast",
    "C50.6": "Axillary tail of breast",
    "C50.8": "Overlapping lesion of breast",
    "C50.9": "Breast, not otherwise specified",
}

# ---------------------------------------------------------------------------
# ICD-O-3 Morphology (histological type)
# ---------------------------------------------------------------------------
ICDO3_MORPHOLOGY: dict[str, str] = {
    "8500": "Infiltrating duct carcinoma, not otherwise specified",
    "8520": "Lobular carcinoma, not otherwise specified",
    "8510": "Medullary carcinoma, not otherwise specified",
    "8470": "Mucinous cystadenocarcinoma, not otherwise specified",
    "8507": "Invasive micropapillary carcinoma of breast",
    "8522": "Infiltrating duct and lobular carcinoma, mixed type",
    "8201": "Cribriform carcinoma, not otherwise specified",
    "8010": "Carcinoma, not otherwise specified",
    "8503": "Intraductal papillary adenocarcinoma with invasion",
    "8140": "Adenocarcinoma, not otherwise specified",
    "8504": "Intracystic carcinoma, not otherwise specified",
    "8480": "Mucinous adenocarcinoma",
    "8453": "Intraductal papillary-mucinous carcinoma, invasive",
    "8540": "Paget disease of breast",
    "8575": "Metaplastic carcinoma, not otherwise specified",
    "8509": "Solid papillary carcinoma",
    "8501": "Comedocarcinoma, not otherwise specified",
    "8050": "Papillary carcinoma, not otherwise specified",
    "8211": "Tubular adenocarcinoma",
    "9020": "Phyllodes tumor, malignant",
    "8032": "Spindle cell carcinoma, not otherwise specified",
    "8200": "Adenoid cystic carcinoma",
    "8502": "Secretory carcinoma of breast",
    "8550": "Acinar cell carcinoma",
}

# ---------------------------------------------------------------------------
# ICD-O-3 Behavior
# ---------------------------------------------------------------------------
ICDO3_BEHAVIOR: dict[str, str] = {
    "0": "Benign neoplasm",
    "2": "Carcinoma in situ, intraepithelial, noninvasive",
    "3": "Malignant, primary site",
    "6": "Malignant, metastatic site",
}

# ---------------------------------------------------------------------------
# ICD-O-3 Grade
# ---------------------------------------------------------------------------
ICDO3_GRADE: dict[str, str] = {
    "1": "Grade I, well differentiated",
    "2": "Grade II, moderately differentiated",
    "3": "Grade III, poorly differentiated",
}

# ---------------------------------------------------------------------------
# ICD-O-3 Laterality
# ---------------------------------------------------------------------------
ICDO3_LATERALITY: dict[str, str] = {
    "L": "Left",
    "R": "Right",
    "B": "Bilateral",
}

# ---------------------------------------------------------------------------
# ICD-11 Stem codes (breast neoplasm classification)
# ---------------------------------------------------------------------------
ICD11_STEM: dict[str, str] = {
    "2C61.0": "Malignant neoplasm of breast, upper outer quadrant",
    "2C61.1": "Malignant neoplasm of breast, upper inner quadrant",
    "2C61.3": "Malignant neoplasm of breast, lower inner quadrant",
    "2C61.4": "Malignant neoplasm of breast, lower outer quadrant",
    "2C61": "Malignant neoplasm of breast, specified quadrant",
    "2C60.0": "Malignant neoplasm of nipple or areola",
    "2C60": "Malignant neoplasm of nipple and areola",
    "2C63": "Malignant neoplasm of axillary tail of breast",
    "2C6Y": "Other specified malignant neoplasm of breast",
    "2C6Z": "Malignant neoplasm of breast, unspecified",
    "2E65.2": "Benign neoplasm of breast, ductal",
    "2E65.3": "Benign neoplasm of breast, lobular",
    "2E65.4": "Benign phyllodes tumour of breast",
    "2E65.5": "Benign neoplasm of nipple",
    "2E65.Y": "Other specified benign neoplasm of breast",
    "2F30.2": "Ductal carcinoma in situ of breast",
    "2F30.5": "Lobular carcinoma in situ of breast",
    "2F30.Y": "Other specified in situ neoplasm of breast",
    "2F75": "In situ neoplasm of other or unspecified site",
    "GB20": "Neoplasm of uncertain behaviour of breast",
    "GB20.0": "Borderline phyllodes tumour of breast",
    "GB20.1": "Papillary neoplasm of breast of uncertain behaviour",
    "GB20.2": "Lobular neoplasm of breast of uncertain behaviour",
    "GB23.0": "Neoplasm of uncertain behaviour, other specified site",
    "GB23.2": "Neoplasm of uncertain behaviour, unspecified",
    "2018": "Malignant neoplasm without specification of site",
    "MA01": "Neoplasm of unspecified behaviour",
}

# ---------------------------------------------------------------------------
# ICD-11 Extension: Laterality
# ---------------------------------------------------------------------------
ICD11_EXT_LATERALITY: dict[str, str] = {
    "XK8G": "Left",
    "XK9K": "Right",
    "XK9G": "Bilateral",
    "XK9J": "Right side, crossing midline",
    "XK8K": "Left side, crossing midline",
    "XS9K": "Laterality not applicable",
}

# ---------------------------------------------------------------------------
# ICD-11 Extension: Grading
# ---------------------------------------------------------------------------
ICD11_EXT_GRADING: dict[str, str] = {
    "XS58": "Grade 1, well differentiated",
    "XS7Z": "Grade 2, moderately differentiated",
    "XS56": "Grade 3, poorly differentiated",
    "XS7H": "Grade not determined",
    "XS7M": "No grading system applicable",
}

# ---------------------------------------------------------------------------
# ICD-11 Extension: Anatomy (post-coordinated anatomical detail)
# ---------------------------------------------------------------------------
ICD11_EXT_ANATOMY: dict[str, str] = {
    "XA9CM2": "Breast, upper outer quadrant",
    "XA2Q54": "Breast, upper inner quadrant",
    "XA3JH9": "Breast, lower outer quadrant",
    "XA3LS6": "Breast, lower inner quadrant",
    "XA0US1": "Breast, central portion",
    "XA2JK3": "Breast, nipple",
    "XA94U2": "Breast, axillary tail",
    "XA0VX8": "Breast, overlapping region",
    "XA3PG5": "Breast, not otherwise specified",
    "XJ3VM": "Axillary lymph node",
    "XA90B2": "Skin of breast",
    "XA5MC5": "Chest wall",
    "XA0UK0": "Pectoralis muscle",
    "XA33X2": "Supraclavicular lymph node",
    "XA5DY0": "Internal mammary lymph node",
    "XA1NS5": "Infraclavicular lymph node",
}

# ---------------------------------------------------------------------------
# ICD-11 Extension: Histopathology (post-coordinated histological detail)
# ---------------------------------------------------------------------------
ICD11_EXT_HISTOPATH: dict[str, str] = {
    "XH1YZ3": "Infiltrating duct carcinoma",
    "XH4TA4": "Lobular carcinoma",
    "XH1390": "Ductal carcinoma in situ",
    "XH9C56": "Lobular carcinoma in situ",
    "XH0GT6": "Invasive micropapillary carcinoma",
    "XH2YP5": "Mucinous carcinoma",
    "XH8UE4": "Medullary carcinoma",
    "XH1S75": "Papillary carcinoma",
    "XH0RD4": "Metaplastic carcinoma",
    "XH8KR8": "Tubular carcinoma",
    "XH5WU3": "Cribriform carcinoma",
    "XH1XB5": "Adenoid cystic carcinoma",
    "XH3E21": "Secretory carcinoma",
    "XH7M": "Phyllodes tumour",
    "XH3RK9": "Paget disease",
    "XH90W1": "Comedocarcinoma",
    "XH9VG0": "Intracystic papillary carcinoma",
    "XH11S9": "Apocrine carcinoma",
    "XH3RZ4": "Solid papillary carcinoma",
    "XH4302": "Neuroendocrine carcinoma",
    "XH60S7": "Spindle cell carcinoma",
    "XH6PY4": "Squamous cell carcinoma",
    "XH9FX2": "Carcinoma, not otherwise specified",
}


# ---------------------------------------------------------------------------
# Master lookup: axis_name → {code: description}
# ---------------------------------------------------------------------------
AXIS_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "icdo3_topography": ICDO3_TOPOGRAPHY,
    "icdo3_morphology": ICDO3_MORPHOLOGY,
    "icdo3_behavior": ICDO3_BEHAVIOR,
    "icdo3_grade": ICDO3_GRADE,
    "icdo3_laterality": ICDO3_LATERALITY,
    "icd11_stem": ICD11_STEM,
    "icd11_ext_laterality": ICD11_EXT_LATERALITY,
    "icd11_ext_grading": ICD11_EXT_GRADING,
    "icd11_ext_anatomy": ICD11_EXT_ANATOMY,
    "icd11_ext_histopath": ICD11_EXT_HISTOPATH,
}


def get_descriptions_for_axis(
    axis_name: str,
    vocab_codes: list[str],
) -> list[str]:
    """Return ordered list of description strings for an axis's vocab codes.

    Parameters
    ----------
    axis_name : str
        Axis name matching AXIS_DESCRIPTIONS keys.
    vocab_codes : list[str]
        Ordered codes from label_vocab.json (index = class index).

    Returns
    -------
    list[str]
        Description strings in the same order as vocab_codes.
        Falls back to the raw code string if no description is found.
    """
    desc_map = AXIS_DESCRIPTIONS.get(axis_name, {})
    return [desc_map.get(code, code) for code in vocab_codes]
