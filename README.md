# MCIS — Medical Coding Intelligence System

**Architecture benchmark repository.** Code, configs, and evaluation infrastructure for the 4-week M1/M2 NLP architecture benchmark. Final model selection feeds the production deployment at the Baheya Foundation for Early Detection and Treatment of Breast Cancer, Cairo, Egypt.

> **Project ID:** ITCS-GP25-57 — Nile University, School of IT and CS
> **Phase:** 2 (final phase, all four modules)
> **Thesis deadline:** Early June 2026

---

## What this repo is for

MCIS automates oncology medical coding from hospital information system (HIS) exports. The system produces four code types across four modules:

| Module | Input | Output | Pipeline type |
|---|---|---|---|
| **M1** | Primary diagnosis text | ICD-O-3 5-tuple + ICD-11 post-coordinated | NLP |
| **M2** | Comorbidity text | ICD-11 codes | NLP |
| **M3** | Service/procedure fields | CPT codes | Deterministic extraction |
| **M4** | Treatment fields | HCPCS drug codes | Deterministic extraction |

This repository covers the **M1 and M2 NLP architecture benchmark only.** M3 and M4 are deterministic extraction pipelines and live in a separate workstream — they are not benchmarked as ML.

---

## Scope of the benchmark

Four weeks. 15 architectural variants across four scenarios:

- **Scenario A** — Loss function (BCE, weighted BCE, focal, APL)
- **Scenario B** — Knowledge integration (hierarchical regularization, label-description fusion)
- **Scenario C** — ICD-11 strategy (flat multi-label vs. two-stage stem→extension with WHO sanctioning)
- **Scenario D** — Encoder choice (PubMedBERT, PathologyBERT, GatorTron, BioLinkBERT, Clinical-Longformer)

The B0 anchor is PubMedBERT + label-wise attention + multi-task heads. The "MCIS-Best" final configuration combines the winning component from each scenario and is reported on nested 5×3 cross-validation.

Authoritative plan: [`docs/MCIS_Benchmark_Handoff.md`](docs/MCIS_Benchmark_Handoff.md) *(copy of the frozen handoff lives here once data lands)*.

---

## Repo layout

```
MCIS/
├── README.md
├── requirements.txt
├── pyproject.toml                # black + ruff config
├── .pre-commit-config.yaml
├── .gitignore                    # excludes data/frozen/ — patient data never enters git
├── data/
│   ├── CHECKSUMS.txt             # SHA256 of every frozen file (committed)
│   └── frozen/                   # READ-ONLY local copy, gitignored
│       ├── baheya_m1_primary_ready.csv
│       ├── baheya_m2_secondary_ready.csv
│       ├── baheya_uncoded_patients.csv
│       └── cv_folds.csv          # produced by Malak Day 2
├── src/
│   ├── data/                     # loaders + splits
│   ├── models/                   # encoder, attention, heads, architectures
│   ├── losses/                   # bce, focal, apl
│   ├── eval/                     # metrics
│   ├── config.py                 # BenchmarkConfig dataclass
│   ├── train.py                  # training entrypoint
│   └── eval_run.py               # evaluation entrypoint
├── configs/                      # one YAML per scenario
│   └── B0.yaml
├── scripts/                      # utility scripts (verify_splits, unk_scan, generate_checksums, …)
├── results/                      # per-week reports + W&B exports
└── tests/                        # pytest unit tests
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Abdelrhman62/MCIS.git
cd MCIS

# 2. Create env (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy frozen data (NOT in git — get it from the preprocessing pipeline)
cp /path/to/preprocessing/output/baheya_m1_primary_ready.csv data/frozen/
cp /path/to/preprocessing/output/baheya_m2_secondary_ready.csv data/frozen/
cp /path/to/preprocessing/output/baheya_uncoded_patients.csv data/frozen/

# 6. Generate and verify checksums
python scripts/generate_checksums.py

# 7. Authenticate W&B
wandb login
```

---

## Running an experiment

```bash
python -m src.train --config configs/B0.yaml
```

Every run logs the data SHA256 checksums to W&B. Drift in the frozen data will surface immediately when the checksum no longer matches `data/CHECKSUMS.txt`.

---

## Team

| Role | Name |
|---|---|
| Benchmark lead, ML dev | Abdelrhman Akram |
| ML dev, evaluation infrastructure | Malak Khalifa |
| M3 / M4 / frontend / validation | Khalid Ahmed, Yousif Metwally, Mohammed Nehad |
| Supervisor | Dr. Mohamed Mysara |
| Supervisor | Dr. Sahar Selim |
| Co-supervisor | Eng. Dina Yahia |

---

## Partner institution

Baheya Foundation for Early Detection and Treatment of Breast Cancer, Cairo, Egypt.

---

## Data privacy

Baheya patient data is **sensitive PHI** and must never enter version control. `data/frozen/` is `.gitignore`d. Only SHA256 checksums of the frozen files are committed, in `data/CHECKSUMS.txt`. If a checksum mismatch is reported by `scripts/generate_checksums.py --verify`, **stop and resolve before training** — the data on disk no longer matches what was reviewed.

---

## License

Internal academic project. Not for external redistribution. Code license to be determined before thesis submission.
