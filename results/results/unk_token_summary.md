# UNK Token Scan Summary

- **Input:** `/Users/abdo/Desktop/MCIS/data/frozen/m1_model_ready/m1_model_ready.parquet`
- **Rows scanned:** 3,055
- **Text columns:** text_section_tagged, text_concatenated (primary: `text_section_tagged`)
- **Tokenizers:** pubmedbert, pathologybert
- **Tokenization parameters:** add_special_tokens=True, truncation=False (matches encoder.py training-time behaviour)
- **Flag threshold:** UNK rate > 5%
- **Segment size for PLM-ICD count:** 128

## Note on section tags

`text_section_tagged` contains literal markers like `[REPORT_TYPE]`, `[DIAGNOSIS]`, `[SPECIMEN]`, `[LATERALITY]`. WordPiece splits these into `[`, `report`, `_`, `type`, `]` — that is expected and is **not** UNK. UNK signal here only flags genuine vocabulary coverage gaps.

## Tokenizer: pubmedbert

### pubmedbert — `text_section_tagged`

_Backbone_: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

**UNK rate distribution**

| stat | value |
|---|---:|
| mean | 0.0000% |
| median | 0.0000% |
| p95 | 0.0000% |
| p99 | 0.0000% |
| max | 0.0000% |
| reports with zero UNK | 3,055 / 3,055 (100.0%) |
| reports flagged (>5%) | **0** |

**Token length distribution (informational; truncation is decided in encoder.py)**

| stat | value |
|---|---:|
| mean tokens | 120 |
| p95 tokens | 175 |
| p99 tokens | 226 |
| max tokens | 387 |
| reports >512 tokens | 0 (0.0%) |
| reports >1536 tokens (12 × 128 ceiling) | 0 |

**Subgroup means (batch / template_flag)**

| subgroup | n | mean UNK rate | mean tokens | n flagged |
|---|---:|---:|---:|---:|
| batch=1 | 1,755 | 0.0000% | 123 | 0 |
| batch=2 | 1,300 | 0.0000% | 117 | 0 |
| template_flag=True (templated) | 475 | 0.0000% | 108 | 0 |
| template_flag=False (free-text) | 2,580 | 0.0000% | 123 | 0 |

### pubmedbert — `text_concatenated`

_Backbone_: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

**UNK rate distribution**

| stat | value |
|---|---:|
| mean | 0.0000% |
| median | 0.0000% |
| p95 | 0.0000% |
| p99 | 0.0000% |
| max | 0.0000% |
| reports with zero UNK | 3,055 / 3,055 (100.0%) |
| reports flagged (>5%) | **0** |

**Token length distribution (informational; truncation is decided in encoder.py)**

| stat | value |
|---|---:|
| mean tokens | 92 |
| p95 tokens | 146 |
| p99 tokens | 196 |
| max tokens | 366 |
| reports >512 tokens | 0 (0.0%) |
| reports >1536 tokens (12 × 128 ceiling) | 0 |

**Subgroup means (batch / template_flag)**

| subgroup | n | mean UNK rate | mean tokens | n flagged |
|---|---:|---:|---:|---:|
| batch=1 | 1,755 | 0.0000% | 94 | 0 |
| batch=2 | 1,300 | 0.0000% | 89 | 0 |
| template_flag=True (templated) | 475 | 0.0000% | 79 | 0 |
| template_flag=False (free-text) | 2,580 | 0.0000% | 94 | 0 |

## Tokenizer: pathologybert

### pathologybert — `text_section_tagged`

_Backbone_: `tsantos/PathologyBERT`

**UNK rate distribution**

| stat | value |
|---|---:|
| mean | 0.7752% |
| median | 0.0000% |
| p95 | 3.3613% |
| p99 | 5.8231% |
| max | 12.0879% |
| reports with zero UNK | 2,020 / 3,055 (66.1%) |
| reports flagged (>5%) | **52** |

**Token length distribution (informational; truncation is decided in encoder.py)**

| stat | value |
|---|---:|
| mean tokens | 116 |
| p95 tokens | 168 |
| p99 tokens | 215 |
| max tokens | 371 |
| reports >512 tokens | 0 (0.0%) |
| reports >1536 tokens (12 × 128 ceiling) | 0 |

**Subgroup means (batch / template_flag)**

| subgroup | n | mean UNK rate | mean tokens | n flagged |
|---|---:|---:|---:|---:|
| batch=1 | 1,755 | 0.7664% | 117 | 18 |
| batch=2 | 1,300 | 0.7870% | 114 | 34 |
| template_flag=True (templated) | 475 | 0.3498% | 104 | 0 |
| template_flag=False (free-text) | 2,580 | 0.8535% | 118 | 52 |

**Top 10 highest UNK-rate reports**

| record_id | n_tokens | n_unk | unk_rate |
|---|---:|---:|---:|
| `BAH-P-1121` | 182 | 22 | 12.0879% |
| `BAH-P-1025` | 165 | 14 | 8.4848% |
| `BAH-P-3011` | 191 | 16 | 8.3770% |
| `BAH-P-2026` | 188 | 14 | 7.4468% |
| `BAH-P-2623` | 217 | 16 | 7.3733% |
| `BAH-P-0045` | 109 | 8 | 7.3394% |
| `BAH-P-2475` | 221 | 16 | 7.2398% |
| `BAH-P-1297` | 168 | 12 | 7.1429% |
| `BAH-P-2792` | 182 | 13 | 7.1429% |
| `BAH-P-2442` | 225 | 16 | 7.1111% |

### pathologybert — `text_concatenated`

_Backbone_: `tsantos/PathologyBERT`

**UNK rate distribution**

| stat | value |
|---|---:|
| mean | 5.8433% |
| median | 5.5556% |
| p95 | 8.6957% |
| p99 | 10.6667% |
| max | 16.1290% |
| reports with zero UNK | 0 / 3,055 (0.0%) |
| reports flagged (>5%) | **2150** |

**Token length distribution (informational; truncation is decided in encoder.py)**

| stat | value |
|---|---:|
| mean tokens | 86 |
| p95 tokens | 137 |
| p99 tokens | 181 |
| max tokens | 349 |
| reports >512 tokens | 0 (0.0%) |
| reports >1536 tokens (12 × 128 ceiling) | 0 |

**Subgroup means (batch / template_flag)**

| subgroup | n | mean UNK rate | mean tokens | n flagged |
|---|---:|---:|---:|---:|
| batch=1 | 1,755 | 5.7755% | 88 | 1214 |
| batch=2 | 1,300 | 5.9348% | 83 | 936 |
| template_flag=True (templated) | 475 | 5.9137% | 74 | 403 |
| template_flag=False (free-text) | 2,580 | 5.8304% | 88 | 1747 |

**Top 10 highest UNK-rate reports**

| record_id | n_tokens | n_unk | unk_rate |
|---|---:|---:|---:|
| `BAH-P-1121` | 155 | 25 | 16.1290% |
| `BAH-P-0045` | 80 | 12 | 15.0000% |
| `BAH-P-0893` | 73 | 10 | 13.6986% |
| `BAH-P-1025` | 135 | 18 | 13.3333% |
| `BAH-P-0053` | 92 | 12 | 13.0435% |
| `BAH-P-2483` | 54 | 7 | 12.9630% |
| `BAH-P-2455` | 86 | 11 | 12.7907% |
| `BAH-P-0436` | 79 | 10 | 12.6582% |
| `BAH-P-0610` | 81 | 10 | 12.3457% |
| `BAH-P-2234` | 57 | 7 | 12.2807% |

## Acceptance

⚠️ **REVIEW** — at least one combination has reports above 5% UNK. See top-10 lists above and inspect those records before E1.
