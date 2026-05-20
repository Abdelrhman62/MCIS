# B2 — faithful PLM-ICD baseline (LAAT, flat 119)

- Folds: 1
- **early_stop (5-fold): 0.0318 ± 0.0000**
- B1 floor: 0.6351  (Δ B2−B1 = -0.6033)
- E1 (per-axis LAAT): 0.5726  (Δ B2−E1 = -0.5408)

| fold | best_epoch | early_stop |
|---:|---:|---:|
| 0 | 0 | 0.0318 |

Interpretation: B2 holds LAAT constant and removes only MCIS's per-axis multi-task decomposition. Δ B2−E1 attributes to that decomposition; Δ B2−B1 places faithful PLM-ICD vs the TF-IDF floor.