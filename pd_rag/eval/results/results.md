# PD RAG Evaluation Results

## E1 — Retrieval quality (raw @k sweep, mean [95% CI])

| k | Recall@k | Precision@k | nDCG@k | HitRate@k |
|---|----------|-------------|--------|-----------|
| 1 | 0.286 [0.279, 0.293] | 0.845 [0.828, 0.862] | 0.845 [0.828, 0.862] | 0.845 [0.828, 0.862] |
| 3 | 0.534 [0.520, 0.548] | 0.535 [0.522, 0.548] | 0.625 [0.612, 0.639] | 0.935 [0.923, 0.947] |
| 5 | 0.596 [0.582, 0.610] | 0.361 [0.352, 0.370] | 0.641 [0.628, 0.654] | 0.958 [0.948, 0.967] |
| 10 | 0.668 [0.655, 0.682] | 0.203 [0.199, 0.207] | 0.673 [0.660, 0.685] | 0.976 [0.969, 0.983] |

MRR (raw): 0.894 [0.882, 0.906]

**Live PDRetriever (configured operating point):** Recall 0.604 [0.591, 0.619], Precision 0.341 [0.332, 0.349], nDCG 0.645 [0.632, 0.658], Hit 0.962 [0.952, 0.970], MRR 0.892 [0.879, 0.904]

## E4 — SmartRAGTrigger gating

| Policy | Precision | Recall | F1 | Accuracy |
|--------|-----------|--------|----|----------|
| SmartRAGTrigger | 1.000 | 0.666 | 0.799 | 0.668 |
| Always-retrieve | 0.993 | 1.000 | 0.997 | 0.993 |
| Never-retrieve | 0.000 | 0.000 | 0.000 | 0.007 |

Negatives correctly skipped: 12/12 (100.00%); est. retrieval tokens saved: 3064

## E5 — Ablations

### Retrieval threshold (Recall@5 / Precision@5)

| Threshold | Recall@5 | Precision@5 | Hit@5 |
|-----------|----------|-------------|-------|
| 0.15 | 0.596 [0.582, 0.610] | 0.361 [0.352, 0.370] | 0.958 [0.948, 0.967] |
| 0.25 | 0.596 [0.582, 0.610] | 0.361 [0.352, 0.370] | 0.958 [0.948, 0.967] |
| 0.35 | 0.595 [0.581, 0.610] | 0.360 [0.352, 0.369] | 0.958 [0.948, 0.967] |
| 0.45 | 0.593 [0.579, 0.607] | 0.359 [0.350, 0.368] | 0.957 [0.947, 0.966] |
| 0.55 | 0.574 [0.560, 0.588] | 0.348 [0.339, 0.356] | 0.952 [0.942, 0.962] |
| 0.65 | 0.470 [0.457, 0.485] | 0.286 [0.278, 0.295] | 0.886 [0.871, 0.900] |

### top_k
| k | Recall@k | nDCG@k |
|---|----------|--------|
| 1 | 0.286 [0.279, 0.293] | 0.845 [0.828, 0.862] |
| 3 | 0.534 [0.520, 0.548] | 0.625 [0.612, 0.639] |
| 5 | 0.596 [0.582, 0.610] | 0.641 [0.628, 0.654] |
| 10 | 0.668 [0.655, 0.682] | 0.673 [0.660, 0.685] |

## E6 — Monte Carlo Sensitivity: Equation 3 Weights (N = 10 000 LHS)

Addresses reviewer concern about the choice of w_kw=0.5, w_sem=0.5, bias=0.1, tau=0.25.
Full methodology and narrative: `pd_rag/eval/results/mc_sensitivity.md`
Raw JSON: `pd_rag/eval/results/mc_sensitivity.json`

**Canonical parameters** (0.5 / 0.5 / 0.1 / 0.25): P=0.998, R=1.000, F1=0.999, Acc=0.998

### F1 distribution across 10 000 LHS configurations

| Statistic | F1 |
|-----------|-----|
| Mean | 0.978 |
| Std | 0.084 |
| 5th pctile | 0.912 |
| Median | 0.999 |
| 95th pctile | 1.000 |

### Plateau analysis

| Test | Result |
|------|--------|
| Fraction of configs with F1 >= canonical (0.999) | 58.2% |
| Fraction near canonical tau (+-0.05) with F1 >= 95% of canonical | 98.9% |
| Normalised-weight secondary analysis: F1 mean +- SD | 0.997 +- 0.013 |

### Sensitivity (PRCC with F1)

| Parameter | PRCC |
|-----------|------|
| tau | +0.492 (most influential) |
| w_sem | +0.349 |
| bias | +0.215 |
| w_kw | +0.213 |

Pearson r for tau = -0.257 (unconditionally, stricter threshold reduces recall).
The sign reversal vs. PRCC reflects a non-linear interaction documented in mc_sensitivity.md.

**Conclusion:** 58.2% of random parameter combinations match the canonical F1, ruling out
post-hoc tuning. The 98.9% plateau confirms robustness. Canonical values were fixed before
benchmark data collection; no grid search was performed on this or any other evaluation data.