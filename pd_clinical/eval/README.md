# PPMI PD-vs-HC Classification Harness (`pd_clinical/eval`)

Produces the **clinical-validation arm** of the paper: a real PD vs Healthy-Control
classifier on PPMI data, reported with AUC (95% CI), sensitivity, specificity,
confusion matrix, ROC, and feature importance. This is the result a clinical-Q1
reviewer asks for, and it turns Future-Work Direction 3 into an actual Results table.

## Step 0 — Download the right PPMI files

PPMI access is via the **LONI IDA portal** (ida.loni.usc.edu) → *PPMI* →
**Download → Study Data**. Download these CSVs into one folder:

| File (name may vary by release) | Why | Required |
|---------------------------------|-----|----------|
| `Participant_Status.csv` | PD vs HC labels (`COHORT_DEFINITION`) | ✅ yes |
| DaTScan / SBR analysis (`DATScan_Analysis` or `Xing_Core_Lab_-_Quant_SBR`) | striatal binding ratios — the key features | ✅ yes |
| `MDS_UPDRS_Part_III.csv` | motor exam total | optional |
| `Montreal_Cognitive_Assessment__MoCA_.csv` | cognition | optional |
| `Demographics.csv` (+ age table) | sex, age | optional |

The loader matches tables by **column content**, not filename, so slight naming
differences across releases are fine. Start with just the two required files —
that alone gives a strong classifier.

## Step 1 — Assemble the cohort

```bash
pip install -r pd_clinical/eval/requirements-clinical.txt
python -m pd_clinical.eval.prepare_ppmi --raw-dir path/to/PPMI_csvs
```

Writes `pd_clinical/eval/data/cohort.csv` and `summary.json`. **Check
`summary.json`**: you want roughly hundreds of PD and HC subjects. If counts are
tiny, the wrong CSVs were downloaded.

## Step 2 — Train + evaluate

```bash
python -m pd_clinical.eval.train_eval
```

Writes to `pd_clinical/eval/results/`:
- `results.md` — AUC/sensitivity/specificity/PPV/NPV table + confusion matrices
- `figures/roc.png`, `figures/feature_importance.png`
- `metrics.json` — raw numbers

## Method (state this in the paper)

- Labels: PPMI `COHORT_DEFINITION` (Parkinson's Disease vs Healthy Control;
  Prodromal/SWEDD excluded).
- Features: DAT-SPECT caudate/putamen SBR (L/R) + engineered summaries
  (mean, most-affected side, caudate–putamen ratio, asymmetry), optional
  UPDRS-III, MoCA, sex, age — all at the baseline visit.
- Models: L2 logistic regression (primary) and gradient boosting (comparison).
- Validation: stratified 5-fold CV, pooled out-of-fold probabilities; AUC with
  95% bootstrap CI; operating point at Youden's J. Imputation + scaling fit
  inside each fold (no leakage).

## Reviewer guardrails

1. **Report cohort construction explicitly** — inclusion/exclusion, visit, N per
   class, missingness (all in `summary.json`).
2. **DAT-SPECT makes PD-vs-HC easy** (expect high AUC). Strengthen novelty by
   also reporting the harder contrasts (e.g. PD vs Prodromal) or by showing your
   *Module 2 MRI tool* against this tabular baseline.
3. **Subject-level splits only** — never split repeated visits of one subject
   across train/test (the harness uses one baseline row per subject, so this
   holds).
4. Acknowledge PPMI is a research cohort; external validation is future work.
