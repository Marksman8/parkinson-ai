# Code and Data Availability Statement

*(Place this section verbatim in the manuscript as a dedicated "Code and Data
Availability" section, per the requirements of the target journal. Adjust the
DOI / version tag once the repository is archived.)*

---

## Code availability

All software described in this article — including the multimodal RAG engine
(`pd_rag`), the Module-0 biomarker analysis pipeline (`module0_biomarker`),
the ODE simulator (`pd_multimodal`), the clinical classification experiments
(`pd_clinical`), and the complete evaluation harness (`pd_rag/eval`) — is
publicly available under the MIT licence at:

> **GitHub repository:**
> https://github.com/[owner]/parkinsons-ai
> (placeholder — replace with the permanent URL before submission)

A citable, version-controlled snapshot of the code used to produce all
reported results will be deposited on Zenodo prior to final submission and
assigned a persistent DOI:

> **Zenodo archive DOI:**
> https://doi.org/10.5281/zenodo.[XXXXXXX]
> (placeholder — deposit via https://zenodo.org/deposit after tagging the
> release `v1.0.0` on GitHub)

The specific Git commit hash corresponding to the results reported in this
paper is recorded in `pd_rag/eval/results/metrics.json` under the key
`config.git_commit` and in `pd_rag/eval/results/mc_sensitivity.json`.

### Software dependencies and versions

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.14 | Runtime |
| sentence-transformers | ≥ 2.7 | Embedding model (`all-MiniLM-L6-v2`) |
| chromadb | ≥ 0.5 | Vector store (HNSW cosine index) |
| numpy | ≥ 1.26 | Numerical computation, Latin Hypercube Sampling |
| scipy | ≥ 1.13 | Statistical tests (Wilcoxon, McNemar) |
| matplotlib | ≥ 3.8 | Figures |
| Qwen2-VL-2B-Instruct | — | Generation backbone (E2/E3 only; HuggingFace model hub) |

A pinned `requirements` file is provided in `pd_rag/eval/requirements-eval.txt`.

---

## Data availability

This study uses **only publicly available datasets**.

### Primary evaluation benchmark

The retrieval and QA evaluation benchmark (`pd_rag/eval/data/`) was
constructed from:

**PubMedQA** (Jin et al., 2019)
- Licence: MIT
- URL: https://pubmedqa.github.io
- Citation: Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. *PubMedQA: A Dataset for
  Biomedical Research Question Answering.* EMNLP 2019.
  https://doi.org/10.18653/v1/D19-1474
- Subset used: PD-related abstracts filtered by KEGG pathway hsa05012 gene
  set; 1 723 questions retained; full reproducible filter logic in
  `pd_rag/eval/build_benchmark.py`.

### Clinical classification dataset

The PD-vs-healthy-control classification experiments (`pd_clinical/`) use
data from:

**Parkinson's Progression Markers Initiative (PPMI)**
- URL: https://www.ppmi-info.org
- Access: Requires registration and data-use agreement (DUA) with The Michael
  J. Fox Foundation. Raw PPMI files are **not** redistributed in this
  repository; researchers must apply directly via the PPMI data portal.
- Cohort used in this paper: 2 043 PD, 423 healthy controls; 11 clinical
  features; full preprocessing script in `pd_clinical/preprocess.py`.

### Biological pathway and interaction data

- **KEGG pathway hsa05012** (Parkinson's disease): accessed programmatically
  via the KEGG REST API (https://www.genome.jp/kegg/rest/); no raw files
  redistributed.
- **STRING v12.0** protein-protein interaction data: accessed via the STRING
  API (https://string-db.org/api); no raw files redistributed.

### Benchmark reproducibility

All benchmark construction is deterministic given a fixed seed (`--seed 42`):

```bash
python -m pd_rag.eval.build_benchmark --distractors 800 --seed 42
```

The resulting files (`data/benchmark.json`, `data/corpus.jsonl`,
`data/qrels.json`, `data/triggers.json`) are checked into the repository
and will remain available indefinitely at the Zenodo archive DOI above.

---

## Monte Carlo sensitivity analysis

The sensitivity analysis for Equation 3 weights and the gating threshold
(Section E6 of the Supplementary) is fully self-contained and requires no
external data beyond the benchmark files above:

```bash
python -m pd_rag.eval.monte_carlo_sensitivity \
    --n-samples 10000 --seed 42 \
    --out-json pd_rag/eval/results/mc_sensitivity.json
```

Pre-computed results are archived at `pd_rag/eval/results/mc_sensitivity.json`
and `pd_rag/eval/results/mc_sensitivity.md`.

---

*For questions about data access or code reproducibility, contact the
corresponding author or open an issue on the GitHub repository.*
