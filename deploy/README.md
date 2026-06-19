# NeuroSync PD — Cloud Demo (Render)

A **lite** single-service deployment for a public paper demo. It serves the real
dashboard UI, the Module 0 biomarker pipeline (KEGG/STRING/centrality, CPU-only),
and the quantitative **Evaluation Results** (PPMI + RAG tables and figures). The
GPU-dependent Qwen2-VL features (chat, MRI, gene, drug, simulation) are cleanly
stubbed with a "disabled in cloud demo" message — they cannot run on Render's
CPU instances.

## What works live vs. stubbed

| Module | Cloud demo |
|--------|-----------|
| Dashboard UI | ✅ live |
| Evaluation Results (PPMI AUC, RAG E1/E4/E5) | ✅ live (tables + figures) |
| Biomarker Discovery (Module 0) | ✅ live (keyword-fallback relevance) |
| Clinical Chat / MRI / Gene / Drug / Simulate | ⛔ stubbed (need GPU) |
| Debugger | ⛔ stubbed |

## Prerequisites

1. **A GitHub repo.** This folder isn't a git repo yet. From the folder that
   contains `deploy/`, `launcher.py`, `module0_biomarker/`, `pd_rag/`,
   `pd_clinical/`:
   ```bash
   git init && git add -A && git commit -m "NeuroSync PD"
   gh repo create neurosync-pd --public --source . --push   # or push manually
   ```
   Make sure `pd_clinical/eval/results/` and `pd_rag/eval/results/` are
   committed (they hold the tables + figures the demo serves). `.venv`, `data/`,
   `deploy_data/`, and model weights are git-ignored.

2. A free **Render** account (render.com).

## Deploy

**Option A — Blueprint (uses `deploy/render.yaml`):**
1. Render dashboard → **New + → Blueprint** → select your repo.
2. Render reads `deploy/render.yaml`. Confirm **Root Directory** points at the
   folder containing `deploy/` (in this repo that inner folder is
   `parkinsons-ai-main`; if you pushed that inner folder *as* the repo root, set
   it to `.`).
3. Click **Apply**. First build takes a few minutes.

**Option B — Manual web service:**
- New + → **Web Service** → your repo.
- Root Directory: the folder containing `deploy/`.
- Build Command: `pip install -r deploy/requirements.txt`
- Start Command: `uvicorn deploy.app:app --host 0.0.0.0 --port $PORT`
- Environment: `PYTHON_VERSION=3.11.9`, `MPLBACKEND=Agg`

Your demo will be at `https://neurosync-pd-demo.onrender.com` (or similar).

## Test locally first

```bash
pip install -r deploy/requirements.txt
uvicorn deploy.app:app --reload --port 5000
# open http://localhost:5000  → click "Evaluation Results" and "Biomarker Discovery"
```

## Notes / gotchas

- **Free-tier cold starts:** Render's free plan spins the service down after
  ~15 min idle; the next request takes ~50 s to wake. Fine for a demo link;
  upgrade to **starter** to keep it warm.
- **Memory:** the lite image (no torch) is small; if the free plan OOMs during
  build/run, switch `plan: free` → `starter` in `render.yaml`.
- **KEGG/STRING:** biomarker discovery calls those public APIs at request time;
  if they're slow/unreachable from Render, Module 0 uses its local fallback.
- **Ephemeral storage:** generated PDFs/graphs live in `deploy_data/` and are
  wiped on redeploy — expected for a stateless demo.
- This deploy does **not** modify the original `launcher.py` workflow; it only
  reuses the dashboard HTML and Module 0 engine.
