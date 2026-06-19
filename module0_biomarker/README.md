# Module 0 — Automated PD Biomarker Discovery Tool

A fully automated pipeline that turns a neurologist's clinical notes into
a structured biomarker report — no coding, no database knowledge required.

## What it does

```
Doctor types:
"Patient has resting tremor, LRRK2 G2019S mutation, dopamine deficiency"
                    ↓
Module 0 automatically:
  1. Checks if input is Parkinson's related (RAG + keywords)
  2. Fetches KEGG pathway hsa05012 (online → local fallback)
  3. Builds STRING PPI network (online → local fallback)
  4. Runs degree centrality analysis
  5. Identifies top 5 hub genes: SNCA, LRRK2, PINK1, PARKIN, DJ-1
  6. Adds clinical explanations in plain language
  7. Retrieves ODE parameters from literature
  8. Generates PPI network visualisation
  9. Creates downloadable PDF report
 10. Saves to shared SQLite memory → Module 2 reads this
```

## Setup

```bash
cd module0_biomarker
python -m venv m0_env
m0_env\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

Open: http://localhost:8001/docs

## Usage

### Via API docs (easiest)
1. Go to http://localhost:8001/docs
2. Click POST /discover
3. Click Try it out
4. Type clinical notes in the box
5. Click Execute
6. Download report from GET /report/{session_id}

### Via curl
```bash
curl -X POST http://localhost:8001/discover \
  -F "clinical_text=Patient has resting tremor, rigidity, bradykinesia.
      Genetic testing shows LRRK2 G2019S mutation.
      DAT scan shows reduced uptake. Started levodopa-carbidopa."
```

### Via PDF upload
```bash
curl -X POST http://localhost:8001/discover/pdf \
  -F "file=@patient_report.pdf"
```

## Connection to Module 2

Module 0 saves results to a shared SQLite database at `data/pd_memory.db`.
Module 2 reads from this database automatically — every Qwen2-VL query is
enriched with the latest biomarker context.

To enable in Module 2, add to `multimodal_engine.py`:
```python
from integration.module0_bridge import Module0Bridge
bridge = Module0Bridge(db_path="path/to/pd_memory.db")
# Then prepend bridge.get_context() to your system prompt
```

## File structure

```
module0_biomarker/
├── api/
│   └── main.py              # FastAPI app (port 8001)
├── core/
│   ├── engine.py            # Main orchestrator
│   ├── memory_store.py      # Shared SQLite DB
│   └── gene_info.py         # Gene explanations + ODE params
├── rag/
│   └── relevance.py         # PD relevance checker
├── network/
│   ├── kegg_string.py       # KEGG + STRING pipeline
│   └── visualizer.py        # PPI network graph
├── report/
│   └── pdf_generator.py     # PDF report generator
├── integration/
│   └── module0_bridge.py    # Module 2 memory bridge
├── data/
│   ├── cache/               # KEGG/STRING cached responses
│   ├── outputs/             # Generated graphs + reports
│   └── pd_memory.db         # Shared SQLite (auto-created)
└── requirements.txt
```
