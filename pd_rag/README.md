# RAG System for Module 2 — PD Clinical AI Tool

Adds literature-grounded answering to Module 2 using 4 knowledge sources.

## What changes with RAG

```
WITHOUT RAG:
Doctor: "What is the mechanism of LRRK2 G2019S in PD?"
Qwen2-VL: answers from training data (may be outdated)

WITH RAG:
Doctor: "What is the mechanism of LRRK2 G2019S in PD?"
System: searches pd_papers + pd_gene_db → finds relevant chunks
Qwen2-VL: answers grounded in actual research papers you provided
```

## Smart trigger — RAG only fires when needed

| Query type | RAG triggered? | Collections searched |
|---|---|---|
| "What causes PD tremor?" | ✅ YES | pd_papers, pd_guidelines, pd_module0 |
| "LRRK2 G2019S mutation risk?" | ✅ YES | pd_papers, pd_gene_db, pd_module0 |
| "What drug for stage 2 PD?" | ✅ YES | pd_papers, pd_guidelines, pd_gene_db |
| MRI image upload | ❌ NO | — |
| PDF report upload | ❌ NO | — |
| ODE simulation | ❌ NO | — |
| "Hello" | ❌ NO | — |

## Setup

```bash
# 1. Install extra dependencies
pip install sentence-transformers chromadb pdfminer.six reportlab

# 2. Add your documents (optional — system works with built-in knowledge)
#    Copy PD research PDFs here:
mkdir -p data/papers
cp your_pd_papers/*.pdf data/papers/

#    Copy clinical guideline PDFs here:
mkdir -p data/guidelines
cp your_guidelines/*.pdf data/guidelines/

#    Copy ClinVar TSV or PharmGKB JSON here:
mkdir -p data/gene_db
cp variant_summary.txt data/gene_db/   # ClinVar export

# 3. Run ingestion (builds ChromaDB)
curl -X POST http://localhost:8000/api/ingest

# 4. Start Module 2 with RAG
uvicorn pd_rag.api_main:app --host 0.0.0.0 --port 8000
```

## Folder structure for your documents

```
data/
├── papers/            ← Add PD research PDFs here
│   ├── spillantini_1997_lewy_bodies.pdf
│   ├── paisan_ruiz_2004_lrrk2.pdf
│   └── ...
├── guidelines/        ← Add clinical guidelines here
│   ├── mds_guidelines_2023.pdf
│   └── ...
├── gene_db/           ← ClinVar/PharmGKB exports
│   ├── variant_summary.txt   (ClinVar TSV)
│   └── pharmgkb_annotations.json
└── vector_db/         ← Auto-created by ChromaDB
    ├── pd_papers/
    ├── pd_guidelines/
    ├── pd_gene_db/
    └── pd_module0/
```

## How Module 0 connects

When Module 0 runs a biomarker discovery session, results are saved to
`pd_memory.db`. The RAG system reads this automatically and includes the
patient's specific gene findings in every query context.

Flow:
```
Module 0 discovers: SNCA hub gene, C_D=0.847
             ↓
Saved to pd_memory.db
             ↓
Module 2 RAG reads this
             ↓
Doctor asks: "what treatment do you recommend?"
             ↓
Qwen2-VL knows: patient has SNCA mutation
              + retrieves SNCA treatment papers
              = personalised, evidence-based answer
```

## New API endpoints

| Endpoint | Description |
|---|---|
| POST /api/ingest | Run full ingestion pipeline |
| POST /api/rag/ingest-paper | Upload single PDF paper |
| GET /api/rag/status | Check collection sizes |
| POST /api/rag/sync-module0 | Manually sync Module 0 results |
| GET /api/rag/search?query=... | Search knowledge base directly |
