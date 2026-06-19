"""
FastAPI — Module 2 with RAG
New endpoints added:
  POST /api/ingest          — add documents to knowledge base
  GET  /api/rag/status      — check RAG collections
  GET  /api/rag/sync-module0 — sync Module 0 results into RAG
"""
import sys
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from pd_rag.multimodal_engine_rag import PDMultiModalEngine
from modules.module6_ode.ode_simulator import (
    HodgkinHuxleyModel, DopamineKineticsModel
)
from modules.module7_router.router import AgentRouter, ModuleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config()
    cfg.ensure_dirs()
    logger.info(f"Loading {cfg.model.model_id} with RAG enabled...")

    engine = PDMultiModalEngine(
        model_id        = cfg.model.model_id,
        vector_db_path  = "data/vector_db",
        module0_db_path = "../module0_biomarker/data/pd_memory.db",
        rag_enabled     = True,
    )
    state["engine"] = engine
    state["router"] = AgentRouter(
        engine={
            ModuleType.CHAT:       engine,
            ModuleType.MRI:        engine,
            ModuleType.REPORT:     engine,
            ModuleType.GENE:       engine,
            ModuleType.DRUG:       engine,
            ModuleType.SIMULATION: HodgkinHuxleyModel(),
        },
        ode_simulator=HodgkinHuxleyModel(),
    )
    logger.info("System ready with RAG.")
    yield
    state.clear()


app = FastAPI(
    title="PD Clinical AI Tool — Module 2 (RAG Enhanced)",
    description=(
        "Multimodal clinical AI for Parkinson's disease. "
        "RAG-enhanced with PD research papers, clinical guidelines, "
        "gene mutation databases, and Module 0 biomarker results."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _tmp(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.flush()
    return tmp.name


# ── EXISTING ENDPOINTS (unchanged) ────────────────────────────────────

@app.get("/health")
async def health():
    rag_status = state["engine"].rag_status() if "engine" in state else {}
    return {
        "status":  "ok",
        "model":   state.get("engine", {}).model_id if "engine" in state else "loading",
        "rag":     rag_status,
        "modules": ["chat", "mri", "report", "gene", "drug", "simulation"],
    }


@app.post("/api/chat")
async def chat(question: str = Form()):
    result = state["engine"].chat(question)
    return JSONResponse(content=result)


@app.post("/api/mri")
async def analyze_mri(file: UploadFile = File()):
    path   = _tmp(file)
    result = state["engine"].analyze_mri(path)
    return JSONResponse(content=result)


@app.post("/api/report")
async def summarize_report(file: UploadFile = File()):
    path   = _tmp(file)
    result = state["engine"].summarize_report(path)
    return JSONResponse(content=result)


@app.post("/api/gene")
async def analyze_genes(report_text: str = Form()):
    result = state["engine"].analyze_genes(report_text)
    return JSONResponse(content=result)


@app.post("/api/drug")
async def recommend_drugs(
    mutations: str = Form(""),
    stage:     str = Form("moderate"),
    symptoms:  str = Form(""),
):
    result = state["engine"].recommend_drugs(
        mutations=[m.strip() for m in mutations.split(",") if m.strip()],
        stage=stage,
        symptoms=[s.strip() for s in symptoms.split(",") if s.strip()],
    )
    return JSONResponse(content=result)


@app.post("/api/simulate")
async def simulate(
    model:   str   = Form("hh"),
    pd_loss: float = Form(0.0),
    I_ext:   float = Form(10.0),
):
    if model == "dopamine":
        sim = DopamineKineticsModel({"pd_loss": pd_loss})
    else:
        sim = HodgkinHuxleyModel({"I_ext": I_ext})
    result = sim.simulate(save_plot=True)
    return JSONResponse(content={
        "model_name":  result.model_name,
        "figure_path": result.figure_path,
        "params":      result.params,
    })


# ── NEW RAG ENDPOINTS ─────────────────────────────────────────────────

@app.post(
    "/api/ingest",
    summary="Add documents to RAG knowledge base",
)
async def ingest_documents():
    """
    Triggers full RAG ingestion pipeline.
    Call this after adding PDFs to:
      data/papers/       — PD research papers
      data/guidelines/   — clinical guidelines
      data/gene_db/      — ClinVar/PharmGKB files
    Module 0 results are synced automatically.
    """
    result = state["engine"].ingest_documents()
    return JSONResponse(content=result)


@app.post(
    "/api/rag/ingest-paper",
    summary="Upload a single research paper PDF",
)
async def ingest_paper(file: UploadFile = File()):
    """Upload a PD research paper PDF to add to the knowledge base."""
    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)
    dest = papers_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Re-ingest
    result = state["engine"].ingest_documents()
    return JSONResponse(content={
        "file":    file.filename,
        "saved":   str(dest),
        "ingest":  result,
    })


@app.get(
    "/api/rag/status",
    summary="Check RAG knowledge base status",
)
async def rag_status():
    """
    Shows how many documents are in each collection:
      pd_papers     — research papers
      pd_guidelines — clinical guidelines
      pd_gene_db    — gene variant database
      pd_module0    — Module 0 biomarker sessions
    """
    return JSONResponse(content=state["engine"].rag_status())


@app.post(
    "/api/rag/sync-module0",
    summary="Sync Module 0 biomarker results into RAG",
)
async def sync_module0():
    """
    Manually trigger sync of Module 0 biomarker discovery
    results into the RAG knowledge base.
    This happens automatically during ingestion.
    """
    try:
        import sys
        sys.path.insert(0, "pd_rag")
        from ingest.ingestor import DocumentIngestor
        ingestor = DocumentIngestor()
        count    = ingestor.sync_module0()
        return JSONResponse(content={
            "status": "ok",
            "synced": count,
        })
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500,
        )


@app.get(
    "/api/rag/search",
    summary="Search the RAG knowledge base directly",
)
async def rag_search(query: str, top_k: int = 5):
    """
    Search across all RAG collections — useful for testing
    what the system knows about a topic.
    """
    retriever = state["engine"]._rag_retriever
    if not retriever:
        return JSONResponse({"error": "RAG not available"}, status_code=503)
    results = retriever.retrieve(
        query=query,
        collections=["pd_papers", "pd_guidelines", "pd_gene_db", "pd_module0"],
        top_k=top_k,
    )
    return JSONResponse(content=[{
        "text":       r["text"][:300],
        "source":     r["source"],
        "collection": r["collection"],
        "score":      r["score"],
    } for r in results])


if __name__ == "__main__":
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=False)
