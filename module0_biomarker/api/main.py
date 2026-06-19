"""
Module 0 — FastAPI Application
Doctor-facing REST API for automated biomarker discovery.

Endpoints:
  POST /discover          — main endpoint, doctor provides text
  POST /discover/pdf      — upload clinical PDF instead of text
  GET  /sessions          — list past discovery sessions
  GET  /session/{id}      — get full session details
  GET  /report/{id}       — download PDF report
  GET  /graph/{id}        — download PPI network graph
  GET  /context/module2   — get context string for Module 2
  GET  /health            — health check
"""
import logging
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.engine       import BiomarkerDiscoveryEngine
    from core.memory_store import PDMemoryStore
    state["engine"] = BiomarkerDiscoveryEngine()
    state["memory"] = state["engine"].memory
    logger.info("Module 0 — Biomarker Discovery Engine ready.")
    yield
    state.clear()


app = FastAPI(
    title="PD Biomarker Discovery Tool — Module 0",
    description=(
        "Automated Parkinson's disease biomarker identification. "
        "Neurologist provides clinical text → system fetches KEGG pathway, "
        "builds STRING PPI network, runs degree centrality, returns top hub genes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# MAIN DISCOVERY ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@app.post("/discover", summary="Automated biomarker discovery from clinical text")
async def discover(
    clinical_text: str = Form(
        ...,
        description=(
            "Clinical notes, symptoms, mutations, or any text describing "
            "the patient's condition. The system will automatically detect "
            "if it is Parkinson's-related and run the full pipeline."
        ),
        example=(
            "Patient presents with resting tremor, rigidity, and bradykinesia. "
            "Genetic testing shows LRRK2 G2019S mutation. Family history of PD. "
            "DAT scan shows reduced uptake in left putamen. "
            "Started on levodopa-carbidopa 100/25mg three times daily."
        ),
    )
):
    """
    Full automated pipeline:
    1. Checks if input is PD-related (RAG + keywords)
    2. Fetches KEGG PD pathway genes
    3. Builds STRING PPI network
    4. Runs degree centrality analysis
    5. Returns top 5 hub genes with explanations
    6. Retrieves ODE parameters for simulation
    7. Generates PPI network visualisation
    8. Creates downloadable PDF report
    9. Saves everything to shared memory for Module 2
    """
    result = state["engine"].run(clinical_text)
    return JSONResponse(content=_safe(result))


@app.post("/discover/pdf", summary="Automated biomarker discovery from uploaded PDF")
async def discover_pdf(file: UploadFile = File(...)):
    """Upload a clinical PDF report — text is extracted automatically."""
    try:
        from pdfminer.high_level import extract_text
        import re
        import os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        shutil.copyfileobj(file.file, tmp)
        tmp.close()  # MUST close on Windows before another handle can read it
        text = extract_text(tmp.name)
        text = re.sub(r"\s+", " ", text).strip()[:3000]
        os.unlink(tmp.name)  # Clean up the file
    except Exception as e:
        return JSONResponse(
            {"error": f"PDF extraction failed: {e}"}, status_code=400
        )

    result = state["engine"].run(text)
    return JSONResponse(content=_safe(result))


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

@app.get("/sessions", summary="List past discovery sessions")
async def list_sessions(limit: int = 10):
    sessions = state["memory"].list_sessions(limit=limit)
    return JSONResponse(content=sessions)


@app.get("/session/{session_id}", summary="Get full session details")
async def get_session(session_id: str):
    genes  = state["memory"].get_top_genes(session_id)
    params = state["memory"].get_ode_parameters(session_id)
    return JSONResponse(content={
        "session_id": session_id,
        "top_genes":  genes,
        "ode_params": params,
    })


# ═══════════════════════════════════════════════════════════════════════
# FILE DOWNLOADS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/report/{session_id}", summary="Download PDF report")
async def download_report(session_id: str):
    path = Path(f"data/outputs/biomarker_report_{session_id}.pdf")
    if path.exists():
        return FileResponse(str(path), media_type="application/pdf",
                            filename=f"pd_biomarker_{session_id}.pdf")
    # Try text fallback
    path_txt = Path(f"data/outputs/biomarker_report_{session_id}.txt")
    if path_txt.exists():
        return FileResponse(str(path_txt), media_type="text/plain",
                            filename=f"pd_biomarker_{session_id}.txt")
    return JSONResponse({"error": "Report not found"}, status_code=404)


@app.get("/graph/{session_id}", summary="Download PPI network graph")
async def download_graph(session_id: str):
    path = Path(f"data/outputs/ppi_network_{session_id}.png")
    if path.exists():
        return FileResponse(str(path), media_type="image/png",
                            filename=f"ppi_network_{session_id}.png")
    return JSONResponse({"error": "Graph not found"}, status_code=404)


# ═══════════════════════════════════════════════════════════════════════
# MODULE 2 CONTEXT
# ═══════════════════════════════════════════════════════════════════════

@app.get("/context/module2", summary="Get memory context string for Module 2")
async def get_module2_context():
    """
    Returns the formatted context string that Module 2 (Qwen2-VL)
    prepends to its system prompt to be aware of the latest
    biomarker discovery session.
    """
    context = state["memory"].get_context_for_module2()
    return JSONResponse(content={
        "context": context,
        "has_context": bool(context),
    })


# ═══════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "module":  "Module 0 — Automated Biomarker Discovery",
        "version": "1.0.0",
        "endpoints": [
            "POST /discover",
            "POST /discover/pdf",
            "GET  /sessions",
            "GET  /session/{id}",
            "GET  /report/{id}",
            "GET  /graph/{id}",
            "GET  /context/module2",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
def _safe(obj):
    """Make objects JSON serialisable."""
    import numpy as np, dataclasses
    if dataclasses.is_dataclass(obj):
        return _safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
