"""
deploy/app.py — Single-service cloud demo of NeuroSync PD (Render-ready).

The full system is four GPU-dependent FastAPI services driven by launcher.py.
Cloud PaaS (Render/Vercel/Netlify) cannot host the Qwen2-VL model, so this
adapter exposes a *lite* demo as ONE FastAPI app bound to $PORT:

    /                  the existing dashboard UI (API base URLs rewritten to
                       same-origin: /mm, /bio, /dbg)
    /bio/*             Module 0 biomarker discovery (KEGG/STRING/centrality) —
                       runs fully on CPU (relevance falls back to keywords)
    /api/eval/*        the quantitative evaluation results (PPMI + RAG) + figures
    /mm/*  /dbg/*       LLM + debugger endpoints, stubbed with a clear
                       "disabled in cloud demo" message

Run locally:   uvicorn deploy.app:app --reload --port 5000
Render start:  uvicorn deploy.app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("deploy")

# repo root = parent of this deploy/ folder (contains launcher.py, modules)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module0_biomarker"))   # only module0 (avoid pkg clashes)

# writable scratch dir for Module 0 outputs (ephemeral on Render — fine for a demo)
DATA = ROOT / "deploy_data"
(DATA / "outputs").mkdir(parents=True, exist_ok=True)
(DATA / "cache").mkdir(parents=True, exist_ok=True)

_EVAL_DIRS = {
    "clinical": ROOT / "pd_clinical" / "eval" / "results",
    "rag":      ROOT / "pd_rag" / "eval" / "results",
}

state: dict = {}

_LLM_DISABLED = (
    "The language-model features (chat, MRI, gene, drug, simulation) are "
    "disabled in this cloud demo because they require a GPU. The Biomarker "
    "Discovery and Evaluation Results modules run fully here."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from core.engine import BiomarkerDiscoveryEngine
        state["engine"] = BiomarkerDiscoveryEngine(
            db_path=str(DATA / "pd_memory.db"),
            cache_dir=str(DATA / "cache"),
            outputs_dir=str(DATA / "outputs"),
            relevance_db=str(DATA / "cache" / "relevance_db"),
        )
        logger.info("Module 0 engine ready (keyword-fallback relevance).")
    except Exception as e:                      # never let the demo hard-fail
        state["engine"] = None
        logger.warning("Module 0 unavailable: %s", e)
    yield
    state.clear()


app = FastAPI(title="NeuroSync PD — Cloud Demo", lifespan=lifespan)


# ── Dashboard (reuse the real UI, rewrite API base URLs to same-origin) ──
def _dashboard_html() -> str:
    try:
        import launcher
        html = launcher.DASHBOARD_HTML
    except Exception as e:
        return f"<h1>NeuroSync PD</h1><p>Dashboard unavailable: {e}</p>"
    return (html
            .replace("http://localhost:8000", "/mm")
            .replace("http://localhost:8001", "/bio")
            .replace("http://localhost:8080", "/dbg"))


@app.get("/", response_class=HTMLResponse)
async def home():
    return _dashboard_html()


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "cloud-demo",
            "biomarker": state.get("engine") is not None}


# ── Evaluation results (file-based; no heavy deps) ───────────────────────
@app.get("/api/eval/{arm}")
async def eval_results(arm: str):
    d = _EVAL_DIRS.get(arm)
    if d is None:
        return JSONResponse({"available": False, "error": "unknown arm"}, status_code=404)
    md = d / "results.md"
    figs_dir = d / "figures"
    figures = sorted(p.name for p in figs_dir.glob("*.png")) if figs_dir.exists() else []
    return JSONResponse({
        "available": md.exists(),
        "markdown": md.read_text(encoding="utf-8") if md.exists() else "",
        "figures": figures,
    })


@app.get("/api/eval/{arm}/figures/{name}")
async def eval_figure(arm: str, name: str):
    d = _EVAL_DIRS.get(arm)
    if d is None:
        return JSONResponse({"error": "unknown arm"}, status_code=404)
    figs_dir = (d / "figures").resolve()
    f = (figs_dir / name).resolve()
    if not str(f).startswith(str(figs_dir)) or not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(f), media_type="image/png")


# ── Module 0 — biomarker discovery (CPU) ─────────────────────────────────
def _safe(obj):
    import dataclasses
    import numpy as np
    if dataclasses.is_dataclass(obj):
        return _safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@app.get("/bio/health")
async def bio_health():
    return {"status": "ok" if state.get("engine") else "unavailable",
            "module": "Module 0 — Biomarker Discovery"}


@app.post("/bio/discover")
async def bio_discover(clinical_text: str = Form(...)):
    if not state.get("engine"):
        return JSONResponse({"error": "Biomarker engine unavailable."}, status_code=503)
    return JSONResponse(content=_safe(state["engine"].run(clinical_text)))


@app.post("/bio/discover/pdf")
async def bio_discover_pdf(file: UploadFile = File(...)):
    if not state.get("engine"):
        return JSONResponse({"error": "Biomarker engine unavailable."}, status_code=503)
    try:
        import os
        import re
        from pdfminer.high_level import extract_text
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
        text = re.sub(r"\s+", " ", extract_text(tmp.name)).strip()[:3000]
        os.unlink(tmp.name)
    except Exception as e:
        return JSONResponse({"error": f"PDF extraction failed: {e}"}, status_code=400)
    return JSONResponse(content=_safe(state["engine"].run(text)))


@app.get("/bio/report/{session_id}")
async def bio_report(session_id: str):
    for ext, mt in [("pdf", "application/pdf"), ("txt", "text/plain")]:
        p = DATA / "outputs" / f"biomarker_report_{session_id}.{ext}"
        if p.exists():
            return FileResponse(str(p), media_type=mt, filename=p.name)
    return JSONResponse({"error": "Report not found"}, status_code=404)


@app.get("/bio/graph/{session_id}")
async def bio_graph(session_id: str):
    p = DATA / "outputs" / f"ppi_network_{session_id}.png"
    if p.exists():
        return FileResponse(str(p), media_type="image/png", filename=p.name)
    return JSONResponse({"error": "Graph not found"}, status_code=404)


# ── Stubs for GPU-dependent services ─────────────────────────────────────
@app.get("/mm/health")
async def mm_health():
    return {"status": "ok", "mode": "stub", "note": _LLM_DISABLED}


@app.get("/dbg/health")
async def dbg_health():
    return {"status": "ok", "mode": "stub"}


@app.post("/mm/api/chat/stream")
async def mm_chat_stream(question: str = Form(default="")):
    def gen():
        yield f"data: {json.dumps({'t': _LLM_DISABLED})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/dbg/run")
async def dbg_run():
    return JSONResponse({
        "timestamp": "cloud-demo",
        "summary": {"total": 1, "ok": 1, "warnings": 0, "errors": 0},
        "checks": [{
            "name": "Cloud demo mode", "status": "ok", "ms": 0,
            "message": "Lite deployment: biomarker + evaluation modules active; "
                       "GPU/LLM services disabled.",
        }],
    })


@app.api_route("/mm/{path:path}", methods=["GET", "POST"])
async def mm_stub(path: str):
    return JSONResponse({"error": _LLM_DISABLED, "module": "disabled"}, status_code=503)


@app.api_route("/dbg/{path:path}", methods=["GET", "POST"])
async def dbg_stub(path: str):
    return JSONResponse({"error": "Debugger not available in cloud demo."},
                        status_code=503)
