"""
FastAPI application — PD Multimodal Clinical AI System
Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""
import sys
import json
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from core.multimodal_engine import PDMultiModalEngine
from modules.module6_ode.ode_simulator import HodgkinHuxleyModel, DopamineKineticsModel
from modules.module7_router.router import AgentRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, release on shutdown."""
    cfg = Config()
    cfg.ensure_dirs()

    logger.info(f"Loading Qwen2-VL model: {cfg.model.model_id}")
    fast_id = cfg.model.fast_chat_model_id if getattr(cfg.model, "fast_chat_enabled", True) else None
    state["engine"] = PDMultiModalEngine(
        model_id=cfg.model.model_id,
        fast_chat_model_id=fast_id,
    )
    state["router"] = AgentRouter(
        engine=state["engine"],
        ode_simulator=HodgkinHuxleyModel(),
    )
    state["config"] = cfg
    logger.info("System ready.")
    yield
    state.clear()


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Parkinson's Disease Clinical AI System",
    description=(
        "Multimodal clinical AI for Parkinson's disease — "
        "chat, MRI analysis, report summarization, gene interpretation, "
        "drug recommendations, and neuron simulation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────
def _tmp_file(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.flush()
    return tmp.name


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """System health check."""
    return {
        "status":   "ok",
        "model":    state.get("config", Config()).model.model_id,
        "modules":  ["chat", "mri", "report", "gene", "drug", "simulation", "router"],
    }


@app.post("/api/route")
async def universal_route(
    text:      Optional[str] = Form(None),
    file:      UploadFile    = File(None),
    model:     str           = Form("hh"),
    pd_loss:   float         = Form(0.0),
    mutations: str           = Form(""),
    stage:     str           = Form("moderate"),
    symptoms:  str           = Form(""),
):
    """
    Universal endpoint — upload any input and it routes automatically.
    Accepts: text question, MRI image, PDF report, gene report text, or ODE params.
    """
    file_path    = _tmp_file(file) if file else None
    extra_params = {
        "model":     model,
        "pd_loss":   pd_loss,
        "mutations": [m.strip() for m in mutations.split(",") if m.strip()],
        "stage":     stage,
        "symptoms":  [s.strip() for s in symptoms.split(",") if s.strip()],
    }
    result = state["router"].route(
        text=text,
        file_path=file_path,
        extra_params=extra_params,
    )
    return JSONResponse(content=_safe_json(result))


@app.post("/api/chat")
async def chat(question: str = Form()):
    """Ask any Parkinson's disease question."""
    try:
        result = state["engine"].chat(question)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/chat/stream")
async def chat_stream(question: str = Form()):
    """Stream the chat answer token-by-token as Server-Sent Events."""
    engine = state["engine"]

    def event_gen():
        try:
            for tok in engine.chat_stream(question):
                yield "data: " + json.dumps({"t": tok}) + "\n\n"
            yield "data: " + json.dumps({"done": True}) + "\n\n"
        except Exception as exc:
            logger.exception("Chat streaming failed")
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/mri")
async def analyze_mri(file: UploadFile = File()):
    """Upload a brain MRI image for Parkinson's analysis."""
    try:
        path   = _tmp_file(file)
        result = state["engine"].analyze_mri(path)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("MRI analysis failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/mri/report")
async def analyze_mri_report(file: UploadFile = File()):
    """Upload an MRI radiology PDF report for structured PD analysis."""
    try:
        path   = _tmp_file(file)
        result = state["engine"].analyze_mri_report(path)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("MRI report analysis failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/report")
async def summarize_report(file: UploadFile = File()):
    """Upload a clinical PDF report for summarization."""
    try:
        path   = _tmp_file(file)
        result = state["engine"].summarize_report(path)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Report summarization failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/gene")
async def analyze_genes(report_text: str = Form()):
    """Analyze gene mutation report text for PD risk."""
    try:
        result = state["engine"].analyze_genes(report_text)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Gene analysis failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/drug")
async def recommend_drugs(
    mutations: str = Form(""),
    stage:     str = Form("moderate"),
    symptoms:  str = Form(""),
):
    """Get drug recommendations based on patient profile."""
    try:
        result = state["engine"].recommend_drugs(
            mutations=[m.strip() for m in mutations.split(",") if m.strip()],
            stage=stage,
            symptoms=[s.strip() for s in symptoms.split(",") if s.strip()],
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Drug recommendation failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/simulate")
async def run_simulation(
    model:   str   = Form("hh"),
    pd_loss: float = Form(0.0),
    I_ext:   float = Form(10.0),
):
    """
    Run ODE neuron simulation.
    model: 'hh' (Hodgkin-Huxley) | 'dopamine' (dopamine kinetics)
    pd_loss: 0.0 (healthy) → 1.0 (severe PD) — dopamine model only
    I_ext: external current (uA/cm²) — HH model only
    """
    try:
        if model == "dopamine":
            sim = DopamineKineticsModel({"pd_loss": pd_loss})
            result = sim.simulate(save_plot=True)
        else:
            sim = HodgkinHuxleyModel({"I_ext": I_ext})
            result = sim.simulate(save_plot=True)

        return JSONResponse(content={
            "model_name":  result.model_name,
            "params":      result.params,
            "figure_path": result.figure_path,
            "status":      "ok",
        })
    except Exception as exc:
        logger.exception("Simulation failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/api/plot/{filename}")
async def get_plot(filename: str):
    """Download a simulation plot by filename."""
    cfg  = state.get("config", Config())
    path = Path(cfg.paths.outputs_dir) / filename
    if not path.exists():
        return JSONResponse({"error": "Plot not found"}, status_code=404)
    return FileResponse(str(path), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


def _safe_json(obj):
    """Recursively make objects JSON serializable."""
    import numpy as np
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return _safe_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj
