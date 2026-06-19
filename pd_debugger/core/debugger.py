"""
PD System Auto-Debugger
Scans the entire pd_multimodal_ai system and reports
every issue, warning, and status check.
"""
import importlib
import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Status(Enum):
    OK      = "ok"
    WARNING = "warning"
    ERROR   = "error"
    SKIP    = "skip"


@dataclass
class CheckResult:
    name:     str
    status:   Status
    message:  str
    detail:   str    = ""
    fix:      str    = ""
    duration: float  = 0.0


@dataclass
class DebugReport:
    timestamp:  str
    system:     Dict
    checks:     List[CheckResult] = field(default_factory=list)
    summary:    Dict              = field(default_factory=dict)

    def add(self, result: CheckResult):
        self.checks.append(result)

    def finalise(self):
        ok      = sum(1 for c in self.checks if c.status == Status.OK)
        warn    = sum(1 for c in self.checks if c.status == Status.WARNING)
        errors  = sum(1 for c in self.checks if c.status == Status.ERROR)
        skipped = sum(1 for c in self.checks if c.status == Status.SKIP)
        self.summary = {
            "total":    len(self.checks),
            "ok":       ok,
            "warnings": warn,
            "errors":   errors,
            "skipped":  skipped,
            "health":   "healthy" if errors == 0 and warn <= 2 else
                        "degraded" if errors == 0 else "critical",
        }


def _run(name: str, fn) -> CheckResult:
    """Run a single check and capture exceptions."""
    t0 = time.perf_counter()
    try:
        result = fn()
        result.duration = round(time.perf_counter() - t0, 3)
        return result
    except Exception as e:
        return CheckResult(
            name     = name,
            status   = Status.ERROR,
            message  = f"Check crashed: {e}",
            fix      = "Check the debugger code or run manually.",
            duration = round(time.perf_counter() - t0, 3),
        )


# ════════════════════════════════════════════════════════════════════
# INDIVIDUAL CHECK FUNCTIONS
# ════════════════════════════════════════════════════════════════════

def check_python_version() -> CheckResult:
    v     = sys.version_info
    valid = v.major == 3 and v.minor >= 10
    return CheckResult(
        name    = "Python version",
        status  = Status.OK if valid else Status.ERROR,
        message = f"Python {v.major}.{v.minor}.{v.micro}",
        fix     = "Install Python 3.10 or higher." if not valid else "",
    )


def check_gpu() -> CheckResult:
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            used  = torch.cuda.memory_allocated(0) / 1e9
            return CheckResult(
                name    = "GPU / CUDA",
                status  = Status.OK,
                message = f"{name}  —  {vram:.1f}GB total, {used:.1f}GB used",
            )
        else:
            return CheckResult(
                name    = "GPU / CUDA",
                status  = Status.WARNING,
                message = "CUDA not available — running on CPU",
                fix     = "Install CUDA 12.1 and matching PyTorch build.",
            )
    except ImportError:
        return CheckResult(
            name    = "GPU / CUDA",
            status  = Status.ERROR,
            message = "PyTorch not installed",
            fix     = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        )


def check_packages() -> CheckResult:
    required = {
        "torch":                 "PyTorch",
        "transformers":          "HuggingFace Transformers",
        "fastapi":               "FastAPI",
        "uvicorn":               "Uvicorn",
        "scipy":                 "SciPy",
        "numpy":                 "NumPy",
        "matplotlib":            "Matplotlib",
        "pdfminer":              "pdfminer.six",
        "chromadb":              "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "networkx":              "NetworkX",
        "requests":              "Requests",
        "reportlab":             "ReportLab",
    }
    missing  = []
    versions = {}
    for pkg, label in required.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            versions[pkg] = ver
        except ImportError:
            missing.append(label)

    if missing:
        return CheckResult(
            name    = "Python packages",
            status  = Status.ERROR,
            message = f"{len(missing)} missing: {', '.join(missing)}",
            fix     = f"pip install {' '.join(m.lower().replace(' ', '-') for m in missing)}",
            detail  = json.dumps(versions, indent=2),
        )
    return CheckResult(
        name    = "Python packages",
        status  = Status.OK,
        message = f"All {len(required)} required packages installed",
        detail  = json.dumps(versions, indent=2),
    )


def check_folder_structure(root: Path) -> CheckResult:
    required_paths = [
        "core/multimodal_engine.py",
        "modules/module6_ode/ode_simulator.py",
        "modules/module6_ode/visualizer.py",
        "modules/module7_router/router.py",
        "api/main.py",
        "config.py",
        "requirements.txt",
    ]
    optional_paths = [
        "pd_rag/core/trigger.py",
        "pd_rag/ingest/ingestor.py",
        "pd_rag/retrieval/retriever.py",
        "module0_biomarker/core/engine.py",
        "module0_biomarker/api/main.py",
        "data/vector_db",
        "data/papers",
    ]
    missing   = [p for p in required_paths if not (root / p).exists()]
    opt_miss  = [p for p in optional_paths if not (root / p).exists()]

    if missing:
        return CheckResult(
            name    = "Folder structure",
            status  = Status.ERROR,
            message = f"{len(missing)} required files missing",
            detail  = "\n".join(f"  MISSING: {p}" for p in missing),
            fix     = "Re-download and extract the project zip.",
        )
    if opt_miss:
        return CheckResult(
            name    = "Folder structure",
            status  = Status.WARNING,
            message = f"Core OK — {len(opt_miss)} optional files missing",
            detail  = "\n".join(f"  optional: {p}" for p in opt_miss),
            fix     = "Add pd_rag/ and module0_biomarker/ for full functionality.",
        )
    return CheckResult(
        name    = "Folder structure",
        status  = Status.OK,
        message = "All required files present",
    )


def check_model_weights(root: Path) -> CheckResult:
    """Check if Qwen2-VL model is downloaded."""
    try:
        from transformers import AutoProcessor
        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        # Check HuggingFace cache
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_slug = "models--Qwen--Qwen2-VL-2B-Instruct"
        if (cache / model_slug).exists():
            size = sum(
                f.stat().st_size
                for f in (cache / model_slug).rglob("*")
                if f.is_file()
            ) / 1e9
            return CheckResult(
                name    = "Model weights (Qwen2-VL-2B)",
                status  = Status.OK,
                message = f"Found in HF cache ({size:.1f}GB)",
            )
        return CheckResult(
            name    = "Model weights (Qwen2-VL-2B)",
            status  = Status.WARNING,
            message = "Not cached — will download on first run (~5GB)",
            fix     = "Run: uvicorn api.main:app and wait for model download.",
        )
    except ImportError:
        return CheckResult(
            name    = "Model weights (Qwen2-VL-2B)",
            status  = Status.ERROR,
            message = "transformers not installed",
            fix     = "pip install transformers",
        )


def check_api_health(port: int = 8000) -> CheckResult:
    """Check if Module 2 FastAPI is running."""
    try:
        import requests
        r = requests.get(f"http://localhost:{port}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            return CheckResult(
                name    = f"Module 2 API (port {port})",
                status  = Status.OK,
                message = f"Running — model: {data.get('model','?')}",
                detail  = json.dumps(data, indent=2),
            )
        return CheckResult(
            name    = f"Module 2 API (port {port})",
            status  = Status.WARNING,
            message = f"Responded with HTTP {r.status_code}",
        )
    except Exception:
        return CheckResult(
            name    = f"Module 2 API (port {port})",
            status  = Status.WARNING,
            message = "Not running",
            fix     = "uvicorn api.main:app --host 0.0.0.0 --port 8000",
        )


def check_module0_api(port: int = 8001) -> CheckResult:
    """Check if Module 0 FastAPI is running."""
    try:
        import requests
        r = requests.get(f"http://localhost:{port}/health", timeout=3)
        if r.status_code == 200:
            return CheckResult(
                name    = f"Module 0 API (port {port})",
                status  = Status.OK,
                message = "Running",
            )
        return CheckResult(
            name    = f"Module 0 API (port {port})",
            status  = Status.WARNING,
            message = f"HTTP {r.status_code}",
        )
    except Exception:
        return CheckResult(
            name    = f"Module 0 API (port {port})",
            status  = Status.WARNING,
            message = "Not running",
            fix     = "cd module0_biomarker && uvicorn api.main:app --port 8001",
        )


def check_sqlite_memory(db_path: Path) -> CheckResult:
    """Check the shared SQLite memory database."""
    if not db_path.exists():
        return CheckResult(
            name    = "SQLite memory (pd_memory.db)",
            status  = Status.WARNING,
            message = "Database not created yet",
            fix     = "Run a biomarker discovery via Module 0 to create it.",
        )
    try:
        conn  = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        sess  = conn.execute("SELECT COUNT(*) as n FROM sessions").fetchone()["n"]
        genes = conn.execute("SELECT COUNT(*) as n FROM top_genes").fetchone()["n"]
        conn.close()
        return CheckResult(
            name    = "SQLite memory (pd_memory.db)",
            status  = Status.OK,
            message = f"{sess} sessions, {genes} gene records",
        )
    except Exception as e:
        return CheckResult(
            name    = "SQLite memory (pd_memory.db)",
            status  = Status.ERROR,
            message = f"DB error: {e}",
            fix     = "Delete pd_memory.db and let it recreate.",
        )


def check_chromadb(vector_db_path: Path) -> CheckResult:
    """Check ChromaDB vector store."""
    if not vector_db_path.exists():
        return CheckResult(
            name    = "ChromaDB (RAG)",
            status  = Status.WARNING,
            message = "Vector DB not created yet",
            fix     = "curl -X POST http://localhost:8000/api/ingest",
        )
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(vector_db_path))
        cols   = client.list_collections()
        counts = {c.name: c.count() for c in cols}
        total  = sum(counts.values())
        if total == 0:
            return CheckResult(
                name    = "ChromaDB (RAG)",
                status  = Status.WARNING,
                message = "Collections exist but empty",
                fix     = "curl -X POST http://localhost:8000/api/ingest",
                detail  = json.dumps(counts),
            )
        return CheckResult(
            name    = "ChromaDB (RAG)",
            status  = Status.OK,
            message = f"{total} total chunks across {len(counts)} collections",
            detail  = json.dumps(counts, indent=2),
        )
    except Exception as e:
        return CheckResult(
            name    = "ChromaDB (RAG)",
            status  = Status.ERROR,
            message = f"ChromaDB error: {e}",
            fix     = "pip install chromadb",
        )


def check_ode_engine() -> CheckResult:
    """Run a quick ODE simulation to verify SciPy is working."""
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
        # Minimal test: exponential decay
        sol = solve_ivp(lambda t, y: [-0.5 * y[0]], (0, 10), [1.0], method="RK45")
        if sol.success and len(sol.t) > 5:
            return CheckResult(
                name    = "ODE engine (SciPy)",
                status  = Status.OK,
                message = f"RK45 OK — {len(sol.t)} steps",
            )
        return CheckResult(
            name    = "ODE engine (SciPy)",
            status  = Status.ERROR,
            message = "solve_ivp returned failure",
            fix     = "pip install scipy --upgrade",
        )
    except Exception as e:
        return CheckResult(
            name    = "ODE engine (SciPy)",
            status  = Status.ERROR,
            message = str(e),
            fix     = "pip install scipy",
        )


def check_network_apis() -> CheckResult:
    """Check connectivity to KEGG and STRING APIs."""
    try:
        import requests
        results = {}
        for name, url in [
            ("KEGG",   "https://rest.kegg.jp/info/pathway"),
            ("STRING", "https://string-db.org/api/json/version"),
        ]:
            try:
                r = requests.get(url, timeout=6)
                results[name] = f"HTTP {r.status_code}"
            except Exception as e:
                results[name] = f"unreachable ({e})"

        all_ok = all("200" in v for v in results.values())
        some_ok = any("200" in v for v in results.values())
        return CheckResult(
            name    = "External APIs (KEGG / STRING)",
            status  = Status.OK if all_ok else (
                      Status.WARNING if some_ok else Status.WARNING),
            message = "  |  ".join(f"{k}: {v}" for k, v in results.items()),
            fix     = "" if all_ok else "Module 0 will use local fallback data.",
        )
    except ImportError:
        return CheckResult(
            name    = "External APIs (KEGG / STRING)",
            status  = Status.WARNING,
            message = "requests not installed — API check skipped",
        )


def check_disk_space() -> CheckResult:
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / 1e9
    pct_used = used / total * 100
    status = (
        Status.ERROR   if free_gb < 5    else
        Status.WARNING if free_gb < 20   else
        Status.OK
    )
    return CheckResult(
        name    = "Disk space",
        status  = status,
        message = f"{free_gb:.1f}GB free  ({pct_used:.0f}% used)",
        fix     = "Free up disk space — models need ~15GB." if status != Status.OK else "",
    )


def check_venv() -> CheckResult:
    """Check if running inside a virtual environment."""
    in_venv = (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    return CheckResult(
        name    = "Virtual environment",
        status  = Status.OK if in_venv else Status.WARNING,
        message = f"Active: {sys.prefix}" if in_venv else "Not in a venv",
        fix     = "" if in_venv else (
            "python -m venv pd_env && pd_env\\Scripts\\activate  (Windows)\n"
            "python -m venv pd_env && source pd_env/bin/activate  (Linux)"
        ),
    )


def check_port_availability() -> CheckResult:
    """Check if ports 8000 and 8001 are free or in use."""
    import socket
    status_map = {}
    for port in [8000, 8001]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            result = s.connect_ex(("127.0.0.1", port))
            status_map[port] = "in use" if result == 0 else "free"
    return CheckResult(
        name    = "Port availability",
        status  = Status.OK,
        message = (
            f"Port 8000 ({status_map[8000]})  |  "
            f"Port 8001 ({status_map[8001]})"
        ),
    )


def check_imports(root: Path) -> CheckResult:
    """Try importing core modules."""
    sys.path.insert(0, str(root))
    failed = []
    for mod, path in [
        ("config",                          str(root / "config.py")),
        ("modules.module6_ode.ode_simulator", None),
        ("modules.module7_router.router",   None),
    ]:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")

    if failed:
        return CheckResult(
            name    = "Core module imports",
            status  = Status.ERROR,
            message = f"{len(failed)} import failures",
            detail  = "\n".join(failed),
            fix     = "Check for syntax errors or missing dependencies.",
        )
    return CheckResult(
        name    = "Core module imports",
        status  = Status.OK,
        message = "config, ODE, router all import cleanly",
    )


def check_vram_budget() -> CheckResult:
    """Check VRAM against model requirements."""
    try:
        import torch
        if not torch.cuda.is_available():
            return CheckResult(
                name    = "VRAM budget",
                status  = Status.SKIP,
                message = "No CUDA GPU detected",
            )
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_vram >= 20:
            rec = "Qwen2-VL-7B-Instruct"
            status = Status.OK
        elif total_vram >= 7:
            rec = "Qwen2-VL-7B-Instruct (fp16)"
            status = Status.OK
        elif total_vram >= 4:
            rec = "Qwen2-VL-2B-Instruct"
            status = Status.OK
        else:
            rec = "Consider CPU-only mode"
            status = Status.WARNING
        return CheckResult(
            name    = "VRAM budget",
            status  = status,
            message = f"{total_vram:.1f}GB VRAM → recommended: {rec}",
            fix     = "" if status == Status.OK else
                      f"Update config.py: model_id = 'Qwen/Qwen2-VL-2B-Instruct'",
        )
    except ImportError:
        return CheckResult(
            name    = "VRAM budget",
            status  = Status.SKIP,
            message = "PyTorch not installed",
        )


# ════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════════

class PDAutoDebugger:
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root).resolve()

    def run(self) -> DebugReport:
        report = DebugReport(
            timestamp = datetime.now().isoformat(),
            system    = {
                "os":       platform.system(),
                "platform": platform.platform(),
                "python":   sys.version,
                "cwd":      str(Path.cwd()),
                "root":     str(self.root),
            },
        )

        checks = [
            ("Python version",           check_python_version),
            ("Virtual environment",      check_venv),
            ("GPU / CUDA",               check_gpu),
            ("VRAM budget",              check_vram_budget),
            ("Disk space",               check_disk_space),
            ("Port availability",        check_port_availability),
            ("Python packages",          check_packages),
            ("Folder structure",         lambda: check_folder_structure(self.root)),
            ("Core module imports",      lambda: check_imports(self.root)),
            ("Model weights",            lambda: check_model_weights(self.root)),
            ("ODE engine",               check_ode_engine),
            ("Module 2 API",             lambda: check_api_health(8000)),
            ("Module 0 API",             lambda: check_module0_api(8001)),
            ("SQLite memory",            lambda: check_sqlite_memory(
                                             self.root / "module0_biomarker" /
                                             "data" / "pd_memory.db")),
            ("ChromaDB RAG",             lambda: check_chromadb(
                                             self.root / "data" / "vector_db")),
            ("External APIs",            check_network_apis),
        ]

        for name, fn in checks:
            result = _run(name, fn)
            report.add(result)
            icon = {"ok": "✓", "warning": "!", "error": "✗", "skip": "-"}[result.status.value]
            logger.info(f"  [{icon}] {name}: {result.message}")

        report.finalise()
        return report

    def run_and_save(self, out_path: str = "debug_report.json") -> DebugReport:
        report = self.run()
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": report.timestamp,
                "system":    report.system,
                "summary":   report.summary,
                "checks":    [
                    {
                        "name":     c.name,
                        "status":   c.status.value,
                        "message":  c.message,
                        "detail":   c.detail,
                        "fix":      c.fix,
                        "duration": c.duration,
                    }
                    for c in report.checks
                ],
            }, f, indent=2)
        return report
