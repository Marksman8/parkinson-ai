# PD System Auto-Debugger

Scans every component of pd_multimodal_ai and reports exactly what's
working, what's broken, and how to fix it.

## What it checks (16 checks total)

| Check | What it verifies |
|---|---|
| Python version | 3.10+ required |
| Virtual environment | Running inside pd_env |
| GPU / CUDA | CUDA availability and GPU name |
| VRAM budget | Recommends correct model size |
| Disk space | Enough space for models |
| Port availability | Ports 8000 and 8001 free or in use |
| Python packages | All 13 required packages installed |
| Folder structure | All required files present |
| Core module imports | config, ODE, router import cleanly |
| Model weights | Qwen2-VL downloaded in HF cache |
| ODE engine | SciPy RK45 runs correctly |
| Module 2 API | FastAPI on port 8000 responding |
| Module 0 API | FastAPI on port 8001 responding |
| SQLite memory | pd_memory.db exists and readable |
| ChromaDB RAG | Vector collections loaded with chunks |
| External APIs | KEGG and STRING reachable |

## 3 ways to use it

### Option 1 — Browser dashboard (recommended)
```bash
cd pd_debugger
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8080
```
Open: http://localhost:8080
Click "Run Diagnostics" — see live dashboard with all checks.

### Option 2 — Terminal CLI
```bash
cd pd_debugger
python run_debug.py --root ../pd_multimodal_ai
```
Color-coded output with fix instructions for every failure.

### Option 3 — Programmatic
```python
from pd_debugger.core.debugger import PDAutoDebugger
debugger = PDAutoDebugger(project_root=".")
report   = debugger.run()
print(report.summary)
for check in report.checks:
    print(check.name, check.status.value, check.message)
```

## Where to put it

```
pd_multimodal_ai/
├── core/
├── modules/
├── api/
├── pd_rag/
├── module0_biomarker/
└── pd_debugger/         ← put this folder here
    ├── core/
    │   └── debugger.py
    ├── api/
    │   └── main.py
    ├── run_debug.py
    └── README.md
```

## Example output

```
  PD SYSTEM AUTO-DEBUGGER
  ─────────────────────────────────────────

  [✓] Python version               Python 3.11.4
  [✓] Virtual environment          Active: C:\...\pd_env
  [✓] GPU / CUDA                   NVIDIA GeForce GTX 1650  — 4.0GB total
  [✓] VRAM budget                  4.0GB → recommended: Qwen2-VL-2B-Instruct
  [✓] Disk space                   48.3GB free (42% used)
  [✓] Port availability            Port 8000 (free) | Port 8001 (free)
  [✓] Python packages              All 13 required packages installed
  [✓] Folder structure             All required files present
  [✓] Core module imports          config, ODE, router all import cleanly
  [!] Model weights                Not cached — will download on first run (~5GB)
      Fix: Run uvicorn api.main:app and wait for model download
  [✓] ODE engine                   RK45 OK — 53 steps
  [!] Module 2 API                 Not running
      Fix: uvicorn api.main:app --host 0.0.0.0 --port 8000
  [—] Module 0 API                 Not running
  [!] SQLite memory                Database not created yet
  [!] ChromaDB RAG                 Vector DB not created yet
      Fix: curl -X POST http://localhost:8000/api/ingest
  [✓] External APIs                KEGG: HTTP 200 | STRING: HTTP 200

  ─────────────────────────────────────────
  RESULT: DEGRADED  [ 11 passed  5 warnings  0 errors ]
```
