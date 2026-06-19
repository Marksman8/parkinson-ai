# Parkinson's Disease Multimodal Clinical AI System

Local, open-source clinical AI for Parkinson's disease — powered by a single
Qwen2-VL multimodal model. No API keys required. Runs entirely on your machine.

## What it does

| Module | Capability | Input |
|--------|-----------|-------|
| Chat AI | PD questions & education | Text |
| MRI Analysis | Brain scan pattern detection | Image |
| Report Summarizer | Clinical PDF summarization | PDF |
| Gene Analyzer | Mutation risk interpretation | Text / VCF |
| Drug Recommender | Pharmacogenomics-aware suggestions | Text params |
| ODE Simulation | Hodgkin-Huxley + dopamine kinetics | Parameters |
| Agent Router | Auto-routes any input | Any |

## GPU Requirements

| Your GPU | Model to use | Set in config.py |
|----------|-------------|-----------------|
| GTX 1650 (4GB) | Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` |
| RTX 3070 (8GB) | Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` |
| RTX 3090 (24GB) | Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` |
| A100 (40GB+) | Qwen2-VL-72B | `Qwen/Qwen2-VL-72B-Instruct` |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/parkinsons-ai.git
cd parkinsons-ai

# 2. Create virtual environment
python -m venv pd_env

# Windows:
pd_env\Scripts\activate
# Linux/Mac:
source pd_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 6. Open API docs
# http://localhost:8000/docs
```

## Usage Examples

```bash
# Chat
curl -X POST http://localhost:8000/api/chat \
  -F "question=What are early symptoms of Parkinson's disease?"

# MRI Analysis
curl -X POST http://localhost:8000/api/mri \
  -F "file=@brain_scan.png"

# Gene Analysis
curl -X POST http://localhost:8000/api/gene \
  -F "report_text=Patient has LRRK2 G2019S mutation and GBA N370S variant"

# Drug Recommendation
curl -X POST http://localhost:8000/api/drug \
  -F "mutations=LRRK2,GBA" \
  -F "stage=moderate" \
  -F "symptoms=tremor,rigidity,bradykinesia"

# ODE Simulation (dopamine with 60% PD loss)
curl -X POST http://localhost:8000/api/simulate \
  -F "model=dopamine" \
  -F "pd_loss=0.6"

# Universal route (auto-detects input type)
curl -X POST http://localhost:8000/api/route \
  -F "file=@patient_report.pdf"
```

## Project Structure

```
parkinsons-ai/
├── core/
│   └── multimodal_engine.py   # Qwen2-VL — handles all AI tasks
├── modules/
│   ├── module6_ode/
│   │   ├── ode_simulator.py   # Hodgkin-Huxley + dopamine kinetics
│   │   └── visualizer.py      # scientific plots
│   └── module7_router/
│       └── router.py          # intent detection + dispatch
├── api/
│   └── main.py                # FastAPI application
├── data/
│   ├── knowledge_base/        # add PD research PDFs here
│   └── outputs/               # simulation plots saved here
├── config.py                  # model selection + paths
└── requirements.txt
```

## Important Notes

- The model downloads automatically on first run (~5GB for 2B, ~15GB for 7B)
- MRI analysis uses vision capabilities of Qwen2-VL directly — no separate CNN needed
- ODE simulation runs on CPU using scipy — no GPU required for that module
- All data stays local — nothing is sent to external servers

## License

MIT License — open source, free to use and modify.
