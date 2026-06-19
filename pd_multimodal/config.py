"""
Centralized configuration for PD Multimodal Clinical AI System.
Change model_id here to switch between GPU sizes.
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).parent


@dataclass
class ModelConfig:
    # ── Multimodal model (MRI / report / gene / drug) ────────────────
    # GTX 1650  (4GB)  → "Qwen/Qwen2-VL-2B-Instruct"
    # RTX 3070  (8GB)  → "Qwen/Qwen2-VL-7B-Instruct"
    # RTX 3090  (24GB) → "Qwen/Qwen2-VL-7B-Instruct"
    # A100      (40GB) → "Qwen/Qwen2-VL-72B-Instruct"
    model_id:      str  = "Qwen/Qwen2-VL-2B-Instruct"
    max_new_tokens: int = 512
    device:        str  = "auto"   # "auto" | "cuda" | "cpu"

    # ── Fast text-only chat model (separate, much faster) ────────────
    # Skips the vision encoder on every chat call, drops TTFT to ~300ms
    # and roughly doubles tokens/sec compared to Qwen2-VL-2B on text.
    fast_chat_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fast_chat_enabled:  bool = True
    chat_max_new_tokens: int = 320


@dataclass
class PathsConfig:
    outputs_dir:   str = str(BASE / "data/outputs")
    knowledge_dir: str = str(BASE / "data/knowledge_base")
    models_dir:    str = str(BASE / "models")


@dataclass
class PortsConfig:
    multimodal: int = 8000
    rag:        int = 8000
    biomarker:  int = 8001
    debugger:   int = 8080


@dataclass
class Config:
    model:  ModelConfig = field(default_factory=ModelConfig)
    paths:  PathsConfig = field(default_factory=PathsConfig)
    ports:  PortsConfig = field(default_factory=PortsConfig)

    def ensure_dirs(self):
        for attr in vars(self.paths).values():
            Path(attr).mkdir(parents=True, exist_ok=True)
