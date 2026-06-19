"""
CORE — Parkinson's Disease Multimodal Engine
Single Qwen2-VL model handling all clinical tasks:
  - Chat Q&A
  - MRI image analysis
  - PDF report summarization
  - Gene mutation interpretation
  - Drug recommendations
"""
import torch
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PDMultiModalEngine:
    """
    Unified multimodal engine using Qwen2-VL.
    Switch model_id for different VRAM budgets:
      - "Qwen/Qwen2-VL-2B-Instruct"  → ~4GB  VRAM (GTX 1650)
      - "Qwen/Qwen2-VL-7B-Instruct"  → ~8GB  VRAM (RTX 3070+)
      - "Qwen/Qwen2-VL-72B-Instruct" → ~40GB VRAM (A100)
    """

    SYSTEM_PROMPT = (
        "You are a specialized clinical AI assistant for Parkinson's Disease analysis. "
        "You can analyze brain MRI scans, interpret genetic mutation reports, "
        "summarize clinical documents, recommend treatments, and answer medical questions. "
        "Always be precise, evidence-based, and recommend specialist consultation "
        "for all clinical decisions. Structure your responses clearly."
    )

    CHAT_SYSTEM_PROMPT = (
        "You are a clinical AI assistant for Parkinson's Disease. "
        "Answer concisely (3–6 sentences max unless the user asks for more), "
        "evidence-based, and recommend specialist consultation for any clinical decision."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        optimize: bool = True,
        load_in_4bit: bool = False,
        fast_chat_model_id: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model_id  = model_id
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {model_id} on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            if optimize:
                # Use bfloat16 if supported for speed/memory, else float16
                model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                # Use PyTorch's native Scaled Dot Product Attention (SDPA)
                # This is natively built into PyTorch 2.0+ and provides similar speed to Flash Attention
                # without requiring ANY extra pip packages locally.
                model_kwargs["attn_implementation"] = "sdpa"
                logger.info("Using PyTorch native SDPA for faster inference.")
            else:
                model_kwargs["torch_dtype"] = torch.float16

            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    # 4-bit quantization reduces VRAM and can speed up generation memory bandwidth
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype", torch.float16)
                    )
                    logger.info("Using 4-bit quantization.")
                except ImportError:
                    logger.warning("bitsandbytes not installed. You can skip 4-bit loading or install it via `pip install bitsandbytes`.")
        else:
            # On CPU, avoid explicit device_map sharding to prevent weight placement mismatches
            # and ensure the model loads into a single device safely.
            model_kwargs["torch_dtype"] = torch.float32

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            **model_kwargs
        )
        self.model.eval()
        logger.info("Model loaded successfully.")

        # ── Module 0 memory bridge ──────────────────────────────────────
        import sys
        module0_path = str(Path(__file__).resolve().parent.parent.parent / "module0_biomarker")
        if module0_path not in sys.path:
            sys.path.append(module0_path)
            
        from integration.module0_bridge import Module0Bridge
        self.m0_bridge = Module0Bridge(
            db_path="../module0_biomarker/data/pd_memory.db"
        )

        # ── Optional: lightweight text-only chat model for fast Q&A ─────
        self.fast_tokenizer = None
        self.fast_model     = None
        if fast_chat_model_id:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info(f"Loading fast chat model: {fast_chat_model_id}")
                self.fast_tokenizer = AutoTokenizer.from_pretrained(fast_chat_model_id)
                fast_kwargs = {}
                if self.device == "cuda":
                    fast_kwargs["device_map"]   = "auto"
                    fast_kwargs["torch_dtype"]  = (
                        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
                    fast_kwargs["attn_implementation"] = "sdpa"
                else:
                    fast_kwargs["torch_dtype"] = torch.float32
                self.fast_model = AutoModelForCausalLM.from_pretrained(
                    fast_chat_model_id, **fast_kwargs
                )
                self.fast_model.eval()
                logger.info("Fast chat model loaded.")
            except Exception as exc:
                logger.warning(
                    f"Failed to load fast chat model ({exc}); chat will fall back to Qwen2-VL."
                )
                self.fast_tokenizer = None
                self.fast_model     = None

    # ─────────────────────────────────────────────────────────────────
    # Internal generation helper
    # ─────────────────────────────────────────────────────────────────
    def _generate(self, messages: list, max_tokens: int = 512) -> str:
        from qwen_vl_utils import process_vision_info

        # ── Inject Module 0 biomarker context ──────────────────────────
        m0_context = self.m0_bridge.get_context()
        if m0_context and messages[0]["role"] == "system":
            messages[0]["content"] += m0_context

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        try:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
        except Exception as exc:
            logger.exception("Qwen2-VL generation failed")
            raise RuntimeError(
                "Multimodal generation failed. Check model compatibility, device settings, and input format. "
                f"Original error: {exc}"
            ) from exc

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    # ─────────────────────────────────────────────────────────────────
    # MODULE 1 — Clinical Chat
    # ─────────────────────────────────────────────────────────────────
    def _chat_messages(self, question: str) -> list:
        sys_prompt = self.CHAT_SYSTEM_PROMPT
        m0_context = self.m0_bridge.get_context()
        if m0_context:
            sys_prompt = sys_prompt + m0_context
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": question},
        ]

    def chat(self, question: str, max_tokens: int = 320) -> dict:
        """Answer Parkinson's disease questions (uses fast text model when available)."""
        messages = self._chat_messages(question)

        if self.fast_model is not None and self.fast_tokenizer is not None:
            answer = self._fast_generate(messages, max_tokens=max_tokens)
        else:
            answer = self._generate(messages, max_tokens=max_tokens)

        return {"question": question, "answer": answer, "module": "chat"}

    def chat_stream(self, question: str, max_tokens: int = 320):
        """Yield decoded chat tokens as they're produced."""
        messages = self._chat_messages(question)
        if self.fast_model is not None and self.fast_tokenizer is not None:
            yield from self._fast_stream(messages, max_tokens=max_tokens)
        else:
            yield from self._vl_stream(messages, max_tokens=max_tokens)

    # ── Fast text-only generation (Qwen2.5-1.5B-Instruct) ──────────────
    def _fast_generate(self, messages: list, max_tokens: int) -> str:
        text = self.fast_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.fast_tokenizer([text], return_tensors="pt").to(self.fast_model.device)
        with torch.no_grad():
            out = self.fast_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.fast_tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.fast_tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _fast_stream(self, messages: list, max_tokens: int):
        from threading import Thread
        from transformers import TextIteratorStreamer

        text = self.fast_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.fast_tokenizer([text], return_tensors="pt").to(self.fast_model.device)
        streamer = TextIteratorStreamer(
            self.fast_tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.fast_tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.fast_model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()

    # ── Streaming through Qwen2-VL (fallback) ──────────────────────────
    def _vl_stream(self, messages: list, max_tokens: int):
        from threading import Thread
        from transformers import TextIteratorStreamer
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()

    # ─────────────────────────────────────────────────────────────────
    # MODULE 2 — MRI Image Analysis
    # ─────────────────────────────────────────────────────────────────
    def analyze_mri(self, image_path: str) -> dict:
        abs_path = str(Path(image_path).resolve())
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{abs_path}"},
                    {
                        "type": "text",
                        "text": (
                            "Analyze this brain MRI for signs of Parkinson's disease. "
                            "Reply STRICTLY in the following structured format and nothing else:\n\n"
                            "VERDICT: <one of: PD-likely | PD-unlikely | Inconclusive | Normal>\n"
                            "CONFIDENCE: <Low | Moderate | High>\n"
                            "KEY FINDINGS:\n"
                            "- <substantia nigra appearance — e.g., loss of swallow-tail sign>\n"
                            "- <basal ganglia / putamen / caudate observations>\n"
                            "- <asymmetry, atrophy, or other notable features>\n"
                            "REASONING: <2–3 sentences linking findings to the verdict>\n"
                            "RECOMMENDATION: <next clinical step — DAT-SPECT, neurologist referral, etc.>\n"
                            "DISCLAIMER: This image-based reading is an AI assistive output, "
                            "not a diagnosis. A specialist must confirm."
                        ),
                    },
                ],
            },
        ]
        analysis = self._generate(messages, max_tokens=420)
        parsed = self._parse_mri_verdict(analysis)
        return {
            "image_path": image_path,
            "analysis":   analysis,
            "verdict":    parsed["verdict"],
            "confidence": parsed["confidence"],
            "findings":   parsed["findings"],
            "reasoning":  parsed["reasoning"],
            "recommendation": parsed["recommendation"],
            "module":     "mri",
        }

    def analyze_mri_report(self, pdf_path: str) -> dict:
        """Analyze a radiology PDF report and return the same verdict schema as analyze_mri."""
        from pdfminer.high_level import extract_text
        raw_text = extract_text(pdf_path)
        cleaned  = re.sub(r"\s+", " ", raw_text).strip()
        if not cleaned:
            return {
                "source": pdf_path, "analysis": "",
                "verdict": "Inconclusive", "confidence": "Low",
                "findings": [], "reasoning":
                "No extractable text in the PDF (it may be a scanned image — try uploading the image directly).",
                "recommendation": "Upload the MRI image directly via the Image tab, or request a text-based report.",
                "module": "mri_report",
            }
        cleaned = cleaned[:3500]

        prompt = (
            "You are analyzing a radiology MRI report for Parkinson's disease findings. "
            "Reply STRICTLY in this exact structured format and nothing else:\n\n"
            "VERDICT: <one of: PD-likely | PD-unlikely | Inconclusive | Normal>\n"
            "CONFIDENCE: <Low | Moderate | High>\n"
            "KEY FINDINGS:\n"
            "- <substantia nigra observations from the report>\n"
            "- <basal ganglia / putamen / caudate observations>\n"
            "- <atrophy, asymmetry, white matter, or other findings>\n"
            "REASONING: <2–3 sentences linking the report's findings to the verdict>\n"
            "RECOMMENDATION: <next clinical step — DAT-SPECT, neurologist referral, repeat imaging, etc.>\n"
            "DISCLAIMER: This is an AI assistive reading of the report, not a diagnosis. "
            "A specialist must confirm.\n\n"
            f"MRI REPORT TEXT:\n{cleaned}"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]

        # Prefer the fast text model when available — this is a pure-text task
        if self.fast_model is not None and self.fast_tokenizer is not None:
            analysis = self._fast_generate(messages, max_tokens=460)
        else:
            analysis = self._generate(messages, max_tokens=460)

        parsed = self._parse_mri_verdict(analysis)
        return {
            "source":         pdf_path,
            "analysis":       analysis,
            "verdict":        parsed["verdict"],
            "confidence":     parsed["confidence"],
            "findings":       parsed["findings"],
            "reasoning":      parsed["reasoning"],
            "recommendation": parsed["recommendation"],
            "module":         "mri_report",
        }

    @staticmethod
    def _parse_mri_verdict(text: str) -> dict:
        """Pull structured fields out of the MRI analysis text."""
        def grab(label):
            m = re.search(rf"{label}\s*:\s*(.+?)(?:\n[A-Z][A-Z ]+:|\Z)",
                          text, re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""

        findings_block = grab("KEY FINDINGS")
        findings = [
            re.sub(r"^[-*•]\s*", "", ln).strip()
            for ln in findings_block.splitlines() if ln.strip()
        ]
        return {
            "verdict":        grab("VERDICT") or "Inconclusive",
            "confidence":     grab("CONFIDENCE") or "Low",
            "findings":       findings,
            "reasoning":      grab("REASONING"),
            "recommendation": grab("RECOMMENDATION"),
        }

    # ─────────────────────────────────────────────────────────────────
    # MODULE 3 — Medical Report Summarization
    # ─────────────────────────────────────────────────────────────────
    def summarize_report(self, pdf_path: str) -> dict:
        """Extract and summarize a clinical PDF report."""
        from pdfminer.high_level import extract_text
        raw_text = extract_text(pdf_path)
        cleaned  = re.sub(r"\s+", " ", raw_text).strip()[:3500]

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Summarize this Parkinson's disease clinical report. "
                    "Extract and structure:\n"
                    "1. Patient summary\n"
                    "2. Key clinical findings\n"
                    "3. Biomarker levels (alpha-synuclein, dopamine, DJ-1, etc.)\n"
                    "4. Current diagnosis and stage\n"
                    "5. Treatment recommendations\n"
                    "6. Follow-up notes\n\n"
                    f"Report text:\n{cleaned}"
                ),
            },
        ]
        summary = self._generate(messages, max_tokens=500)

        # detect mentioned biomarkers
        biomarkers_list = [
            "alpha-synuclein", "DJ-1", "LRRK2", "dopamine",
            "urate", "neurofilament", "parkin", "GBA",
        ]
        found_biomarkers = [b for b in biomarkers_list if b.lower() in cleaned.lower()]

        return {
            "source":      pdf_path,
            "summary":     summary,
            "biomarkers":  found_biomarkers,
            "module":      "report",
        }

    # ─────────────────────────────────────────────────────────────────
    # MODULE 4 — Gene Mutation Analysis
    # ─────────────────────────────────────────────────────────────────
    def analyze_genes(self, report_text: str) -> dict:
        """Interpret gene mutation report for PD risk."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze these genetic mutations for Parkinson's disease risk. "
                    "For each identified mutation provide:\n"
                    "1. Gene name and variant\n"
                    "2. Risk level (High / Moderate / Low / Unknown)\n"
                    "3. Affected biological pathway\n"
                    "4. Clinical significance for Parkinson's disease\n"
                    "5. Recommended genetic counseling actions\n\n"
                    f"Genetic Report:\n{report_text}"
                ),
            },
        ]
        interpretation = self._generate(messages, max_tokens=600)
        return {
            "report_text":   report_text,
            "interpretation": interpretation,
            "module":        "gene",
        }

    # ─────────────────────────────────────────────────────────────────
    # MODULE 5 — Drug Recommendation
    # ─────────────────────────────────────────────────────────────────
    def recommend_drugs(
        self,
        mutations: list,
        stage: str,
        symptoms: list,
    ) -> dict:
        """Generate pharmacogenomics-aware drug recommendations."""
        profile = (
            f"Gene mutations: {', '.join(mutations) if mutations else 'None reported'}\n"
            f"Disease stage: {stage}\n"
            f"Current symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Recommend Parkinson's disease medications for this patient. "
                    "For each recommended drug provide:\n"
                    "1. Drug name and class\n"
                    "2. Mechanism of action\n"
                    "3. Why it suits this patient's genetic/symptom profile\n"
                    "4. Dosage considerations\n"
                    "5. Key drug interactions to watch\n"
                    "6. Side effects relevant to this patient\n\n"
                    f"Patient Profile:\n{profile}"
                ),
            },
        ]
        recommendations = self._generate(messages, max_tokens=600)
        return {
            "patient_profile":  profile,
            "recommendations":  recommendations,
            "module":           "drug",
        }
