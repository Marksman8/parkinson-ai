"""
MODULE 2 — Parkinson's Clinical AI Tool (RAG-Enhanced)
Qwen2-VL multimodal model with:
  - Smart RAG trigger (searches literature only when needed)
  - 4 ChromaDB collections (papers, guidelines, gene_db, module0)
  - Module 0 memory bridge (patient biomarker context)
  - All original 5 clinical modules preserved
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import re
import torch

logger = logging.getLogger(__name__)


class PDMultiModalEngine:
    """
    Unified multimodal engine — Qwen2-VL + RAG + Module 0 memory.

    Model sizes:
      GTX 1650  (4GB)  → "Qwen/Qwen2-VL-2B-Instruct"
      RTX 3070  (8GB)  → "Qwen/Qwen2-VL-7B-Instruct"
      RTX 3090  (24GB) → "Qwen/Qwen2-VL-7B-Instruct"
      A100      (40GB) → "Qwen/Qwen2-VL-72B-Instruct"
    """

    SYSTEM_PROMPT = (
        "You are a specialized clinical AI assistant for Parkinson's Disease analysis. "
        "You can analyze brain MRI scans, interpret genetic mutation reports, "
        "summarize clinical documents, recommend treatments, and answer medical questions. "
        "Always be precise, evidence-based, and recommend specialist consultation "
        "for all clinical decisions. Structure your responses clearly."
    )

    def __init__(
        self,
        model_id:        str  = "Qwen/Qwen2-VL-2B-Instruct",
        vector_db_path:  str  = "data/vector_db",
        module0_db_path: str  = "../module0_biomarker/data/pd_memory.db",
        rag_enabled:     bool = True,
    ):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.model_id   = model_id
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag_enabled = rag_enabled

        logger.info(f"Loading {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model     = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Qwen2-VL loaded.")

        # ── RAG components ────────────────────────────────────────────
        self._rag_trigger  = None
        self._rag_retriever = None
        if rag_enabled:
            self._init_rag(vector_db_path)

        # ── Module 0 memory bridge ────────────────────────────────────
        self._m0_bridge = None
        self._init_module0_bridge(module0_db_path)

    # ── INITIALISATION ────────────────────────────────────────────────
    def _init_rag(self, vector_db_path: str):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "pd_rag"))
            from core.trigger         import SmartRAGTrigger
            from retrieval.retriever  import PDRetriever

            self._rag_trigger   = SmartRAGTrigger()
            self._rag_retriever = PDRetriever(vector_db_path=vector_db_path)
            stats = self._rag_retriever.collection_stats()
            logger.info(f"RAG ready. Collections: {stats}")
        except Exception as e:
            logger.warning(f"RAG init failed ({e}) — running without RAG.")
            self._rag_enabled = False

    def _init_module0_bridge(self, db_path: str):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "module0_biomarker"))
            from integration.module0_bridge import Module0Bridge
            self._m0_bridge = Module0Bridge(db_path=db_path)
            logger.info("Module 0 memory bridge ready.")
        except Exception as e:
            logger.warning(f"Module 0 bridge unavailable: {e}")

    # ── CORE GENERATION ───────────────────────────────────────────────
    def _generate(
        self,
        messages:   list,
        max_tokens: int = 512,
        query:      Optional[str] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        # 1. Inject Module 0 biomarker context
        m0_context = (
            self._m0_bridge.get_context()
            if self._m0_bridge else ""
        )

        # 2. Smart RAG retrieval
        rag_context = ""
        if self.rag_enabled and self._rag_trigger and query:
            q_type, needs_rag, collections, top_k = \
                self._rag_trigger.classify(query)
            logger.info(self._rag_trigger.describe(q_type, needs_rag, collections))

            if needs_rag and self._rag_retriever:
                rag_context, _ = self._rag_retriever.retrieve_and_format(
                    query       = query,
                    collections = collections,
                    top_k       = top_k,
                    query_type  = q_type.value,
                )

        # 3. Inject both contexts into system message
        if messages and messages[0]["role"] == "system":
            extra = ""
            if m0_context:
                extra += m0_context
            if rag_context:
                extra += rag_context
            if extra:
                messages[0]["content"] = messages[0]["content"] + extra

        # 4. Qwen2-VL inference
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

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

    # ── MODULE 2.1 — CLINICAL CHAT ────────────────────────────────────
    def chat(self, question: str) -> Dict:
        messages = [
            {"role": "system",  "content": self.SYSTEM_PROMPT},
            {"role": "user",    "content": question},
        ]
        answer = self._generate(messages, max_tokens=600, query=question)
        return {"question": question, "answer": answer, "module": "chat"}

    # ── MODULE 2.2 — MRI ANALYSIS ─────────────────────────────────────
    def analyze_mri(self, image_path: str) -> Dict:
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
                            "Look at this brain MRI image and describe what you observe. "
                            "Focus on:\n"
                            "1. Overall brain structure and symmetry\n"
                            "2. Basal ganglia region appearance\n"
                            "3. Substantia nigra if visible\n"
                            "4. White matter changes\n"
                            "5. Any notable findings relevant to Parkinson's disease\n"
                            "Describe what you see objectively and clinically."
                        ),
                    },
                ],
            },
        ]
        # MRI does not trigger RAG — image task
        analysis = self._generate(messages, max_tokens=500, query=None)
        return {"image_path": image_path, "analysis": analysis, "module": "mri"}

    # ── MODULE 2.3 — REPORT SUMMARISATION ────────────────────────────
    def summarize_report(self, pdf_path: str) -> Dict:
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
                    "3. Biomarker levels\n"
                    "4. Current diagnosis and stage\n"
                    "5. Treatment recommendations\n\n"
                    f"Report:\n{cleaned}"
                ),
            },
        ]
        # Report summary does not need RAG — document already provided
        summary = self._generate(messages, max_tokens=500, query=None)

        biomarkers = [
            b for b in [
                "alpha-synuclein", "DJ-1", "LRRK2", "dopamine",
                "urate", "neurofilament", "parkin", "GBA",
            ]
            if b.lower() in cleaned.lower()
        ]
        return {
            "source":    pdf_path,
            "summary":   summary,
            "biomarkers":biomarkers,
            "module":    "report",
        }

    # ── MODULE 2.4 — GENE MUTATION ANALYSIS ──────────────────────────
    def analyze_genes(self, report_text: str) -> Dict:
        query = f"gene mutation Parkinson risk {report_text[:100]}"
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
                    "4. Clinical significance\n"
                    "5. Recommended actions\n\n"
                    f"Genetic Report:\n{report_text}"
                ),
            },
        ]
        # Gene query triggers RAG over pd_papers + pd_gene_db + pd_module0
        interpretation = self._generate(messages, max_tokens=600, query=query)
        return {
            "report_text":    report_text,
            "interpretation": interpretation,
            "module":         "gene",
        }

    # ── MODULE 2.5 — DRUG RECOMMENDATION ─────────────────────────────
    def recommend_drugs(
        self,
        mutations: List[str],
        stage:     str,
        symptoms:  List[str],
    ) -> Dict:
        profile = (
            f"Gene mutations: {', '.join(mutations) if mutations else 'None reported'}\n"
            f"Disease stage: {stage}\n"
            f"Symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}"
        )
        query = f"Parkinson drug treatment {stage} stage {' '.join(mutations)}"
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Recommend Parkinson's disease medications for this patient. "
                    "For each recommended drug provide:\n"
                    "1. Drug name and class\n"
                    "2. Mechanism of action\n"
                    "3. Why it suits this patient's profile\n"
                    "4. Dosage considerations\n"
                    "5. Key interactions and side effects\n\n"
                    f"Patient Profile:\n{profile}"
                ),
            },
        ]
        # Drug query triggers RAG over pd_papers + pd_guidelines + pd_gene_db
        recommendations = self._generate(messages, max_tokens=600, query=query)
        return {
            "patient_profile": profile,
            "recommendations": recommendations,
            "module":          "drug",
        }

    # ── RAG STATUS ────────────────────────────────────────────────────
    def rag_status(self) -> Dict:
        """Return current RAG and memory bridge status."""
        stats = {}
        if self._rag_retriever:
            stats["collections"] = self._rag_retriever.collection_stats()
        stats["module0_context"] = bool(
            self._m0_bridge and self._m0_bridge.has_context()
        )
        stats["rag_enabled"] = self.rag_enabled
        return stats

    def ingest_documents(self, force: bool = False) -> Dict:
        """
        Trigger full RAG ingestion pipeline.
        Call this once after adding documents to data/papers/ etc.
        """
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "pd_rag"))
            from ingest.ingestor import DocumentIngestor
            ingestor = DocumentIngestor()
            counts   = ingestor.ingest_all()
            logger.info(f"Ingestion complete: {counts}")
            return {"status": "ok", "counts": counts}
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return {"status": "error", "error": str(e)}
