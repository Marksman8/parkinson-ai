"""
MODULE 7 — AI Agent Router
Classifies input and dispatches to the correct handler.
With the multimodal engine, most tasks go to one model.
ODE simulation is the only separate component.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Input type labels
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".nii"}
PDF_EXTS   = {".pdf"}
GENE_EXTS  = {".vcf", ".txt"}

GENE_KEYWORDS = ["mutation", "gene", "lrrk2", "snca", "gba", "pink1",
                 "prkn", "variant", "vcf", "genomic", "allele"]
DRUG_KEYWORDS = ["drug", "medication", "treatment", "prescribe",
                 "levodopa", "pramipexole", "therapy", "recommend"]
SIM_KEYWORDS  = ["simulate", "ode", "dopamine level", "neuron",
                 "hodgkin", "huxley", "kinetics", "differential"]
MRI_KEYWORDS  = ["mri", "scan", "brain image", "neuroimaging"]
REPORT_KEYWORDS = ["report", "pdf", "lab result", "clinical note", "diagnosis"]


class AgentRouter:
    """
    Routes user input to:
      - PDMultiModalEngine  (chat, MRI, report, gene, drug)
      - ODE Simulator       (neuron/dopamine simulations)
    """

    def __init__(self, engine: Any, ode_simulator: Any = None):
        self.engine = engine
        self.ode    = ode_simulator

    def _detect_intent(
        self,
        text: Optional[str],
        file_path: Optional[str],
    ) -> str:
        # File extension takes highest priority
        if file_path:
            suffix = Path(file_path).suffix.lower()
            # Handle .nii.gz
            if file_path.endswith(".nii.gz"):
                return "mri"
            if suffix in IMAGE_EXTS:
                return "mri"
            if suffix in PDF_EXTS:
                return "report"
            if suffix in GENE_EXTS:
                return "gene"

        if not text:
            return "chat"

        text_lower = text.lower()

        # Score each module by keyword hits
        scores = {
            "gene":   sum(1 for k in GENE_KEYWORDS   if k in text_lower),
            "drug":   sum(1 for k in DRUG_KEYWORDS   if k in text_lower),
            "sim":    sum(1 for k in SIM_KEYWORDS    if k in text_lower),
            "mri":    sum(1 for k in MRI_KEYWORDS    if k in text_lower),
            "report": sum(1 for k in REPORT_KEYWORDS if k in text_lower),
        }

        best       = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score == 0:
            return "chat"
        return best

    def route(
        self,
        text:        Optional[str]  = None,
        file_path:   Optional[str]  = None,
        extra_params: Dict          = None,
    ) -> Dict:
        """
        Route input and return structured result.

        extra_params for ODE simulation:
          {
            "model":    "hh" | "dopamine",
            "pd_loss":  0.0–1.0,
            "I_ext":    float   (HH model only),
          }
        """
        extra   = extra_params or {}
        intent  = self._detect_intent(text, file_path)
        logger.info(f"AgentRouter detected intent: {intent}")

        try:
            if intent == "mri":
                if not file_path:
                    return {"error": "MRI analysis requires an image file.", "module": "mri"}
                result = self.engine.analyze_mri(file_path)

            elif intent == "report":
                if not file_path:
                    return {"error": "Report summarization requires a PDF file.", "module": "report"}
                result = self.engine.summarize_report(file_path)

            elif intent == "gene":
                content = text or (open(file_path).read() if file_path else "")
                if not content:
                    return {"error": "Gene analysis requires report text or file.", "module": "gene"}
                result = self.engine.analyze_genes(content)

            elif intent == "drug":
                mutations = extra.get("mutations", [])
                stage     = extra.get("stage",     "moderate")
                symptoms  = extra.get("symptoms",  [])
                result    = self.engine.recommend_drugs(mutations, stage, symptoms)

            elif intent == "sim":
                if not self.ode:
                    return {"error": "ODE simulator not loaded.", "module": "sim"}
                model_type = extra.get("model", "hh")
                if model_type == "dopamine":
                    from modules.module6_ode.ode_simulator import DopamineKineticsModel
                    sim    = DopamineKineticsModel({"pd_loss": extra.get("pd_loss", 0.0)})
                else:
                    from modules.module6_ode.ode_simulator import HodgkinHuxleyModel
                    sim    = HodgkinHuxleyModel({"I_ext": extra.get("I_ext", 10.0)})
                result = sim.simulate(save_plot=True).to_dict()
                result["module"] = "simulation"

            else:
                # Default — general clinical chat
                result = self.engine.chat(text or "Tell me about Parkinson's disease.")

            return {"status": "ok", "intent": intent, "result": result}

        except Exception as e:
            logger.error(f"Router error [{intent}]: {e}", exc_info=True)
            return {"status": "error", "intent": intent, "error": str(e)}
