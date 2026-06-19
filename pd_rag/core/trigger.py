"""
Smart RAG Trigger Classifier
Decides whether a query needs literature retrieval or not.
Uses keyword scoring + query type detection.
No extra model needed — fast rule-based classification.
"""
from enum import Enum
from typing import Tuple


class QueryType(Enum):
    CHAT_CLINICAL   = "chat_clinical"    # needs RAG — clinical question
    GENE_QUERY      = "gene_query"       # needs RAG — gene/mutation question
    DRUG_QUERY      = "drug_query"       # needs RAG — treatment question
    GUIDELINE_QUERY = "guideline_query"  # needs RAG — protocol question
    MRI_ANALYSIS    = "mri_analysis"     # no RAG — image task
    REPORT_SUMMARY  = "report_summary"   # no RAG — document already provided
    ODE_SIMULATION  = "ode_simulation"   # no RAG — math task
    GENERAL_CHAT    = "general_chat"     # no RAG — greeting/simple


# ── Keyword maps ──────────────────────────────────────────────────────
_GENE_KW = [
    "gene", "mutation", "variant", "lrrk2", "snca", "pink1",
    "parkin", "prkn", "dj-1", "park7", "gba", "vcf", "allele",
    "pathogenic", "benign", "clinvar", "genotype", "polymorphism",
    "alpha-synuclein", "aggregation", "lewy", "ubiquitin",
]
_DRUG_KW = [
    "drug", "medication", "treatment", "therapy", "levodopa",
    "carbidopa", "pramipexole", "ropinirole", "rasagiline",
    "selegiline", "amantadine", "entacapone", "dose", "dosage",
    "side effect", "interaction", "prescribe", "pharmacology",
    "pharmacokinetic", "plasma concentration", "half-life",
]
_GUIDELINE_KW = [
    "guideline", "protocol", "recommendation", "standard of care",
    "clinical trial", "evidence", "meta-analysis", "systematic review",
    "diagnosis criteria", "staging", "hoehn yahr", "updrs",
    "mds", "nice guideline", "management", "best practice",
]
_CLINICAL_KW = [
    "symptom", "tremor", "rigidity", "bradykinesia", "dyskinesia",
    "progression", "stage", "prognosis", "cause", "mechanism",
    "pathophysiology", "substantia nigra", "dopamine", "basal ganglia",
    "neurodegeneration", "explain", "what is", "how does", "why",
    "research", "study", "paper", "found", "evidence", "literature",
]
_MRI_KW    = ["mri", "scan", "image", "neuroimaging", "brain scan", "t1", "t2"]
_REPORT_KW = ["report", "summarise", "summarize", "pdf", "lab result", "clinical note"]
_ODE_KW    = ["simulate", "ode", "differential", "kinetics", "hodgkin", "model"]
_GREET_KW  = ["hello", "hi", "hey", "thanks", "thank you", "ok", "okay", "good"]


class SmartRAGTrigger:
    """
    Classifies a query and decides:
    - Should RAG be triggered?
    - Which collections to search?
    - How many chunks to retrieve?
    """

    def classify(self, query: str) -> Tuple[QueryType, bool, list, int]:
        """
        Returns:
          query_type    : QueryType enum
          needs_rag     : bool
          collections   : list of ChromaDB collection names to search
          top_k         : number of chunks to retrieve
        """
        q = query.lower().strip()

        # ── No RAG cases (fast path) ─────────────────────────────────
        if any(k in q for k in _MRI_KW):
            return QueryType.MRI_ANALYSIS, False, [], 0

        if any(k in q for k in _REPORT_KW) and len(q) < 80:
            return QueryType.REPORT_SUMMARY, False, [], 0

        if any(k in q for k in _ODE_KW):
            return QueryType.ODE_SIMULATION, False, [], 0

        if any(k in q for k in _GREET_KW) and len(q) < 40:
            return QueryType.GENERAL_CHAT, False, [], 0

        # ── Score each RAG-needed category ───────────────────────────
        scores = {
            "gene":      sum(1 for k in _GENE_KW      if k in q),
            "drug":      sum(1 for k in _DRUG_KW      if k in q),
            "guideline": sum(1 for k in _GUIDELINE_KW if k in q),
            "clinical":  sum(1 for k in _CLINICAL_KW  if k in q),
        }

        total = sum(scores.values())
        if total == 0:
            return QueryType.GENERAL_CHAT, False, [], 0

        # ── Pick query type and collections ──────────────────────────
        best = max(scores, key=scores.get)

        if best == "gene" or scores["gene"] >= 2:
            return (
                QueryType.GENE_QUERY, True,
                ["pd_papers", "pd_gene_db", "pd_module0"], 6,
            )
        elif best == "drug" or scores["drug"] >= 2:
            return (
                QueryType.DRUG_QUERY, True,
                ["pd_papers", "pd_guidelines", "pd_gene_db"], 5,
            )
        elif best == "guideline" or scores["guideline"] >= 1:
            return (
                QueryType.GUIDELINE_QUERY, True,
                ["pd_guidelines", "pd_papers"], 5,
            )
        else:
            return (
                QueryType.CHAT_CLINICAL, True,
                ["pd_papers", "pd_guidelines", "pd_module0"], 5,
            )

    def describe(self, query_type: QueryType, needs_rag: bool, collections: list) -> str:
        """Human-readable description of routing decision."""
        if not needs_rag:
            return f"[{query_type.value}] → direct model inference (no RAG)"
        return (
            f"[{query_type.value}] → RAG search in: "
            f"{', '.join(collections)}"
        )
