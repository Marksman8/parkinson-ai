"""
Module 0 — Automated Biomarker Discovery Engine
Main orchestrator. Doctor provides clinical text → system does everything.

Flow:
  1. RAG relevance check (is this PD-related?)
  2. KEGG pathway fetch (online → local fallback)
  3. STRING PPI network build (online → local fallback)
  4. Degree centrality analysis → top 5 hub genes
  5. Clinical explanations attached
  6. ODE parameters retrieved from literature
  7. PPI network visualised
  8. PDF report generated
  9. Results saved to shared SQLite memory (Module 2 reads this)
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BiomarkerDiscoveryEngine:
    """
    One method call does everything.
    Doctor just provides text.
    """

    def __init__(
        self,
        db_path:       str = "data/pd_memory.db",
        cache_dir:     str = "data/cache",
        outputs_dir:   str = "data/outputs",
        relevance_db:  str = "data/cache/relevance_db",
    ):
        from rag.relevance    import PDRelevanceChecker
        from network.kegg_string import KEGGStringPipeline
        from core.memory_store   import PDMemoryStore

        self.relevance   = PDRelevanceChecker(db_path=relevance_db)
        self.pipeline    = KEGGStringPipeline(cache_dir=cache_dir)
        self.memory      = PDMemoryStore(db_path=db_path)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def run(self, doctor_input: str) -> Dict:
        """
        Full automated pipeline from doctor text to biomarker report.
        Returns structured result dictionary.
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info(f"Starting biomarker discovery — session {session_id}")

        # ── STEP 1: Relevance check ───────────────────────────────────
        is_pd, confidence, method = self.relevance.check(doctor_input)
        if not is_pd:
            return {
                "status":    "not_pd_related",
                "message":   (
                    "The provided clinical text does not appear to be related to "
                    "Parkinson's disease. Please include symptoms, gene names, or "
                    "clinical findings related to PD."
                ),
                "confidence": confidence,
            }

        logger.info(f"PD relevance confirmed (conf={confidence:.2f}, method={method})")

        # ── STEP 2: Fetch KEGG genes ──────────────────────────────────
        genes, kegg_source = self.pipeline.fetch_kegg_genes()
        logger.info(f"Genes fetched: {len(genes)} from {kegg_source}")

        # ── STEP 3: Build STRING PPI ──────────────────────────────────
        G, ppi_source = self.pipeline.build_ppi_network(genes)
        logger.info(f"PPI: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from {ppi_source}")

        # ── STEP 4: Centrality → top genes ───────────────────────────
        top_genes = self.pipeline.compute_centrality(G, top_n=5)

        # ── STEP 5: Attach clinical explanations ─────────────────────
        from core.gene_info import get_explanation, get_ode_params, GENE_EXPLANATIONS
        for i, g in enumerate(top_genes):
            g["rank"]        = i + 1
            g["explanation"] = get_explanation(g["gene"])
            g["gene_info"]   = GENE_EXPLANATIONS.get(g["gene"], {})

        # ── STEP 6: ODE parameters ────────────────────────────────────
        top_gene_names = [g["gene"] for g in top_genes]
        ode_params = get_ode_params(top_gene_names)

        # ── STEP 7: Visualise network ─────────────────────────────────
        graph_path = str(self.outputs_dir / f"ppi_network_{session_id}.png")
        try:
            from network.visualizer import draw_ppi_network
            draw_ppi_network(G, top_genes, save_path=graph_path)
        except Exception as e:
            logger.warning(f"Visualisation failed: {e}")
            graph_path = None

        # ── STEP 8: Generate PDF report ───────────────────────────────
        report_path = str(self.outputs_dir / f"biomarker_report_{session_id}.pdf")
        try:
            from report.pdf_generator import generate_pdf_report
            generate_pdf_report(
                session_id=session_id,
                doctor_input=doctor_input,
                top_genes=top_genes,
                ode_params=ode_params,
                graph_path=graph_path,
                save_path=report_path,
            )
        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            report_path = None

        # ── STEP 9: Save to shared SQLite memory ─────────────────────
        self.memory.save_session(
            session_id=session_id,
            doctor_input=doctor_input,
            confidence=confidence,
            kegg_source=kegg_source,
            ppi_source=ppi_source,
        )
        self.memory.save_top_genes(session_id, top_genes)
        self.memory.save_ode_parameters(session_id, ode_params)
        if graph_path or report_path:
            self.memory.save_report_paths(
                session_id,
                report_path or "",
                graph_path or "",
            )

        logger.info(f"Session {session_id} complete.")

        return {
            "status":      "success",
            "session_id":  session_id,
            "confidence":  confidence,
            "kegg_source": kegg_source,
            "ppi_source":  ppi_source,
            "network_stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            },
            "top_genes":   top_genes,
            "ode_params":  ode_params,
            "graph_path":  graph_path,
            "report_path": report_path,
            "memory_saved": True,
            "module2_context": self.memory.get_context_for_module2(),
        }
