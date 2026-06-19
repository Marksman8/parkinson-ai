"""
Module 2 Memory Integration Patch
Add this to your existing pd_multimodal/core/multimodal_engine.py

This patches PDMultiModalEngine to automatically read
Module 0's SQLite memory and inject biomarker context
into every Qwen2-VL query.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Shared DB path — must match Module 0 ─────────────────────────────
# Update this path to wherever Module 0 saves its database
MODULE0_DB_PATH = "../module0_biomarker/data/pd_memory.db"


def get_module0_context(db_path: str = MODULE0_DB_PATH) -> str:
    """
    Read latest biomarker session from Module 0's SQLite DB.
    Returns empty string if no session exists or DB not found.
    """
    if not Path(db_path).exists():
        return ""
    try:
        import sys
        sys.path.insert(0, str(Path(db_path).parent.parent))
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Get latest session
        session = conn.execute("""
            SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        if not session:
            conn.close()
            return ""

        sid = session["session_id"]

        # Get top genes
        genes = conn.execute("""
            SELECT * FROM top_genes WHERE session_id = ? ORDER BY rank ASC
        """, (sid,)).fetchall()

        # Get ODE params
        params = conn.execute("""
            SELECT * FROM ode_parameters WHERE session_id = ? LIMIT 15
        """, (sid,)).fetchall()

        conn.close()

        # Format context
        lines = [
            "\n=== PATIENT BIOMARKER CONTEXT (from Module 0 Analysis) ===",
            f"Analysis date: {session['created_at'][:10]}",
            f"Clinical input summary: {session['doctor_input'][:150]}...",
            "",
            "Identified Hub Genes (Parkinson's PPI Network):",
        ]
        for g in genes:
            lines.append(
                f"  {g['rank']}. {g['gene']}  "
                f"[Degree Centrality={g['degree_c']:.3f}, "
                f"Hub Score={g['hub_score']:.3f}]"
                + (f"  — {g['explanation'][:80]}" if g['explanation'] else "")
            )

        if params:
            lines.append("")
            lines.append("Calibrated ODE Parameters:")
            current_gene = ""
            for p in params:
                if p["gene"] != current_gene:
                    current_gene = p["gene"]
                    lines.append(f"  {current_gene}:")
                lines.append(
                    f"    {p['param_name']} = {p['param_value']} {p['param_unit']}"
                )

        lines.append("=== END BIOMARKER CONTEXT ===\n")
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Could not read Module 0 memory: {e}")
        return ""


# ── Monkey-patch for existing PDMultiModalEngine ──────────────────────
# Add this to your existing multimodal_engine.py __init__:
#
#   from integration.module0_bridge import get_module0_context
#   self._module0_context_fn = get_module0_context
#
# Then modify _generate() to inject context:
#
#   def _generate(self, messages, max_tokens=512):
#       # Inject Module 0 biomarker context into system message
#       m0_context = self._module0_context_fn()
#       if m0_context and messages[0]["role"] == "system":
#           messages[0]["content"] = messages[0]["content"] + m0_context
#       # ... rest of existing _generate code


# ── Standalone context getter for Module 2 API ───────────────────────
class Module0Bridge:
    """
    Used by Module 2's FastAPI to enrich responses with
    Module 0 biomarker context.
    """
    def __init__(self, db_path: str = MODULE0_DB_PATH):
        self.db_path = db_path

    def get_context(self) -> str:
        return get_module0_context(self.db_path)

    def has_context(self) -> bool:
        return bool(get_module0_context(self.db_path))

    def get_latest_genes(self):
        if not Path(self.db_path).exists():
            return []
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            session = conn.execute("""
                SELECT session_id FROM sessions
                ORDER BY created_at DESC LIMIT 1
            """).fetchone()
            if not session:
                return []
            genes = conn.execute("""
                SELECT gene, rank, degree_c, hub_score
                FROM top_genes WHERE session_id = ?
                ORDER BY rank
            """, (session["session_id"],)).fetchall()
            conn.close()
            return [dict(g) for g in genes]
        except Exception:
            return []
