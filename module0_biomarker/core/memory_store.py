"""
Shared SQLite Memory Store
Module 0 writes biomarker discovery results here.
Module 2 (Qwen2-VL) reads from here to enrich its responses
with patient-specific biomarker context.
"""
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared DB path — both modules point to this
DEFAULT_DB_PATH = "data/pd_memory.db"


class PDMemoryStore:
    """
    Persistent SQLite store shared between Module 0 and Module 2.
    Stores biomarker sessions, top genes, and ODE parameters.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT UNIQUE NOT NULL,
                    created_at  TEXT NOT NULL,
                    doctor_input TEXT,
                    confidence  REAL,
                    kegg_source TEXT,
                    ppi_source  TEXT
                );

                CREATE TABLE IF NOT EXISTS top_genes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    gene        TEXT NOT NULL,
                    rank        INTEGER,
                    degree      INTEGER,
                    degree_c    REAL,
                    betweenness_c REAL,
                    closeness_c REAL,
                    hub_score   REAL,
                    explanation TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS ode_parameters (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    gene        TEXT NOT NULL,
                    param_name  TEXT NOT NULL,
                    param_value REAL,
                    param_unit  TEXT,
                    source_ref  TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    report_path TEXT,
                    graph_path  TEXT,
                    created_at  TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
            """)
        logger.info(f"Memory store initialised at {self.db_path}")

    # ── Write ─────────────────────────────────────────────────────────
    def save_session(
        self,
        session_id: str,
        doctor_input: str,
        confidence: float,
        kegg_source: str,
        ppi_source: str,
    ):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, doctor_input, confidence, kegg_source, ppi_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(),
                doctor_input,
                confidence,
                kegg_source,
                ppi_source,
            ))

    def save_top_genes(self, session_id: str, top_genes: List[Dict]):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM top_genes WHERE session_id = ?", (session_id,)
            )
            for rank, g in enumerate(top_genes, 1):
                conn.execute("""
                    INSERT INTO top_genes
                    (session_id, gene, rank, degree, degree_c,
                     betweenness_c, closeness_c, hub_score, explanation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    g["gene"],
                    rank,
                    g.get("degree", 0),
                    g.get("degree_c", 0),
                    g.get("betweenness_c", 0),
                    g.get("closeness_c", 0),
                    g.get("hub_score", 0),
                    g.get("explanation", ""),
                ))

    def save_ode_parameters(self, session_id: str, ode_params: List[Dict]):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM ode_parameters WHERE session_id = ?", (session_id,)
            )
            for p in ode_params:
                conn.execute("""
                    INSERT INTO ode_parameters
                    (session_id, gene, param_name, param_value, param_unit, source_ref)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    p.get("gene", ""),
                    p.get("name", ""),
                    p.get("value", 0.0),
                    p.get("unit", ""),
                    p.get("source", ""),
                ))

    def save_report_paths(
        self,
        session_id: str,
        report_path: str,
        graph_path: str,
    ):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO reports
                (session_id, report_path, graph_path, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                session_id, report_path, graph_path,
                datetime.now().isoformat(),
            ))

    # ── Read ──────────────────────────────────────────────────────────
    def get_latest_session(self) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1
            """).fetchone()
            return dict(row) if row else None

    def get_top_genes(self, session_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM top_genes
                WHERE session_id = ?
                ORDER BY rank ASC
            """, (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_ode_parameters(self, session_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM ode_parameters
                WHERE session_id = ?
            """, (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_context_for_module2(self) -> str:
        """
        Returns a formatted string that Module 2 (Qwen2-VL)
        can prepend to its system prompt to be aware of
        the latest biomarker discovery session.
        """
        session = self.get_latest_session()
        if not session:
            return ""

        genes = self.get_top_genes(session["session_id"])
        params = self.get_ode_parameters(session["session_id"])

        lines = [
            "=== BIOMARKER DISCOVERY CONTEXT (Module 0) ===",
            f"Session: {session['session_id']}",
            f"Date: {session['created_at'][:10]}",
            f"Doctor input: {session['doctor_input'][:200]}",
            "",
            "Top Hub Genes Identified:",
        ]
        for g in genes:
            lines.append(
                f"  {g['rank']}. {g['gene']}  "
                f"[degree_C={g['degree_c']:.3f}  hub={g['hub_score']:.3f}]"
                + (f"  — {g['explanation']}" if g.get("explanation") else "")
            )

        if params:
            lines.append("")
            lines.append("ODE Parameters (from Module 1 calibration):")
            for p in params[:10]:
                lines.append(
                    f"  {p['gene']} · {p['param_name']} = "
                    f"{p['param_value']} {p['param_unit']}"
                )

        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)

    def list_sessions(self, limit: int = 10) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT session_id, created_at, doctor_input
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]
