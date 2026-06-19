"""
RAG Relevance Checker
Determines if doctor's input is Parkinson's disease related.
Uses sentence-transformers + ChromaDB for semantic similarity.
Falls back to keyword matching if model not available.
"""
import logging
import re
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# ── PD keyword fallback set ───────────────────────────────────────────
PD_KEYWORDS = [
    "parkinson", "parkinsons", "parkinson's",
    "tremor", "rigidity", "bradykinesia", "dopamine", "dopaminergic",
    "substantia nigra", "lewy body", "alpha synuclein", "snca",
    "lrrk2", "pink1", "parkin", "prkn", "dj1", "dj-1", "park7",
    "nigrostriatal", "basal ganglia", "motor symptoms", "dyskinesia",
    "levodopa", "carbidopa", "dopamine deficiency", "neurodegeneration",
    "movement disorder", "resting tremor", "pill rolling", "festination",
    "micrographia", "hypomimia", "anosmia", "rem sleep", "constipation",
    "deep brain stimulation", "dbs", "mao-b", "comt inhibitor",
    "ropinirole", "pramipexole", "rasagiline", "selegiline",
    "ubiquitin", "proteasome", "mitophagy", "autophagy",
    "oxidative stress", "mitochondrial dysfunction",
]

# ── PD knowledge base sentences for ChromaDB ─────────────────────────
PD_KNOWLEDGE = [
    "Parkinson's disease is a neurodegenerative disorder affecting dopaminergic neurons.",
    "SNCA gene encodes alpha-synuclein which forms Lewy bodies in Parkinson's disease.",
    "LRRK2 G2019S mutation is the most common genetic cause of Parkinson's disease.",
    "PINK1 and PARKIN cooperate in mitophagy to eliminate damaged mitochondria.",
    "DJ-1 protein acts as an oxidative stress sensor in Parkinson's disease neurons.",
    "Tremor, rigidity, and bradykinesia are the cardinal motor symptoms of Parkinson's.",
    "Levodopa is the gold standard treatment for Parkinson's disease symptoms.",
    "Substantia nigra degeneration causes dopamine deficiency in Parkinson's disease.",
    "Degree centrality analysis identifies hub genes in Parkinson's PPI networks.",
    "KEGG pathway hsa05012 contains the complete Parkinson's disease pathway.",
    "STRING database provides protein-protein interaction data for Parkinson's genes.",
    "Basal ganglia dysfunction underlies the motor symptoms of Parkinson's disease.",
    "Alpha-synuclein aggregation is the hallmark pathology of Parkinson's disease.",
    "Ubiquitin-proteasome system impairment contributes to Parkinson's disease.",
    "Deep brain stimulation targets the subthalamic nucleus in Parkinson's disease.",
]


class PDRelevanceChecker:
    """
    Two-layer relevance detection:
    Layer 1: Fast keyword matching
    Layer 2: Semantic similarity via sentence-transformers + ChromaDB
    """

    def __init__(self, db_path: str = "data/cache/relevance_db"):
        self.db_path = db_path
        self._chroma = None
        self._embedder = None
        self._semantic_ready = False
        self._init_semantic()

    def _init_semantic(self):
        """Try to initialise semantic layer — fail gracefully."""
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            self._embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            client = chromadb.PersistentClient(path=self.db_path)
            col = client.get_or_create_collection(
                name="pd_relevance",
                metadata={"hnsw:space": "cosine"},
            )
            # Seed if empty
            if col.count() == 0:
                embeddings = self._embedder.encode(PD_KNOWLEDGE).tolist()
                col.upsert(
                    ids=[f"pd_{i}" for i in range(len(PD_KNOWLEDGE))],
                    embeddings=embeddings,
                    documents=PD_KNOWLEDGE,
                )
            self._collection = col
            self._semantic_ready = True
            logger.info("Semantic relevance layer ready.")
        except Exception as e:
            logger.warning(f"Semantic layer unavailable ({e}). Using keywords only.")
            self._semantic_ready = False

    def _keyword_check(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        hits = sum(1 for kw in PD_KEYWORDS if kw in text_lower)
        score = min(hits / 3.0, 1.0)   # 3+ hits = full confidence
        return hits > 0, score

    def _semantic_check(self, text: str) -> Tuple[bool, float]:
        if not self._semantic_ready:
            return False, 0.0
        embedding = self._embedder.encode(text).tolist()
        results = self._collection.query(
            query_embeddings=[embedding], n_results=3
        )
        distances = results["distances"][0]        # cosine distances
        # cosine similarity = 1 - distance
        similarities = [1 - d for d in distances]
        best_sim = max(similarities) if similarities else 0.0
        return best_sim > 0.55, best_sim

    def check(self, text: str) -> Tuple[bool, float, str]:
        """
        Returns (is_pd_related, confidence, method_used)
        confidence: 0.0 – 1.0
        """
        kw_hit, kw_score = self._keyword_check(text)
        sem_hit, sem_score = self._semantic_check(text)

        # Combine: keyword takes priority for direct mentions
        if kw_hit and sem_hit:
            confidence = min(0.5 * kw_score + 0.5 * sem_score + 0.1, 1.0)
            method = "keyword+semantic"
        elif kw_hit:
            confidence = kw_score
            method = "keyword"
        elif sem_hit:
            confidence = sem_score
            method = "semantic"
        else:
            confidence = max(kw_score, sem_score)
            method = "keyword+semantic"

        is_pd = confidence > 0.25
        logger.info(f"Relevance check: {is_pd} (conf={confidence:.2f}, method={method})")
        return is_pd, confidence, method
