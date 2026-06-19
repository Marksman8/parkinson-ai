"""
RAG Retriever
Searches ChromaDB collections and formats retrieved
chunks into a clean context string for Qwen2-VL.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PDRetriever:
    """
    Retrieves relevant chunks from ChromaDB collections
    and formats them for injection into Qwen2-VL prompts.
    """

    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(embedder_model)
        client        = chromadb.PersistentClient(path=vector_db_path)

        # Load collections — handle missing gracefully
        self.collections = {}
        for name in ["pd_papers", "pd_guidelines", "pd_gene_db", "pd_module0"]:
            try:
                self.collections[name] = client.get_or_create_collection(
                    name, metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning(f"Could not load collection {name}: {e}")

    def retrieve(
        self,
        query:       str,
        collections: List[str],
        top_k:       int = 5,
    ) -> List[Dict]:
        """
        Search specified collections and return ranked results.
        Returns list of {text, source, collection, score}.
        """
        if not query.strip() or not collections:
            return []

        query_embedding = self.embedder.encode(query).tolist()
        all_results     = []

        for col_name in collections:
            col = self.collections.get(col_name)
            if not col or col.count() == 0:
                continue
            try:
                k = min(top_k, col.count())
                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                )
                docs      = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for doc, meta, dist in zip(docs, metadatas, distances):
                    # cosine similarity = 1 - distance
                    score = round(1 - dist, 4)
                    if score > 0.25:    # relevance threshold
                        all_results.append({
                            "text":       doc,
                            "source":     meta.get("source", col_name),
                            "collection": col_name,
                            "score":      score,
                            "type":       meta.get("type", ""),
                        })
            except Exception as e:
                logger.warning(f"Retrieval failed for {col_name}: {e}")

        # Sort by score, deduplicate, take top_k overall
        all_results.sort(key=lambda x: x["score"], reverse=True)
        seen  = set()
        final = []
        for r in all_results:
            key = r["text"][:80]
            if key not in seen:
                seen.add(key)
                final.append(r)
            if len(final) >= top_k:
                break

        logger.info(
            f"Retrieved {len(final)} chunks from {collections} "
            f"for query: '{query[:60]}...'"
        )
        return final

    def format_context(
        self,
        results:    List[Dict],
        query_type: str = "",
    ) -> str:
        """
        Format retrieved chunks into a clean context string
        for Qwen2-VL system prompt injection.
        """
        if not results:
            return ""

        lines = [
            f"\n=== RETRIEVED LITERATURE CONTEXT ({query_type}) ===",
            "The following information was retrieved from the PD knowledge base:",
            "",
        ]

        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] Source: {r['source']}  "
                f"(relevance: {r['score']:.2f})"
            )
            lines.append(r["text"])
            lines.append("")

        lines.append(
            "Use the above context to provide an accurate, evidence-based answer. "
            "Cite sources when appropriate."
        )
        lines.append("=== END RETRIEVED CONTEXT ===\n")
        return "\n".join(lines)

    def retrieve_and_format(
        self,
        query:       str,
        collections: List[str],
        top_k:       int = 5,
        query_type:  str = "",
    ) -> Tuple[str, List[Dict]]:
        """
        Convenience method: retrieve + format in one call.
        Returns (formatted_context_string, raw_results_list).
        """
        results = self.retrieve(query, collections, top_k)
        context = self.format_context(results, query_type)
        return context, results

    def collection_stats(self) -> Dict[str, int]:
        return {
            name: col.count()
            for name, col in self.collections.items()
        }
