"""
KEGG + STRING Network Pipeline
Fetches Parkinson's pathway and builds PPI network.
Online first → fallback to local cached data.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ── KEGG PD pathway ID ────────────────────────────────────────────────
KEGG_PD_PATHWAY  = "hsa05012"
KEGG_API_BASE    = "https://rest.kegg.jp"
STRING_API_BASE  = "https://string-db.org/api"
STRING_SPECIES   = "9606"   # Homo sapiens
STRING_THRESHOLD = 700      # high confidence (0–1000)

# ── Local fallback gene list (core PD pathway genes) ─────────────────
LOCAL_PD_GENES = [
    "SNCA", "LRRK2", "PINK1", "PRKN", "PARK7",
    "UCHL1", "MAPT", "GBA", "ATP13A2", "FBXO7",
    "VPS35", "EIF4G1", "DNAJC13", "CHCHD2", "TMEM230",
    "TH", "SLC6A3", "DRD2", "COMT", "MAOB",
    "CASP3", "CASP9", "CYCS", "APAF1", "BCL2",
    "SOD1", "SOD2", "CAT", "GPX1", "NFE2L2",
]

# ── Local fallback PPI edges (literature-confirmed interactions) ──────
LOCAL_PPI_EDGES = [
    ("SNCA", "LRRK2"), ("SNCA", "PINK1"), ("SNCA", "PRKN"),
    ("SNCA", "PARK7"), ("SNCA", "TH"),    ("SNCA", "UCHL1"),
    ("SNCA", "GBA"),   ("SNCA", "MAOB"),  ("SNCA", "SOD1"),
    ("SNCA", "BCL2"),  ("LRRK2", "PINK1"),("LRRK2", "PRKN"),
    ("LRRK2", "VPS35"),("LRRK2", "TH"),   ("PINK1", "PRKN"),
    ("PINK1", "PARK7"),("PRKN", "PARK7"), ("PRKN", "UCHL1"),
    ("PRKN", "CASP3"), ("PARK7", "PINK1"),("PARK7", "SOD2"),
    ("TH", "SLC6A3"),  ("TH", "DRD2"),    ("COMT", "TH"),
    ("MAOB", "TH"),    ("GBA", "SNCA"),   ("ATP13A2", "SNCA"),
    ("CASP3", "CASP9"),("CASP9", "CYCS"), ("CYCS", "APAF1"),
    ("BCL2", "CYCS"),  ("SOD2", "CAT"),   ("NFE2L2", "SOD2"),
]


class KEGGStringPipeline:
    """
    Fetches KEGG PD pathway genes and builds STRING PPI network.
    Online → local fallback strategy.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir  = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gene_cache_path = self.cache_dir / "kegg_pd_genes.json"
        self._ppi_cache_path  = self.cache_dir / "string_ppi.json"

    # ── KEGG ──────────────────────────────────────────────────────────
    def fetch_kegg_genes(self) -> Tuple[List[str], str]:
        """Returns (gene_list, source) where source is 'online'|'cache'|'local'."""
        # 1. Try cached
        if self._gene_cache_path.exists():
            try:
                data = json.loads(self._gene_cache_path.read_text())
                logger.info(f"KEGG genes loaded from cache ({len(data['genes'])} genes)")
                return data["genes"], "cache"
            except Exception:
                pass

        # 2. Try online
        try:
            import requests
            url = f"{KEGG_API_BASE}/link/hsa/{KEGG_PD_PATHWAY}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            genes = []
            for line in resp.text.strip().split("\n"):
                if "\t" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        gene_id = parts[1].replace("hsa:", "").strip()
                        genes.append(gene_id)
            # Convert NCBI gene IDs to symbols
            symbols = self._resolve_gene_symbols(genes[:50])
            # Cache result
            self._gene_cache_path.write_text(
                json.dumps({"genes": symbols, "pathway": KEGG_PD_PATHWAY})
            )
            logger.info(f"KEGG online: {len(symbols)} genes from {KEGG_PD_PATHWAY}")
            return symbols, "online"
        except Exception as e:
            logger.warning(f"KEGG online failed: {e}. Using local fallback.")

        # 3. Local fallback
        return LOCAL_PD_GENES, "local"

    def _resolve_gene_symbols(self, gene_ids: List[str]) -> List[str]:
        """Convert NCBI IDs to gene symbols via KEGG API."""
        symbols = []
        try:
            import requests
            for gid in gene_ids[:40]:
                url = f"{KEGG_API_BASE}/get/hsa:{gid}"
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    for line in r.text.split("\n"):
                        if line.startswith("SYMBOL"):
                            sym = line.split()[1].strip().rstrip(",")
                            symbols.append(sym)
                            break
                time.sleep(0.1)   # KEGG rate limit
        except Exception as e:
            logger.warning(f"Symbol resolution failed: {e}")
            symbols = gene_ids[:30]
        return symbols if symbols else LOCAL_PD_GENES

    # ── STRING ────────────────────────────────────────────────────────
    def build_ppi_network(
        self, genes: List[str]
    ) -> Tuple[nx.Graph, str]:
        """Returns (networkx_graph, source)."""
        # 1. Try cache
        if self._ppi_cache_path.exists():
            try:
                data = json.loads(self._ppi_cache_path.read_text())
                G = nx.Graph()
                G.add_nodes_from(data["nodes"])
                G.add_edges_from(
                    [(e[0], e[1], {"weight": e[2]}) for e in data["edges"]]
                )
                logger.info(f"PPI loaded from cache: {G.number_of_nodes()} nodes, "
                            f"{G.number_of_edges()} edges")
                return G, "cache"
            except Exception:
                pass

        # 2. Try STRING online
        try:
            import requests
            gene_str = "%0d".join(genes[:30])
            url = (
                f"{STRING_API_BASE}/json/network?"
                f"identifiers={gene_str}"
                f"&species={STRING_SPECIES}"
                f"&required_score={STRING_THRESHOLD}"
                f"&caller_identity=pd_biomarker_tool"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            interactions = resp.json()

            G = nx.Graph()
            edges_data = []
            for item in interactions:
                g1 = item.get("preferredName_A", "")
                g2 = item.get("preferredName_B", "")
                score = item.get("score", 0) / 1000.0
                if g1 and g2 and g1 != g2:
                    G.add_edge(g1, g2, weight=score)
                    edges_data.append([g1, g2, score])

            # Cache
            self._ppi_cache_path.write_text(json.dumps({
                "nodes": list(G.nodes()),
                "edges": edges_data,
            }))
            logger.info(f"STRING online: {G.number_of_nodes()} nodes, "
                        f"{G.number_of_edges()} edges")
            return G, "online"

        except Exception as e:
            logger.warning(f"STRING online failed: {e}. Using local fallback.")

        # 3. Local fallback
        G = nx.Graph()
        for gene in LOCAL_PD_GENES:
            G.add_node(gene)
        for g1, g2 in LOCAL_PPI_EDGES:
            G.add_edge(g1, g2, weight=0.9)
        logger.info(f"Local PPI fallback: {G.number_of_nodes()} nodes")
        return G, "local"

    # ── Centrality ────────────────────────────────────────────────────
    def compute_centrality(
        self,
        G: nx.Graph,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Compute degree + betweenness centrality.
        Returns sorted list of top_n hub genes.
        """
        if G.number_of_nodes() == 0:
            return []

        degree_c      = nx.degree_centrality(G)
        betweenness_c = nx.betweenness_centrality(G, normalized=True)
        closeness_c   = nx.closeness_centrality(G)

        results = []
        for node in G.nodes():
            results.append({
                "gene":         node,
                "degree":       G.degree(node),
                "degree_c":     round(degree_c.get(node, 0), 4),
                "betweenness_c":round(betweenness_c.get(node, 0), 4),
                "closeness_c":  round(closeness_c.get(node, 0), 4),
                # composite score
                "hub_score":    round(
                    0.5 * degree_c.get(node, 0) +
                    0.3 * betweenness_c.get(node, 0) +
                    0.2 * closeness_c.get(node, 0), 4
                ),
            })

        results.sort(key=lambda x: x["hub_score"], reverse=True)
        return results[:top_n]
