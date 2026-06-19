"""
PPI Network Visualizer
Generates a publication-quality PPI network graph
with hub genes highlighted.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ── Color scheme ─────────────────────────────────────────────────────
BG       = "#0d1117"
HUB_COLORS = {
    "SNCA":  "#4f8ef7",
    "LRRK2": "#f59e0b",
    "PINK1": "#a78bfa",
    "PRKN":  "#10b981",
    "PARKIN":"#10b981",
    "PARK7": "#ef4444",
    "DJ-1":  "#ef4444",
}
NODE_DEFAULT  = "#3a4a5a"
EDGE_COLOR    = "#2a3a4a"
HUB_EDGE      = "#ffffff"
TEXT_COLOR    = "#e2e8f0"


def draw_ppi_network(
    G: nx.Graph,
    top_genes: List[Dict],
    save_path: str = "data/outputs/ppi_network.png",
    title: str = "Parkinson's Disease PPI Network",
) -> str:
    """
    Draw the PPI network with hub genes highlighted.
    Returns the path to the saved image.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    top_gene_names = {g["gene"] for g in top_genes}

    fig, ax = plt.subplots(figsize=(13, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # Layout — spring layout gives good separation
    np.random.seed(42)
    pos = nx.spring_layout(G, k=2.2, iterations=80, seed=42)

    # ── Draw edges ────────────────────────────────────────────────────
    # Regular edges
    regular_edges = [
        (u, v) for u, v in G.edges()
        if u not in top_gene_names and v not in top_gene_names
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=regular_edges, ax=ax,
        edge_color=EDGE_COLOR, alpha=0.4, width=0.8,
    )

    # Hub edges — brighter
    hub_edges = [
        (u, v) for u, v in G.edges()
        if u in top_gene_names or v in top_gene_names
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=hub_edges, ax=ax,
        edge_color="#4a6a8a", alpha=0.7, width=1.4,
    )

    # ── Draw nodes ────────────────────────────────────────────────────
    # Regular nodes
    regular_nodes = [n for n in G.nodes() if n not in top_gene_names]
    nx.draw_networkx_nodes(
        G, pos, nodelist=regular_nodes, ax=ax,
        node_color=NODE_DEFAULT, node_size=280,
        alpha=0.75,
    )

    # Hub nodes — larger, colored, glowing
    for gene_info in top_genes:
        gene = gene_info["gene"]
        if gene not in G.nodes():
            continue
        color = HUB_COLORS.get(gene, "#ff6b6b")
        rank  = top_genes.index(gene_info)
        size  = max(1800 - rank * 200, 900)

        # Glow effect — draw multiple layers
        for alpha, scale in [(0.08, 3.5), (0.15, 2.5), (0.3, 1.8)]:
            nx.draw_networkx_nodes(
                G, pos, nodelist=[gene], ax=ax,
                node_color=color, node_size=int(size * scale),
                alpha=alpha,
            )
        nx.draw_networkx_nodes(
            G, pos, nodelist=[gene], ax=ax,
            node_color=color, node_size=size,
            alpha=1.0,
            edgecolors="white", linewidths=2.0,
        )

    # ── Labels ────────────────────────────────────────────────────────
    # Regular node labels (small)
    regular_labels = {n: n for n in regular_nodes if G.degree(n) >= 3}
    nx.draw_networkx_labels(
        G, pos, labels=regular_labels, ax=ax,
        font_size=5.5, font_color="#8a9aaa", font_family="monospace",
    )

    # Hub gene labels (larger, white)
    hub_labels = {g["gene"]: g["gene"] for g in top_genes if g["gene"] in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=hub_labels, ax=ax,
        font_size=9, font_color="white",
        font_weight="bold", font_family="monospace",
    )

    # ── Legend ────────────────────────────────────────────────────────
    legend_patches = []
    for i, gene_info in enumerate(top_genes):
        gene  = gene_info["gene"]
        color = HUB_COLORS.get(gene, "#ff6b6b")
        label = (
            f"{gene}  "
            f"[C_D={gene_info['degree_c']:.3f}  "
            f"hub={gene_info['hub_score']:.3f}]"
        )
        legend_patches.append(mpatches.Patch(color=color, label=label))

    legend = ax.legend(
        handles=legend_patches,
        loc="lower left",
        framealpha=0.85,
        facecolor="#0d1117",
        edgecolor="#3a4a5a",
        labelcolor="white",
        fontsize=8,
        title="Hub Genes (by degree centrality)",
        title_fontsize=8.5,
    )
    legend.get_title().set_color("#a0b0c0")

    # ── Title & annotations ───────────────────────────────────────────
    ax.set_title(
        title,
        color="white", fontsize=14, fontweight="bold",
        pad=16, fontfamily="monospace",
    )
    ax.text(
        0.99, 0.01,
        f"Nodes: {G.number_of_nodes()}  ·  Edges: {G.number_of_edges()}  ·  "
        f"STRING confidence ≥ 0.70",
        transform=ax.transAxes,
        ha="right", va="bottom",
        color="#5a6a7a", fontsize=7, fontfamily="monospace",
    )

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"PPI network saved to {save_path}")
    return save_path
