"""
run_eval.py
===========
Runs the PD RAG evaluation experiments E1-E5 over the *existing* system
components (PDRetriever, SmartRAGTrigger, PDMultiModalEngine) and writes
paper-ready tables (Markdown + CSV) and figures (PNG).

Experiments
-----------
E1  Retrieval quality   : Recall@k, Precision@k, MRR, nDCG@k.
                          - "as-configured": the live PDRetriever at its
                            trigger-chosen operating point (the real system).
                          - "@k sweep": raw cosine retrieval for clean curves.
E2  End-to-end QA       : token-F1 / EM of RAG vs no-RAG answers vs reference.
E3  Faithfulness        : groundedness, hallucination rate, answer relevancy
                          (RAG vs no-RAG), with McNemar + Wilcoxon tests.
E4  Smart trigger       : precision/recall/F1 of SmartRAGTrigger needs_rag
                          vs "always-retrieve" and "never-retrieve" baselines,
                          plus estimated retrieval tokens saved.
E5  Ablations           : retrieval threshold sweep, top_k sweep, and
                          (optional) embedder / chunk-size ablations.

Design choices that make this defensible for review:
  * The retrieval target is a throwaway ChromaDB built from corpus.jsonl, so
    your production collections are never modified.
  * Passage ids are stored as ChromaDB metadata `source`, so the live
    PDRetriever results map straight back to the gold qrels.
  * Faithfulness for BOTH RAG and no-RAG answers is scored against the SAME
    gold reference passages — a fair, fixed denominator for the hallucination
    comparison.
  * Module-0 memory injection is disabled during eval so answers are grounded
    only in retrieval (controls the experiment).

Quick start
-----------
    python -m pd_rag.eval.build_benchmark
    python -m pd_rag.eval.run_eval --skip-generation        # E1, E4, E5 (fast)
    python -m pd_rag.eval.run_eval --model Qwen/Qwen2-VL-2B-Instruct --gen-sample 60
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# make sibling package imports work both as module and as script
_PD_RAG_DIR = Path(__file__).resolve().parent.parent
if str(_PD_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_PD_RAG_DIR))

from eval import metrics as M   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_eval")

K_VALUES = [1, 3, 5, 10]
THRESHOLDS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]


# --------------------------------------------------------------------------- #
#  Throwaway retrieval index (for raw / ablation retrieval)
# --------------------------------------------------------------------------- #
class RawIndex:
    """Minimal ChromaDB-backed index over the benchmark corpus.

    Stores passage id as both the Chroma id and metadata `source` so that the
    production PDRetriever (which surfaces metadata.source) maps to qrels.
    Supports re-indexing with a different embedder or chunk size for E5.
    """

    def __init__(self, db_path: str, embedder_model: str,
                 collection: str = "pd_papers"):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(embedder_model)
        self.client = chromadb.PersistentClient(path=db_path)
        # fresh collection each build
        try:
            self.client.delete_collection(collection)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def build(self, corpus: List[Dict], chunk_words: Optional[int] = None,
              batch: int = 128):
        records = corpus if not chunk_words else _rechunk(corpus, chunk_words)
        ids = [r["id"] for r in records]
        docs = [r["text"] for r in records]
        metas = [{"source": r["parent"], "type": "paper"} if "parent" in r
                 else {"source": r["id"], "type": "paper"} for r in records]
        for i in range(0, len(docs), batch):
            embs = self.embedder.encode(docs[i:i + batch]).tolist()
            self.col.upsert(ids=ids[i:i + batch], embeddings=embs,
                            documents=docs[i:i + batch], metadatas=metas[i:i + batch])
        logger.info("RawIndex built: %d passages.", len(docs))

    def query(self, text: str, k: int) -> List[Tuple[str, float]]:
        """Return [(parent_passage_id, cosine_sim)] ranked desc, deduped by parent."""
        emb = self.embedder.encode(text).tolist()
        k_eff = min(max(k * 3, k), max(self.col.count(), 1))   # over-fetch then dedup
        res = self.col.query(query_embeddings=[emb], n_results=k_eff)
        out: List[Tuple[str, float]] = []
        seen = set()
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            pid = meta.get("source")
            if pid in seen:
                continue
            seen.add(pid)
            out.append((pid, 1.0 - dist))   # cosine sim
            if len(out) >= k:
                break
        return out


def _rechunk(corpus: List[Dict], chunk_words: int, overlap: int = 20) -> List[Dict]:
    out = []
    step = max(chunk_words - overlap, 1)
    for rec in corpus:
        words = rec["text"].split()
        if len(words) <= chunk_words:
            out.append({"id": rec["id"], "text": rec["text"], "parent": rec["id"]})
            continue
        for j, s in enumerate(range(0, len(words), step)):
            chunk = " ".join(words[s:s + chunk_words])
            if len(chunk) > 40:
                out.append({"id": f"{rec['id']}__w{j}", "text": chunk,
                            "parent": rec["id"]})
    return out


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path):
    bench = json.loads((data_dir / "benchmark.json").read_text(encoding="utf-8"))
    qrels = json.loads((data_dir / "qrels.json").read_text(encoding="utf-8"))
    triggers = json.loads((data_dir / "triggers.json").read_text(encoding="utf-8"))
    corpus = []
    with (data_dir / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    corpus_text = {r["id"]: r["text"] for r in corpus}
    return bench, qrels, triggers, corpus, corpus_text


# --------------------------------------------------------------------------- #
#  E1 — retrieval quality
# --------------------------------------------------------------------------- #
def e1_retrieval(items: List[Dict], qrels: Dict, raw: RawIndex,
                 live_retriever, trigger) -> Dict:
    # --- raw @k sweep (clean curves) ---
    per_k = {k: {"recall": [], "precision": [], "ndcg": [], "hit": []} for k in K_VALUES}
    rr = []
    for it in items:
        qid, q = it["id"], it["question"]
        rel = qrels.get(qid, [])
        ranked = [pid for pid, _ in raw.query(q, max(K_VALUES))]
        rr.append(M.reciprocal_rank(ranked, rel))
        for k in K_VALUES:
            per_k[k]["recall"].append(M.recall_at_k(ranked, rel, k))
            per_k[k]["precision"].append(M.precision_at_k(ranked, rel, k))
            per_k[k]["ndcg"].append(M.ndcg_at_k(ranked, rel, k))
            per_k[k]["hit"].append(M.hit_rate_at_k(ranked, rel, k))

    sweep = {k: {m: M.summarize(v) for m, v in d.items()} for k, d in per_k.items()}
    sweep_mrr = M.summarize(rr)

    # --- live PDRetriever at its configured operating point ---
    live = {"recall": [], "precision": [], "ndcg": [], "hit": [], "rr": []}
    for it in items:
        qid, q = it["id"], it["question"]
        rel = qrels.get(qid, [])
        _qt, needs, cols, top_k = trigger.classify(q)
        if not needs:
            cols, top_k = ["pd_papers"], 5
        results = live_retriever.retrieve(q, collections=cols, top_k=top_k)
        ranked = [r["source"] for r in results]      # metadata.source == passage id
        live["recall"].append(M.recall_at_k(ranked, rel, top_k))
        live["precision"].append(M.precision_at_k(ranked, rel, top_k))
        live["ndcg"].append(M.ndcg_at_k(ranked, rel, top_k))
        live["hit"].append(M.hit_rate_at_k(ranked, rel, top_k))
        live["rr"].append(M.reciprocal_rank(ranked, rel))

    live_summary = {m: M.summarize(v) for m, v in live.items()}
    return {"raw_sweep": sweep, "raw_mrr": sweep_mrr,
            "live_configured": live_summary}


# --------------------------------------------------------------------------- #
#  E2 / E3 — generation: accuracy + faithfulness, RAG vs no-RAG
# --------------------------------------------------------------------------- #
def e2_e3_generation(items: List[Dict], corpus_text: Dict, embed: Callable,
                     engine, support_fn, ground_threshold: float,
                     correct_tau: float) -> Dict:
    rows = []
    for it in items:
        qid, q = it["id"], it["question"]
        ref = it.get("reference_answer", "")
        gold_ctx = [corpus_text[p] for p in it.get("relevant_ids", [])
                    if p in corpus_text]

        engine.rag_enabled = True
        ans_rag = engine.chat(q)["answer"]
        engine.rag_enabled = False
        ans_norag = engine.chat(q)["answer"]

        f1_rag = M.token_f1(ans_rag, ref)
        f1_norag = M.token_f1(ans_norag, ref)
        rows.append({
            "id": qid,
            "f1_rag": f1_rag, "f1_norag": f1_norag,
            "em_rag": M.exact_match(ans_rag, ref),
            "em_norag": M.exact_match(ans_norag, ref),
            "ground_rag": M.groundedness(ans_rag, gold_ctx, embed,
                                         ground_threshold, support_fn),
            "ground_norag": M.groundedness(ans_norag, gold_ctx, embed,
                                           ground_threshold, support_fn),
            "halluc_rag": M.hallucination_rate(ans_rag, gold_ctx, embed,
                                               ground_threshold, support_fn),
            "halluc_norag": M.hallucination_rate(ans_norag, gold_ctx, embed,
                                                 ground_threshold, support_fn),
            "relevancy_rag": M.answer_relevancy(ans_rag, q, embed),
            "relevancy_norag": M.answer_relevancy(ans_norag, q, embed),
            "correct_rag": f1_rag >= correct_tau,
            "correct_norag": f1_norag >= correct_tau,
        })

    def col(name):
        return [r[name] for r in rows]

    summary = {
        "n": len(rows),
        "token_f1":      {"rag": M.summarize(col("f1_rag")),
                          "no_rag": M.summarize(col("f1_norag"))},
        "exact_match":   {"rag": M.summarize(col("em_rag")),
                          "no_rag": M.summarize(col("em_norag"))},
        "groundedness":  {"rag": M.summarize(col("ground_rag")),
                          "no_rag": M.summarize(col("ground_norag"))},
        "hallucination": {"rag": M.summarize(col("halluc_rag")),
                          "no_rag": M.summarize(col("halluc_norag"))},
        "answer_relevancy": {"rag": M.summarize(col("relevancy_rag")),
                             "no_rag": M.summarize(col("relevancy_norag"))},
        "tests": {
            "accuracy_mcnemar": M.mcnemar(col("correct_rag"), col("correct_norag")),
            "faithfulness_wilcoxon": M.wilcoxon(col("ground_rag"), col("ground_norag")),
            "token_f1_wilcoxon": M.wilcoxon(col("f1_rag"), col("f1_norag")),
        },
    }
    return {"summary": summary, "per_item": rows}


# --------------------------------------------------------------------------- #
#  E4 — SmartRAGTrigger gating
# --------------------------------------------------------------------------- #
def e4_trigger(items: List[Dict], negatives: List[Dict], triggers: Dict,
               trigger, avg_ctx_tokens: float) -> Dict:
    y_true, y_pred = [], []
    queries = ([{"id": it["id"], "question": it["question"]} for it in items]
               + [{"id": n["id"], "question": n["question"]} for n in negatives])
    for qrec in queries:
        gold = bool(triggers.get(qrec["id"], True))
        _qt, needs, _cols, _k = trigger.classify(qrec["question"])
        y_true.append(gold)
        y_pred.append(bool(needs))

    smart = M.binary_prf(y_true, y_pred)
    always = M.binary_prf(y_true, [True] * len(y_true))
    never = M.binary_prf(y_true, [False] * len(y_true))

    # token-savings vs always-retrieve: queries correctly skipped * avg ctx tokens
    correctly_skipped = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    n_neg = sum(1 for t in y_true if not t)
    savings = {
        "avg_retrieval_tokens_per_query": avg_ctx_tokens,
        "queries_correctly_skipped": correctly_skipped,
        "negative_queries": n_neg,
        "skip_rate_on_negatives": correctly_skipped / n_neg if n_neg else 0.0,
        "estimated_tokens_saved": correctly_skipped * avg_ctx_tokens,
    }
    return {"smart_trigger": smart, "always_retrieve": always,
            "never_retrieve": never, "token_savings": savings}


# --------------------------------------------------------------------------- #
#  E5 — ablations
# --------------------------------------------------------------------------- #
def e5_threshold(items: List[Dict], qrels: Dict, raw: RawIndex,
                 thresholds: Sequence[float], k: int = 5) -> Dict:
    out = {}
    for th in thresholds:
        recs, precs, hits = [], [], []
        for it in items:
            rel = qrels.get(it["id"], [])
            ranked = [pid for pid, sim in raw.query(it["question"], k) if sim >= th]
            recs.append(M.recall_at_k(ranked, rel, k))
            precs.append(M.precision_at_k(ranked, rel, k) if ranked else 0.0)
            hits.append(M.hit_rate_at_k(ranked, rel, k))
        out[f"{th:.2f}"] = {"recall@%d" % k: M.summarize(recs),
                            "precision@%d" % k: M.summarize(precs),
                            "hit@%d" % k: M.summarize(hits)}
    return out


def e5_topk(items: List[Dict], qrels: Dict, raw: RawIndex,
            k_values: Sequence[int]) -> Dict:
    out = {}
    for k in k_values:
        recs, ndcgs = [], []
        for it in items:
            rel = qrels.get(it["id"], [])
            ranked = [pid for pid, _ in raw.query(it["question"], k)]
            recs.append(M.recall_at_k(ranked, rel, k))
            ndcgs.append(M.ndcg_at_k(ranked, rel, k))
        out[str(k)] = {"recall": M.summarize(recs), "ndcg": M.summarize(ndcgs)}
    return out


def e5_embedder(items: List[Dict], qrels: Dict, corpus: List[Dict],
                db_path: str, embedder_models: Sequence[str], k: int = 5) -> Dict:
    out = {}
    for model in embedder_models:
        logger.info("E5 embedder ablation: %s", model)
        try:
            idx = RawIndex(db_path + "_emb", model)
            idx.build(corpus)
            recs, ndcgs = [], []
            for it in items:
                rel = qrels.get(it["id"], [])
                ranked = [pid for pid, _ in idx.query(it["question"], k)]
                recs.append(M.recall_at_k(ranked, rel, k))
                ndcgs.append(M.ndcg_at_k(ranked, rel, k))
            out[model] = {"recall@%d" % k: M.summarize(recs),
                          "ndcg@%d" % k: M.summarize(ndcgs)}
        except Exception as e:
            logger.warning("Embedder %s failed: %s", model, e)
            out[model] = {"error": str(e)}
    return out


def e5_chunk(items: List[Dict], qrels: Dict, corpus: List[Dict],
             db_path: str, embedder_model: str,
             chunk_sizes: Sequence[int], k: int = 5) -> Dict:
    out = {}
    for cw in chunk_sizes:
        logger.info("E5 chunk ablation: %d words", cw)
        idx = RawIndex(db_path + "_chunk", embedder_model)
        idx.build(corpus, chunk_words=cw)
        recs, ndcgs = [], []
        for it in items:
            rel = qrels.get(it["id"], [])
            ranked = [pid for pid, _ in idx.query(it["question"], k)]
            recs.append(M.recall_at_k(ranked, rel, k))
            ndcgs.append(M.ndcg_at_k(ranked, rel, k))
        out[str(cw)] = {"recall@%d" % k: M.summarize(recs),
                        "ndcg@%d" % k: M.summarize(ndcgs)}
    return out


# --------------------------------------------------------------------------- #
#  Reporting: tables + figures
# --------------------------------------------------------------------------- #
def _fmt(ci: Dict) -> str:
    if ci is None or np.isnan(ci.get("mean", float("nan"))):
        return "n/a"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def write_tables(results: Dict, out_dir: Path):
    md = ["# PD RAG Evaluation Results", ""]

    # E1
    md.append("## E1 — Retrieval quality (raw @k sweep, mean [95% CI])\n")
    md.append("| k | Recall@k | Precision@k | nDCG@k | HitRate@k |")
    md.append("|---|----------|-------------|--------|-----------|")
    for k in K_VALUES:
        s = results["E1"]["raw_sweep"][k]
        md.append(f"| {k} | {_fmt(s['recall'])} | {_fmt(s['precision'])} "
                  f"| {_fmt(s['ndcg'])} | {_fmt(s['hit'])} |")
    md.append(f"\nMRR (raw): {_fmt(results['E1']['raw_mrr'])}\n")
    lc = results["E1"]["live_configured"]
    md.append("**Live PDRetriever (configured operating point):** "
              f"Recall {_fmt(lc['recall'])}, Precision {_fmt(lc['precision'])}, "
              f"nDCG {_fmt(lc['ndcg'])}, Hit {_fmt(lc['hit'])}, MRR {_fmt(lc['rr'])}\n")

    # E2/E3
    if "E2E3" in results:
        s = results["E2E3"]["summary"]
        md.append("## E2/E3 — Generation: RAG vs no-RAG (mean [95% CI])\n")
        md.append("| Metric | RAG | no-RAG |")
        md.append("|--------|-----|--------|")
        for label, key in [("Token-F1", "token_f1"), ("Exact match", "exact_match"),
                           ("Groundedness", "groundedness"),
                           ("Hallucination rate", "hallucination"),
                           ("Answer relevancy", "answer_relevancy")]:
            md.append(f"| {label} | {_fmt(s[key]['rag'])} | {_fmt(s[key]['no_rag'])} |")
        t = s["tests"]
        md.append(f"\nMcNemar (accuracy@F1>=tau): b={t['accuracy_mcnemar']['b']}, "
                  f"c={t['accuracy_mcnemar']['c']}, p={t['accuracy_mcnemar']['p_value']:.4g}")
        md.append(f"Wilcoxon (groundedness): p={t['faithfulness_wilcoxon']['p_value']}")
        md.append(f"Wilcoxon (token-F1): p={t['token_f1_wilcoxon']['p_value']}\n")

    # E4
    e4 = results["E4"]
    md.append("## E4 — SmartRAGTrigger gating\n")
    md.append("| Policy | Precision | Recall | F1 | Accuracy |")
    md.append("|--------|-----------|--------|----|----------|")
    for label, key in [("SmartRAGTrigger", "smart_trigger"),
                       ("Always-retrieve", "always_retrieve"),
                       ("Never-retrieve", "never_retrieve")]:
        p = e4[key]
        md.append(f"| {label} | {p['precision']:.3f} | {p['recall']:.3f} "
                  f"| {p['f1']:.3f} | {p['accuracy']:.3f} |")
    ts = e4["token_savings"]
    md.append(f"\nNegatives correctly skipped: {ts['queries_correctly_skipped']}/"
              f"{ts['negative_queries']} ({ts['skip_rate_on_negatives']:.2%}); "
              f"est. retrieval tokens saved: {ts['estimated_tokens_saved']:.0f}\n")

    # E5
    md.append("## E5 — Ablations\n")
    md.append("### Retrieval threshold (Recall@5 / Precision@5)\n")
    md.append("| Threshold | Recall@5 | Precision@5 | Hit@5 |")
    md.append("|-----------|----------|-------------|-------|")
    for th, d in results["E5"]["threshold"].items():
        md.append(f"| {th} | {_fmt(d['recall@5'])} | {_fmt(d['precision@5'])} "
                  f"| {_fmt(d['hit@5'])} |")
    md.append("\n### top_k\n| k | Recall@k | nDCG@k |\n|---|----------|--------|")
    for k, d in results["E5"]["topk"].items():
        md.append(f"| {k} | {_fmt(d['recall'])} | {_fmt(d['ndcg'])} |")
    if results["E5"].get("embedder"):
        md.append("\n### Embedder\n| Model | Recall@5 | nDCG@5 |\n|-------|----------|--------|")
        for model, d in results["E5"]["embedder"].items():
            if "error" in d:
                md.append(f"| {model} | error | {d['error'][:40]} |")
            else:
                md.append(f"| {model} | {_fmt(d['recall@5'])} | {_fmt(d['ndcg@5'])} |")
    if results["E5"].get("chunk"):
        md.append("\n### Chunk size (words)\n| Size | Recall@5 | nDCG@5 |\n|------|----------|--------|")
        for cw, d in results["E5"]["chunk"].items():
            md.append(f"| {cw} | {_fmt(d['recall@5'])} | {_fmt(d['ndcg@5'])} |")

    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")

    # machine-readable CSV for E1 sweep
    with (out_dir / "e1_retrieval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "recall", "recall_lo", "recall_hi", "precision",
                    "ndcg", "hit"])
        for k in K_VALUES:
            s = results["E1"]["raw_sweep"][k]
            w.writerow([k, s["recall"]["mean"], s["recall"]["lo"], s["recall"]["hi"],
                        s["precision"]["mean"], s["ndcg"]["mean"], s["hit"]["mean"]])
    logger.info("Tables written to %s", out_dir / "results.md")


def write_figures(results: Dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable (%s) — skipping figures.", e)
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Fig 1: Recall@k & nDCG@k curve
    ks = K_VALUES
    rec = [results["E1"]["raw_sweep"][k]["recall"]["mean"] for k in ks]
    ndcg = [results["E1"]["raw_sweep"][k]["ndcg"]["mean"] for k in ks]
    plt.figure(figsize=(5, 4))
    plt.plot(ks, rec, "o-", label="Recall@k")
    plt.plot(ks, ndcg, "s--", label="nDCG@k")
    plt.xlabel("k"); plt.ylabel("score"); plt.ylim(0, 1.02)
    plt.title("E1: Retrieval quality vs k"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fig_dir / "e1_retrieval.png", dpi=150); plt.close()

    # Fig 2: RAG vs no-RAG generation bars
    if "E2E3" in results:
        s = results["E2E3"]["summary"]
        labels = ["Token-F1", "Ground.", "Halluc.", "Relevancy"]
        keys = ["token_f1", "groundedness", "hallucination", "answer_relevancy"]
        rag = [s[k]["rag"]["mean"] for k in keys]
        nor = [s[k]["no_rag"]["mean"] for k in keys]
        x = np.arange(len(labels)); wd = 0.38
        plt.figure(figsize=(6, 4))
        plt.bar(x - wd / 2, rag, wd, label="RAG")
        plt.bar(x + wd / 2, nor, wd, label="no-RAG")
        plt.xticks(x, labels); plt.ylim(0, 1.02)
        plt.title("E2/E3: RAG vs no-RAG"); plt.legend(); plt.grid(axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(fig_dir / "e2e3_rag_vs_norag.png", dpi=150); plt.close()

    # Fig 3: trigger policies F1
    e4 = results["E4"]
    pols = ["smart_trigger", "always_retrieve", "never_retrieve"]
    f1s = [e4[p]["f1"] for p in pols]
    plt.figure(figsize=(5, 4))
    plt.bar(["Smart", "Always", "Never"], f1s, color=["#2b8cbe", "#a6bddb", "#ccc"])
    plt.ylabel("F1 (needs_rag)"); plt.ylim(0, 1.02)
    plt.title("E4: Gating policy F1"); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(fig_dir / "e4_trigger.png", dpi=150); plt.close()

    # Fig 4: threshold ablation
    th = results["E5"]["threshold"]
    xs = [float(t) for t in th]
    recs = [th[t]["recall@5"]["mean"] for t in th]
    precs = [th[t]["precision@5"]["mean"] for t in th]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, recs, "o-", label="Recall@5")
    plt.plot(xs, precs, "s--", label="Precision@5")
    plt.axvline(0.25, color="r", ls=":", alpha=.6, label="current (0.25)")
    plt.xlabel("cosine threshold"); plt.ylabel("score"); plt.ylim(0, 1.02)
    plt.title("E5: Threshold ablation"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fig_dir / "e5_threshold.png", dpi=150); plt.close()

    logger.info("Figures written to %s", fig_dir)


# --------------------------------------------------------------------------- #
#  Engine / embedder setup
# --------------------------------------------------------------------------- #
def load_engine(model_id: str, eval_db_path: str):
    """Load the real PDMultiModalEngine pointed at the throwaway eval DB.

    Module-0 memory injection is disabled so generation is grounded only in
    retrieval (experiment control)."""
    sys.path.insert(0, str(_PD_RAG_DIR))
    from multimodal_engine_rag import PDMultiModalEngine
    engine = PDMultiModalEngine(model_id=model_id, vector_db_path=eval_db_path,
                                rag_enabled=True)
    engine._m0_bridge = None
    return engine


def make_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    return lambda texts: np.asarray(enc.encode(texts))


def estimate_avg_ctx_tokens(items: List[Dict], corpus_text: Dict) -> float:
    """Rough retrieval-token estimate (~0.75 words/token) for E4 savings."""
    counts = []
    for it in items:
        words = sum(len(corpus_text[p].split())
                    for p in it.get("relevant_ids", []) if p in corpus_text)
        counts.append(words / 0.75)
    return float(np.mean(counts)) if counts else 0.0


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run PD RAG evaluation E1-E5.")
    ap.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--eval-db", default=str(Path(__file__).parent / "eval_vector_db"))
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Embedder for the index + groundedness (match retriever).")
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct",
                    help="Qwen2-VL model id for generation (2B fits ~8GB VRAM).")
    ap.add_argument("--skip-generation", action="store_true",
                    help="Run only E1/E4/E5 (no model load) — fast.")
    ap.add_argument("--gen-sample", type=int, default=60,
                    help="Number of questions to use for E2/E3 generation.")
    ap.add_argument("--ground-threshold", type=float, default=0.55)
    ap.add_argument("--correct-tau", type=float, default=0.30,
                    help="Token-F1 threshold counted as 'correct' for McNemar.")
    ap.add_argument("--llm-judge", action="store_true",
                    help="Use Qwen as an entailment judge for faithfulness "
                         "(slower; report alongside embedding proxy).")
    ap.add_argument("--ablate-embedder", default="",
                    help="Comma-separated embedder model ids for E5.")
    ap.add_argument("--ablate-chunk", default="",
                    help="Comma-separated chunk sizes (words) for E5.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "benchmark.json").exists():
        raise SystemExit(f"No benchmark in {data_dir}. Run build_benchmark first.")

    bench, qrels, triggers, corpus, corpus_text = load_data(data_dir)
    items = bench["items"]
    negatives = bench.get("negatives", [])
    logger.info("Loaded %d questions, %d negatives, %d corpus passages.",
                len(items), len(negatives), len(corpus))

    # build retrieval index
    raw = RawIndex(args.eval_db, args.embedder)
    raw.build(corpus)

    # live PDRetriever + SmartRAGTrigger over the same eval DB
    from retrieval.retriever import PDRetriever
    from core.trigger import SmartRAGTrigger
    live_retriever = PDRetriever(vector_db_path=args.eval_db,
                                 embedder_model=args.embedder)
    trigger = SmartRAGTrigger()

    results: Dict = {"config": vars(args)}

    # E1
    logger.info("Running E1 (retrieval)...")
    results["E1"] = e1_retrieval(items, qrels, raw, live_retriever, trigger)

    # E4
    logger.info("Running E4 (trigger gating)...")
    avg_ctx = estimate_avg_ctx_tokens(items, corpus_text)
    results["E4"] = e4_trigger(items, negatives, triggers, trigger, avg_ctx)

    # E5
    logger.info("Running E5 (ablations)...")
    e5 = {"threshold": e5_threshold(items, qrels, raw, THRESHOLDS),
          "topk": e5_topk(items, qrels, raw, K_VALUES)}
    if args.ablate_embedder.strip():
        e5["embedder"] = e5_embedder(items, qrels, corpus, args.eval_db,
                                     [m.strip() for m in args.ablate_embedder.split(",")])
    if args.ablate_chunk.strip():
        e5["chunk"] = e5_chunk(items, qrels, corpus, args.eval_db, args.embedder,
                               [int(c) for c in args.ablate_chunk.split(",")])
    results["E5"] = e5

    # E2/E3 (generation)
    if not args.skip_generation:
        logger.info("Loading generator %s ...", args.model)
        engine = load_engine(args.model, args.eval_db)
        embed = make_embedder(args.embedder)
        support_fn = None
        if args.llm_judge:
            support_fn = M.make_llm_judge_support(
                lambda p: engine.chat(p)["answer"])
        gen_items = items[:args.gen_sample]
        logger.info("Running E2/E3 on %d questions (RAG vs no-RAG)...", len(gen_items))
        t0 = time.time()
        results["E2E3"] = e2_e3_generation(
            gen_items, corpus_text, embed, engine, support_fn,
            args.ground_threshold, args.correct_tau)
        logger.info("Generation done in %.1fs", time.time() - t0)

    # persist + report
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2),
                                           encoding="utf-8")
    write_tables(results, out_dir)
    write_figures(results, out_dir)
    logger.info("All results in %s", out_dir.resolve())


if __name__ == "__main__":
    main()
