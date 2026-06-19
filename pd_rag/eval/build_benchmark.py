"""
build_benchmark.py
==================
Builds a Parkinson's-disease question-answering gold set for RAG evaluation.

Each benchmark item maps:
    query  ->  reference answer  ->  the source passage(s) that support it

Primary source : PubMedQA (`qiaojin/PubMedQA`, config `pqa_labeled`) — fully open,
                 downloadable through the HuggingFace `datasets` library. Every item
                 carries the abstract context passages that justify the answer, which
                 gives us *gold relevance labels* for retrieval metrics (E1).

Optional source: BioASQ (Task B). BioASQ requires (free) registration, so we do not
                 download it automatically. If you have a BioASQ training JSON
                 (e.g. `training12b.json`), pass --bioasq-file and its PD questions
                 + snippets are merged in.

Outputs (written to --out-dir, default pd_rag/eval/data/):
    benchmark.json   list of QA items (see _SCHEMA below)
    corpus.jsonl     every candidate passage (gold + distractors), one JSON per line
    qrels.json       query_id -> [relevant_passage_id, ...]   (gold relevance)
    triggers.json    query_id -> needs_rag (bool)   (gold labels for E4)
    summary.json     counts + provenance for the paper's Methods section

Usage
-----
    python -m pd_rag.eval.build_benchmark --out-dir pd_rag/eval/data
    python -m pd_rag.eval.build_benchmark --distractors 1000 --bioasq-file training12b.json

The corpus is the retrieval target: at eval time run_eval.py embeds corpus.jsonl
into a *throwaway* ChromaDB so your production collections are never touched.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_benchmark")

# --- item schema (documented for the paper) --------------------------------
# _SCHEMA = {
#     "id":               str,        # unique query id, e.g. "pubmedqa_25429730"
#     "question":         str,        # the natural-language query
#     "reference_answer": str,        # long-form reference answer
#     "decision":         str,        # "yes"/"no"/"maybe" (PubMedQA) or ""
#     "relevant_ids":     [str],      # corpus passage ids that support the answer
#     "source":           str,        # "PubMedQA" | "BioASQ"
#     "external_id":      str,        # PMID / BioASQ id for traceability
#     "query_type":       str,        # heuristic: gene/drug/guideline/clinical
#     "needs_rag":        bool,       # gold label for the SmartRAGTrigger (E4)
# }

# Parkinson's-relevance filter. An item is kept if any term appears in the
# question or its context passages.
PD_TERMS = [
    "parkinson", "parkinsonism", "parkinsonian",
    "lrrk2", "snca", "alpha-synuclein", "alpha synuclein", "α-synuclein",
    "pink1", "parkin", "prkn", "park7", "dj-1", "gba", "glucocerebrosidase",
    "substantia nigra", "dopaminergic", "dopamine neuron", "nigrostriatal",
    "levodopa", "l-dopa", "carbidopa", "dopamine agonist", "pramipexole",
    "lewy body", "lewy bodies", "bradykinesia", "resting tremor",
    "deep brain stimulation", "subthalamic", "updrs", "hoehn",
    "neurodegeneration of the basal ganglia",
]

# Routing keywords reused to assign an expected query_type (mirrors trigger.py
# categories so E4/E5 can stratify by type without importing the live trigger).
_TYPE_KW = {
    "gene":      ["gene", "mutation", "variant", "lrrk2", "snca", "pink1",
                  "parkin", "prkn", "gba", "allele", "genotype", "pathogenic"],
    "drug":      ["drug", "treatment", "therapy", "levodopa", "carbidopa",
                  "agonist", "dose", "medication", "pharmacolog"],
    "guideline": ["guideline", "criteria", "staging", "hoehn", "updrs",
                  "recommendation", "diagnosis", "management"],
}

# Negative (no-retrieval) queries with gold needs_rag=False, used in E4 to test
# whether the SmartRAGTrigger correctly *skips* retrieval. These mirror the
# non-RAG branches of SmartRAGTrigger (greetings, MRI, report, ODE).
NEGATIVE_QUERIES = [
    ("neg_greet_1",  "Hello, how are you today?",                         False),
    ("neg_greet_2",  "Thanks, that was helpful.",                         False),
    ("neg_greet_3",  "Hi there",                                          False),
    ("neg_greet_4",  "ok good",                                           False),
    ("neg_mri_1",    "Analyze this brain MRI scan for me.",               False),
    ("neg_mri_2",    "Here is a T2 neuroimaging image, what do you see?", False),
    ("neg_mri_3",    "Please look at the attached brain scan.",           False),
    ("neg_report_1", "Summarize this clinical report PDF.",               False),
    ("neg_report_2", "Summarise the attached lab result.",                False),
    ("neg_ode_1",    "Simulate the dopamine kinetics ODE model.",         False),
    ("neg_ode_2",    "Run a Hodgkin-Huxley differential equation model.", False),
    ("neg_ode_3",    "Simulate neuron firing with these parameters.",     False),
]


def _is_pd(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in PD_TERMS)


def _guess_query_type(text: str) -> str:
    t = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in t) for cat, kws in _TYPE_KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "clinical"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


# --------------------------------------------------------------------------- #
#  PubMedQA
# --------------------------------------------------------------------------- #
def load_pubmedqa(configs: Sequence[str] = ("pqa_labeled",),
                  max_items: Optional[int] = None,
                  distractor_cap: int = 20000):
    """Load one or more PubMedQA configs and keep PD-related items.

    `pqa_labeled`   (~1k)   expert yes/no/maybe gold answers — highest quality.
    `pqa_unlabeled` (~61k)  real questions + abstract contexts, no final_decision.
    `pqa_artificial`(~211k) auto-generated yes/no questions from PubMed abstracts.

    Pulling the larger configs is what scales the PD slice from a handful of
    items to hundreds. All three share the same structure (question, abstract
    `contexts` = gold passages, `long_answer`), so gold relevance labels are
    consistent. Items are de-duplicated by PMID across configs.

    Non-PD items contribute their passages to an (in-domain biomedical)
    distractor pool, bounded by `distractor_cap` to keep memory in check.

    Returns (items, distractor_pool, per_config_counts).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required for PubMedQA.\n"
            "    pip install datasets\n"
            "Then re-run, or supply --bioasq-file only."
        )

    items: List[Dict] = []
    distractor_pool: List[Dict] = []      # in-domain (non-PD) passages -> retrieval noise
    seen_pmids: Set[str] = set()
    per_config: Dict[str, int] = {}

    for cfg in configs:
        logger.info("Downloading PubMedQA (qiaojin/PubMedQA, %s)...", cfg)
        try:
            ds = load_dataset("qiaojin/PubMedQA", cfg, split="train")
        except Exception as e:
            logger.warning("Could not load config %s (%s) — skipping.", cfg, e)
            per_config[cfg] = 0
            continue

        kept = 0
        for row in ds:
            pmid = str(row.get("pubid", ""))
            if not pmid or pmid in seen_pmids:
                continue
            question = _clean(row.get("question", ""))
            ctx = row.get("context", {}) or {}
            passages = [_clean(p) for p in ctx.get("contexts", []) if _clean(p)]
            long_answer = _clean(row.get("long_answer", ""))
            decision = row.get("final_decision", "") or ""

            if not question or not passages:
                continue

            joined = question + " " + " ".join(passages)
            if _is_pd(joined):
                seen_pmids.add(pmid)
                relevant_ids, passage_records = [], []
                for j, p in enumerate(passages):
                    pid = f"pubmedqa_{pmid}_c{j}"
                    relevant_ids.append(pid)
                    passage_records.append({"id": pid, "text": p,
                                            "source": f"PubMedQA:{pmid}"})
                items.append({
                    "id":               f"pubmedqa_{pmid}",
                    "question":         question,
                    "reference_answer": long_answer,
                    "decision":         decision,
                    "relevant_ids":     relevant_ids,
                    "source":           f"PubMedQA/{cfg}",
                    "external_id":      pmid,
                    "query_type":       _guess_query_type(joined),
                    "needs_rag":        True,
                    "_passages":        passage_records,   # stripped before writing
                })
                kept += 1
            elif len(distractor_pool) < distractor_cap:
                for j, p in enumerate(passages):
                    distractor_pool.append({
                        "id": f"distractor_{pmid}_c{j}", "text": p,
                        "source": f"PubMedQA-distractor:{pmid}",
                    })
        per_config[cfg] = kept
        logger.info("  %s: kept %d new PD items (running total %d).",
                    cfg, kept, len(items))

    if max_items:
        items = items[:max_items]
    logger.info("PubMedQA total: %d PD items, %d distractor passages.",
                len(items), len(distractor_pool))
    return items, distractor_pool, per_config


# --------------------------------------------------------------------------- #
#  BioASQ (optional, local file)
# --------------------------------------------------------------------------- #
def load_bioasq(path: str) -> List[Dict]:
    """Parse a BioASQ Task B training JSON and keep PD questions.

    BioASQ format: {"questions": [{"id","body","ideal_answer",[snippets]}, ...]}.
    Each snippet's text becomes a gold passage `bioasq_<id>_s<j>`.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("BioASQ file not found: %s — skipping.", path)
        return []

    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    questions = data.get("questions", data if isinstance(data, list) else [])
    items: List[Dict] = []

    for q in questions:
        qid = str(q.get("id", ""))
        body = _clean(q.get("body", ""))
        ideal = q.get("ideal_answer", "")
        if isinstance(ideal, list):
            ideal = _clean(" ".join(ideal))
        else:
            ideal = _clean(ideal)
        snippets = [_clean(s.get("text", "")) for s in q.get("snippets", [])]
        snippets = [s for s in snippets if s]

        joined = body + " " + " ".join(snippets) + " " + ideal
        if not body or not snippets or not _is_pd(joined):
            continue

        relevant_ids, passage_records = [], []
        for j, s in enumerate(snippets):
            pid = f"bioasq_{qid}_s{j}"
            relevant_ids.append(pid)
            passage_records.append({"id": pid, "text": s, "source": f"BioASQ:{qid}"})

        items.append({
            "id":               f"bioasq_{qid}",
            "question":         body,
            "reference_answer": ideal,
            "decision":         "",
            "relevant_ids":     relevant_ids,
            "source":           "BioASQ",
            "external_id":      qid,
            "query_type":       _guess_query_type(joined),
            "needs_rag":        True,
            "_passages":        passage_records,
        })

    logger.info("BioASQ: kept %d PD items.", len(items))
    return items


# --------------------------------------------------------------------------- #
#  Assemble + write
# --------------------------------------------------------------------------- #
def build(out_dir: str, distractors: int, bioasq_file: Optional[str],
          max_pubmedqa: Optional[int], pubmedqa_configs: Sequence[str],
          seed: int = 42) -> Dict:
    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pubmed_items, distractor_pool, per_config = load_pubmedqa(
        pubmedqa_configs, max_pubmedqa)
    bioasq_items = load_bioasq(bioasq_file) if bioasq_file else []
    items = pubmed_items + bioasq_items

    if not items:
        raise SystemExit(
            "No PD items found. Check network access for PubMedQA, or pass "
            "--bioasq-file with a BioASQ training JSON."
        )

    # --- corpus = gold passages (dedup by id) + sampled distractors ----------
    corpus: Dict[str, Dict] = {}
    for it in items:
        for rec in it["_passages"]:
            corpus[rec["id"]] = rec

    random.shuffle(distractor_pool)
    for rec in distractor_pool[:distractors]:
        corpus[rec["id"]] = rec

    # --- qrels + trigger labels ---------------------------------------------
    qrels: Dict[str, List[str]] = {}
    triggers: Dict[str, bool] = {}
    for it in items:
        qrels[it["id"]] = it["relevant_ids"]
        triggers[it["id"]] = it["needs_rag"]

    # negatives (no question -> corpus mapping; only used for trigger eval)
    negatives = []
    for nid, ntext, needs in NEGATIVE_QUERIES:
        triggers[nid] = needs
        negatives.append({"id": nid, "question": ntext, "needs_rag": needs,
                           "source": "synthetic_negative"})

    # --- strip internal field before writing benchmark ----------------------
    clean_items = []
    for it in items:
        c = {k: v for k, v in it.items() if not k.startswith("_")}
        clean_items.append(c)

    # --- write ---------------------------------------------------------------
    (out / "benchmark.json").write_text(
        json.dumps({"items": clean_items, "negatives": negatives}, indent=2),
        encoding="utf-8")

    with (out / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for rec in corpus.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (out / "qrels.json").write_text(json.dumps(qrels, indent=2), encoding="utf-8")
    (out / "triggers.json").write_text(json.dumps(triggers, indent=2), encoding="utf-8")

    n_gold = sum(len(v) for v in qrels.values())
    summary = {
        "n_questions":          len(clean_items),
        "n_negative_queries":   len(negatives),
        "n_corpus_passages":    len(corpus),
        "n_gold_relevant":      n_gold,
        "n_distractors":        min(distractors, len(distractor_pool)),
        "sources": {
            "PubMedQA": len(pubmed_items),
            "BioASQ":   len(bioasq_items),
        },
        "pubmedqa_per_config":     per_config,
        "query_type_distribution": _count_by(clean_items, "query_type"),
        "decision_distribution":   _count_by(clean_items, "decision"),
        "seed": seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Benchmark written to %s", out.resolve())
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    if len(clean_items) < 30:
        logger.warning(
            "Only %d questions — still underpowered for a Q1 paper. Add "
            "more configs via --pubmedqa-configs (include pqa_artificial) "
            "and/or --bioasq-file.", len(clean_items))
    return summary


def _count_by(items: List[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it.get(key, "")] = out.get(it.get(key, ""), 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Build PD RAG QA benchmark.")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "data"),
                    help="Output directory for benchmark files.")
    ap.add_argument("--distractors", type=int, default=800,
                    help="Number of in-domain (non-PD) passages added as retrieval noise.")
    ap.add_argument("--pubmedqa-configs", default="pqa_labeled,pqa_artificial",
                    help="Comma-separated PubMedQA configs to merge. Options: "
                         "pqa_labeled, pqa_unlabeled, pqa_artificial. Include the "
                         "larger configs to scale the PD slice to hundreds of items.")
    ap.add_argument("--bioasq-file", default=None,
                    help="Optional path to a BioASQ Task B training JSON.")
    ap.add_argument("--max-pubmedqa", type=int, default=None,
                    help="Cap PubMedQA PD items (for a quick smoke test).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    configs = [c.strip() for c in args.pubmedqa_configs.split(",") if c.strip()]
    build(args.out_dir, args.distractors, args.bioasq_file,
          args.max_pubmedqa, configs, args.seed)


if __name__ == "__main__":
    main()
