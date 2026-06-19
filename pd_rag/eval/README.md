# PD RAG Evaluation Harness (`pd_rag/eval`)

Turns the PD RAG system into a **quantitatively evaluated** study suitable for a
Q1 methods/informatics journal. Everything runs on **public data** (PubMedQA,
optional BioASQ), so results are reproducible and citable.

## Contribution framing for the paper

The headline is **selective-retrieval RAG for Parkinson's-disease clinical QA**:
a domain-routed knowledge base (`SmartRAGTrigger` decides *when* to retrieve and
*which* collections to search). Plain RAG wrappers are not novel — the gating +
domain benchmark + ablations are what make this defensible.

## Pipeline

```
build_benchmark.py   →  data/benchmark.json, corpus.jsonl, qrels.json, triggers.json
run_eval.py          →  results/results.md, metrics.json, e1_retrieval.csv, figures/*.png
```

`build_benchmark` keeps PD-related PubMedQA items (each carries the abstract
passages that justify its answer → **gold relevance labels**), adds PD-unrelated
passages as retrieval **distractors**, and adds synthetic **negative** queries
(greetings/MRI/ODE) whose gold label is *no retrieval* — for the gating test.

`run_eval` builds a **throwaway** ChromaDB from `corpus.jsonl` (your production
collections are never touched) and runs:

| Exp | What | Key outputs |
|-----|------|-------------|
| E1 | Retrieval quality | Recall@k, Precision@k, MRR, nDCG@k (raw sweep + live `PDRetriever`) |
| E2 | QA accuracy | token-F1 / EM, RAG vs no-RAG |
| E3 | Faithfulness | groundedness, hallucination rate, answer relevancy + McNemar/Wilcoxon |
| E4 | Gating | `SmartRAGTrigger` P/R/F1 vs always/never-retrieve + token savings |
| E5 | Ablations | retrieval threshold, top_k, embedder, chunk size |

## Run it

```bash
pip install -r pd_rag/eval/requirements-eval.txt

# 1. Build the benchmark (public PubMedQA; add --bioasq-file for BioASQ)
python -m pd_rag.eval.build_benchmark --distractors 800

# 2a. Fast pass — retrieval + gating + ablations only (no GPU/model load)
python -m pd_rag.eval.run_eval --skip-generation

# 2b. Full pass — adds RAG-vs-no-RAG generation (RTX 5060 = 2B model)
python -m pd_rag.eval.run_eval --model Qwen/Qwen2-VL-2B-Instruct --gen-sample 60

# optional ablations
python -m pd_rag.eval.run_eval --skip-generation \
    --ablate-embedder "sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,NeuML/pubmedbert-base-embeddings" \
    --ablate-chunk "128,256,400,512"
```

Run from the **repo root** so `pd_rag` is importable, or `cd pd_rag` and call the
files directly.

## Reviewer guardrails (read before writing the paper)

1. **Validate the LLM judge.** `--llm-judge` reports LLM-graded faithfulness, but
   the default is an embedding-based proxy (RAGAS-style). For either, hand-rate a
   ~50-item subset and report agreement (Cohen's κ), or reviewers discount it.
2. **Faithfulness denominator.** Both RAG and no-RAG answers are scored against
   the *same gold reference passages* — this is the fair test of "does RAG reduce
   hallucination," stated explicitly in `run_eval` docstring.
3. **Don't overclaim clinical use.** Frame as a retrieval/QA methods paper, not a
   validated diagnostic. Suggested venues: *J. Biomedical Informatics*,
   *Artificial Intelligence in Medicine*, *Computers in Biology and Medicine*,
   *JMIR Medical Informatics*.
4. **Report the operating point honestly.** The live `PDRetriever` uses a 0.25
   cosine threshold; E5 shows the full threshold curve so the choice is justified
   rather than assumed.
```
