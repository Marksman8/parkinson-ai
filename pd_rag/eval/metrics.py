"""
metrics.py
==========
Evaluation metrics for the PD RAG harness. Pure functions — no model loading —
so they are unit-testable and reusable in the paper's analysis notebooks.

Retrieval metrics (binary relevance from qrels):
    recall_at_k, precision_at_k, hit_rate_at_k, reciprocal_rank, ndcg_at_k

Generation metrics:
    token_f1, exact_match            (extractive overlap with reference answer)
    groundedness                     (sentence-level support from retrieved context)
    hallucination_rate               (= 1 - groundedness)
    answer_relevancy                 (answer<->question semantic similarity)

Classifier metrics (SmartRAGTrigger, E4):
    binary_prf                       (precision / recall / F1 / accuracy)

Statistics:
    bootstrap_ci                     (nonparametric CI for any per-item metric)
    mcnemar                          (paired accuracy comparison, RAG vs no-RAG)
    wilcoxon                         (paired score comparison)

The groundedness / relevancy metrics take an `embed` callable
(texts -> np.ndarray) so the caller decides which SentenceTransformer to use.
This is the standard embedding-based proxy for RAGAS faithfulness / answer
relevancy; for a stricter measure pass an NLI/LLM judge via `support_fn`.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
#  Retrieval metrics
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of gold-relevant passages found in the top-k retrieved."""
    rel = set(relevant)
    if not rel:
        return float("nan")
    topk = set(retrieved[:k])
    return len(topk & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    rel = set(relevant)
    topk = retrieved[:k]
    return sum(1 for r in topk if r in rel) / k


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant passage appears in top-k, else 0.0."""
    rel = set(relevant)
    return 1.0 if (set(retrieved[:k]) & rel) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant passage (0 if none retrieved)."""
    rel = set(relevant)
    for i, r in enumerate(retrieved, 1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized DCG with binary relevance gains."""
    rel = set(relevant)
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], 1):
        if r in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Generation: extractive overlap
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between answer and reference."""
    pred, ref = _tokens(prediction), _tokens(reference)
    if not pred or not ref:
        return 0.0
    common: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if ref_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _tokens(prediction) == _tokens(reference) else 0.0


# --------------------------------------------------------------------------- #
#  Generation: groundedness / faithfulness / relevancy (embedding-based)
# --------------------------------------------------------------------------- #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) > 12]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def groundedness(
    answer: str,
    contexts: Sequence[str],
    embed: Callable[[List[str]], np.ndarray],
    threshold: float = 0.55,
    support_fn: Optional[Callable[[str, Sequence[str]], bool]] = None,
) -> float:
    """Fraction of answer sentences supported by the retrieved context.

    Default: a sentence is "supported" if its max cosine similarity to any
    context passage >= threshold (embedding-based faithfulness proxy, in the
    spirit of RAGAS faithfulness). Pass `support_fn` to substitute an NLI or
    LLM-judge entailment check for a stricter measure.

    Returns NaN if there are no scorable sentences (caller should drop these).
    """
    sents = split_sentences(answer)
    if not sents:
        return float("nan")
    if not contexts:
        return 0.0

    if support_fn is not None:
        supported = sum(1 for s in sents if support_fn(s, contexts))
        return supported / len(sents)

    sent_emb = embed(sents)
    ctx_emb = embed(list(contexts))
    sims = _cos(np.asarray(sent_emb), np.asarray(ctx_emb))   # (n_sent, n_ctx)
    max_sim = sims.max(axis=1)
    return float((max_sim >= threshold).mean())


def hallucination_rate(
    answer: str,
    contexts: Sequence[str],
    embed: Callable[[List[str]], np.ndarray],
    threshold: float = 0.55,
    support_fn: Optional[Callable[[str, Sequence[str]], bool]] = None,
) -> float:
    """1 - groundedness. NaN if answer has no scorable sentences."""
    g = groundedness(answer, contexts, embed, threshold, support_fn)
    return float("nan") if math.isnan(g) else 1.0 - g


def answer_relevancy(
    answer: str,
    question: str,
    embed: Callable[[List[str]], np.ndarray],
) -> float:
    """Cosine similarity between answer and question embeddings."""
    if not answer.strip() or not question.strip():
        return float("nan")
    emb = np.asarray(embed([answer, question]))
    return float(_cos(emb[0:1], emb[1:2])[0, 0])


def make_llm_judge_support(generate_fn: Callable[[str], str]) -> Callable[[str, Sequence[str]], bool]:
    """Build a stricter `support_fn` backed by an LLM yes/no entailment judge.

    `generate_fn` is any text-in/text-out callable (e.g. a wrapper around your
    Qwen2-VL `chat`). Returns True iff the model judges the claim entailed by
    the context. Use with --llm-judge to report LLM-graded faithfulness
    alongside the embedding proxy (and remember to validate against human
    ratings on a subset, per the paper guardrail).
    """
    def support(claim: str, contexts: Sequence[str]) -> bool:
        ctx = "\n".join(f"- {c}" for c in contexts)
        prompt = (
            "You are a strict fact-checker. Given CONTEXT and a CLAIM, answer "
            "with a single word: YES if the claim is fully supported by the "
            "context, otherwise NO.\n\n"
            f"CONTEXT:\n{ctx}\n\nCLAIM: {claim}\n\nAnswer (YES/NO):"
        )
        out = generate_fn(prompt).strip().lower()
        return out.startswith("y")
    return support


# --------------------------------------------------------------------------- #
#  Classifier metrics (SmartRAGTrigger)
# --------------------------------------------------------------------------- #
def binary_prf(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    """Precision/recall/F1/accuracy for the positive class (needs_rag=True)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Nonparametric bootstrap CI for the mean of a per-item metric.

    NaNs are dropped (e.g. items with no scorable sentences)."""
    arr = np.asarray([v for v in values if not (v is None or math.isnan(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.integers(0, arr.size, arr.size)]
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """McNemar's test on paired correctness (e.g. RAG vs no-RAG accuracy).

    b = A correct & B wrong ; c = A wrong & B correct. Uses an exact binomial
    p-value (robust for small discordant counts). Falls back gracefully if
    scipy is absent.
    """
    b = sum(1 for a, x in zip(correct_a, correct_b) if a and not x)
    c = sum(1 for a, x in zip(correct_a, correct_b) if not a and x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "stat": 0.0}
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    except Exception:
        # exact two-sided binomial without scipy
        k = min(b, c)
        cum = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2 * cum)
    stat = (abs(b - c) - 1) ** 2 / n if n else 0.0      # continuity-corrected chi2
    return {"b": b, "c": c, "p_value": float(p), "stat": float(stat)}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired scores (e.g. faithfulness)."""
    pairs = [(x, y) for x, y in zip(a, b)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 6:
        return {"p_value": float("nan"), "stat": float("nan"), "n": len(pairs)}
    try:
        from scipy.stats import wilcoxon as _w
        xa = [x for x, _ in pairs]
        xb = [y for _, y in pairs]
        stat, p = _w(xa, xb)
        return {"p_value": float(p), "stat": float(stat), "n": len(pairs)}
    except Exception:
        return {"p_value": float("nan"), "stat": float("nan"), "n": len(pairs)}


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Convenience: mean + 95% bootstrap CI for a metric column."""
    return bootstrap_ci(values)
