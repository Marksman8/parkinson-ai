"""
pd_rag.eval — Reproducible evaluation harness for the PD RAG system.

Modules
-------
build_benchmark : Build a PD question-answer gold set (query -> answer -> relevant
                  source passages) from PubMedQA + optional BioASQ.
metrics         : Retrieval metrics (Recall@k, Precision@k, MRR, nDCG) and
                  generation metrics (groundedness / faithfulness, hallucination
                  rate, answer relevancy, token-F1), plus statistical helpers.
run_eval        : Runs experiments E1-E5 over the existing PDRetriever /
                  SmartRAGTrigger / PDMultiModalEngine and writes tables + plots.

The benchmark uses only public data, so results are reproducible and citable.
"""
