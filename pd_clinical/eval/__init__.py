"""
pd_clinical.eval — PD vs Healthy-Control classification on PPMI tabular data.

Two stages:
  prepare_ppmi : assemble a clean cohort table (label + features) from the raw
                 PPMI Study-Data CSVs, joining on PATNO at the baseline visit.
  train_eval   : train logistic-regression + gradient-boosting classifiers with
                 stratified CV and a held-out test split; report AUC (95% CI),
                 sensitivity, specificity, confusion matrix; write tables + ROC
                 and feature-importance figures for the paper.

This produces the clinical-validation arm a Q1 reviewer expects: real cohort,
real metrics, proper statistics — the analogue of the RAG harness for Module 2.
"""
