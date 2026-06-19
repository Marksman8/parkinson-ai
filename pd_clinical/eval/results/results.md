# PD vs Healthy-Control Classification (PPMI)

Cohort: 2043 PD, 423 HC; 11 features.

| Model | AUC [95% CI] | Sensitivity | Specificity | PPV | NPV | Accuracy |
|-------|--------------|-------------|-------------|-----|-----|----------|
| logistic_regression | 0.966 [0.957, 0.973] | 0.965 | 0.827 | 0.964 | 0.831 | 0.942 |
| gradient_boosting | 0.979 [0.973, 0.984] | 0.881 | 0.962 | 0.991 | 0.625 | 0.895 |

**logistic_regression confusion matrix** (Youden threshold 0.286): TP=1972 FP=73 FN=71 TN=350
**gradient_boosting confusion matrix** (Youden threshold 0.901): TP=1799 FP=16 FN=244 TN=407