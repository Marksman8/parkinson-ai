"""
train_eval.py
=============
Train and evaluate PD-vs-Healthy-Control classifiers on the PPMI cohort table
produced by prepare_ppmi.py, and write paper-ready tables + figures.

Protocol (clinically standard, reviewer-defensible):
  * Subject-level stratified 5-fold cross-validation; out-of-fold predicted
    probabilities are pooled to compute AUC, ROC, and an operating point.
  * Operating point = Youden's J (max sensitivity+specificity-1) on the pooled
    CV ROC; sensitivity, specificity, PPV, NPV, accuracy, and the confusion
    matrix are reported at that threshold.
  * AUC is reported with a 95% bootstrap confidence interval.
  * Two models: L2 logistic regression (primary, interpretable) and gradient
    boosting (nonlinear comparison) — gives reviewers a baseline contrast.
  * Median imputation + standardisation are fit inside each CV fold (no leakage).

Outputs (results/ dir):
  results.md, metrics.json, figures/roc.png, figures/feature_importance.png

Usage
-----
    python -m pd_clinical.eval.train_eval
    python -m pd_clinical.eval.train_eval --data data/cohort.csv --folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_eval")


def _require_sklearn():
    try:
        import sklearn  # noqa
    except ImportError:
        raise SystemExit("scikit-learn required:  pip install scikit-learn pandas")


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:      # need both classes in the resample
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    if not aucs:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"auc": float(roc_auc_score(y, scores)), "lo": float(lo), "hi": float(hi)}


def youden_operating_point(y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    k = int(np.argmax(j))
    threshold = float(thr[k])
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"threshold": threshold, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "accuracy": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_models(seed: int):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
    ])
    gb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=seed)),
    ])
    return {"logistic_regression": logreg, "gradient_boosting": gb}


def evaluate_model(name, model, X, y, feature_names, folds, seed) -> Dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    auc = bootstrap_auc_ci(y, proba, seed=seed)
    op = youden_operating_point(y, proba)

    # feature importance from a full-data fit (for reporting only)
    model.fit(X, y)
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        imp = dict(zip(feature_names, np.abs(clf.coef_[0]).tolist()))
    elif hasattr(clf, "feature_importances_"):
        imp = dict(zip(feature_names, clf.feature_importances_.tolist()))
    else:
        imp = {}

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, proba)
    logger.info("%s: AUC=%.3f [%.3f, %.3f]  sens=%.3f spec=%.3f",
                name, auc["auc"], auc["lo"], auc["hi"],
                op["sensitivity"], op["specificity"])
    return {"auc": auc, "operating_point": op, "feature_importance": imp,
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}}


def write_outputs(results: Dict, feature_names: List[str],
                  summary: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = ["# PD vs Healthy-Control Classification (PPMI)", ""]
    md.append(f"Cohort: {summary['n_pd']} PD, {summary['n_hc']} HC; "
              f"{len(feature_names)} features.\n")
    md.append("| Model | AUC [95% CI] | Sensitivity | Specificity | PPV | NPV | Accuracy |")
    md.append("|-------|--------------|-------------|-------------|-----|-----|----------|")
    for name, r in results["models"].items():
        a, op = r["auc"], r["operating_point"]
        md.append(f"| {name} | {a['auc']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] "
                  f"| {op['sensitivity']:.3f} | {op['specificity']:.3f} "
                  f"| {op['ppv']:.3f} | {op['npv']:.3f} | {op['accuracy']:.3f} |")
    md.append("")
    for name, r in results["models"].items():
        op = r["operating_point"]
        md.append(f"**{name} confusion matrix** (Youden threshold "
                  f"{op['threshold']:.3f}): "
                  f"TP={op['tp']} FP={op['fp']} FN={op['fn']} TN={op['tn']}")
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "results.md")

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable (%s) — skipping figures.", e)
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(5, 5))
    for name, r in results["models"].items():
        a = r["auc"]
        plt.plot(r["roc"]["fpr"], r["roc"]["tpr"],
                 label=f"{name} (AUC={a['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title("PD vs HC — ROC (pooled CV)"); plt.legend(loc="lower right")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(fig_dir / "roc.png", dpi=150); plt.close()

    # feature importance of the primary (logistic) model
    primary = results["models"].get("logistic_regression")
    if primary and primary["feature_importance"]:
        items = sorted(primary["feature_importance"].items(),
                       key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        plt.figure(figsize=(6, max(3, 0.4 * len(names))))
        plt.barh(names[::-1], vals[::-1], color="#2b8cbe")
        plt.xlabel("|standardised coefficient|")
        plt.title("Feature importance (logistic regression)")
        plt.tight_layout(); plt.savefig(fig_dir / "feature_importance.png", dpi=150)
        plt.close()
    logger.info("Figures in %s", fig_dir)


def main():
    _require_sklearn()
    import pandas as pd

    ap = argparse.ArgumentParser(description="Train/evaluate PD-vs-HC classifier.")
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "cohort.csv"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"No cohort at {data_path}. Run prepare_ppmi.py first.")
    df = pd.read_csv(data_path)
    feature_names = [c for c in df.columns if c not in ("PATNO", "label")]
    if not feature_names:
        raise SystemExit("No feature columns in cohort.csv.")

    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise SystemExit("Cohort has only one class — check label assembly.")
    logger.info("Cohort: %d subjects (%d PD, %d HC), %d features.",
                len(y), int((y == 1).sum()), int((y == 0).sum()), len(feature_names))

    models = build_models(args.seed)
    out = {"models": {}, "features": feature_names,
           "n_pd": int((y == 1).sum()), "n_hc": int((y == 0).sum()),
           "folds": args.folds}
    for name, model in models.items():
        out["models"][name] = evaluate_model(
            name, model, X, y, feature_names, args.folds, args.seed)

    summary = {"n_pd": out["n_pd"], "n_hc": out["n_hc"]}
    write_outputs(out, feature_names, summary, Path(args.out_dir))
    logger.info("Done. Results in %s", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
