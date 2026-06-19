"""
prepare_ppmi.py
===============
Assemble a clean PD-vs-Healthy-Control cohort table from the raw PPMI
Study-Data CSV exports. The output (`cohort.csv`) is the single input to
`train_eval.py`.

PPMI file/column names drift between releases, so this loader is deliberately
tolerant: it scans every CSV in --raw-dir and locates the tables it needs by
matching *column patterns* rather than hard-coded filenames.

What it joins (all keyed on PATNO at the baseline visit):
  * Label   — Participant Status: COHORT_DEFINITION ("Parkinson's Disease" vs
              "Healthy Control"). Prodromal / SWEDD are excluded by default.
  * DAT-SPECT striatal binding ratios — the strongest features:
              caudate (L/R), putamen (L/R), plus engineered summaries
              (mean / most-affected-side / caudate-putamen ratio / asymmetry).
  * MDS-UPDRS Part III total (motor exam)            [optional]
  * MoCA total (cognition)                           [optional]
  * Demographics: sex, age-at-visit                  [optional]

Baseline visit is chosen per PATNO by EVENT_ID preference (SC, BL, V01, ...);
if a table has no EVENT_ID, the first row per PATNO is used.

Usage
-----
    python -m pd_clinical.eval.prepare_ppmi --raw-dir path/to/PPMI_csvs
    # -> pd_clinical/eval/data/cohort.csv  +  summary.json

Download checklist (LONI IDA -> PPMI -> Download -> Study Data):
    Participant_Status.csv          (labels — REQUIRED)
    DaTScan / SBR analysis CSV      (features — REQUIRED, e.g. "DATScan_Analysis"
                                     or "Xing_Core_Lab_-_Quant_SBR")
    MDS_UPDRS_Part_III.csv          (optional)
    Montreal_Cognitive_Assessment__MoCA_.csv   (optional)
    Demographics.csv / Age tables   (optional)
Put them all in one folder and point --raw-dir at it. Exact filenames may
differ by release — matching is by column content, so that's fine.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_ppmi")

# EVENT_ID preference for the "baseline" visit (first available wins).
VISIT_ORDER = ["SC", "BL", "V01", "V02", "V03", "V04"]


def _load_csvs(raw_dir: Path) -> Dict[str, "pd.DataFrame"]:
    import pandas as pd
    # search recursively — the PPMI "all tabular" export nests CSVs in
    # per-category subfolders (Imaging/, Motor Assessments/, ...).
    files = [p for p in raw_dir.rglob("*") if p.suffix.lower() == ".csv"
             and "archived" not in p.name.lower()]   # prefer current over -Archived
    if not files:
        raise SystemExit(f"No CSV files found under {raw_dir} (searched recursively).")
    out = {}
    for f in files:
        try:
            # key by name; if duplicate names across folders, keep the larger file
            df = pd.read_csv(f, low_memory=False)
            if f.name not in out or len(df) > len(out[f.name]):
                out[f.name] = df
        except Exception as e:
            logger.warning("Could not read %s: %s", f.name, e)
    logger.info("Loaded %d CSV files (recursive) from %s", len(out), raw_dir)
    return out


def _cols_ci(df) -> Dict[str, str]:
    """Map UPPERCASE column name -> actual column name (case-insensitive lookup)."""
    return {c.upper(): c for c in df.columns}


def _find_table(dfs: Dict[str, "pd.DataFrame"], required_upper: List[str]):
    """Return the first dataframe whose columns include all `required_upper`."""
    for name, df in dfs.items():
        cols = set(_cols_ci(df).keys())
        if all(any(req in c for c in cols) for req in required_upper):
            return name, df
    return None, None


def _pick_baseline(df):
    """Reduce to one row per PATNO at the preferred visit."""
    import pandas as pd
    ci = _cols_ci(df)
    patno = ci.get("PATNO")
    if patno is None:
        return df
    if "EVENT_ID" in ci:
        ev = ci["EVENT_ID"]
        rank = {v: i for i, v in enumerate(VISIT_ORDER)}
        df = df.copy()
        df["_rank"] = df[ev].map(lambda x: rank.get(str(x).strip().upper(), 999))
        df = df.sort_values([patno, "_rank"]).groupby(patno, as_index=False).first()
        df = df.drop(columns=["_rank"])
    else:
        df = df.drop_duplicates(subset=[patno], keep="first")
    return df


# --------------------------------------------------------------------------- #
#  Label assembly
# --------------------------------------------------------------------------- #
def build_labels(dfs) -> "pd.DataFrame":
    import pandas as pd
    # prefer the table with the explicit COHORT_DEFINITION text column
    name, df = _find_table(dfs, ["PATNO", "COHORT_DEFINITION"])
    if df is None:
        name, df = _find_table(dfs, ["PATNO", "COHORT"])
    if df is None:
        raise SystemExit(
            "Could not find a Participant Status table (needs PATNO + COHORT/"
            "COHORT_DEFINITION). Download Participant_Status.csv.")
    logger.info("Labels from: %s", name)
    ci = _cols_ci(df)
    patno = ci["PATNO"]

    # Prefer the text definition; fall back to numeric COHORT (1=PD, 2=HC).
    def_col = next((ci[c] for c in ci if "COHORT_DEFINITION" in c), None)
    rows = []
    for _, r in df.iterrows():
        label = None
        if def_col is not None:
            d = str(r[def_col]).lower()
            if "parkinson" in d:
                label = 1
            elif "healthy" in d or "control" in d:
                label = 0
        if label is None:
            num = ci.get("COHORT")
            if num is not None:
                try:
                    v = int(float(r[num]))
                    label = 1 if v == 1 else (0 if v == 2 else None)
                except Exception:
                    label = None
        if label is not None:
            rows.append({"PATNO": r[patno], "label": label})
    out = pd.DataFrame(rows).drop_duplicates("PATNO")
    logger.info("Labels: %d PD, %d HC (Prodromal/SWEDD excluded).",
                int((out.label == 1).sum()), int((out.label == 0).sum()))
    return out


# --------------------------------------------------------------------------- #
#  Feature assembly
# --------------------------------------------------------------------------- #
def build_datscan(dfs) -> Optional["pd.DataFrame"]:
    import numpy as np
    import pandas as pd
    # require the SBR table: must actually carry PUTAMEN binding-ratio columns
    name, df = _find_table(dfs, ["PATNO", "PUTAMEN"])
    if df is None:
        name, df = _find_table(dfs, ["PATNO", "DATSCAN"])
    if df is None:
        logger.warning("No DaTscan/SBR table found — features will be weak.")
        return None
    logger.info("DaTscan from: %s", name)
    df = _pick_baseline(df)
    ci = _cols_ci(df)
    patno = ci["PATNO"]

    def col(*pats):
        for up, actual in ci.items():
            if all(p in up for p in pats):
                return actual
        return None

    cr, cl = col("CAUDATE", "R"), col("CAUDATE", "L")
    pr, pl = col("PUTAMEN", "R"), col("PUTAMEN", "L")
    # exclude anterior-putamen specific cols from the main L/R if a plain one exists
    feats = pd.DataFrame({"PATNO": df[patno]})
    for nm, c in [("caudate_r", cr), ("caudate_l", cl),
                  ("putamen_r", pr), ("putamen_l", pl)]:
        if c is not None:
            feats[nm] = pd.to_numeric(df[c], errors="coerce")

    # engineered, clinically meaningful summaries
    if {"caudate_r", "caudate_l"}.issubset(feats.columns):
        feats["caudate_mean"] = feats[["caudate_r", "caudate_l"]].mean(axis=1)
    if {"putamen_r", "putamen_l"}.issubset(feats.columns):
        feats["putamen_mean"] = feats[["putamen_r", "putamen_l"]].mean(axis=1)
        feats["putamen_min"] = feats[["putamen_r", "putamen_l"]].min(axis=1)  # most-affected side
        denom = feats[["putamen_r", "putamen_l"]].sum(axis=1).replace(0, np.nan)
        feats["putamen_asym"] = (feats["putamen_r"] - feats["putamen_l"]).abs() / denom
    if {"putamen_mean", "caudate_mean"}.issubset(feats.columns):
        feats["putamen_caudate_ratio"] = feats["putamen_mean"] / feats["caudate_mean"].replace(0, np.nan)
    return feats


def build_optional(dfs) -> Dict[str, "pd.DataFrame"]:
    import pandas as pd
    out = {}

    # MDS-UPDRS Part III total
    name, df = _find_table(dfs, ["PATNO", "NP3"])
    if df is not None:
        df = _pick_baseline(df)
        ci = _cols_ci(df)
        tot = next((ci[c] for c in ci if c in ("NP3TOT", "MDS_UPDRS_PART_III")), None)
        sub = pd.DataFrame({"PATNO": df[ci["PATNO"]]})
        if tot:
            sub["updrs3_total"] = pd.to_numeric(df[tot], errors="coerce")
        else:  # sum NP3* item columns
            np3 = [ci[c] for c in ci if c.startswith("NP3")]
            if np3:
                sub["updrs3_total"] = df[np3].apply(
                    pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        if "updrs3_total" in sub.columns:
            out["updrs"] = sub
            logger.info("UPDRS-III from: %s", name)

    # MoCA
    name, df = _find_table(dfs, ["PATNO", "MCATOT"])
    if df is not None:
        df = _pick_baseline(df)
        ci = _cols_ci(df)
        out["moca"] = pd.DataFrame({"PATNO": df[ci["PATNO"]],
                                    "moca_total": pd.to_numeric(df[ci["MCATOT"]],
                                                                errors="coerce")})
        logger.info("MoCA from: %s", name)

    # Demographics: sex + age
    name, df = _find_table(dfs, ["PATNO", "SEX"])
    if df is None:
        name, df = _find_table(dfs, ["PATNO", "GENDER"])
    if df is not None:
        df = _pick_baseline(df)
        ci = _cols_ci(df)
        sub = pd.DataFrame({"PATNO": df[ci["PATNO"]]})
        sexc = ci.get("SEX") or ci.get("GENDER")
        if sexc:
            sub["sex_male"] = df[sexc].map(
                lambda x: 1 if str(x).strip().upper() in ("M", "MALE", "1") else 0)
        agec = ci.get("AGE") or ci.get("AGE_AT_VISIT") or next(
            (ci[c] for c in ci if "AGE" in c and "AGEONSET" not in c), None)
        if agec:
            sub["age"] = pd.to_numeric(df[agec], errors="coerce")
        if len(sub.columns) > 1:
            out["demo"] = sub
            logger.info("Demographics from: %s", name)

    return out


# --------------------------------------------------------------------------- #
#  Assemble + write
# --------------------------------------------------------------------------- #
def build(raw_dir: str, out_dir: str) -> Dict:
    import pandas as pd
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dfs = _load_csvs(raw)
    labels = build_labels(dfs)
    datscan = build_datscan(dfs)
    optional = build_optional(dfs)

    cohort = labels
    if datscan is not None:
        cohort = cohort.merge(datscan, on="PATNO", how="left")
    for sub in optional.values():
        cohort = cohort.merge(sub, on="PATNO", how="left")

    # drop rows with no usable features at all
    feat_cols = [c for c in cohort.columns if c not in ("PATNO", "label")]
    cohort = cohort.dropna(subset=feat_cols, how="all")

    cohort.to_csv(out / "cohort.csv", index=False)

    summary = {
        "n_total":    int(len(cohort)),
        "n_pd":       int((cohort.label == 1).sum()),
        "n_hc":       int((cohort.label == 0).sum()),
        "features":   feat_cols,
        "missingness": {c: float(cohort[c].isna().mean().round(3)) for c in feat_cols},
        "raw_dir":    str(raw.resolve()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Cohort written: %s", (out / "cohort.csv").resolve())
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    if summary["n_pd"] < 50 or summary["n_hc"] < 50:
        logger.warning("Small cohort (PD=%d, HC=%d) — check that the right CSVs "
                       "were downloaded.", summary["n_pd"], summary["n_hc"])
    return summary


def main():
    ap = argparse.ArgumentParser(description="Assemble PPMI PD-vs-HC cohort table.")
    ap.add_argument("--raw-dir", required=True,
                    help="Folder containing the raw PPMI Study-Data CSV exports.")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "data"))
    args = ap.parse_args()
    build(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
