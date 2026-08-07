#!/usr/bin/env python3
"""B-2 Headline intersectional subgroup computation (reproducible).

The manuscript's headline fairness finding is "low-SES non-native immigrant
students are the most underserved subgroup (AUC 0.779 vs 0.880, gap 0.101,
F1 0.750 vs 0.484)". The prior pipeline (15_fairness_evaluation.py) only
emitted pairwise two-variable intersections, so this merged "non-native x Q1"
headline group was not reproducible from pipeline outputs.

This script computes the merged headline groups and persists them, so the
claim is directly reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def main() -> int:
    require_package("pandas", "pip install -r requirements.txt")
    require_package("numpy", "pip install -r requirements.txt")
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])

    df = load_table(processed_dir / "predictions" / "holdout_predictions.parquet")

    escs = df["ESCS"].copy()
    valid = escs.notna()
    escs_q = pd.Series("missing", index=df.index)
    escs_q[valid] = pd.qcut(escs[valid].rank(method="first"), 4,
                            labels=["Q1", "Q2", "Q3", "Q4"])
    df["ESCS_Q"] = escs_q

    y = df["LOW_PERFORMER_MATH"].astype(int).values
    p = df["best_classification_score"].values
    w = (df["W_FSTUWT"] / df["W_FSTUWT"].mean()).values

    s = df["IMMIG"].astype(str).str.strip().str.lower()
    nonnative = s.str.startswith(("first", "second")).fillna(False).values
    native = s.str.startswith("native").fillna(False).values
    q1 = (escs_q == "Q1").values
    q4 = (escs_q == "Q4").values

    def weighted_metrics(mask):
        n = int(mask.sum())
        pred = (p[mask] >= 0.5).astype(int)
        auc = roc_auc_score(y[mask], p[mask], sample_weight=w[mask])
        f1 = f1_score(y[mask], pred, sample_weight=w[mask])
        return n, auc, f1

    rows = []
    for label, mask in [
        ("low-SES non-native immigrant (Q1 x non-native, merged)", nonnative & q1),
        ("high-SES native (Q4 x native)", native & q4),
        ("low-SES (Q1, all)", q1),
        ("high-SES (Q4, all)", q4),
        ("non-native (all)", nonnative),
        ("native (all)", native),
    ]:
        n, auc, f1 = weighted_metrics(mask)
        rows.append({"group": label, "n_holdout": n,
                     "weighted_auc": round(auc, 4), "weighted_f1": round(f1, 4)})
        print(f"{label}: n={n}, AUC={auc:.4f}, F1={f1:.4f}")

    out = pd.DataFrame(rows)
    out_path = tables_dir / "headline_intersectional.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    low = out[out["group"].str.startswith("low-SES non")].iloc[0]
    high = out[out["group"].str.startswith("high-SES")].iloc[0]
    print(f"\nHEADLINE: AUC gap = {high['weighted_auc'] - low['weighted_auc']:.3f} "
          f"({low['weighted_auc']} vs {high['weighted_auc']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
