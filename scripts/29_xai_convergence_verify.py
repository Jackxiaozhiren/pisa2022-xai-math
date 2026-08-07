#!/usr/bin/env python3
"""B-3 XAI convergence recomputation (clean, correctly-labelled).

Recomputes the multi-method / cross-model rank-correlation values that the
manuscript reports as "SHAP cross-model rho=0.83" and "ALE rho=0.76".

Fixes two defects in the prior pipeline:
  1. 12_multi_xai_comparison.py labelled SHAP(XGBoost) vs Perm(LightGBM) as
     "SHAP vs Permutation" and the manuscript read that as cross-model SHAP.
  2. ALE-vs-SHAP rho was never computed or saved (only a plot existed).

Output: reports/tables/xai_convergence_verified.csv + prints the rho values
needed to update the manuscript.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def main() -> int:
    require_package("joblib", "pip install joblib")
    require_package("numpy", "pip install -r requirements.txt")
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr, kendalltau

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    model_dir = processed_dir / "models"

    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features].copy()
    for c in x.columns:
        if str(x[c].dtype) == "category":
            x[c] = x[c].cat.codes.astype("int8")

    seed = config["sample"]["random_state"]
    n_sample = 5000
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n_sample, replace=False)
    x_s = x.iloc[idx]

    import shap

    models = {
        "xgboost_tuned": joblib.load(model_dir / "classification_xgboost_tuned.joblib"),
        "lightgbm_tuned": joblib.load(model_dir / "classification_lightgbm_tuned.joblib"),
    }

    shap_imps = {}
    for name, model in models.items():
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(x_s.values)
        if isinstance(vals, list):
            vals = vals[1]
        if np.ndim(vals) == 3:
            vals = vals[:, :, 1]
        imp = pd.DataFrame({
            "feature": features,
            f"shap_importance_{name}": np.abs(vals).mean(axis=0),
        })
        imp[f"shap_rank_{name}"] = imp[f"shap_importance_{name}"].rank(ascending=False, method="min").astype(int)
        shap_imps[name] = imp
        print(f"{name}: SHAP computed on deterministic {n_sample}-row sample (seed {seed})")

    # Permutation importance for the same tuned XGBoost model
    perm_path = tables_dir / "classification_xgboost_tuned_permutation_importance.csv"
    perm = pd.read_csv(perm_path).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm["perm_rank_xgb"] = perm["importance_mean"].rank(ascending=False, method="min").astype(int)

    # ALE importance (computed on the same tuned XGBoost model by 14_ale_analysis.py)
    ale = pd.read_csv(tables_dir / "ale_importance_summary.csv")
    ale["ale_rank"] = ale["ale_abs_mean"].rank(ascending=False, method="min").astype(int)

    # LIME ranks from the existing multi-XAI comparison
    lime = pd.read_csv(tables_dir / "multi_xai_comparison.csv")[["feature", "lime_rank"]]

    # Merge all rankings on feature
    merged = shap_imps["xgboost_tuned"][["feature", "shap_rank_xgboost_tuned"]]
    merged = merged.merge(shap_imps["lightgbm_tuned"][["feature", "shap_rank_lightgbm_tuned"]], on="feature")
    merged = merged.merge(perm[["feature", "perm_rank_xgb"]], on="feature")
    merged = merged.merge(ale[["feature", "ale_rank"]], on="feature")
    merged = merged.merge(lime, on="feature")
    n_feat = len(merged)

    def corr(a, b):
        a = merged[a].astype(float)
        b = merged[b].astype(float)
        rho, p = spearmanr(a, b)
        tau, _ = kendalltau(a, b)
        return rho, p, tau

    comparisons = {
        "SHAP(XGB) vs Permutation(XGB)": corr("shap_rank_xgboost_tuned", "perm_rank_xgb"),
        "SHAP(XGB) vs SHAP(LGBM) [cross-model]": corr("shap_rank_xgboost_tuned", "shap_rank_lightgbm_tuned"),
        "SHAP(XGB) vs ALE(XGB)": corr("shap_rank_xgboost_tuned", "ale_rank"),
        "SHAP(XGB) vs LIME": corr("shap_rank_xgboost_tuned", "lime_rank"),
    }

    rows = []
    for label, (rho, p, tau) in comparisons.items():
        rows.append({"comparison": label, "spearman_r": rho, "spearman_p": p,
                     "kendall_tau": tau, "n_features": n_feat})
        print(f"{label}: Spearman rho={rho:.4f} (p={p:.2e}), Kendall tau={tau:.4f}")

    out = pd.DataFrame(rows)
    out_path = tables_dir / "xai_convergence_verified.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Also persist the merged ranking matrix for the B-4 convergence figure
    merged.to_csv(tables_dir / "xai_rankings_verified.csv", index=False)
    print(f"Saved rankings: {tables_dir / 'xai_rankings_verified.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
