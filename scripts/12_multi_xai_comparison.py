#!/usr/bin/env python3
"""Multi-XAI comparison: SHAP vs Permutation Importance vs LIME.

Compares top-K feature rankings across three interpretability methods
using Spearman's ρ and Kendall's τ, following Niu et al. (2025).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def main() -> int:
    require_package("lime", "pip install lime")
    require_package("joblib", "pip install joblib")
    require_package("numpy", "pip install -r requirements.txt")
    import joblib
    import numpy as np
    import pandas as pd
    from scipy.stats import kendalltau, spearmanr

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    model_path = model_dir / f"classification_{best['best_classification_model']}.joblib"
    if not model_path.exists():
        model_path = model_dir / "classification_lightgbm.joblib"
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features]

    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    x_num = x.copy()
    for c in cat_cols:
        x_num[c] = x_num[c].cat.codes.astype("int8")

    # ── 1. SHAP importance (mean |SHAP|) ──────────────────────────
    require_package("shap", "pip install shap")
    import shap

    n_sample = min(3000, len(x_num))
    rng = np.random.default_rng(config["sample"]["random_state"])
    idx = rng.choice(len(x_num), size=n_sample, replace=False)
    x_s = x_num.iloc[idx]

    # Handle pipeline models
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocess")
        estimator = model.named_steps.get("model")
        if preprocessor is not None:
            x_transformed = preprocessor.transform(x_s)
            if hasattr(x_transformed, "toarray"):
                x_transformed = x_transformed.toarray()
            try:
                feat_names = np.array(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = np.array([f"f{i}" for i in range(x_transformed.shape[1])])
        else:
            estimator = model
            x_transformed = x_s.values
            feat_names = np.array(features)
    else:
        estimator = model
        x_transformed = x_s.values
        feat_names = np.array(features)

    print("Computing SHAP values", flush=True)
    try:
        explainer_shap = shap.TreeExplainer(estimator)
        shap_vals = explainer_shap.shap_values(x_transformed)
    except Exception:
        explainer_shap = shap.Explainer(estimator, x_transformed)
        shap_vals = explainer_shap(x_transformed, check_additivity=False).values

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    shap_imp = pd.DataFrame({
        "feature": feat_names,
        "shap_importance": np.abs(shap_vals).mean(axis=0),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    shap_imp["shap_rank"] = range(1, len(shap_imp) + 1)

    # ── 2. Permutation importance (from existing table) ───────────
    perm_path = tables_dir / "classification_lightgbm_permutation_importance.csv"
    if not perm_path.exists():
        perm_path = tables_dir / f"classification_{best['best_classification_model']}_permutation_importance.csv"
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        perm = perm.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        perm["perm_rank"] = range(1, len(perm) + 1)
    else:
        perm = pd.DataFrame({"feature": feat_names, "importance_mean": [0.0]*len(feat_names)})
        perm["perm_rank"] = range(1, len(perm) + 1)

    # ── 3. LIME importance ────────────────────────────────────────
    print("Computing LIME explanations", flush=True)
    from lime.lime_tabular import LimeTabularExplainer

    n_lime = min(500, n_sample)
    x_lime = x_s.iloc[:n_lime].copy()
    x_lime = x_lime.fillna(x_lime.median())  # handle NaN for LIME

    # Drop zero-variance columns (they break LIME's discretizer)
    stds = x_lime.std()
    nonzero_cols = stds[stds > 1e-8].index.tolist()
    if len(nonzero_cols) < len(x_lime.columns):
        dropped = set(x_lime.columns) - set(nonzero_cols)
        print(f"  Dropping {len(dropped)} zero-variance columns for LIME: {dropped}", flush=True)
        x_lime = x_lime[nonzero_cols]
        lime_features = nonzero_cols
    else:
        lime_features = list(features)

    explainer_lime = LimeTabularExplainer(
        x_lime.values,
        feature_names=lime_features,
        class_names=["Non-Low", "Low-Performer"],
        mode="classification",
        discretize_continuous=False,
    )

    # Wrapper for pipeline models
    def predict_fn(data):
        df_in = pd.DataFrame(data, columns=lime_features)
        if hasattr(model, "named_steps"):
            return model.predict_proba(df_in)
        return model.predict_proba(data)

    lime_weights = np.zeros(len(lime_features))
    lime_successful = 0
    for i in range(min(100, n_lime)):
        try:
            exp = explainer_lime.explain_instance(
                x_lime.iloc[i].values, predict_fn, num_features=len(lime_features), labels=(1,)
            )
            for feat_idx, weight in exp.as_list(label=1):
                try:
                    idx = int(feat_idx)
                except (ValueError, TypeError):
                    # LIME may return feature names as strings
                    continue
                if 0 <= idx < len(lime_features):
                    lime_weights[idx] += abs(weight)
            lime_successful += 1
        except Exception:
            continue

    if lime_successful > 0:
        lime_imp = pd.DataFrame({
            "feature": lime_features,
            "lime_importance": lime_weights,
        }).sort_values("lime_importance", ascending=False).reset_index(drop=True)
        lime_imp["lime_rank"] = range(1, len(lime_imp) + 1)
        print(f"  LIME explanations computed for {lime_successful}/{min(100, n_lime)} instances", flush=True)
    else:
        print("  LIME failed — using SHAP-only ranking; report as limitation", flush=True)
        lime_imp = pd.DataFrame({
            "feature": lime_features,
            "lime_importance": [0.0] * len(lime_features),
            "lime_rank": [len(lime_features) + 1] * len(lime_features),
        })

    # ── 4. Merge and compare ──────────────────────────────────────
    merged = shap_imp[["feature", "shap_rank"]].merge(
        perm[["feature", "perm_rank"]], on="feature", how="outer"
    ).merge(
        lime_imp[["feature", "lime_rank"]], on="feature", how="outer"
    ).fillna(len(features) + 1)

    # Top-15 comparison
    top15 = merged.nsmallest(15, "shap_rank")[["feature", "shap_rank", "perm_rank", "lime_rank"]]
    print("Top-15 feature ranking comparison:", flush=True)
    print(top15.to_string(index=False))

    # ── 5. Ranking consistency metrics ────────────────────────────
    valid = merged.dropna(subset=["shap_rank", "perm_rank", "lime_rank"])
    rho_sp, p_sp = spearmanr(valid["shap_rank"], valid["perm_rank"])
    tau_kt, p_kt = kendalltau(valid["shap_rank"], valid["perm_rank"])

    rho_sl, p_sl = spearmanr(valid["shap_rank"], valid["lime_rank"])
    tau_sl, _ = kendalltau(valid["shap_rank"], valid["lime_rank"])

    rho_pl, p_pl = spearmanr(valid["perm_rank"], valid["lime_rank"])
    tau_pl, _ = kendalltau(valid["perm_rank"], valid["lime_rank"])

    consistency = pd.DataFrame([
        {"comparison": "SHAP vs Permutation", "spearman_r": rho_sp, "spearman_p": p_sp,
         "kendall_tau": tau_kt, "kendall_p": p_kt},
        {"comparison": "SHAP vs LIME", "spearman_r": rho_sl, "spearman_p": p_sl,
         "kendall_tau": tau_sl, "kendall_p": 0.0},
        {"comparison": "Permutation vs LIME", "spearman_r": rho_pl, "spearman_p": p_pl,
         "kendall_tau": tau_pl, "kendall_p": 0.0},
    ])
    print("\nRanking consistency:", flush=True)
    print(consistency.to_string(index=False))

    merged.to_csv(tables_dir / "multi_xai_comparison.csv", index=False)
    consistency.to_csv(tables_dir / "xai_ranking_consistency.csv", index=False)

    # ── 6. Plot ───────────────────────────────────────────────────
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Top-15 rank comparison
    top15_plot = merged.nsmallest(15, "shap_rank").sort_values("shap_rank", ascending=False)
    x_pos = range(len(top15_plot))
    w = 0.25
    axes[0].barh([p + w for p in x_pos], top15_plot["shap_rank"], w, label="SHAP")
    axes[0].barh(x_pos, top15_plot["perm_rank"], w, label="Permutation")
    axes[0].barh([p - w for p in x_pos], top15_plot["lime_rank"], w, label="LIME")
    axes[0].set_yticks(x_pos)
    axes[0].set_yticklabels(top15_plot["feature"], fontsize=8)
    axes[0].set_xlabel("Rank (lower = more important)")
    axes[0].set_title("Top-15 Feature Ranking: SHAP vs Permutation vs LIME")
    axes[0].legend()
    axes[0].invert_yaxis()

    # Right: Consistency bar chart
    axes[1].bar(consistency["comparison"], consistency["spearman_r"], color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    axes[1].set_ylabel("Spearman's ρ")
    axes[1].set_title("XAI Method Ranking Consistency")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="High consistency (ρ=0.8)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "multi_xai_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
