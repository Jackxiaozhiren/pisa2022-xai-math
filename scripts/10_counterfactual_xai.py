#!/usr/bin/env python3
"""Counterfactual XAI analysis using SHAP-based approximate counterfactuals.

For low-performing students, computes:
1. Which features would need to change (and by how much) for the model
   to predict "not low-performing"
2. Counterfactual reachability: how large the required changes are
   relative to realistic variation in each feature

Reference: Wachter et al. (2018), Khine et al. (2025)
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

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load best model and data ──────────────────────────────────
    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    model_path = model_dir / f"classification_{best['best_classification_model']}.joblib"
    if not model_path.exists():
        model_path = model_dir / "classification_lightgbm.joblib"
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features]

    # Convert category columns to numeric
    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    x_num = x.copy()
    for c in cat_cols:
        x_num[c] = x_num[c].cat.codes.astype("int8")

    # ── SHAP-based approximate counterfactuals ────────────────────
    require_package("shap", "pip install shap")
    y = df["LOW_PERFORMER_MATH"]
    import shap

    # Handle both Pipeline and raw estimator models
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocess")
        estimator = model.named_steps.get("model")
    else:
        preprocessor = None
        estimator = model

    if preprocessor is not None and estimator is not None:
        x_transformed = preprocessor.transform(x_num)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()
        try:
            feature_names = np.array(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(x_transformed.shape[1])])
    else:
        # No pipeline — use estimator directly
        estimator = model
        x_transformed = x_num.values
        feature_names = np.array(features)

    # Compute SHAP values on a representative sample
    n_shap = min(10000, len(x_transformed))
    rng_cf = np.random.default_rng(config["sample"]["random_state"])
    shap_idx = rng_cf.choice(len(x_transformed), size=n_shap, replace=False)
    x_shap = x_transformed[shap_idx]
    low_shap_mask = y.values[shap_idx] == 1

    print(f"Computing SHAP values on {n_shap}-row sample", flush=True)
    try:
        explainer = shap.Explainer(estimator, x_shap)
        shap_values = explainer(x_shap, check_additivity=False)
    except Exception:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(x_shap)

    if getattr(shap_values, "values", None) is not None and shap_values.values.ndim == 3:
        shap_vals = shap_values.values[:, :, 1]  # class 1 (low-performer)
    else:
        shap_vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    # ── For low-performing students in sample: compute counterfactual features ─
    low_shap = shap_vals[low_shap_mask]
    low_indices = df.index[shap_idx][low_shap_mask]
    low_x_sample = x_shap[low_shap_mask]

    n_low = min(2000, low_shap.shape[0])
    sample_idx = rng_cf.choice(low_shap.shape[0], size=n_low, replace=False)

    cf_rows = []
    preds_full = model.predict_proba(x_num)[:, 1]

    for i, idx in enumerate(sample_idx):
        global_idx = low_indices[idx]
        shap_i = low_shap[idx]
        base_pred = preds_full[shap_idx][low_shap_mask][idx]

        # Get top-5 features pushing this student toward "low-performer"
        top_indices = np.argsort(shap_i)[-10:]  # top 10 contributors to class 1
        for rank, feat_i in enumerate(reversed(top_indices)):
            cf_rows.append(
                {
                    "student_idx": int(global_idx),
                    "predicted_probability": float(base_pred),
                    "feature": str(feature_names[feat_i]),
                    "shap_contribution": float(shap_i[feat_i]),
                    "rank": rank + 1,
                }
            )

    cf_detail = pd.DataFrame(cf_rows)

    # ── Aggregate: which features are most targeted for counterfactuals ─
    agg = (
        cf_detail.groupby("feature")
        .agg(
            mean_shap_contribution=("shap_contribution", "mean"),
            std_shap_contribution=("shap_contribution", "std"),
            median_shap_contribution=("shap_contribution", "median"),
            n_appearances=("student_idx", "nunique"),
            pct_students=("student_idx", lambda x: len(x) / n_low),
        )
        .sort_values("pct_students", ascending=False)
        .reset_index()
    )

    # ── Counterfactual magnitude: for top features, estimate change needed ─
    cf_magnitude = []
    for feat_name in agg.head(20)["feature"]:
        feat_idx = None
        for fi, fn in enumerate(feature_names):
            if feat_name in str(fn) or str(fn) in feat_name:
                feat_idx = fi
                break
        if feat_idx is None:
            continue

        feat_vals = low_x_sample[:, feat_idx]
        feat_shap = low_shap[:, feat_idx]

        # Mean change needed to offset positive SHAP contribution
        positive_contrib = feat_shap > 0
        if positive_contrib.sum() > 0:
            mean_change = feat_shap[positive_contrib].mean()
            # normalize by feature std for comparability
            feat_std = feat_vals.std()
            if feat_std > 0:
                normalized_change = mean_change / feat_std
            else:
                normalized_change = float("nan")
        else:
            mean_change = 0.0
            normalized_change = 0.0

        cf_magnitude.append(
            {
                "feature": feat_name,
                "mean_shap_for_low_perf": float(feat_shap.mean()),
                "mean_required_change": float(mean_change),
                "normalized_required_change": float(normalized_change),
                "n_low_students": int(low_shap.shape[0]),
            }
        )

    magnitude_df = pd.DataFrame(cf_magnitude).sort_values(
        "normalized_required_change", ascending=True
    )

    # ── Counterfactual by ESCS ────────────────────────────────────
    if "ESCS" in df.columns:
        low_escs = df.loc[low_indices, "ESCS"]
        escs_quintile = pd.qcut(low_escs.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        cf_detail_with_escs = cf_detail.merge(
            pd.DataFrame({"ESCS_QUINTILE": escs_quintile}),
            left_on="student_idx",
            right_index=True,
            how="inner",
        )
        escs_reach = (
            cf_detail_with_escs.groupby("ESCS_QUINTILE", observed=False)
            .agg(
                mean_shap=("shap_contribution", "mean"),
                n_students=("student_idx", "nunique"),
            )
            .reset_index()
        )
        escs_reach.to_csv(tables_dir / "counterfactual_reachability_by_escs.csv", index=False)

    # ── Save outputs ──────────────────────────────────────────────
    cf_detail.to_csv(tables_dir / "counterfactual_detail.csv", index=False)
    agg.to_csv(tables_dir / "counterfactual_aggregate.csv", index=False)
    magnitude_df.to_csv(tables_dir / "counterfactual_magnitude.csv", index=False)

    print("Top-10 counterfactual features (by % of students):", flush=True)
    print(agg.head(10)[["feature", "pct_students", "mean_shap_contribution"]].to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    # Plot 1: Top counterfactual features
    top15 = agg.head(15).sort_values("pct_students")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top15["feature"], top15["pct_students"] * 100)
    ax.set_xlabel("% of low-performing students where feature is a top contributor")
    ax.set_title(
        "Top-15 Features Driving Low-Performer Predictions\n(Counterfactual Targets)"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "counterfactual_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Counterfactual change required
    mag10 = magnitude_df.head(10).sort_values("normalized_required_change", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = [
        "#d62728" if v < 0 else "#2ca02c" for v in mag10["normalized_required_change"]
    ]
    ax2.barh(mag10["feature"], mag10["normalized_required_change"], color=colors)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Normalized required change (SD units)")
    ax2.set_title(
        "Counterfactual Change Required to Shift Low-Performer Prediction\n(negative = reducing this feature helps)"
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "counterfactual_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: ESCS reachability
    if "ESCS" in df.columns:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(
            escs_reach["ESCS_QUINTILE"].astype(str), escs_reach["mean_shap"], color="#1f77b4"
        )
        ax3.set_xlabel("ESCS Quintile")
        ax3.set_ylabel("Mean SHAP Contribution (Low-Performer Direction)")
        ax3.set_title(
            "Counterfactual Burden by SES Quintile\n(higher = more features pushing toward low-performance)"
        )
        plt.tight_layout()
        plt.savefig(figures_dir / "counterfactual_reachability_escs.png", dpi=300, bbox_inches="tight")
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
