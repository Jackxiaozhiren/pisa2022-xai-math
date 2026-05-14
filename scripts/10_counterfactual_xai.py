#!/usr/bin/env python3
"""Counterfactual XAI analysis using DiCE on the best classification model.

Generates counterfactual explanations for low-performing students, answering:
"What would need to change for this student to be predicted as NOT low-performing?"

Reference: Mothilal et al. (2020), Khine et al. (2025)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def main() -> int:
    require_package("dice_ml", "pip install dice-ml")
    require_package("joblib", "pip install joblib")
    require_package("pandas", "pip install -r requirements.txt")
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
        # Fallback to lightgbm if tuned model not available
        model_path = model_dir / "classification_lightgbm.joblib"
        print("Tuned model not found; using default lightgbm", flush=True)
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features]

    # Convert category columns to numeric for DiCE compatibility
    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    x_num = x.copy()
    for c in cat_cols:
        x_num[c] = x_num[c].cat.codes.astype("int8")

    y = df["LOW_PERFORMER_MATH"]

    # ── Select low-performing students for counterfactual queries ─
    low_perf = df[df["LOW_PERFORMER_MATH"] == 1]
    n_cf = min(500, len(low_perf))
    cf_indices = low_perf.sample(n_cf, random_state=config["sample"]["random_state"]).index
    cf_x = x_num.loc[cf_indices]

    print(f"Generating counterfactuals for {n_cf} low-performing students", flush=True)

    # ── Build DiCE explainer ─────────────────────────────────────
    from dice_ml import Data, Dice, Model

    # Build a minimal tabular data description
    cf_data = x_num.copy()
    cf_data["LOW_PERFORMER_MATH"] = y.values

    # Identify continuous vs discrete features (heuristic: >10 unique values = continuous)
    continuous_features = []
    for col in features:
        n_unique = cf_data[col].nunique()
        if n_unique > 10:
            continuous_features.append(col)

    d = Data(
        dataframe=cf_data,
        continuous_features=continuous_features,
        outcome_name="LOW_PERFORMER_MATH",
    )

    # Wrap sklearn model for DiCE
    class SklearnWrapper:
        def __init__(self, model, features):
            self.model = model
            self.features = list(features)

        def predict(self, x):
            if isinstance(x, np.ndarray):
                x = pd.DataFrame(x, columns=self.features)
            proba = self.model.predict_proba(x)[:, 1]
            return proba

        def predict_proba(self, x):
            if isinstance(x, np.ndarray):
                x = pd.DataFrame(x, columns=self.features)
            return self.model.predict_proba(x)

    backend_model = Model(model=SklearnWrapper(model, features), backend="sklearn")
    explainer = Dice(d, backend_model, method="random")

    # ── Generate counterfactuals ─────────────────────────────────
    desired_class = 0  # "not low-performing"
    try:
        cf_explanations = explainer.generate_counterfactuals(
            cf_x,
            total_CFs=3,  # 3 counterfactuals per query instance
            desired_class=desired_class,
            proximity_weight=0.5,
            diversity_weight=0.5,
        )
        cf_df = cf_explanations.cf_examples_list[0].final_cfs_df
    except Exception as e:
        print(f"DiCE generation failed: {e}", flush=True)
        # Fallback: compute simple feature perturbation directions
        cf_rows = []
        preds = model.predict_proba(cf_x)[:, 1]
        for i, idx in enumerate(cf_indices[:100]):
            base_pred = preds[i]
            row = {"query_index": int(idx), "baseline_prediction": float(base_pred)}
            # Record feature values for the query instance
            for f in features:
                row[f"base_{f}"] = float(x_num.loc[idx, f])
            cf_rows.append(row)
        pd.DataFrame(cf_rows).to_csv(tables_dir / "counterfactual_query_baselines.csv", index=False)
        print("Saved baseline query features (DiCE unavailable)", flush=True)
        return 1

    # ── Analyze counterfactual feature perturbations ─────────────
    cf_records = []
    for i, idx in enumerate(cf_indices):
        cf_subset = cf_df[cf_df.iloc[:, -1] == i] if cf_df.shape[1] > len(features) else cf_df
        if len(cf_subset) == 0:
            continue
        base_vals = x_num.loc[idx]
        for _, cf_row in cf_subset.iterrows():
            for f in features:
                base_val = base_vals[f]
                cf_val = cf_row[f]
                if abs(cf_val - base_val) > 1e-8:
                    cf_records.append(
                        {
                            "query_index": int(idx),
                            "feature": f,
                            "base_value": float(base_val),
                            "counterfactual_value": float(cf_val),
                            "change": float(cf_val - base_val),
                            "abs_change": float(abs(cf_val - base_val)),
                        }
                    )

    if not cf_records:
        print("No counterfactual perturbations found", flush=True)
        return 1

    cf_summary = pd.DataFrame(cf_records)

    # ── Aggregate across all counterfactuals ─────────────────────
    agg = (
        cf_summary.groupby("feature")
        .agg(
            mean_change=("change", "mean"),
            mean_abs_change=("abs_change", "mean"),
            perturbation_count=("feature", "count"),
            perturbation_frequency=("feature", lambda x: len(x) / len(cf_summary["query_index"].unique())),
        )
        .sort_values("perturbation_count", ascending=False)
        .reset_index()
    )

    cf_summary.to_csv(tables_dir / "counterfactual_perturbations.csv", index=False)
    agg.to_csv(tables_dir / "counterfactual_aggregate.csv", index=False)

    print("Top-10 counterfactual features by perturbation frequency:", flush=True)
    print(agg.head(10).to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    top15 = agg.head(15).sort_values("mean_abs_change")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top15["feature"], top15["mean_abs_change"])
    ax.set_xlabel("Mean absolute counterfactual change (standardized)")
    ax.set_title("Top-15 Counterfactual Features for Low-Performer → Non-Low-Performer")
    plt.tight_layout()
    plt.savefig(figures_dir / "counterfactual_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Counterfactual reachability by ESCS ──────────────────────
    if "ESCS" in df.columns and "ESCS" in features:
        cf_summary_with_escs = cf_summary.merge(
            df[["ESCS"]], left_on="query_index", right_index=True, how="left"
        )
        cf_summary_with_escs["ESCS_QUINTILE"] = pd.qcut(
            cf_summary_with_escs["ESCS"].dropna(), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
        )
        escs_reach = (
            cf_summary_with_escs.groupby("ESCS_QUINTILE", observed=False)
            .agg(
                mean_abs_change=("abs_change", "mean"),
                n_queries=("query_index", "nunique"),
                n_perturbations=("abs_change", "count"),
            )
            .reset_index()
        )
        escs_reach.to_csv(tables_dir / "counterfactual_reachability_by_escs.csv", index=False)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(escs_reach["ESCS_QUINTILE"].astype(str), escs_reach["mean_abs_change"])
        ax2.set_xlabel("ESCS Quintile")
        ax2.set_ylabel("Mean absolute counterfactual change")
        ax2.set_title("Counterfactual Reachability by SES Quintile\n(lower = more reachable)")
        plt.tight_layout()
        plt.savefig(figures_dir / "counterfactual_reachability_escs.png", dpi=300, bbox_inches="tight")
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
