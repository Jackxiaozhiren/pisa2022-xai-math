#!/usr/bin/env python3
"""ALE (Accumulated Local Effects) analysis for best classification model.

Computes first-order ALE for top features and second-order ALE for key
feature pairs. ALE is preferred over SHAP/PDP when features are correlated
(e.g., ICT resources and SES in PISA data).

References:
    Herbinger et al. (2024). "Decomposing Global Feature Effects Based on
    Feature Interactions." JMLR, 25(381).
    Molnar (2022). "Interpretable Machine Learning: A Guide for Making
    Black Box Models Explainable."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.explain import (
    compute_ale,
    compute_second_order_ale,
    permutation_importance_table,
    save_ale_plots,
    save_shap_vs_ale_comparison,
)
from pisa_xai.io import load_table, require_package


TOP_N_ALE = 12
SECOND_ORDER_PAIRS = [
    ("ICTRES", "HOMEPOS"),
    ("ICTEFFIC", "MATHEFF"),
    ("ICTRES", "ICTEFFIC"),
    ("HOMEPOS", "MATHEFF"),
    ("ICTRES", "ESCS"),
    ("ANXMAT", "MATHEFF"),
]


def main() -> int:
    require_package("joblib", "pip install joblib")
    require_package("pandas", "pip install -r requirements.txt")
    import joblib
    import pandas as pd

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    model_dir = processed_dir / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features]

    max_rows = config["models"].get("explanation_max_rows", 5000)
    random_state = config["sample"]["random_state"]

    print("=" * 70)
    print("ALE Analysis (Accumulated Local Effects)")
    print("=" * 70)

    best_clf_name = f"classification_{best['best_classification_model']}"
    best_clf_path = model_dir / f"{best_clf_name}.joblib"
    if not best_clf_path.exists():
        print(f"Model not found: {best_clf_path}")
        return 1

    best_clf = joblib.load(best_clf_path)
    print(f"Computing ALE for: {best_clf_name}")

    # Pre-filter to features present in the data
    available_features = [f for f in x.columns[:TOP_N_ALE * 3] if f in x.columns]

    ale_result = compute_ale(
        best_clf, x, available_features, n_bins=20, max_rows=max_rows
    )
    if ale_result is None:
        print("ALE computation returned None — check model pipeline structure")
        return 1

    print(f"ALE computed for {len(ale_result['ale_values'])} features")

    # Save ALE plots
    ale_paths = save_ale_plots(ale_result, figures_dir / "ale", top_n=TOP_N_ALE)
    print(f"Saved {len(ale_paths)} ALE plots")

    # SHAP vs ALE comparison
    x_imp = x.sample(min(max_rows, len(x)), random_state=random_state)
    y_imp = df.loc[x_imp.index, "LOW_PERFORMER_MATH"]
    importance = permutation_importance_table(
        best_clf, x_imp, y_imp, scoring="roc_auc", n_repeats=5
    )
    shap_vs_ale_path = save_shap_vs_ale_comparison(
        importance, ale_result, figures_dir / "shap_vs_ale_rankings.png"
    )
    if shap_vs_ale_path:
        print(f"Saved SHAP vs ALE comparison: {shap_vs_ale_path}")

    # Second-order ALE for key feature pairs
    available_pairs = [
        (a, b) for (a, b) in SECOND_ORDER_PAIRS
        if a in x.columns and b in x.columns
    ]
    if available_pairs:
        print(f"Computing second-order ALE for {len(available_pairs)} feature pairs")
        ale_2d = compute_second_order_ale(
            best_clf, x, available_pairs, n_bins=10, max_rows=max_rows
        )
        if ale_2d:
            require_package("matplotlib", "pip install -r requirements.txt")
            import matplotlib.pyplot as plt
            import numpy as np

            ale2d_dir = figures_dir / "ale" / "second_order"
            ale2d_dir.mkdir(parents=True, exist_ok=True)
            for (fa, fb), result in ale_2d.items():
                fig, ax = plt.subplots(figsize=(7, 5.5))
                im = ax.contourf(
                    result["bin_centers_b"],
                    result["bin_centers_a"],
                    result["ale_values"],
                    levels=15,
                    cmap="RdBu_r",
                )
                plt.colorbar(im, ax=ax, label="Second-order ALE")
                ax.set_xlabel(fb)
                ax.set_ylabel(fa)
                ax.set_title(f"Second-Order ALE: {fa} × {fb}")
                fname = ale2d_dir / f"ale2d_{fa}_x_{fb}.png"
                fig.tight_layout()
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {fname}")

    # Save ALE importance summary
    import pandas as pd
    ale_summary = pd.DataFrame([
        {"feature": f, "ale_abs_mean": float(np.abs(vals).mean()),
         "ale_range": float(vals.max()) - float(vals.min())}
        for f, vals in ale_result["ale_values"].items()
    ]).sort_values("ale_abs_mean", ascending=False)
    summary_path = tables_dir / "ale_importance_summary.csv"
    ale_summary.to_csv(summary_path, index=False)
    print(f"Saved ALE importance summary: {summary_path}")
    print(ale_summary.head(15).to_string(index=False))

    print("\nALE analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
