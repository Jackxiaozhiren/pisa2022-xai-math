#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.explain import permutation_importance_table, save_shap_summary
from pisa_xai.io import load_table, require_package


DIGITAL_FEATURES = {
    "ICTRES",
    "ICTHOME",
    "ICTSCH",
    "ICTEFFIC",
    "ICTINFO",
    "ICTDISTR",
    "ICTSUBJ",
    "LEARNRES",
    "DISTICT",
    "STUDYHMW",
}


def sample_for_explanation(x, y, max_rows: int, random_state: int):
    if len(x) <= max_rows:
        return x, y
    sampled_index = x.sample(max_rows, random_state=random_state).index
    return x.loc[sampled_index], y.loc[sampled_index]


def plot_digital_importance(importance, output_path: Path) -> None:
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    digital = importance[importance["feature"].isin(DIGITAL_FEATURES)].copy()
    if digital.empty:
        return
    digital = digital.sort_values("importance_mean", ascending=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, max(3, 0.35 * len(digital))))
    plt.barh(digital["feature"], digital["importance_mean"], xerr=digital["importance_std"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Digital-learning feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    require_package("joblib", "pip install joblib")
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

    max_rows = config["models"].get("permutation_max_rows", 10000)
    shap_rows = config["models"].get("explanation_max_rows", 5000)
    random_state = config["sample"]["random_state"]
    explanation_rows = []
    digital_tables = []

    candidates = [
        (
            f"classification_{best['best_classification_model']}",
            "LOW_PERFORMER_MATH",
            "roc_auc",
        ),
        (
            f"regression_{best['best_regression_model']}",
            "MATH_PV_MEAN",
            "neg_root_mean_squared_error",
        ),
    ]
    for model_name, target, scoring in candidates:
        model_path = model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            continue
        print(f"Explaining {model_name}", flush=True)
        model = joblib.load(model_path)

        shap_path = save_shap_summary(
            model,
            x,
            figures_dir / f"{model_name}_shap_summary.png",
            max_rows=shap_rows,
        )
        if shap_path:
            explanation_rows.append(
                {
                    "model": model_name,
                    "artifact_type": "shap_summary",
                    "path": str(shap_path),
                }
            )
            print(f"Saved SHAP summary: {shap_path}")

        x_imp, y_imp = sample_for_explanation(x, df[target], max_rows, random_state)
        importance = permutation_importance_table(
            model,
            x_imp,
            y_imp,
            scoring=scoring,
            n_repeats=5,
        )
        table_path = tables_dir / f"{model_name}_permutation_importance.csv"
        importance.to_csv(table_path, index=False)
        explanation_rows.append(
            {
                "model": model_name,
                "artifact_type": "permutation_importance",
                "path": str(table_path),
            }
        )
        digital = importance[importance["feature"].isin(DIGITAL_FEATURES)].copy()
        digital.insert(0, "model", model_name)
        digital_tables.append(digital)
        if model_name.startswith("classification_"):
            plot_digital_importance(importance, figures_dir / "digital_feature_importance.png")
        print(importance.head(20).to_string(index=False))

    if digital_tables:
        pd.concat(digital_tables, ignore_index=True).to_csv(
            tables_dir / "digital_feature_importance.csv",
            index=False,
        )
    pd.DataFrame(explanation_rows).to_csv(tables_dir / "explanation_artifacts.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
