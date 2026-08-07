#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.explain import (
    compute_shap_interactions,
    permutation_importance_table,
    save_shap_dependence_plots,
    save_shap_summary,
)
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

# Digital divide level mapping for color coding
DD_LEVEL = {
    "ICTRES": "Access",
    "ICTHOME": "Access",
    "ICTSCH": "Access",
    "ICTEFFIC": "Skills/Confidence",
    "ICTINFO": "Usage",
    "ICTDISTR": "Usage",
    "ICTSUBJ": "Usage",
    "STUDYHMW": "Usage",
    "LEARNRES": "Usage",
    "DISTICT": "Usage",
}

DD_COLORS = {
    "Access": "#2563eb",
    "Skills/Confidence": "#16a34a",
    "Usage": "#ea580c",
}


def sample_for_explanation(x, y, max_rows: int, random_state: int):
    if len(x) <= max_rows:
        return x, y
    sampled_index = x.sample(max_rows, random_state=random_state).index
    return x.loc[sampled_index], y.loc[sampled_index]


def plot_digital_importance(importance, output_path: Path) -> None:
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    digital = importance[importance["feature"].isin(DIGITAL_FEATURES)].copy()
    if digital.empty:
        return
    digital = digital.sort_values("importance_mean", ascending=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(digital))))
    colors = [DD_COLORS.get(DD_LEVEL.get(f, ""), "#6b7280") for f in digital["feature"]]
    bars = ax.barh(digital["feature"], digital["importance_mean"],
                   xerr=digital.get("importance_std", None),
                   color=colors, edgecolor="white", linewidth=0.5)

    # Legend showing digital divide levels
    legend_elements = [
        Patch(facecolor="#2563eb", label="Access"),
        Patch(facecolor="#16a34a", label="Skills/Confidence"),
        Patch(facecolor="#ea580c", label="Usage"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    ax.set_xlabel("Permutation Importance", fontsize=10)
    ax.set_ylabel("Digital-Learning Feature", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    # ── SHAP interaction analysis (best classification model only) ──
    best_clf_name = f"classification_{best['best_classification_model']}"
    best_clf_path = model_dir / f"{best_clf_name}.joblib"
    if best_clf_path.exists():
        print(f"Computing SHAP interactions for {best_clf_name}", flush=True)
        best_clf = joblib.load(best_clf_path)
        x_interact = x.sample(min(shap_rows, len(x)), random_state=random_state)
        shap_interact = compute_shap_interactions(best_clf, x_interact, max_rows=shap_rows)
        if shap_interact is not None:
            interact_table = f"shap_interactions_{best['best_classification_model']}.csv"
            shap_interact.to_csv(tables_dir / interact_table, index=False)
            explanation_rows.append(
                {"model": best_clf_name, "artifact_type": "shap_interactions", "path": str(tables_dir / interact_table)}
            )
            print(f"Saved SHAP interactions: {tables_dir / interact_table}")

        # SHAP dependence plots for key ICT feature pairs
        ict_feature_pairs = []
        ict_available = [f for f in ["ICTRES", "ICTEFFIC", "ICTSUBJ", "ICTHOME", "ICTSCH"] if f in x.columns]
        for i in range(len(ict_available)):
            for j in range(i + 1, len(ict_available)):
                ict_feature_pairs.append((ict_available[i], ict_available[j]))
        # Also add ICT × HOMEPOS and ICT × MATHEFF context pairs
        for base in ["HOMEPOS", "MATHEFF"]:
            if base in x.columns:
                for ict_f in ict_available[:3]:  # top 3 ICT features
                    ict_feature_pairs.append((ict_f, base))
        if ict_feature_pairs:
            print(f"Generating {len(ict_feature_pairs)} SHAP dependence plots", flush=True)
            dep_paths = save_shap_dependence_plots(
                best_clf, x_interact, figures_dir, ict_feature_pairs, max_rows=shap_rows
            )
            for dep_path in dep_paths or []:
                explanation_rows.append(
                    {"model": best_clf_name, "artifact_type": "shap_dependence", "path": str(dep_path)}
                )

    pd.DataFrame(explanation_rows).to_csv(tables_dir / "explanation_artifacts.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
