#!/usr/bin/env python3
"""AXE-inspired explanation quality evaluation.

Computes predictiveness-based explanation quality without requiring ground-truth
explanations, following the AXE framework (Rawal et al., ACM FAccT 2025).

Three components:
1. Explanation predictiveness: Can we predict model output from explanations?
2. Explanation stability: How stable are explanations under input perturbation?
3. Explanation fairwashing detection: Do explanations hide protected attribute use?

References:
    Rawal et al. (2025). "Evaluating Model Explanations without Ground Truth." ACM FAccT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from pisa_xai.config import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def explanation_predictiveness(
    model, x_sample, shap_values, top_k: int = 10
) -> dict:
    """Measure how well top-K SHAP features predict model output.

    If top features identified by SHAP can reconstruct model predictions with
    high fidelity, explanations are predictive (good).
    """
    require_package("numpy", "pip install numpy")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    n_samples = len(shap_values.values)
    feature_importance = np.abs(shap_values.values).mean(axis=0)

    top_indices = np.argsort(feature_importance)[-top_k:]
    x_top = shap_values.data[:, top_indices]

    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, x_top, model.predict(x_sample), cv=5, scoring="r2")
    return {
        "mean_r2": float(scores.mean()),
        "std_r2": float(scores.std()),
        "n_top_features": top_k,
    }


def explanation_stability_perturbation(
    model, x_sample, shap_explainer, perturbation_std: float = 0.05, n_repeats: int = 10
) -> dict:
    """Measure SHAP stability under small input perturbations.

    AXE principle: Good explanations should be stable under minor perturbations.
    """
    require_package("numpy", "pip install numpy")
    import numpy as np

    base_values = np.abs(shap_explainer(x_sample).values).mean(axis=0)
    cosine_similarities = []

    rng = np.random.RandomState(20260510)
    for _ in range(n_repeats):
        x_perturbed = x_sample + rng.normal(0, perturbation_std, x_sample.shape)
        perturbed_values = np.abs(shap_explainer(x_perturbed).values).mean(axis=0)

        cos_sim = np.dot(base_values, perturbed_values) / (
            np.linalg.norm(base_values) * np.linalg.norm(perturbed_values) + 1e-10
        )
        cosine_similarities.append(float(cos_sim))

    return {
        "mean_cosine_similarity": float(np.mean(cosine_similarities)),
        "std_cosine_similarity": float(np.std(cosine_similarities)),
        "n_repeats": n_repeats,
        "perturbation_std": perturbation_std,
    }


def explanation_fairwashing_check(
    model, x_sample, shap_values, protected_var_name: str, protected_var_idx: int
) -> dict:
    """AXE fairwashing detection: Compare SHAP importance of protected attribute
    with its actual predictive importance (via permutation).

    If a protected attribute has high permutation importance but low SHAP
    importance, explanations may be "fairwashing" — hiding protected
    attribute use.
    """
    require_package("numpy", "pip install numpy")
    import numpy as np

    shap_importance = np.abs(shap_values.values[:, protected_var_idx]).mean()
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    shap_rank = int(np.sum(feature_importance > shap_importance))

    from sklearn.inspection import permutation_importance
    n_perm = min(2000, len(x_sample))
    x_perm = x_sample[:n_perm]
    perm_imp = permutation_importance(
        model, x_perm,
        model.predict(x_perm),
        n_repeats=5,
        random_state=20260510,
        scoring="neg_mean_squared_error",
    )
    perm_rank = int(np.sum(
        np.abs(perm_imp.importances_mean) >
        np.abs(perm_imp.importances_mean[protected_var_idx])
    ))

    rank_discrepancy = shap_rank - perm_rank
    return {
        "protected_variable": protected_var_name,
        "shap_rank": shap_rank,
        "permutation_rank": perm_rank,
        "rank_discrepancy": rank_discrepancy,
        "fairwashing_concern": "YES" if abs(rank_discrepancy) > 5 else "NO",
    }


def main() -> int:
    require_package("pandas", "pip install pandas")
    require_package("numpy", "pip install numpy")
    require_package("shap", "pip install shap")
    import joblib
    import pandas as pd
    import numpy as np

    config = load_config(DEFAULT_CONFIG_PATH)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    best_clf_name = f"classification_{best['best_classification_model']}"
    best_clf = joblib.load(model_dir / f"{best_clf_name}.joblib")

    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    features = [f for f in features if f in df.columns]
    x = df[features]

    n_sample = min(2000, len(x))
    x_sample = x.sample(n_sample, random_state=20260510)
    import numpy as np
    cat_cols = [c for c in x_sample.columns if str(x_sample[c].dtype) == "category"]
    for c in cat_cols:
        x_sample[c] = x_sample[c].cat.codes.astype("int8")
    x_transformed = x_sample.fillna(x_sample.median()).values.astype(np.float64)

    import shap
    explainer = shap.Explainer(best_clf, x_transformed)
    shap_values = explainer(x_transformed)

    print("=" * 70)
    print("AXE-Inspired Explanation Quality Evaluation")
    print("Reference: Rawal et al. (2025), ACM FAccT")
    print("=" * 70)

    # 1. Predictiveness
    print("\n[1] Explanation Predictiveness")
    pred = explanation_predictiveness(best_clf, x_transformed, shap_values, top_k=10)
    print(f"    Top-{pred['n_top_features']} features reconstruct model output with R² = {pred['mean_r2']:.4f} ± {pred['std_r2']:.4f}")
    if pred["mean_r2"] > 0.8:
        print("    ✓ Explanations are highly predictive of model behavior")
    elif pred["mean_r2"] > 0.5:
        print("    ~ Explanations are moderately predictive")
    else:
        print("    ⚠ Explanations have low predictiveness — consider alternative XAI methods")

    # 2. Stability
    print("\n[2] Explanation Stability under Perturbation")
    stab = explanation_stability_perturbation(
        best_clf, x_transformed, explainer
    )
    print(f"    Cosine similarity under perturbation: {stab['mean_cosine_similarity']:.4f} ± {stab['std_cosine_similarity']:.4f}")
    if stab["mean_cosine_similarity"] > 0.95:
        print("    ✓ Explanations are highly stable under perturbation")
    elif stab["mean_cosine_similarity"] > 0.85:
        print("    ~ Explanations are moderately stable")
    else:
        print("    ⚠ Explanation instability detected")

    # 3. Fairwashing check for known protected variables
    print("\n[3] Explanation Fairwashing Detection")
    protected_vars = {
        "ST004D01T": "Gender",
        "IMMIG": "Immigrant background",
    }

    fairwash_results = []
    for var_name, var_label in protected_vars.items():
        if var_name in features:
            idx = features.index(var_name)
            result = explanation_fairwashing_check(
                best_clf, x_sample, shap_values, var_label, idx
            )
            fairwash_results.append(result)
            print(f"    {var_label}: SHAP rank = {result['shap_rank']}, "
                  f"Permutation rank = {result['permutation_rank']}, "
                  f"Discrepancy = {result['rank_discrepancy']}, "
                  f"Fairwashing concern: {result['fairwashing_concern']}")

    if fairwash_results:
        fw_df = pd.DataFrame(fairwash_results)
        fw_df.to_csv(tables_dir / "axe_fairwashing_check.csv", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("AXE EVALUATION SUMMARY")
    print("=" * 70)
    results = {"predictiveness_r2": pred["mean_r2"], "stability_cosine": stab["mean_cosine_similarity"]}
    overall = "PASS" if (results["predictiveness_r2"] > 0.5 and results["stability_cosine"] > 0.85) else "REVIEW"
    print(f"  Overall: {overall}")
    print(f"  Predictiveness R²: {results['predictiveness_r2']:.4f}")
    print(f"  Stability (cosine): {results['stability_cosine']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
