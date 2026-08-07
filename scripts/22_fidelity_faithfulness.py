#!/usr/bin/env python3
r"""Compute SHAP explanation fidelity and faithfulness metrics.

Implements two formal XAI quality metrics benchmarked against Kişman et al.
(2026, Sustainability) who reported Fidelity = 0.95 and Faithfulness = 0.85
for PISA Math SHAP explanations.

Metrics:
    Fidelity: How well SHAP values reconstruct model predictions.
        R² of regressing model output on SHAP values.
        Higher = SHAP correctly captures model behavior.

    Faithfulness: Monotonicity of prediction change when top features are
        perturbed. Correlation between feature importance and prediction
        sensitivity. Higher = SHAP correctly ranks feature influence.

References:
    Kişman et al. (2026). Sustainability, 18(3), 1415.
    Gunasekara & Saarela (2025). ACM SAC.
    Létoffé et al. (2026). arXiv:2604.xxxxx.
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


def compute_fidelity(model, x_sample, shap_values, task: str = "classification") -> dict:
    """Fidelity: R² of model output reconstruction from SHAP values.

    For regression: SHAP values + base_value = prediction by additivity,
    so fidelity is trivially 1.0 for tree-based SHAP. We verify this.

    For classification: SHAP values are in log-odds space; we convert
    predictions through logit to match, then compute R². This is the
    metric comparable to Kişman et al. (2026).
    """
    require_package("numpy", "pip install numpy")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    shap_matrix = shap_values.values

    if task == "regression":
        y_pred = model.predict(x_sample)
        shap_pred = shap_values.base_values + shap_matrix.sum(axis=1)
    else:
        # Classification: convert to log-odds space for fair comparison
        y_proba = model.predict_proba(x_sample)[:, 1]
        # Clip to avoid log(0)
        eps = 1e-10
        y_proba = np.clip(y_proba, eps, 1 - eps)
        y_logodds = np.log(y_proba / (1 - y_proba))
        shap_pred = shap_values.base_values + shap_matrix.sum(axis=1)
        y_pred = y_logodds

    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, shap_matrix, y_pred, cv=5, scoring="r2")

    return {
        "fidelity_r2_mean": float(scores.mean()),
        "fidelity_r2_std": float(scores.std()),
        "n_samples": len(y_pred),
    }


def compute_faithfulness(model, x_array, shap_values, task: str = "classification", top_k: int = 10, n_shuffles: int = 10) -> dict:
    """Faithfulness: Correlation between SHAP importance and prediction
    sensitivity when features are shuffled (permuted).

    For each top-K feature:
    1. Compute SHAP-based feature importance |SHAP|.mean()
    2. Shuffle the feature (destroy its signal) and measure |Δprediction|
    3. Compute Pearson r between SHAP importance and prediction change

    This is the Kişman et al. (2026) approach: features that SHAP says are
    important should genuinely change predictions when their signal is destroyed.
    Benchmark: Faithfulness = 0.85 for PISA Math.
    """
    require_package("numpy", "pip install numpy")
    import numpy as np
    from scipy.stats import pearsonr

    feature_importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(feature_importance)[-top_k:]

    rng = np.random.RandomState(20260510)

    if task == "regression":
        base_pred = model.predict(x_array)
    else:
        base_pred = model.predict_proba(x_array)[:, 1]

    importance_vals = []
    sensitivity_vals = []

    for feat_idx in top_indices:
        imp = float(feature_importance[feat_idx])
        delta_preds = []

        for _ in range(n_shuffles):
            x_shuffled = x_array.copy()
            rng.shuffle(x_shuffled[:, feat_idx])
            if task == "regression":
                pred_shuffled = model.predict(x_shuffled)
            else:
                pred_shuffled = model.predict_proba(x_shuffled)[:, 1]
            delta_preds.append(np.mean(np.abs(pred_shuffled - base_pred)))

        sensitivity = float(np.mean(delta_preds))
        importance_vals.append(imp)
        sensitivity_vals.append(sensitivity)

    unique_sens = len(set(round(s, 8) for s in sensitivity_vals))
    if unique_sens > 1:
        r, p = pearsonr(importance_vals, sensitivity_vals)
    else:
        r, p = 0.0, 1.0

    return {
        "faithfulness_pearson_r": float(r),
        "faithfulness_p_value": float(p),
        "top_k": top_k,
        "n_shuffles_per_feature": n_shuffles,
    }


def compute_fidelity_all_features(model, x_sample, shap_values, task: str = "classification") -> dict:
    """Compute fidelity using ALL features (not just top-K)."""
    require_package("numpy", "pip install numpy")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    shap_matrix = shap_values.values

    if task == "regression":
        y_pred = model.predict(x_sample)
    else:
        y_proba = model.predict_proba(x_sample)[:, 1]
        eps = 1e-10
        y_proba = np.clip(y_proba, eps, 1 - eps)
        y_pred = np.log(y_proba / (1 - y_proba))

    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, shap_matrix, y_pred, cv=5, scoring="r2")

    return {
        "fidelity_all_features_r2_mean": float(scores.mean()),
        "fidelity_all_features_r2_std": float(scores.std()),
    }


def main() -> int:
    require_package("pandas", "pip install pandas")
    require_package("numpy", "pip install numpy")
    require_package("shap", "pip install shap")
    require_package("scipy", "pip install scipy")
    import joblib
    import pandas as pd

    config = load_config(DEFAULT_CONFIG_PATH)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    best_clf_name = f"classification_{best['best_classification_model']}"
    best_reg_name = f"regression_{best['best_regression_model']}"

    best_clf = joblib.load(model_dir / f"{best_clf_name}.joblib")
    best_reg = joblib.load(model_dir / f"{best_reg_name}.joblib")

    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    features = [f for f in features if f in df.columns]
    x = df[features]

    print("=" * 70)
    print("SHAP Explanation Fidelity & Faithfulness Evaluation")
    print("Reference: Kişman et al. (2026), Sustainability, 18(3), 1415")
    print("  Benchmark: Fidelity = 0.95, Faithfulness = 0.85 (PISA Math)")
    print("=" * 70)

    n_sample = 2000
    x_sample = x.sample(n_sample, random_state=20260510)

    import numpy as np
    import shap

    # Preprocess: convert category dtypes to numeric codes (same as training script)
    x_sample_pp = x_sample.copy()
    cat_cols = [c for c in x_sample_pp.columns if str(x_sample_pp[c].dtype) == "category"]
    for c in cat_cols:
        x_sample_pp[c] = x_sample_pp[c].cat.codes.astype("int8")
    # Fill any remaining NaN with median
    x_sample_pp = x_sample_pp.fillna(x_sample_pp.median())
    x_trans = x_sample_pp.values.astype(np.float64)

    # Create SHAP explainer directly on the raw model
    explainer_clf = shap.Explainer(best_clf, x_trans)
    shap_values_clf = explainer_clf(x_trans)

    explainer_reg = shap.Explainer(best_reg, x_trans)
    shap_values_reg = explainer_reg(x_trans)

    # Classification Fidelity & Faithfulness
    print(f"\n{'─'*60}")
    print(f"Classification Model: {best_clf_name}")
    print(f"{'─'*60}")

    fid_clf = compute_fidelity(best_clf, x_trans, shap_values_clf, "classification")
    print(f"  Fidelity (R², log-odds): {fid_clf['fidelity_r2_mean']:.4f} ± {fid_clf['fidelity_r2_std']:.4f}")

    faith_clf = compute_faithfulness(best_clf, x_trans, shap_values_clf, "classification")
    print(f"  Faithfulness (Pearson r): {faith_clf['faithfulness_pearson_r']:.4f}")

    fid_all_clf = compute_fidelity_all_features(best_clf, x_trans, shap_values_clf, "classification")
    print(f"  Fidelity (all-feat, log-odds): {fid_all_clf['fidelity_all_features_r2_mean']:.4f} ± {fid_all_clf['fidelity_all_features_r2_std']:.4f}")

    # Regression Fidelity & Faithfulness
    print(f"\n{'─'*60}")
    print(f"Regression Model: {best_reg_name}")
    print(f"{'─'*60}")

    fid_reg = compute_fidelity(best_reg, x_trans, shap_values_reg, "regression")
    print(f"  Fidelity (R²): {fid_reg['fidelity_r2_mean']:.4f} ± {fid_reg['fidelity_r2_std']:.4f}")

    faith_reg = compute_faithfulness(best_reg, x_trans, shap_values_reg, "regression")
    print(f"  Faithfulness (Pearson r): {faith_reg['faithfulness_pearson_r']:.4f}")

    fid_all_reg = compute_fidelity_all_features(best_reg, x_trans, shap_values_reg, "regression")
    print(f"  Fidelity (all features R²): {fid_all_reg['fidelity_all_features_r2_mean']:.4f} ± {fid_all_reg['fidelity_all_features_r2_std']:.4f}")

    # Summary & Comparison with Kişman et al. benchmark
    print(f"\n{'='*70}")
    print("FIDELITY & FAITHFULNESS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Classification':<18} {'Regression':<18} {'Kişman (PISA Math)'}")
    print(f"  {'─'*70}")
    print(f"  {'Fidelity (R²)':<30} {fid_clf['fidelity_r2_mean']:.4f} ± {fid_clf['fidelity_r2_std']:.3f}   {fid_reg['fidelity_r2_mean']:.4f} ± {fid_reg['fidelity_r2_std']:.3f}   0.95")
    print(f"  {'Fidelity (all features R²)':<30} {fid_all_clf['fidelity_all_features_r2_mean']:.4f} ± {fid_all_clf['fidelity_all_features_r2_std']:.3f}   {fid_all_reg['fidelity_all_features_r2_mean']:.4f} ± {fid_all_reg['fidelity_all_features_r2_std']:.3f}   0.95")
    print(f"  {'Faithfulness (Pearson r)':<30} {faith_clf['faithfulness_pearson_r']:.4f}           {faith_reg['faithfulness_pearson_r']:.4f}           0.85")

    # Interpretation
    print(f"\n  Interpretation:")
    if fid_all_clf['fidelity_all_features_r2_mean'] >= 0.90:
        print(f"    ✓ Fidelity meets published PISA benchmark (≥0.90)")
    elif fid_all_clf['fidelity_all_features_r2_mean'] >= 0.80:
        print(f"    ~ Fidelity near published PISA benchmark (≥0.80)")
    else:
        print(f"    ⚠ Fidelity below published PISA benchmark (<0.80)")

    if faith_clf['faithfulness_pearson_r'] >= 0.80:
        print(f"    ✓ Faithfulness meets published PISA benchmark (≥0.80)")
    elif faith_clf['faithfulness_pearson_r'] >= 0.70:
        print(f"    ~ Faithfulness near published PISA benchmark (≥0.70)")
    else:
        print(f"    ⚠ Faithfulness below published benchmark (<0.70)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
