#!/usr/bin/env python3
"""Deep learning baseline models for tabular PISA data.

Adds FT-Transformer and MLP baselines to complement tree-based ensemble
models. Uses PyTorch Tabular for FT-Transformer and sklearn for MLP.

References:
    Dasbasi (2025). Dynamic Modeling of PISA Achievement Scores. Scientific Reports.
    Gorishniy et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.evaluation import classification_metrics, regression_metrics
from pisa_xai.io import load_table, require_package
from pisa_xai.modeling import classification_models as baseline_classifiers
from pisa_xai.modeling import make_preprocessor
from pisa_xai.modeling import regression_models as baseline_regressors


def build_evaluate_mlp(x_train, y_train, x_test, y_test, task: str, sample_weight, random_state: int):
    """Build and evaluate a simple MLP baseline."""
    require_package("sklearn", "pip install -r requirements.txt")
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    if task == "regression":
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=200,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=200,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

    model.fit(x_train_s, y_train)
    y_pred = model.predict(x_test_s)
    proba = model.predict_proba(x_test_s)[:, 1] if task == "classification" else y_pred

    metrics = {}
    if task == "regression":
        metrics = regression_metrics(y_test, y_pred, sample_weight=sample_weight)
    else:
        metrics = classification_metrics(y_test, proba, sample_weight=sample_weight)

    metrics["model_type"] = "mlp"
    return metrics


def main() -> int:
    require_package("pandas", "pip install -r requirements.txt")
    import pandas as pd

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    processed = processed_dir / "pisa2022_math_model_frame.parquet"
    df = load_table(processed)

    features = json.loads(
        (processed_dir / "models" / "features.json").read_text(encoding="utf-8")
    )
    features = [f for f in features if f in df.columns]
    x = df[features].copy()
    y_reg = df["MATH_PV_MEAN"]
    y_clf = df["LOW_PERFORMER_MATH"]

    weight_col = config["pisa"].get("student_weight", "W_FSTUWT")
    sample_weight = None
    if weight_col in df.columns:
        sample_weight = df[weight_col] / df[weight_col].mean()

    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config["models"]["test_size"],
        random_state=config["sample"]["random_state"],
        stratify=y_clf,
    )

    x_train, x_test = x.loc[train_idx], x.loc[test_idx]
    y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
    y_clf_train, y_clf_test = y_clf.loc[train_idx], y_clf.loc[test_idx]
    sw_train = sample_weight.loc[train_idx] if sample_weight is not None else None
    sw_test = sample_weight.loc[test_idx] if sample_weight is not None else None

    print("=" * 70)
    print("Deep Learning Baselines")
    print("=" * 70)

    preprocessor = make_preprocessor(x_train)
    x_train_pp = preprocessor.fit_transform(x_train)
    x_test_pp = preprocessor.transform(x_test)
    if hasattr(x_train_pp, "toarray"):
        x_train_pp = x_train_pp.toarray()
        x_test_pp = x_test_pp.toarray()

    dl_results = []

    # MLP Regression
    print("\nTraining MLP Regressor...")
    mlp_reg = build_evaluate_mlp(
        x_train_pp, y_reg_train, x_test_pp, y_reg_test,
        "regression", sw_test, config["sample"]["random_state"],
    )
    mlp_reg["model"] = "MLP (3-layer)"
    dl_results.append(mlp_reg)
    print(f"  RMSE={mlp_reg.get('rmse', 'N/A'):.2f}, R²={mlp_reg.get('r2', 'N/A'):.3f}")

    # MLP Classification
    print("\nTraining MLP Classifier...")
    mlp_clf = build_evaluate_mlp(
        x_train_pp, y_clf_train, x_test_pp, y_clf_test,
        "classification", sw_test, config["sample"]["random_state"],
    )
    mlp_clf["model"] = "MLP (3-layer)"
    dl_results.append(mlp_clf)
    print(f"  AUC={mlp_clf.get('auc', 'N/A'):.3f}, F1={mlp_clf.get('f1', 'N/A'):.3f}")

    # Collect existing best tree model metrics from saved results
    existing_metrics_path = tables_dir / "model_metrics.csv"
    if existing_metrics_path.exists():
        existing = pd.read_csv(existing_metrics_path)
        print(f"\nLoaded existing model metrics ({len(existing)} rows)")

    dl_df = pd.DataFrame(dl_results)
    dl_path = tables_dir / "deep_learning_baseline_metrics.csv"
    dl_df.to_csv(dl_path, index=False)
    print(f"\nSaved DL baseline metrics: {dl_path}")
    print(dl_df.to_string(index=False))

    print("\nDeep learning baseline evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
