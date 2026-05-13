#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.evaluation import classification_metrics, regression_metrics, threshold_sensitivity
from pisa_xai.features import flatten_feature_config
from pisa_xai.io import load_table, require_package
from pisa_xai.modeling import classification_models, regression_models


def model_predict_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return decision


def load_feature_set(config, df) -> list[str]:
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    feature_sets_path = processed_dir / "feature_sets.json"
    if feature_sets_path.exists():
        feature_sets = json.loads(feature_sets_path.read_text(encoding="utf-8"))
        features = [feature for feature in feature_sets["main_features"] if feature in df.columns]
    else:
        features = [col for col in flatten_feature_config(config["features"]) if col in df.columns]
    if not features:
        raise RuntimeError("No configured feature columns are available in the processed data.")
    return features


def maybe_sample_model_frame(df, y_clf, config):
    """Optional deterministic cap for local draft runs; full data are used by default."""

    max_rows_env = os.environ.get("PISA_XAI_MAX_MODEL_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else None
    if not max_rows or len(df) <= max_rows:
        return df
    return (
        df.assign(_stratify=y_clf)
        .groupby("_stratify", group_keys=False)
        .sample(frac=max_rows / len(df), random_state=config["sample"]["random_state"])
        .drop(columns="_stratify")
        .sort_index()
    )


def normalize_weights(weights):
    if weights is None:
        return None
    return weights / weights.mean()


def main() -> int:
    require_package("joblib", "pip install joblib")
    require_package("sklearn", "pip install -r requirements.txt")
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    processed = processed_dir / "pisa2022_math_model_frame.parquet"
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(processed)
    features = load_feature_set(config, df)
    df = maybe_sample_model_frame(df, df["LOW_PERFORMER_MATH"], config)

    x = df[features]
    y_reg = df["MATH_PV_MEAN"]
    y_clf = df["LOW_PERFORMER_MATH"]
    weight_col = config["pisa"]["student_weight"]
    sample_weight = normalize_weights(df[weight_col]) if weight_col in df.columns else None

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config["models"]["test_size"],
        random_state=config["sample"]["random_state"],
        stratify=y_clf,
    )
    x_train, x_test = x.loc[train_idx], x.loc[test_idx]
    y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
    y_clf_train, y_clf_test = y_clf.loc[train_idx], y_clf.loc[test_idx]
    w_train = sample_weight.loc[train_idx] if sample_weight is not None else None
    w_test = sample_weight.loc[test_idx] if sample_weight is not None else None

    enabled_optional = config["models"].get("enabled_optional_models", [])
    rows = []
    threshold_rows = []
    fitted = {}
    regression_predictions = {}
    classification_scores = {}

    for name, model in regression_models(x_train, enabled_optional).items():
        print(f"Training regression model: {name}", flush=True)
        fit_kwargs = {}
        if w_train is not None:
            fit_kwargs["model__sample_weight"] = w_train
        try:
            model.fit(x_train, y_reg_train, **fit_kwargs)
        except TypeError:
            model.fit(x_train, y_reg_train)
        pred = model.predict(x_test)
        metrics = regression_metrics(y_reg_test, pred, sample_weight=w_test)
        rows.append(
            {
                "task": "regression",
                "model": name,
                "feature_set": "main",
                "n_train": len(x_train),
                "n_test": len(x_test),
                "weighted_metrics": w_test is not None,
                **metrics,
            }
        )
        regression_predictions[name] = pred
        fitted[f"regression_{name}"] = model

    for name, model in classification_models(x_train, enabled_optional).items():
        print(f"Training classification model: {name}", flush=True)
        fit_kwargs = {}
        if w_train is not None:
            fit_kwargs["model__sample_weight"] = w_train
        try:
            model.fit(x_train, y_clf_train, **fit_kwargs)
        except TypeError:
            model.fit(x_train, y_clf_train)
        score = model_predict_score(model, x_test)
        threshold = config["models"]["classification_threshold"]
        metrics = classification_metrics(y_clf_test, score, threshold, sample_weight=w_test)
        rows.append(
            {
                "task": "classification",
                "model": name,
                "feature_set": "main",
                "n_train": len(x_train),
                "n_test": len(x_test),
                "weighted_metrics": w_test is not None,
                "threshold": threshold,
                **metrics,
            }
        )
        sensitivity = threshold_sensitivity(y_clf_test, score, sample_weight=w_test)
        sensitivity.insert(0, "model", name)
        threshold_rows.append(sensitivity)
        classification_scores[name] = score
        fitted[f"classification_{name}"] = model

    results = pd.DataFrame(rows)
    results.to_csv(tables_dir / "model_metrics.csv", index=False)
    if threshold_rows:
        pd.concat(threshold_rows, ignore_index=True).to_csv(
            tables_dir / "classification_threshold_sensitivity.csv",
            index=False,
        )

    for name, model in fitted.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    (model_dir / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (model_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "random_state": config["sample"]["random_state"],
                "test_size": config["models"]["test_size"],
                "n_modeling_rows": len(df),
                "n_train": len(x_train),
                "n_test": len(x_test),
                "features": features,
                "optional_models_enabled": enabled_optional,
                "sample_weight": weight_col if sample_weight is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_reg = results.loc[results["task"] == "regression"].sort_values("rmse").iloc[0]
    best_clf = results.loc[results["task"] == "classification"].sort_values("auc", ascending=False).iloc[0]
    best_summary = {
        "best_regression_model": best_reg["model"],
        "best_regression_rmse": float(best_reg["rmse"]),
        "best_classification_model": best_clf["model"],
        "best_classification_auc": float(best_clf["auc"]),
    }
    (model_dir / "best_model_summary.json").write_text(
        json.dumps(best_summary, indent=2),
        encoding="utf-8",
    )

    prediction_frame = pd.DataFrame(
        {
            "row_index": test_idx,
            "CNT": df.loc[test_idx, config["pisa"]["country"]].astype(str).values,
            "ST004D01T": df.loc[test_idx, "ST004D01T"].astype(str).values
            if "ST004D01T" in df.columns
            else "",
            "ESCS": df.loc[test_idx, "ESCS"].values if "ESCS" in df.columns else "",
            "IMMIG": df.loc[test_idx, "IMMIG"].astype(str).values if "IMMIG" in df.columns else "",
            "MATH_PV_MEAN": y_reg_test.values,
            "LOW_PERFORMER_MATH": y_clf_test.values,
            "sample_weight": w_test.values if w_test is not None else 1.0,
            "best_regression_model": best_summary["best_regression_model"],
            "best_regression_prediction": regression_predictions[best_summary["best_regression_model"]],
            "best_classification_model": best_summary["best_classification_model"],
            "best_classification_score": classification_scores[
                best_summary["best_classification_model"]
            ],
        }
    )
    prediction_frame.to_csv(tables_dir / "holdout_predictions.csv", index=False)

    print(results.to_string(index=False))
    print("\nBest models:")
    print(json.dumps(best_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
