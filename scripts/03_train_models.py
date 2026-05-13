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
from pisa_xai.modeling import (
    classification_models,
    regression_models,
    tune_lightgbm_regressor,
    tune_lightgbm_classifier,
    tune_xgboost_regressor,
    tune_xgboost_classifier,
    build_stacking_regressor,
    build_stacking_classifier,
)


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

    # ── Optuna hyperparameter tuning ─────────────────────────────────
    tuning_rows = []
    n_trials = config["models"].get("optuna_n_trials", 50)
    # Create a validation split from the training set for tuning
    val_size = 0.15  # 15% of training = 12% of total
    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_size,
        random_state=config["sample"]["random_state"],
        stratify=y_clf.loc[train_idx],
    )
    x_tune, x_val = x.loc[train_sub_idx], x.loc[val_idx]
    y_reg_tune, y_reg_val = y_reg.loc[train_sub_idx], y_reg.loc[val_idx]
    y_clf_tune, y_clf_val = y_clf.loc[train_sub_idx], y_clf.loc[val_idx]

    # XGBoost does not support pandas category dtypes — convert to numeric codes
    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    if cat_cols:
        x_tune_xgb = x_tune.copy()
        x_val_xgb = x_val.copy()
        x_train_xgb = x_train.copy()
        x_test_xgb = x_test.copy()
        for c in cat_cols:
            x_tune_xgb[c] = x_tune_xgb[c].cat.codes.astype("int8")
            x_val_xgb[c] = x_val_xgb[c].cat.codes.astype("int8")
            x_train_xgb[c] = x_train_xgb[c].cat.codes.astype("int8")
            x_test_xgb[c] = x_test_xgb[c].cat.codes.astype("int8")
    else:
        x_tune_xgb, x_val_xgb, x_train_xgb, x_test_xgb = x_tune, x_val, x_train, x_test

    if "lightgbm" in enabled_optional:
        print(f"Optuna tuning: LightGBM regressor ({n_trials} trials)", flush=True)
        best_params = tune_lightgbm_regressor(
            x_tune, y_reg_tune, x_val, y_reg_val,
            n_trials=n_trials, random_state=config["sample"]["random_state"],
        )
        if best_params:
            import lightgbm as lgb
            lgb_reg_tuned = lgb.LGBMRegressor(**best_params, n_jobs=-1, verbose=-1)
            lgb_reg_tuned.fit(x_train, y_reg_train)
            pred_reg = lgb_reg_tuned.predict(x_test)
            metrics_reg = regression_metrics(y_reg_test, pred_reg, sample_weight=w_test)
            tuning_rows.append({"task": "regression", "model": "lightgbm_tuned", "feature_set": "main",
                                "n_train": len(x_train), "n_test": len(x_test),
                                "weighted_metrics": w_test is not None, **metrics_reg})
            fitted["regression_lightgbm_tuned"] = lgb_reg_tuned
            regression_predictions["lightgbm_tuned"] = pred_reg
            pd.DataFrame([best_params]).to_csv(tables_dir / "hyperparameter_tuning_regression.csv", index=False)

        print(f"Optuna tuning: LightGBM classifier ({n_trials} trials)", flush=True)
        best_params_clf = tune_lightgbm_classifier(
            x_tune, y_clf_tune, x_val, y_clf_val,
            n_trials=n_trials, random_state=config["sample"]["random_state"],
        )
        if best_params_clf:
            lgb_clf_tuned = lgb.LGBMClassifier(**best_params_clf, n_jobs=-1, verbose=-1)
            lgb_clf_tuned.fit(x_train, y_clf_train)
            score_clf = model_predict_score(lgb_clf_tuned, x_test)
            metrics_clf = classification_metrics(
                y_clf_test, score_clf, config["models"]["classification_threshold"], sample_weight=w_test
            )
            tuning_rows.append({"task": "classification", "model": "lightgbm_tuned", "feature_set": "main",
                                "n_train": len(x_train), "n_test": len(x_test),
                                "weighted_metrics": w_test is not None,
                                "threshold": config["models"]["classification_threshold"], **metrics_clf})
            fitted["classification_lightgbm_tuned"] = lgb_clf_tuned
            classification_scores["lightgbm_tuned"] = score_clf
            pd.DataFrame([best_params_clf]).to_csv(tables_dir / "hyperparameter_tuning_classification.csv", index=False)

    if "xgboost" in enabled_optional:
        print(f"Optuna tuning: XGBoost regressor ({n_trials} trials)", flush=True)
        best_params_xgb = tune_xgboost_regressor(
            x_tune_xgb, y_reg_tune, x_val_xgb, y_reg_val,
            n_trials=n_trials, random_state=config["sample"]["random_state"],
        )
        if best_params_xgb:
            import xgboost as xgb
            xgb_reg_tuned = xgb.XGBRegressor(**best_params_xgb, n_jobs=-1, verbosity=0)
            xgb_reg_tuned.fit(x_train_xgb, y_reg_train)
            pred_reg = xgb_reg_tuned.predict(x_test_xgb)
            metrics_reg = regression_metrics(y_reg_test, pred_reg, sample_weight=w_test)
            tuning_rows.append({"task": "regression", "model": "xgboost_tuned", "feature_set": "main",
                                "n_train": len(x_train), "n_test": len(x_test),
                                "weighted_metrics": w_test is not None, **metrics_reg})
            fitted["regression_xgboost_tuned"] = xgb_reg_tuned
            regression_predictions["xgboost_tuned"] = pred_reg

        print(f"Optuna tuning: XGBoost classifier ({n_trials} trials)", flush=True)
        best_params_xgb_clf = tune_xgboost_classifier(
            x_tune_xgb, y_clf_tune, x_val_xgb, y_clf_val,
            n_trials=n_trials, random_state=config["sample"]["random_state"],
        )
        if best_params_xgb_clf:
            xgb_clf_tuned = xgb.XGBClassifier(**best_params_xgb_clf, n_jobs=-1, verbosity=0)
            xgb_clf_tuned.fit(x_train_xgb, y_clf_train)
            score_clf = model_predict_score(xgb_clf_tuned, x_test_xgb)
            metrics_clf = classification_metrics(
                y_clf_test, score_clf, config["models"]["classification_threshold"], sample_weight=w_test
            )
            tuning_rows.append({"task": "classification", "model": "xgboost_tuned", "feature_set": "main",
                                "n_train": len(x_train), "n_test": len(x_test),
                                "weighted_metrics": w_test is not None,
                                "threshold": config["models"]["classification_threshold"], **metrics_clf})
            fitted["classification_xgboost_tuned"] = xgb_clf_tuned
            classification_scores["xgboost_tuned"] = score_clf

    # ── Stacking ensemble ────────────────────────────────────────────
    if "lightgbm" in enabled_optional and "xgboost" in enabled_optional:
        x_train_stack = x_train_xgb if cat_cols else x_train
        x_test_stack = x_test_xgb if cat_cols else x_test

        print("Building stacking regressor", flush=True)
        stack_reg = build_stacking_regressor(random_state=config["sample"]["random_state"])
        stack_reg.fit(x_train_stack, y_reg_train)
        pred_reg = stack_reg.predict(x_test_stack)
        metrics_reg = regression_metrics(y_reg_test, pred_reg, sample_weight=w_test)
        tuning_rows.append({"task": "regression", "model": "stacking_ensemble", "feature_set": "main",
                            "n_train": len(x_train), "n_test": len(x_test),
                            "weighted_metrics": w_test is not None, **metrics_reg})
        fitted["regression_stacking_ensemble"] = stack_reg
        regression_predictions["stacking_ensemble"] = pred_reg

        print("Building stacking classifier", flush=True)
        stack_clf = build_stacking_classifier(random_state=config["sample"]["random_state"])
        stack_clf.fit(x_train_stack, y_clf_train)
        score_clf = model_predict_score(stack_clf, x_test_stack)
        metrics_clf = classification_metrics(
            y_clf_test, score_clf, config["models"]["classification_threshold"], sample_weight=w_test
        )
        tuning_rows.append({"task": "classification", "model": "stacking_ensemble", "feature_set": "main",
                            "n_train": len(x_train), "n_test": len(x_test),
                            "weighted_metrics": w_test is not None,
                            "threshold": config["models"]["classification_threshold"], **metrics_clf})
        fitted["classification_stacking_ensemble"] = stack_clf
        classification_scores["stacking_ensemble"] = score_clf

    if tuning_rows:
        results = pd.concat([pd.DataFrame(rows), pd.DataFrame(tuning_rows)], ignore_index=True)
    else:
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
