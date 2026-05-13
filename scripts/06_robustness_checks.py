#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.evaluation import (
    calibration_bins,
    calibration_summary,
    classification_metrics,
    regression_metrics,
)
from pisa_xai.io import load_table, require_package
from pisa_xai.modeling import classification_models, make_preprocessor, regression_models


def sample_rows(df, max_rows: int, random_state: int):
    if len(df) <= max_rows:
        return df
    return (
        df.assign(_stratify=df["LOW_PERFORMER_MATH"])
        .groupby("_stratify", group_keys=False)
        .sample(frac=max_rows / len(df), random_state=random_state)
        .drop(columns="_stratify")
        .sort_index()
    )


def add_escs_quintile(df):
    require_package("pandas", "pip install -r requirements.txt")
    import pandas as pd

    result = df.copy()
    result["ESCS_QUINTILE"] = "missing"
    if "ESCS" not in result.columns:
        return result
    valid = result["ESCS"].notna()
    result.loc[valid, "ESCS_QUINTILE"] = pd.qcut(
        result.loc[valid, "ESCS"].rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    ).astype(str)
    return result


def subgroup_metrics(predictions, weight_col: str):
    rows = []
    grouped = add_escs_quintile(predictions)
    for variable in ["ST004D01T", "IMMIG", "ESCS_QUINTILE"]:
        if variable not in grouped.columns:
            continue
        for value, group in grouped.groupby(variable, dropna=False):
            if group["LOW_PERFORMER_MATH"].nunique() < 2:
                continue
            rows.append(
                {
                    "group_variable": variable,
                    "group_value": value,
                    "n_holdout": len(group),
                    **classification_metrics(
                        group["LOW_PERFORMER_MATH"],
                        group["best_classification_score"],
                        sample_weight=group[weight_col],
                    ),
                    **{
                        f"regression_{key}": value
                        for key, value in regression_metrics(
                            group["MATH_PV_MEAN"],
                            group["best_regression_prediction"],
                            sample_weight=group[weight_col],
                        ).items()
                    },
                }
            )
    return rows


def model_predict_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def fit_with_optional_weight(model, x_train, y_train, sample_weight):
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    try:
        model.fit(x_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(x_train, y_train)
    return model


def choose_model(models, preferred_order: list[str]):
    for name in preferred_order:
        if name in models:
            return name, models[name]
    return next(iter(models.items()))


def calibration_outputs(predictions, weight_col: str):
    summary = calibration_summary(
        predictions["LOW_PERFORMER_MATH"],
        predictions["best_classification_score"],
        sample_weight=predictions[weight_col],
    )
    bins = calibration_bins(
        predictions["LOW_PERFORMER_MATH"],
        predictions["best_classification_score"],
        sample_weight=predictions[weight_col],
        n_bins=10,
    )
    summary["expected_calibration_error"] = bins.attrs["expected_calibration_error"]
    return summary, bins


def country_group_holdout_metrics(df, features, config):
    require_package("sklearn", "pip install -r requirements.txt")
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    random_state = config["sample"]["random_state"]
    max_rows = config["models"].get("robustness_max_rows", 120000)
    sampled = sample_rows(df, max_rows, random_state)
    weight_col = config["pisa"]["student_weight"]
    country_col = config["pisa"]["country"]
    feature_list = [feature for feature in features if feature in sampled.columns]
    groups = sampled[country_col].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config["models"]["test_size"],
        random_state=random_state,
    )
    train_pos, test_pos = next(splitter.split(sampled, groups=groups))
    train_idx = sampled.index[train_pos]
    test_idx = sampled.index[test_pos]

    x = sampled[feature_list]
    y_reg = sampled["MATH_PV_MEAN"]
    y_clf = sampled["LOW_PERFORMER_MATH"]
    weights = (
        sampled[weight_col] / sampled[weight_col].mean()
        if weight_col in sampled.columns
        else None
    )
    x_train, x_test = x.loc[train_idx], x.loc[test_idx]
    w_train = weights.loc[train_idx] if weights is not None else None
    w_test = weights.loc[test_idx] if weights is not None else None

    enabled_optional = config["models"].get("enabled_optional_models", [])
    reg_name, reg_model = choose_model(
        regression_models(x_train, enabled_optional),
        ["lightgbm", "hist_gradient_boosting", "xgboost", "random_forest", "ridge"],
    )
    clf_name, clf_model = choose_model(
        classification_models(x_train, enabled_optional),
        ["lightgbm", "hist_gradient_boosting", "xgboost", "random_forest", "logistic_l2"],
    )

    fit_with_optional_weight(reg_model, x_train, y_reg.loc[train_idx], w_train)
    fit_with_optional_weight(clf_model, x_train, y_clf.loc[train_idx], w_train)
    reg_metrics = regression_metrics(
        y_reg.loc[test_idx],
        reg_model.predict(x_test),
        sample_weight=w_test,
    )
    clf_metrics = classification_metrics(
        y_clf.loc[test_idx],
        model_predict_score(clf_model, x_test),
        sample_weight=w_test,
    )
    heldout_countries = sorted(sampled.loc[test_idx, country_col].astype(str).unique())
    row = {
        "robustness_check": "country_group_holdout",
        "n_rows": len(sampled),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_train_countries": sampled.loc[train_idx, country_col].nunique(dropna=True),
        "n_test_countries": len(heldout_countries),
        "heldout_countries": "; ".join(heldout_countries),
        "n_features": len(feature_list),
        "regression_model": reg_name,
        "classification_model": clf_name,
        **reg_metrics,
        **{f"classification_{key}": value for key, value in clf_metrics.items()},
    }
    return pd.DataFrame([row])


def country_effect_metrics(df, features, config):
    require_package("sklearn", "pip install -r requirements.txt")
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    random_state = config["sample"]["random_state"]
    max_rows = config["models"].get("robustness_max_rows", 120000)
    sampled = sample_rows(df, max_rows, random_state)
    weight_col = config["pisa"]["student_weight"]
    country_col = config["pisa"]["country"]

    rows = []
    for label, feature_list in [
        ("without_country_fixed_effects", features),
        ("with_country_fixed_effects", features + [country_col]),
    ]:
        feature_list = [feature for feature in feature_list if feature in sampled.columns]
        x = sampled[feature_list]
        y_reg = sampled["MATH_PV_MEAN"]
        y_clf = sampled["LOW_PERFORMER_MATH"]
        w = sampled[weight_col] / sampled[weight_col].mean() if weight_col in sampled.columns else None
        train_idx, test_idx = train_test_split(
            sampled.index,
            test_size=config["models"]["test_size"],
            random_state=random_state,
            stratify=y_clf,
        )
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        w_train = w.loc[train_idx] if w is not None else None
        w_test = w.loc[test_idx] if w is not None else None

        reg = Pipeline(
            [
                ("preprocess", make_preprocessor(x_train)),
                ("model", Ridge(alpha=1.0, random_state=random_state)),
            ]
        )
        clf = Pipeline(
            [
                ("preprocess", make_preprocessor(x_train)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        fit_kwargs = {"model__sample_weight": w_train} if w_train is not None else {}
        reg.fit(x_train, y_reg.loc[train_idx], **fit_kwargs)
        clf.fit(x_train, y_clf.loc[train_idx], **fit_kwargs)
        reg_metrics = regression_metrics(
            y_reg.loc[test_idx],
            reg.predict(x_test),
            sample_weight=w_test,
        )
        clf_metrics = classification_metrics(
            y_clf.loc[test_idx],
            clf.predict_proba(x_test)[:, 1],
            sample_weight=w_test,
        )
        rows.append(
            {
                "robustness_check": label,
                "n_rows": len(sampled),
                "n_features": len(feature_list),
                **reg_metrics,
                **{f"classification_{key}": value for key, value in clf_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    require_package("pandas", "pip install -r requirements.txt")
    import pandas as pd

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    predictions = pd.read_csv(tables_dir / "holdout_predictions.csv")
    features = json.loads((processed_dir / "models" / "features.json").read_text(encoding="utf-8"))

    subgroup = pd.DataFrame(subgroup_metrics(predictions, "sample_weight"))
    subgroup.to_csv(tables_dir / "subgroup_holdout_metrics.csv", index=False)

    calibration, calibration_bin_table = calibration_outputs(predictions, "sample_weight")
    pd.DataFrame([calibration]).to_csv(tables_dir / "calibration_metrics.csv", index=False)
    calibration_bin_table.to_csv(tables_dir / "calibration_bins.csv", index=False)

    oecd = set(config.get("robustness", {}).get("oecd_countries", []))
    oecd_holdout = predictions[predictions["CNT"].isin(oecd)]
    if len(oecd_holdout) and oecd_holdout["LOW_PERFORMER_MATH"].nunique() > 1:
        oecd_metrics = {
            **classification_metrics(
                oecd_holdout["LOW_PERFORMER_MATH"],
                oecd_holdout["best_classification_score"],
                sample_weight=oecd_holdout["sample_weight"],
            ),
            **{
                f"regression_{key}": value
                for key, value in regression_metrics(
                    oecd_holdout["MATH_PV_MEAN"],
                    oecd_holdout["best_regression_prediction"],
                    sample_weight=oecd_holdout["sample_weight"],
                ).items()
            },
        }
        pd.DataFrame([{ "scope": "OECD holdout evaluation", "n_holdout": len(oecd_holdout), **oecd_metrics }]).to_csv(
            tables_dir / "oecd_holdout_metrics.csv",
            index=False,
        )

    complete_case = df[features].notna().all(axis=1)
    pd.DataFrame(
        [
            {
                "feature_set": "main",
                "n_total": len(df),
                "n_complete_case": int(complete_case.sum()),
                "complete_case_rate": float(complete_case.mean()),
                "low_performer_rate_all": float(df["LOW_PERFORMER_MATH"].mean()),
                "low_performer_rate_complete_case": float(df.loc[complete_case, "LOW_PERFORMER_MATH"].mean()),
            }
        ]
    ).to_csv(tables_dir / "complete_case_sensitivity.csv", index=False)

    threshold = config["pisa"]["low_performer_threshold"]
    pv_cols = [f"PV{i}MATH" for i in range(1, config["pisa"]["math_pv_count"] + 1) if f"PV{i}MATH" in df.columns]
    alt_rows = [
        {
            "label_rule": "mean_plausible_value",
            "low_performer_rate": float((df["MATH_PV_MEAN"] < threshold).mean()),
        }
    ]
    for pv_col in pv_cols:
        alt_rows.append(
            {
                "label_rule": pv_col,
                "low_performer_rate": float((df[pv_col] < threshold).mean()),
            }
        )
    pd.DataFrame(alt_rows).to_csv(tables_dir / "alternative_low_performer_labels.csv", index=False)

    country_effect_metrics(df, features, config).to_csv(
        tables_dir / "country_fixed_effects_sensitivity.csv",
        index=False,
    )
    country_group_holdout_metrics(df, features, config).to_csv(
        tables_dir / "country_group_holdout_metrics.csv",
        index=False,
    )

    print("Robustness outputs written to reports/tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
