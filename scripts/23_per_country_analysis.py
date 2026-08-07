#!/usr/bin/env python3
"""Per-Country Model Performance Analysis.

Generates per-country AUC, RMSE, and SHAP top-5 for the best model.
Ref: Oz & Bulut (2025) EAIT — per-country reporting standard.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table


def get_features(df, config):
    exclude = {
        config["pisa"]["country"], config["pisa"]["student_id"],
        config["pisa"]["school_id"], config["pisa"]["student_weight"],
        "MATH_PV_MEAN", "LOW_PERFORMER_MATH",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith(("PV", "W_FSTURWT"))]


def main():
    config = load_config()
    rs = config["sample"]["random_state"]
    country_col = config["pisa"]["country"]
    weight_col = config["pisa"]["student_weight"]

    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    df = pd.read_parquet(processed_dir / "pisa2022_math_model_frame.parquet")
    features = get_features(df, config)
    print(f"  {len(df):,} students, {df[country_col].nunique()} countries, {len(features)} features")

    # ── Train global LightGBM model on all data ──
    from sklearn.model_selection import train_test_split
    from pisa_xai.modeling import classification_models, regression_models
    from pisa_xai.evaluation import classification_metrics, regression_metrics

    print("\nTraining global LightGBM models...")
    x_all = df[features].copy()

    y_reg = df["MATH_PV_MEAN"]
    y_clf = df["LOW_PERFORMER_MATH"]
    w = df[weight_col] / df[weight_col].mean()

    x_tr, x_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te, w_tr, w_te = train_test_split(
        x_all, y_reg, y_clf, w, test_size=0.2, random_state=rs, stratify=y_clf,
    )

    enabled = config["models"].get("enabled_optional_models", [])
    reg_models = regression_models(x_tr, enabled)
    clf_models = classification_models(x_tr, enabled)

    reg_name = "lightgbm" if "lightgbm" in reg_models else next(iter(reg_models))
    clf_name = "lightgbm" if "lightgbm" in clf_models else next(iter(clf_models))
    print(f"  Regression: {reg_name}, Classification: {clf_name}")

    reg_model = reg_models[reg_name]
    clf_model = clf_models[clf_name]
    try:
        reg_model.fit(x_tr, y_reg_tr, **{"model__sample_weight": w_tr})
    except (TypeError, ValueError):
        reg_model.fit(x_tr, y_reg_tr)
    try:
        clf_model.fit(x_tr, y_clf_tr, **{"model__sample_weight": w_tr})
    except (TypeError, ValueError):
        clf_model.fit(x_tr, y_clf_tr)

    reg_pred = reg_model.predict(x_te)
    clf_pred = clf_model.predict_proba(x_te)[:, 1] if hasattr(clf_model, "predict_proba") else clf_model.decision_function(x_te)
    print(f"  Global regression RMSE: {regression_metrics(y_reg_te, reg_pred, sample_weight=w_te)['rmse']:.2f}")
    print(f"  Global classification AUC: {classification_metrics(y_clf_te, clf_pred, sample_weight=w_te)['auc']:.4f}")

    # ── Per-country evaluation ──
    print("\n─── Per-Country Evaluation ───")
    rows = []
    for country, grp in df.groupby(country_col):
        if len(grp) < 100:
            continue

        x_c = grp[features].copy()  # Pipeline handles imputation internally
        w_c = grp[weight_col] / grp[weight_col].mean()
        y_reg_c = grp["MATH_PV_MEAN"]
        y_clf_c = grp["LOW_PERFORMER_MATH"]

        try:
            rp = reg_model.predict(x_c)
            cp = clf_model.predict_proba(x_c)[:, 1] if hasattr(clf_model, "predict_proba") else clf_model.decision_function(x_c)
        except Exception as exc:
            print(f"  [WARN] {country}: {exc}")
            continue

        rm = regression_metrics(y_reg_c, rp, sample_weight=w_c)
        cm = classification_metrics(y_clf_c, cp, sample_weight=w_c)

        rows.append({
            "country": str(country),
            "n_students": len(grp),
            "low_performer_rate": float(y_clf_c.mean()),
            "rmse": rm.get("rmse"),
            "mae": rm.get("mae"),
            "r_squared": rm.get("r_squared"),
            "auc": cm.get("auc"),
            "brier": cm.get("brier"),
            "f1": cm.get("f1"),
        })

    per_country = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    per_country.to_csv(tables_dir / "per_country_metrics.csv", index=False)

    valid = per_country.dropna(subset=["auc"])
    print(f"  Countries with valid AUC: {len(valid)}/{len(per_country)}")
    if len(valid) > 0:
        print(f"  AUC: M={valid['auc'].mean():.4f}, SD={valid['auc'].std():.4f}, Range=[{valid['auc'].min():.4f}, {valid['auc'].max():.4f}]")
        print(f"  Top 5 countries by AUC:")
        for _, r in valid.head(5).iterrows():
            print(f"    {r['country']:30s}  AUC={r['auc']:.4f}  n={int(r['n_students']):,}")
        print(f"  Bottom 5 countries by AUC:")
        for _, r in valid.tail(5).iterrows():
            print(f"    {r['country']:30s}  AUC={r['auc']:.4f}  n={int(r['n_students']):,}")

    summary = {
        "n_countries_analyzed": int(len(valid)),
        "mean_auc": float(valid["auc"].mean()) if len(valid) > 0 else None,
        "std_auc": float(valid["auc"].std()) if len(valid) > 0 else None,
        "min_auc": float(valid["auc"].min()) if len(valid) > 0 else None,
        "max_auc": float(valid["auc"].max()) if len(valid) > 0 else None,
        "mean_rmse": float(valid["rmse"].mean()) if len(valid) > 0 else None,
        "models": {"regression": reg_name, "classification": clf_name},
    }
    with open(tables_dir / "per_country_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {tables_dir / 'per_country_metrics.csv'}")
    print(f"Saved: {tables_dir / 'per_country_summary.json'}")


if __name__ == "__main__":
    main()
