#!/usr/bin/env python3
"""5-fold CV for primary models using default parameters (~1-2h)."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table
from pisa_xai.evaluation import classification_metrics, regression_metrics

RS = 20260510; N_FOLDS = 5

def main():
    t0 = time.time()
    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    feature_sets = json.loads((processed_dir / "feature_sets.json").read_text(encoding="utf-8"))
    features = [f for f in feature_sets["main_features"] if f in df.columns]

    cat_cols = [c for c in features if str(df[c].dtype) == "category"]
    print(f"Features: {len(features)} ({len(cat_cols)} cat)")

    # Numeric version (for XGBoost which needs int codes)
    X_xgb = df[features].copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes.astype("int8")

    # Categorical version (for LGBM/HistGB which support native categorical)
    X_cat = df[features].copy()

    y_reg = df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)
    y_clf = (y_reg < 420.07).astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)

    metrics = {"XGBoost": {"auc": [], "rmse": [], "r2": []},
               "LightGBM": {"auc": [], "rmse": [], "r2": []},
               "HistGB": {"auc": [], "rmse": [], "r2": []}}

    for fold, (train_i, test_i) in enumerate(skf.split(df, y_clf)):
        print(f"\nFold {fold+1}/{N_FOLDS} ...", end=" ", flush=True)
        tf = time.time()

        # XGBoost (numeric)
        xgb_c = xgb.XGBClassifier(random_state=RS, verbosity=0, n_jobs=-1)
        xgb_c.fit(X_xgb.iloc[train_i], y_clf.iloc[train_i])
        metrics["XGBoost"]["auc"].append(classification_metrics(y_clf.iloc[test_i], xgb_c.predict_proba(X_xgb.iloc[test_i])[:, 1])["auc"])
        xgb_r = xgb.XGBRegressor(random_state=RS, verbosity=0, n_jobs=-1)
        xgb_r.fit(X_xgb.iloc[train_i], y_reg.iloc[train_i])
        metrics["XGBoost"]["rmse"].append(regression_metrics(y_reg.iloc[test_i], xgb_r.predict(X_xgb.iloc[test_i]))["rmse"])
        metrics["XGBoost"]["r2"].append(regression_metrics(y_reg.iloc[test_i], xgb_r.predict(X_xgb.iloc[test_i]))["r2"])

        # LightGBM (categorical)
        lgb_c = lgb.LGBMClassifier(random_state=RS, verbose=-1, n_jobs=-1)
        lgb_c.fit(X_cat.iloc[train_i], y_clf.iloc[train_i])
        metrics["LightGBM"]["auc"].append(classification_metrics(y_clf.iloc[test_i], lgb_c.predict_proba(X_cat.iloc[test_i])[:, 1])["auc"])
        lgb_r = lgb.LGBMRegressor(random_state=RS, verbose=-1, n_jobs=-1)
        lgb_r.fit(X_cat.iloc[train_i], y_reg.iloc[train_i])
        metrics["LightGBM"]["rmse"].append(regression_metrics(y_reg.iloc[test_i], lgb_r.predict(X_cat.iloc[test_i]))["rmse"])
        metrics["LightGBM"]["r2"].append(regression_metrics(y_reg.iloc[test_i], lgb_r.predict(X_cat.iloc[test_i]))["r2"])

        # HistGB (categorical)
        hgb_c = HistGradientBoostingClassifier(random_state=RS, categorical_features=cat_cols)
        hgb_c.fit(X_cat.iloc[train_i], y_clf.iloc[train_i])
        metrics["HistGB"]["auc"].append(classification_metrics(y_clf.iloc[test_i], hgb_c.predict_proba(X_cat.iloc[test_i])[:, 1])["auc"])
        hgb_r = HistGradientBoostingRegressor(random_state=RS, categorical_features=cat_cols)
        hgb_r.fit(X_cat.iloc[train_i], y_reg.iloc[train_i])
        metrics["HistGB"]["rmse"].append(regression_metrics(y_reg.iloc[test_i], hgb_r.predict(X_cat.iloc[test_i]))["rmse"])
        metrics["HistGB"]["r2"].append(regression_metrics(y_reg.iloc[test_i], hgb_r.predict(X_cat.iloc[test_i]))["r2"])

        print(f"{time.time()-tf:.0f}s")

    # ── Summary ──
    out_path = tables_dir / "kfold_cv_results.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 65)
    print(f"5-Fold CV (default params, {N_FOLDS} folds, unweighted)")
    print("=" * 65)
    for model in ["XGBoost", "LightGBM", "HistGB"]:
        m = metrics[model]
        print(f"\n{model}:")
        for metric in ["auc", "rmse", "r2"]:
            vals = m[metric]
            print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  [{min(vals):.4f}, {max(vals):.4f}]")

    print(f"\nTotal: {(time.time()-t0)/60:.0f} min | Results: {out_path}")

if __name__ == "__main__":
    main()
