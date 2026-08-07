#!/usr/bin/env python3
"""Fast baseline comparison: EBM (20K sample) + HistGradientBoosting vs XGBoost-tuned."""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table
from pisa_xai.evaluation import classification_metrics, regression_metrics

def main():
    t0 = time.time()
    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    model_dir = processed_dir / "models"
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    RS = 20260510

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    feature_sets = json.loads((processed_dir / "feature_sets.json").read_text(encoding="utf-8"))
    features = [f for f in feature_sets["main_features"] if f in df.columns]
    cat_cols = [c for c in features if str(df[c].dtype) == "category"]
    print(f"Features: {len(features)} ({len(cat_cols)} cat)")

    # Numeric version
    X_all = df[features].copy()
    for c in cat_cols:
        X_all[c] = X_all[c].cat.codes.astype("int8")

    y_reg = df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)
    y_clf = (y_reg < 420.07).astype(int)

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RS, stratify=y_clf)
    X_train_full = X_all.loc[train_idx]
    X_test = X_all.loc[test_idx]
    y_reg_train = y_reg.loc[train_idx]; y_reg_test = y_reg.loc[test_idx]
    y_clf_train = y_clf.loc[train_idx]; y_clf_test = y_clf.loc[test_idx]

    # Sample for slower models
    SAMPLE_N = 30_000
    sample_idx = X_train_full.sample(SAMPLE_N, random_state=RS).index
    X_tr_s = X_train_full.loc[sample_idx]
    y_clf_tr_s = y_clf_train.loc[sample_idx]
    y_reg_tr_s = y_reg_train.loc[sample_idx]

    # XGBoost test data (needs categorical)
    X_test_xgb = df.loc[test_idx, features]

    # ── Load XGBoost ──
    xgb_clf = joblib.load(model_dir / "classification_xgboost_tuned.joblib")
    xgb_reg = joblib.load(model_dir / "regression_xgboost_tuned.joblib")
    xgb_clf_score = xgb_clf.predict_proba(X_test_xgb)[:, 1]
    xgb_reg_pred = xgb_reg.predict(X_test_xgb)
    xgb_clf_m = classification_metrics(y_clf_test, xgb_clf_score)
    xgb_reg_m = regression_metrics(y_reg_test, xgb_reg_pred)

    results = {"XGBoost_tuned": {"classification": xgb_clf_m, "regression": xgb_reg_m}}

    # ── HistGradientBoosting (fast sklearn baseline) ──
    print("Training HistGradientBoosting (full data)...")
    t = time.time()
    hgb_clf = HistGradientBoostingClassifier(random_state=RS, categorical_features=cat_cols)
    hgb_clf.fit(df.loc[train_idx, features], y_clf_train)
    hgb_clf_score = hgb_clf.predict_proba(X_test_xgb)[:, 1]
    results["HistGB"] = {
        "classification": classification_metrics(y_clf_test, hgb_clf_score),
    }
    hgb_reg = HistGradientBoostingRegressor(random_state=RS, categorical_features=cat_cols)
    hgb_reg.fit(df.loc[train_idx, features], y_reg_train)
    hgb_reg_pred = hgb_reg.predict(X_test_xgb)
    results["HistGB"]["regression"] = regression_metrics(y_reg_test, hgb_reg_pred)
    print(f"  Done in {time.time() - t:.0f}s")

    # ── EBM (small sample, fast config) ──
    print(f"Training EBM ({SAMPLE_N:,} rows, max_bins=32)...")
    t = time.time()
    ebm_clf = ExplainableBoostingClassifier(random_state=RS, interactions=0, max_bins=32)
    ebm_clf.fit(X_tr_s, y_clf_tr_s)
    ebm_clf_score = ebm_clf.predict_proba(X_test)[:, 1]
    results["EBM"] = {
        "classification": classification_metrics(y_clf_test, ebm_clf_score),
        "note": f"Trained on {SAMPLE_N:,} sample (vs XGBoost on full ~490K)",
    }
    ebm_reg = ExplainableBoostingRegressor(random_state=RS, interactions=0, max_bins=32)
    ebm_reg.fit(X_tr_s, y_reg_tr_s)
    ebm_reg_pred = ebm_reg.predict(X_test)
    results["EBM"]["regression"] = regression_metrics(y_reg_test, ebm_reg_pred)
    print(f"  Done in {time.time() - t:.0f}s")

    # ── Summary ──
    out_path = tables_dir / "ebm_baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Model Comparison — Same 20% holdout")
    print("=" * 70)
    print(f"{'Model':<20} {'AUC':>8} {'Brier':>8} {'F1':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 70)
    for name in ["XGBoost_tuned", "HistGB", "EBM"]:
        r = results[name]
        print(f"{name:<20} {r['classification']['auc']:>8.4f} {r['classification']['brier']:>8.4f} {r['classification']['f1']:>8.4f} {r['regression']['rmse']:>8.2f} {r['regression']['r2']:>8.4f}")
    print("-" * 70)
    print(f"\nTotal: {time.time() - t0:.0f}s | Results: {out_path}")

if __name__ == "__main__":
    main()
