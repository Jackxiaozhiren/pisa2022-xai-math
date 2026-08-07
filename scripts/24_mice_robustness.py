#!/usr/bin/env python3
"""MICE robustness: compare median imputation vs iterative imputation.

Confirms that main findings are stable across imputation methods.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from pisa_xai.config import load_config, resolve_project_path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pisa_xai.evaluation import classification_metrics, regression_metrics


def get_features(df, config):
    exclude = {
        config["pisa"]["country"], config["pisa"]["student_id"],
        config["pisa"]["school_id"], config["pisa"]["student_weight"],
        "MATH_PV_MEAN", "LOW_PERFORMER_MATH",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith(("PV", "W_FSTURWT"))]


def compare_imputation(x_tr, y_tr, x_te, y_te, w_tr, w_te, task, rs):
    """Compare median imputation vs MICE on a lightweight model."""
    results = []

    # Prep: separate numeric / categorical
    num_cols = list(x_tr.select_dtypes(include=[np.number]).columns)
    cat_cols = list(x_tr.select_dtypes(exclude=[np.number]).columns)

    # ── Method 1: Median / Mode imputation ──
    x_tr_m = x_tr.copy()
    x_te_m = x_te.copy()
    for c in num_cols:
        med = x_tr[c].median()
        x_tr_m[c] = x_tr_m[c].fillna(med)
        x_te_m[c] = x_te_m[c].fillna(med)
    for c in cat_cols:
        mode_v = x_tr[c].mode().iloc[0] if not x_tr[c].mode().empty else "missing"
        x_tr_m[c] = x_tr_m[c].fillna(mode_v)
        x_te_m[c] = x_te_m[c].fillna(mode_v)

    # One-hot encode
    if cat_cols:
        x_tr_m = pd.get_dummies(x_tr_m, columns=cat_cols)
        x_te_m = pd.get_dummies(x_te_m, columns=cat_cols)
        common = x_tr_m.columns.intersection(x_te_m.columns)
        x_tr_m, x_te_m = x_tr_m[common], x_te_m[common]

    scaler = StandardScaler()
    x_tr_ms = scaler.fit_transform(x_tr_m)
    x_te_ms = scaler.transform(x_te_m)

    if task == "classification":
        m = HistGradientBoostingClassifier(random_state=rs, max_iter=200)
        m.fit(x_tr_ms, y_tr)
        pred = m.predict_proba(x_te_ms)[:, 1]
        metrics = classification_metrics(y_te, pred, sample_weight=w_te)
    else:
        m = HistGradientBoostingRegressor(random_state=rs, max_iter=200)
        m.fit(x_tr_ms, y_tr)
        pred = m.predict(x_te_ms)
        metrics = regression_metrics(y_te, pred, sample_weight=w_te)
    results.append({"method": "median_imputation", **metrics})

    # ── Method 2: MICE (IterativeImputer) ──
    # Only use numeric + encoded categorical for MICE
    x_tr_enc = pd.get_dummies(x_tr.fillna(x_tr.median(numeric_only=True)), columns=cat_cols)
    x_te_enc = pd.get_dummies(x_te.fillna(x_te.median(numeric_only=True)), columns=cat_cols)
    common_cols = x_tr_enc.columns.intersection(x_te_enc.columns)
    x_tr_enc, x_te_enc = x_tr_enc[common_cols], x_te_enc[common_cols]

    # Remove constant columns
    const_cols = [c for c in x_tr_enc.columns if x_tr_enc[c].nunique() <= 1]
    x_tr_enc = x_tr_enc.drop(columns=const_cols, errors="ignore")
    x_te_enc = x_te_enc.drop(columns=const_cols, errors="ignore")

    n_imp = 3
    mice_metrics_list = []
    for i in range(n_imp):
        seed = rs + i * 1000
        imp = IterativeImputer(
            estimator=HistGradientBoostingRegressor(random_state=seed, max_iter=50),
            max_iter=5, random_state=seed, sample_posterior=True,
        )
        x_tr_imp = pd.DataFrame(imp.fit_transform(x_tr_enc), columns=x_tr_enc.columns)
        x_te_imp = pd.DataFrame(imp.transform(x_te_enc), columns=x_te_enc.columns)

        scaler_i = StandardScaler()
        x_tr_imp_s = scaler_i.fit_transform(x_tr_imp)
        x_te_imp_s = scaler_i.transform(x_te_imp)

        if task == "classification":
            mdl = HistGradientBoostingClassifier(random_state=rs, max_iter=200)
            mdl.fit(x_tr_imp_s, y_tr)
            p = mdl.predict_proba(x_te_imp_s)[:, 1]
            met = classification_metrics(y_te, p, sample_weight=w_te)
        else:
            mdl = HistGradientBoostingRegressor(random_state=rs, max_iter=200)
            mdl.fit(x_tr_imp_s, y_tr)
            p = mdl.predict(x_te_imp_s)
            met = regression_metrics(y_te, p, sample_weight=w_te)
        mice_metrics_list.append(met)

    # Pool MICE results
    pooled = {}
    for key in mice_metrics_list[0]:
        vals = [m[key] for m in mice_metrics_list if key in m and m[key] is not None]
        if vals:
            pooled[key] = float(np.mean(vals))
            pooled[f"{key}_std"] = float(np.std(vals, ddof=1))
    results.append({"method": "mice_pooled", **pooled})
    return results


def main():
    config = load_config()
    rs = config["sample"]["random_state"]
    weight_col = config["pisa"]["student_weight"]

    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(processed_dir / "pisa2022_math_model_frame.parquet")
    features = get_features(df, config)

    # Missingness report
    x_features = df[features]
    missing = (x_features.isnull().sum() / len(x_features) * 100).sort_values(ascending=False)
    print("Top 10 variables by missingness:")
    for var, pct in missing.head(10).items():
        flag = " *** HIGH" if pct > 50 else (" ** MOD" if pct > 20 else "")
        print(f"  {var:20s}: {pct:6.2f}%{flag}")
    print(f"Complete cases: {x_features.dropna().shape[0] / len(x_features) * 100:.2f}%")

    # Split
    y_reg = df["MATH_PV_MEAN"]
    y_clf = df["LOW_PERFORMER_MATH"]
    w = df[weight_col] / df[weight_col].mean()

    x_tr, x_te, y_reg_tr, y_reg_te, y_clf_tr, y_clf_te, w_tr, w_te = train_test_split(
        x_features, y_reg, y_clf, w, test_size=0.2, random_state=rs, stratify=y_clf,
    )
    print(f"\nTrain: {len(x_tr):,}, Test: {len(x_te):,}")

    # Compare imputation methods
    print("\n─── Classification: Imputation Comparison ───")
    clf_res = compare_imputation(x_tr, y_clf_tr, x_te, y_clf_te, w_tr, w_te, "classification", rs)
    for r in clf_res:
        a = r.get("auc")
        b = r.get("brier")
        print(f"  {r['method']:25s}  AUC={'N/A' if a is None else f'{a:.4f}'}  Brier={'N/A' if b is None else f'{b:.4f}'}")

    print("\n─── Regression: Imputation Comparison ───")
    reg_res = compare_imputation(x_tr, y_reg_tr, x_te, y_reg_te, w_tr, w_te, "regression", rs)
    for r in reg_res:
        rm = r.get("rmse")
        r2 = r.get("r_squared")
        print(f"  {r['method']:25s}  RMSE={'N/A' if rm is None else f'{rm:.2f}'}  R²={'N/A' if r2 is None else f'{r2:.4f}'}")

    # Save
    output = {
        "missingness_pct": {var: float(pct) for var, pct in missing.items()},
        "complete_case_pct": float(x_features.dropna().shape[0] / len(x_features) * 100),
        "classification": clf_res,
        "regression": reg_res,
    }
    out_path = tables_dir / "mice_robustness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
