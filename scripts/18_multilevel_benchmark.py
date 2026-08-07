#!/usr/bin/env python3
"""Multilevel benchmarking: per-country OLS pooling + ICC decomposition.

PISA data violates ML model independence assumptions (students nested
in countries). This script provides multilevel benchmarks to complement
tree-based models.

Approach:
  (1) ICC decomposition (country-level, school-level)
  (2) Per-country Ridge regression with meta-analytic coefficient pooling
  (3) Spearman rank correlation between multilevel |z| and SHAP importance

Refs: Soares (2024) JPA; Kim et al. (2024) SAGE Open.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from pisa_xai.config import load_config, resolve_project_path
from scipy.stats import spearmanr


def get_features(df, config):
    exclude = {
        config["pisa"]["country"], config["pisa"]["student_id"],
        config["pisa"]["school_id"], config["pisa"]["student_weight"],
        "MATH_PV_MEAN", "LOW_PERFORMER_MATH",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith(("PV", "W_FSTURWT"))]


def compute_icc(y, groups):
    """ANOVA ICC: σ²_between / (σ²_between + σ²_within)."""
    y, groups = np.asarray(y, float), np.asarray(groups, str)
    grand = np.nanmean(y)
    unique = np.unique(groups)
    n, k = len(y), len(unique)
    ssb = ssw = 0.0
    for g in unique:
        m = groups == g
        gm = np.nanmean(y[m])
        ssb += m.sum() * (gm - grand) ** 2
        ssw += np.nansum((y[m] - gm) ** 2)
    msb = ssb / (k - 1) if k > 1 else 0.0
    msw = ssw / (n - k) if n > k else 1.0
    n0 = (n - sum((groups == g).sum()**2 for g in unique) / n) / (k - 1) if k > 1 else 1.0
    vb = max((msb - msw) / n0, 0.0) if n0 > 0 else 0.0
    vw = msw
    icc = vb / (vb + vw) if (vb + vw) > 0 else 0.0
    return {"between_var": float(vb), "within_var": float(vw), "icc": float(icc),
            "n_groups": k, "n_obs": n}


def per_country_pooled_coefficients(df, features, outcome, country_col, max_rows=80000):
    """Fit Ridge per country, pool coefficients via meta-analytic mean."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=20260510)

    num_features = [f for f in features if f in df.columns and str(df[f].dtype) in ("float64", "int64")]
    X = df[num_features].copy().fillna(df[num_features].median())
    y = df[outcome].values
    countries = df[country_col].astype(str).values

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    data = X_scaled.copy()
    data["_y"] = y
    data["_country"] = countries
    data = data.dropna()

    coefs = {}
    for grp_name, grp in data.groupby("_country"):
        if len(grp) < 50 or grp["_y"].nunique() < 5:
            continue
        model = Ridge(alpha=1.0, random_state=20260510)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(grp[X_scaled.columns], grp["_y"])
        for j, col in enumerate(X_scaled.columns):
            coefs.setdefault(col, []).append(model.coef_[j])

    results = []
    for col in X_scaled.columns:
        vals = coefs.get(col, [])
        if len(vals) < 5:
            continue
        mean_c = np.mean(vals)
        se_c = np.std(vals, ddof=1) / np.sqrt(len(vals))
        z = mean_c / se_c if se_c > 0 else 0.0
        results.append({
            "variable": col, "coefficient": float(mean_c),
            "std_error": float(se_c), "z_statistic": float(z),
            "p_value": float(2 * (1 - _ncdf(abs(z)))),
            "abs_z": abs(float(z)), "n_countries": len(vals),
        })
    results.sort(key=lambda r: r["abs_z"], reverse=True)
    return {"coefficients": results, "n_countries_used": data["_country"].nunique(),
            "n_obs": len(data)}


def _ncdf(x):
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def compare_with_shap(ml_results, tables_dir):
    """Compare multilevel |z| rankings with SHAP rankings."""
    shap_path = Path(tables_dir) / "shap_importance_classification.csv"
    if not shap_path.exists():
        return None
    shap_df = pd.read_csv(shap_path)
    if "variable" not in shap_df.columns or "importance" not in shap_df.columns:
        return None

    ml_rank = pd.DataFrame(ml_results["coefficients"]).set_index("variable")["abs_z"].rank(ascending=False)
    shap_rank = shap_df.set_index("variable")["importance"].rank(ascending=False)
    common = ml_rank.index.intersection(shap_rank.index)
    if len(common) < 5:
        return None
    rho, pval = spearmanr(ml_rank.loc[common], shap_rank.loc[common])
    return {"n_common_vars": len(common), "spearman_rho": float(rho), "p_value": float(pval)}


def main():
    config = load_config()
    country_col = config["pisa"]["country"]
    school_col = config["pisa"]["school_id"]

    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PISA 2022 model frame...")
    df = pd.read_parquet(processed_dir / "pisa2022_math_model_frame.parquet")
    features = get_features(df, config)
    print(f"  {len(df):,} students, {df[country_col].nunique()} countries, {len(features)} features")

    # 1. ICC decomposition
    print("\n─── ICC Decomposition ───")
    for name, outcome in [("Math Score", "MATH_PV_MEAN"), ("Low Performer", "LOW_PERFORMER_MATH")]:
        icc_c = compute_icc(df[outcome], df[country_col])
        icc_s = compute_icc(df[outcome], df[school_col])
        print(f"  {name}: Country ICC={icc_c['icc']:.4f}, School ICC={icc_s['icc']:.4f}")
        if name == "Math Score":
            reg_icc_c, reg_icc_s = icc_c, icc_s
        else:
            clf_icc_c, clf_icc_s = icc_c, icc_s

    # 2. Per-country pooled coefficients
    print("\n─── Per-Country Pooled Regression ───")
    reg_res = per_country_pooled_coefficients(df, features, "MATH_PV_MEAN", country_col)
    print(f"  N countries: {reg_res['n_countries_used']}, N obs: {reg_res['n_obs']:,}")
    print("  Top 10 by |z|:")
    for c in reg_res["coefficients"][:10]:
        print(f"    {c['variable']:20s}  z={c['z_statistic']:+.3f}  p={c['p_value']:.4f}  n_ctry={c['n_countries']}")

    print("\n─── Per-Country Pooled Classification ───")
    clf_res = per_country_pooled_coefficients(df, features, "LOW_PERFORMER_MATH", country_col)
    print(f"  N countries: {clf_res['n_countries_used']}, N obs: {clf_res['n_obs']:,}")
    print("  Top 10 by |z|:")
    for c in clf_res["coefficients"][:10]:
        print(f"    {c['variable']:20s}  z={c['z_statistic']:+.3f}  p={c['p_value']:.4f}  n_ctry={c['n_countries']}")

    # 3. Compare with SHAP
    print("\n─── ML vs Multilevel Rank Comparison ───")
    comp = compare_with_shap(clf_res, tables_dir)
    if comp:
        print(f"  Spearman's ρ (SHAP vs Multilevel |z|): {comp['spearman_rho']:.3f}")
        print(f"  p={comp['p_value']:.4f}, N common vars={comp['n_common_vars']}")
    else:
        # Fallback: report top-5 overlap
        shap_path = tables_dir / "shap_importance_classification.csv"
        if shap_path.exists():
            shap = pd.read_csv(shap_path)
            ml_top5 = {c["variable"] for c in clf_res["coefficients"][:5]}
            shap_top5 = set(shap.iloc[:5, 0].tolist() if "variable" in shap.columns else shap.columns[:5])
            overlap = ml_top5 & shap_top5
            print(f"  Top-5 overlap (ML ∩ SHAP): {len(overlap)}/5 — {overlap}")
        else:
            print("  SHAP importance file not found; skipping")

    # 4. Save
    output = {
        "icc": {"regression_country": reg_icc_c, "regression_school": reg_icc_s,
                "classification_country": clf_icc_c, "classification_school": clf_icc_s},
        "classification_top15": clf_res["coefficients"][:15],
        "regression_top15": reg_res["coefficients"][:15],
        "ml_multilevel_comparison": comp,
    }
    with open(tables_dir / "multilevel_benchmark.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {tables_dir / 'multilevel_benchmark.json'}")

    for task, res in [("classification", clf_res), ("regression", reg_res)]:
        pd.DataFrame(res["coefficients"]).to_csv(tables_dir / f"multilevel_coefficients_{task}.csv", index=False)
        print(f"Saved: {tables_dir / f'multilevel_coefficients_{task}.csv'}")


if __name__ == "__main__":
    main()
