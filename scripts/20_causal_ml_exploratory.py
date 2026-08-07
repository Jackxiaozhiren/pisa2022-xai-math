#!/usr/bin/env python3
r"""Exploratory causal machine learning analysis for key ICT variables.

Estimates Conditional Average Treatment Effects (CATE) of ICT resources and
ICT self-efficacy on mathematics achievement using Double Machine Learning
(Double ML) and Causal Forest.

⚠️ CRITICAL CAVEAT: This is an EXPLORATORY analysis. PISA 2022 is
observational/cross-sectional data. The following causal identification
assumptions are NOT guaranteed:
  1. Unconfoundedness (no unobserved confounders)
  2. Overlap/positivity (sufficient variation in treatment assignment)
  3. Stable Unit Treatment Value Assumption (SUTVA)

Results serve as hypothesis-generating evidence, NOT causal proof. They
should be interpreted as "what the data would suggest IF identification
assumptions hold" rather than "what policy should do."

References:
    Chernozhukov et al. (2018). Double ML. The Econometrics Journal, 21(1).
    Athey & Imbens (2016). Recursive partitioning for heterogeneous CATE. PNAS.
    Athey, Tibshirani & Wager (2019). Generalized Random Forests. Annals of Statistics.
    McJames et al. (2024). Bayesian Causal Forests for educational assessment.
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


def prepare_treatment_outcome(df, features, treatment_var, outcome_var, confounders):
    """Prepare treatment, outcome, and confounder matrices.

    Parameters
    ----------
    treatment_var : str
        Continuous variable to binarize at median (e.g., "ICTRES").
        For binary treatments, pass directly.
    confounders : list[str]
        Variables to control for (must satisfy unconfoundedness assumption
        for valid causal interpretation).
    """
    import numpy as np
    import pandas as pd

    valid = df[treatment_var].notna() & df[outcome_var].notna()
    for c in confounders:
        valid = valid & df[c].notna()

    df_clean = df[valid].copy()

    if df_clean[treatment_var].nunique() > 2:
        median_val = df_clean[treatment_var].median()
        df_clean["_treatment"] = (df_clean[treatment_var] > median_val).astype(int)
        treat_type = "binary (above median)"
    else:
        df_clean["_treatment"] = df_clean[treatment_var].astype(int)
        treat_type = "binary"

    confounder_cols = [c for c in confounders if c in df_clean.columns]
    return df_clean, treat_type


def double_ml_estimate(
    df, treatment_var, outcome_var, confounders, random_state: int = 20260510
):
    """Estimate ATE/CATE using Double ML with XGBoost nuisance models."""
    require_package("numpy", "pip install numpy")
    require_package("sklearn", "pip install -r requirements.txt")
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    df_clean, treat_type = prepare_treatment_outcome(
        df, [], treatment_var, outcome_var, confounders
    )

    y = df_clean[outcome_var].values
    d = df_clean["_treatment"].values
    x_conf_df = df_clean[[c for c in confounders if c in df_clean.columns]].copy()
    for col in x_conf_df.columns:
        if x_conf_df[col].dtype == object:
            x_conf_df[col] = x_conf_df[col].astype('category').cat.codes
        elif str(x_conf_df[col].dtype) == 'category':
            x_conf_df[col] = x_conf_df[col].cat.codes
        x_conf_df[col] = x_conf_df[col].astype(float)
    x_conf = x_conf_df.fillna(0.0).values

    scaler = StandardScaler()
    x_conf = scaler.fit_transform(x_conf)

    n = len(y)
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Cross-fitting: estimate nuisance functions and ATE
    residuals_y = np.zeros(n)
    residuals_d = np.zeros(n)

    for train_idx, test_idx in kf.split(range(n)):
        # Outcome model: E[Y | X]
        g_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=random_state
        )
        g_y.fit(x_conf[train_idx], y[train_idx])
        residuals_y[test_idx] = y[test_idx] - g_y.predict(x_conf[test_idx])

        # Propensity model: E[D | X]
        g_d = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=random_state
        )
        g_d.fit(x_conf[train_idx], d[train_idx])
        p_d = g_d.predict_proba(x_conf[test_idx])[:, 1]
        residuals_d[test_idx] = d[test_idx] - p_d

    # R-Learner: regress residual_y on residual_d
    ate = np.mean(residuals_y * residuals_d) / np.mean(residuals_d ** 2)

    # Bootstrap standard error
    n_boot = 200
    ate_boot = np.zeros(n_boot)
    rng = np.random.RandomState(random_state)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        num = np.mean(residuals_y[idx] * residuals_d[idx])
        den = np.mean(residuals_d[idx] ** 2)
        ate_boot[b] = num / den if den > 1e-10 else 0

    ate_se = np.std(ate_boot)
    pval = 2 * min(
        np.mean(ate_boot <= 0),
        np.mean(ate_boot >= 0),
    )

    return {
        "treatment": treatment_var,
        "outcome": outcome_var,
        "n_confounders": len(confounders),
        "n_observations": int(n),
        "treatment_proportion": float(np.mean(d)),
        "ate": float(ate),
        "ate_se": float(ate_se),
        "ate_ci_lower": float(ate - 1.96 * ate_se),
        "ate_ci_upper": float(ate + 1.96 * ate_se),
        "p_value_approx": float(pval),
    }


def causal_forest_estimate(
    df, treatment_var, outcome_var, confounders, random_state: int = 20260510
):
    """Estimate CATE using Causal Forest (via econml if available).

    Falls back to simplified heterogeneous effect analysis using subgroup
    ATE estimation if econml is not installed.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df_clean, treat_type = prepare_treatment_outcome(
        df, [], treatment_var, outcome_var, confounders
    )

    y = df_clean[outcome_var].values
    d = df_clean["_treatment"].values
    x_conf_df = df_clean[[c for c in confounders if c in df_clean.columns]].copy()
    for col in x_conf_df.columns:
        if x_conf_df[col].dtype == object:
            x_conf_df[col] = x_conf_df[col].astype('category').cat.codes
        elif str(x_conf_df[col].dtype) == 'category':
            x_conf_df[col] = x_conf_df[col].cat.codes
        x_conf_df[col] = x_conf_df[col].astype(float)
    x_conf = x_conf_df.fillna(0.0).values

    scaler = StandardScaler()
    x_conf = scaler.fit_transform(x_conf)

    # Try econml Causal Forest
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor

        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=random_state),
            model_t=GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=random_state),
            discrete_treatment=True,
            n_estimators=500,
            min_samples_leaf=50,
            max_depth=6,
            random_state=random_state,
        )
        cf.fit(y, d, X=x_conf)
        cate = cf.effect(x_conf)

        # Heterogeneity by ESCS quintile
        if "ESCS" in df_clean.columns:
            escs = df_clean["ESCS"].values
            escs_valid = ~np.isnan(escs)
            if escs_valid.sum() > 100:
                qtiles = pd.qcut(
                    pd.Series(escs[escs_valid]).rank(method="first"),
                    4,
                    labels=["Q1", "Q2", "Q3", "Q4"],
                )
                heterogeneity = {}
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    mask = (qtiles == q).values
                    if mask.sum() >= 20:
                        heterogeneity[str(q)] = {
                            "mean_cate": float(np.mean(cate[mask])),
                            "std_cate": float(np.std(cate[mask])),
                            "n": int(mask.sum()),
                        }
                return {"method": "econml.CausalForestDML", "heterogeneity_by_escs": heterogeneity}

        return {"method": "econml.CausalForestDML", "mean_cate": float(np.mean(cate))}

    except ImportError:
        pass

    # Fallback: subgroup ATE by ESCS quartile
    try:
        if "ESCS" not in df_clean.columns:
            return {"method": "subgroup_ate_fallback", "error": "ESCS not available"}

        import pandas as pd
        from sklearn.linear_model import LinearRegression

        escs_clean = df_clean["ESCS"].dropna()
        valid_idx = escs_clean.index
        escs_q = pd.qcut(
            escs_clean.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        results = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_idx = escs_q[escs_q == q].index
            y_q = df_clean.loc[q_idx, outcome_var].values
            d_q = df_clean.loc[q_idx, "_treatment"].values
            x_q = df_clean.loc[q_idx, [c for c in confounders if c in df_clean.columns]].values

            # Simple OLS: Y ~ D + X
            from sklearn.linear_model import LinearRegression
            import numpy as np

            scaler_q = StandardScaler()
            x_q = scaler_q.fit_transform(x_q)
            x_with_d = np.column_stack([d_q, x_q])
            ols = LinearRegression()
            ols.fit(x_with_d, y_q)
            ate_q = ols.coef_[0]

            # Bootstrap SE
            n_q = len(y_q)
            boot_ates = []
            rng = np.random.RandomState(random_state)
            for _ in range(200):
                idx_b = rng.choice(n_q, n_q, replace=True)
                ols_b = LinearRegression()
                ols_b.fit(x_with_d[idx_b], y_q[idx_b])
                boot_ates.append(ols_b.coef_[0])

            results[str(q)] = {
                "ate": float(ate_q),
                "ate_se": float(np.std(boot_ates)),
                "ate_ci_lower": float(ate_q - 1.96 * np.std(boot_ates)),
                "ate_ci_upper": float(ate_q + 1.96 * np.std(boot_ates)),
                "n": int(n_q),
            }

        return {"method": "subgroup_ate", "heterogeneity_by_escs": results}
    except Exception as exc:
        return {"method": "subgroup_ate_fallback", "error": str(exc)}


def main() -> int:
    require_package("pandas", "pip install pandas")
    require_package("numpy", "pip install numpy")
    import pandas as pd
    import numpy as np

    config = load_config(DEFAULT_CONFIG_PATH)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    processed = processed_dir / "pisa2022_math_model_frame.parquet"
    df = load_table(processed)

    features = json.loads(
        (processed_dir / "models" / "features.json").read_text(encoding="utf-8")
    )
    features = [f for f in features if f in df.columns]

    # Define treatment-outcome pairs and confounders
    # Confounders selected based on theoretical knowledge, not data-driven selection
    base_confounders = [
        "ESCS",      # Socioeconomic status
        "HOMEPOS",   # Home possessions
        "HISEI",     # Parental occupation
        "IMMIG",     # Immigrant background (if available)
        "ST001D01T", # Grade
    ]
    base_confounders = [c for c in base_confounders if c in df.columns]

    analyses = [
        {
            "treatment": "ICTRES",
            "outcome": "MATH_PV_MEAN",
            "confounders": base_confounders,
            "note": "Effect of ICT resources on math achievement",
        },
        {
            "treatment": "ICTEFFIC",
            "outcome": "MATH_PV_MEAN",
            "confounders": base_confounders,
            "note": "Effect of ICT self-efficacy on math achievement",
        },
    ]

    print("=" * 70)
    print("Exploratory Causal ML Analysis")
    print("=" * 70)
    print()
    print("⚠️  CAVEAT: PISA 2022 is observational/cross-sectional data.")
    print("    Results are HYPOTHESIS-GENERATING, not causal proof.")
    print("    See unconfoundedness, overlap, and SUTVA assumptions.")
    print()

    all_results = []

    for analysis in analyses:
        treatment = analysis["treatment"]
        outcome = analysis["outcome"]
        confounders = analysis["confounders"]
        note = analysis["note"]

        print(f"\n{'-'*60}")
        print(f"Treatment: {treatment} → Outcome: {outcome}")
        print(f"  {note}")
        print(f"  Confounders: {confounders}")

        # Double ML estimation
        print(f"\n  [Double ML]")
        dml_result = double_ml_estimate(df, treatment, outcome, confounders)
        dml_result["analysis_note"] = note
        all_results.append({"method": "double_ml", **dml_result})
        print(f"    ATE = {dml_result['ate']:.4f} (SE = {dml_result['ate_se']:.4f})")
        print(f"    95% CI: [{dml_result['ate_ci_lower']:.4f}, {dml_result['ate_ci_upper']:.4f}]")
        print(f"    N = {dml_result['n_observations']:,}")

        # Causal Forest / subgroup ATE
        print(f"\n  [Causal Forest / Subgroup ATE]")
        cf_result = causal_forest_estimate(df, treatment, outcome, confounders)
        cf_result["treatment"] = treatment
        cf_result["outcome"] = outcome
        cf_result["analysis_note"] = note
        print(f"    Method: {cf_result.get('method', 'unknown')}")

        if cf_result.get("mean_cate") is not None:
            print(f"    Mean CATE: {cf_result['mean_cate']:.4f}")

        heterogeneity = cf_result.get("heterogeneity_by_escs", {})
        if heterogeneity:
            print(f"    Heterogeneity by ESCS quartile:")
            for q, stats in heterogeneity.items():
                if "ate" in stats:
                    print(f"      {q}: ATE = {stats['ate']:.3f} "
                          f"(CI: [{stats['ate_ci_lower']:.3f}, {stats['ate_ci_upper']:.3f}], "
                          f"n = {stats['n']})")
                elif "mean_cate" in stats:
                    print(f"      {q}: Mean CATE = {stats['mean_cate']:.4f} "
                          f"(std = {stats['std_cate']:.4f}, n = {stats['n']})")

        all_results.append({"method": cf_result.get("method", "causal_forest"), **cf_result})

    # Save results
    results_df = pd.DataFrame([
        {
            "method": r.get("method", ""),
            "treatment": r.get("treatment", ""),
            "outcome": r.get("outcome", ""),
            "ate": r.get("ate", None),
            "ate_se": r.get("ate_se", None),
            "ate_ci_lower": r.get("ate_ci_lower", None),
            "ate_ci_upper": r.get("ate_ci_upper", None),
            "n": r.get("n_observations", r.get("n", None)),
            "note": r.get("analysis_note", ""),
        }
        for r in all_results if "ate" in r
    ])

    results_path = tables_dir / "causal_ml_exploratory_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved causal ML results: {results_path}")

    # Interpretation guidance
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDANCE")
    print("=" * 70)
    print("""
    1. These ATE/CATE estimates should NOT be interpreted as causal effects.
       They describe conditional associations after controlling for observed
       confounders, which is a weaker claim than causal identification.

    2. The Double ML estimates control for observable confounding but cannot
       address unobserved confounders (e.g., student motivation, teacher
       quality, peer effects).

    3. Subgroup ATE heterogeneity (by ESCS quartile) is exploratory and may
       reflect differential confounding rather than true effect heterogeneity.

    4. For the manuscript, frame these results as:
       - "Exploratory causal estimates using Double ML" (not "causal effects")
       - "Conditional associations after confounder adjustment"
       - "Hypothesis-generating evidence consistent with the digital divide
          framework" (if direction aligns)

    5. Primary interpretative weight should remain on the predictive/XAI
       analysis. Causal estimates supplement but do not replace predictive
       evidence.
    """)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
