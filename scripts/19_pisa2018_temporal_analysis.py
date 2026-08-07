#!/usr/bin/env python3
"""PISA 2018 vs PISA 2022 temporal comparison analysis.

Downloads PISA 2018 public-use data and performs:
1. Cross-wave feature importance stability (SHAP ranking correlation)
2. Model transfer performance (train on 2018, test on 2022 and reverse)
3. Digital divide variable importance stability over time

References:
    Tiukhova et al. (2024). Decision Support Systems, 182, 114229.
    Liu et al. (2024). iScience, 27(10), 110848.

PISA 2018 data download:
    https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip (~489 MB)
    https://webfs.oecd.org/pisa2018/SPSS_SCH_QQQ.zip

The script expects the .SAV files to be placed in data/raw/pisa2018/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from pisa_xai.config import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from pisa_xai.io import load_table, require_package

PISA2018_URLS = {
    "student": "https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip",
}
# School file optional — uncomment if needed:
# "school": "https://webfs.oecd.org/pisa2018/SPSS_SCH_QQQ.zip",

PISA2018_MATH_VARS = [
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV6MATH", "PV7MATH", "PV8MATH", "PV9MATH", "PV10MATH",
]

# PISA 2018 student weight and replicate weights
PISA2018_WEIGHTS = {
    "student_weight": "W_FSTUWT",
    "replicate_prefix": "W_FSTR",
    "senate_weight": "SENWT",
}

PISA2018_KEY_VARS = {
    "CNT": "CNT",       # Country
    "CNTSCHID": "CNTSCHID",
    "STRATUM": "STRATUM",
    "ST004D01T": "ST004D01T",  # Gender
    "IMMIG": "IMMIG",           # Immigrant background
    "ESCS": "ESCS",             # SES index
    "HOMEPOS": "HOMEPOS",
    "HISEI": "HISEI",
    "ST001D01T": "ST001D01T",  # Grade
}

# ICT variables that exist in both 2018 and 2022
ICT_VARS_2018 = [
    "ICTRES", "ICTSCH", "ICTHOME", "ICTEFFIC",
    "ICTINFO", "ICTSUBJ", "STUDYHMW",
]

# Subject-specific variables common across waves
COMMON_STUDENT_VARS = [
    "MATHEFF", "MATHEINT", "ANXMAT", "MATHPER",
    "FAMCON", "BELONG", "PERCOOP", "COGNACT",
    "TEACHBEHA", "STUBEHA", "EDUSHORT", "STAFFSHORT",
    "STIMREAD", "STIMHOME",
]


def download_pisa2018(raw_dir: Path) -> dict:
    """Download PISA 2018 data files. Returns path to extracted files."""
    import urllib.request
    import zipfile
    import io

    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for name, url in PISA2018_URLS.items():
        zip_path = raw_dir / f"pisa2018_{name}.zip"
        if zip_path.exists():
            print(f"  {name} zip already exists: {zip_path}")
            downloaded[name] = zip_path
            continue

        print(f"  Downloading {name} data from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=600) as resp:
                data = resp.read()
            zip_path.write_bytes(data)
            downloaded[name] = zip_path
            print(f"    Saved: {zip_path} ({len(data) / 1e6:.1f} MB)")
        except Exception as exc:
            print(f"    Download failed: {exc}")
            print(f"    Please download manually from: {url}")
            print(f"    and save to: {zip_path}")
            continue

    return downloaded


def extract_pisa2018(zip_paths: dict, extract_dir: Path) -> dict:
    """Extract PISA 2018 SAV files."""
    import zipfile

    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted = {}

    for name, zip_path in zip_paths.items():
        with zipfile.ZipFile(zip_path, "r") as zf:
            sav_files = [n for n in zf.namelist() if n.endswith('.sav')]
            if not sav_files:
                print(f"  No .sav file found in {zip_path}")
                continue
            sav_name = sav_files[0]  # Take first .sav file
            sav_path = extract_dir / sav_name
            if not sav_path.exists():
                print(f"  Extracting {sav_name} from {zip_path}...")
                zf.extract(sav_name, extract_dir)
            extracted[name] = sav_path
            print(f"    Extracted: {sav_path}")

    return extracted


def load_pisa2018_student(sav_path: Path, max_rows: int | None = None) -> "pd.DataFrame":
    """Load PISA 2018 student SAV file into a pandas DataFrame."""
    require_package("pandas", "pip install pandas")
    import pandas as pd

    print(f"  Loading PISA 2018 student data from {sav_path}...")
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(sav_path)
    except ImportError:
        try:
            df = pd.read_spss(sav_path)
        except Exception:
            raise ImportError(
                "Need pyreadstat or pandas.read_spss. Install: pip install pyreadstat"
            )

    print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=20260510)
        print(f"    Subsampled to {len(df):,} rows for efficiency")

    return df


def prepare_pisa2018_features(df) -> "pd.DataFrame":
    """Create PISA 2018 feature set comparable to PISA 2022 analysis."""
    import numpy as np
    import pandas as pd

    result = df.copy()

    # Math PV mean (outcome)
    pv_cols = [c for c in PISA2018_MATH_VARS if c in result.columns]
    result["MATH_PV_MEAN"] = result[pv_cols].mean(axis=1)

    # Low-performer label (same threshold as 2022: 420.07)
    result["LOW_PERFORMER_MATH"] = (result["MATH_PV_MEAN"] < 420.07).astype(int)

    # Normalize student weight
    wt_col = PISA2018_WEIGHTS["student_weight"]
    if wt_col in result.columns:
        result["W_FSTUWT_NORM"] = result[wt_col] / result[wt_col].mean()

    # Select available common features
    available_features = []
    all_common = ICT_VARS_2018 + COMMON_STUDENT_VARS + list(PISA2018_KEY_VARS.values())

    for feat in all_common:
        if feat in result.columns and feat not in ["CNTSCHID", "STRATUM", "CNT"]:
            available_features.append(feat)

    # Remove non-predictor columns
    exclude = ["W_FSTUWT", "SENWT"] + [c for c in result.columns if c.startswith("W_FSTR")]
    available_features = [f for f in available_features if f not in exclude]
    available_features = [f for f in available_features if not f.startswith("PV")]

    print(f"    Available common features: {len(available_features)}")
    return result, available_features


def train_basic_model(x_train, y_train, task: str, random_state: int = 20260510):
    """Train a basic XGBoost model for temporal comparison."""
    require_package("xgboost", "pip install xgboost")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    import numpy as np

    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    pipeline.fit(x_train, y_train)
    return pipeline


def main() -> int:
    require_package("pandas", "pip install pandas")
    import pandas as pd
    import numpy as np

    import shap
    from sklearn.model_selection import train_test_split

    config = load_config(DEFAULT_CONFIG_PATH)
    project_root = _PROJECT_DIR
    raw_dir = project_root / "data" / "raw" / "pisa2018"
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PISA 2018 vs 2022 Temporal Comparison Analysis")
    print("=" * 70)

    # Step 1: Download/load PISA 2018 data
    print("\n[1] Data acquisition...")
    zip_paths = download_pisa2018(raw_dir)
    if not zip_paths:
        print("ERROR: Could not download data. Please download manually.")
        return 1

    extract_dir = raw_dir / "extracted"
    sav_files = extract_pisa2018(zip_paths, extract_dir)
    if "student" not in sav_files:
        print("ERROR: Student file not found.")
        return 1

    # Step 2: Load PISA 2018
    print("\n[2] Loading PISA 2018 student data...")
    df18 = load_pisa2018_student(sav_files["student"], max_rows=200000)
    df18, features18 = prepare_pisa2018_features(df18)

    # Step 3: Load PISA 2022 data for comparison
    print("\n[3] Loading PISA 2022 data...")
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    df22 = load_table(processed_dir / "pisa2022_math_model_frame.parquet")

    # Find common features (name match only, handle dtype differences)
    common_features = sorted(set(features18) & set(df22.columns))
    # Exclude non-numeric identifiers
    exclude = {'CNT', 'CNTSCHID', 'STRATUM', 'OECD'}
    common_features = [f for f in common_features if f not in exclude and not f.startswith('PV') and not f.startswith('W_FST')]
    print(f"    Common features for cross-wave comparison: {len(common_features)}")
    print(f"    Features: {common_features[:20]}...")

    if len(common_features) < 5:
        print("ERROR: Too few common features for meaningful comparison.")
        return 1

    # Prepare feature matrices
    x18 = df18[common_features].copy()
    y18_reg = df18["MATH_PV_MEAN"]
    y18_clf = df18["LOW_PERFORMER_MATH"]

    x22 = df22[common_features].copy()
    y22_reg = df22["MATH_PV_MEAN"]
    y22_clf = df22["LOW_PERFORMER_MATH"]

    # Convert categoricals to numeric, then impute
    for df_tmp in [x18, x22]:
        for col in df_tmp.columns:
            if df_tmp[col].dtype == object:
                df_tmp[col] = df_tmp[col].astype('category').cat.codes
            elif str(df_tmp[col].dtype) == 'category':
                df_tmp[col] = df_tmp[col].cat.codes

    from sklearn.impute import SimpleImputer
    imp18 = SimpleImputer(strategy="median")
    imp22 = SimpleImputer(strategy="median")

    x18_imp = pd.DataFrame(imp18.fit_transform(x18), columns=common_features, index=x18.index)
    x22_imp = pd.DataFrame(imp22.fit_transform(x22), columns=common_features, index=x22.index)

    # Step 4: Within-wave models
    print("\n[4] Training within-wave models...")
    rs = config["sample"]["random_state"]

    # 2018 model
    x18_train, x18_test, y18c_train, y18c_test = train_test_split(
        x18_imp, y18_clf, test_size=0.2, random_state=rs, stratify=y18_clf
    )
    model18 = train_basic_model(x18_train, y18c_train, "classification", rs)

    # 2022 model (on common features)
    x22_train, x22_test, y22c_train, y22c_test = train_test_split(
        x22_imp, y22_clf, test_size=0.2, random_state=rs, stratify=y22_clf
    )
    model22 = train_basic_model(x22_train, y22c_train, "classification", rs)

    # Step 5: Feature importance comparison across waves
    print("\n[5] Feature importance stability (2018 vs 2022)...")

    explainer18 = shap.Explainer(model18.named_steps["model"], x18_train.iloc[:2000])
    shap_values18 = explainer18(x18_train.iloc[:2000])

    explainer22 = shap.Explainer(model22.named_steps["model"], x22_train.iloc[:2000])
    shap_values22 = explainer22(x22_train.iloc[:2000])

    imp18 = pd.DataFrame({
        "feature": common_features,
        "shap_importance_2018": np.abs(shap_values18.values).mean(axis=0),
    }).sort_values("shap_importance_2018", ascending=False)

    imp22 = pd.DataFrame({
        "feature": common_features,
        "shap_importance_2022": np.abs(shap_values22.values).mean(axis=0),
    }).sort_values("shap_importance_2022", ascending=False)

    imp_merged = imp18.merge(imp22, on="feature", how="outer").fillna(0)
    imp_merged["rank_2018"] = imp_merged["shap_importance_2018"].rank(ascending=False)
    imp_merged["rank_2022"] = imp_merged["shap_importance_2022"].rank(ascending=False)
    imp_merged["rank_change"] = imp_merged["rank_2022"] - imp_merged["rank_2018"]

    from scipy.stats import spearmanr, kendalltau
    rho, pval_s = spearmanr(imp_merged["shap_importance_2018"], imp_merged["shap_importance_2022"])
    tau, pval_k = kendalltau(imp_merged["shap_importance_2018"], imp_merged["shap_importance_2022"])

    print(f"    Spearman's ρ (feature importance 2018 vs 2022): {rho:.3f} (p={pval_s:.4f})")
    print(f"    Kendall's τ (feature importance 2018 vs 2022): {tau:.3f} (p={pval_k:.4f})")

    # ICT variable stability specifically
    ict_in_both = [f for f in ICT_VARS_2018 if f in common_features]
    if ict_in_both:
        ict_stability = imp_merged[imp_merged["feature"].isin(ict_in_both)]
        print(f"\n    ICT variable importance stability:")
        for _, row in ict_stability.sort_values("shap_importance_2022", ascending=False).iterrows():
            print(f"      {row['feature']}: rank {int(row['rank_2018'])} → {int(row['rank_2022'])} "
                  f"(Δ={int(row['rank_change'])})")

    imp_path = tables_dir / "temporal_feature_importance_stability.csv"
    imp_merged.to_csv(imp_path, index=False)
    print(f"    Saved: {imp_path}")

    # Step 6: Cross-wave model transfer
    print("\n[6] Cross-wave model transfer performance...")
    from sklearn.metrics import roc_auc_score, mean_squared_error

    # Both directions
    y22_pred_from18 = model18.predict_proba(x22_imp)[:, 1]
    y18_pred_from22 = model22.predict_proba(x18_imp)[:, 1]

    auc_18on22 = roc_auc_score(y22_clf, y22_pred_from18)
    auc_22on18 = roc_auc_score(y18_clf, y18_pred_from22)

    print(f"    Train 2018 → Test 2022 (global): AUC = {auc_18on22:.3f}")
    print(f"    Train 2022 → Test 2018 (global): AUC = {auc_22on18:.3f}")

    # Country-level transfer for major systems
    if "CNT" in df18.columns and "CNT" in df22.columns:
        transfer_rows = []
        common_countries = sorted(set(df18["CNT"].unique()) & set(df22["CNT"].unique()))
        print(f"\n    Country-level transfer ({len(common_countries)} common countries):")

        for country in common_countries:
            mask18 = df18["CNT"] == country
            mask22 = df22[df22["CNT"] == country].index

            if len(mask22) < 100:
                continue

            y_c22 = df22.loc[mask22, "LOW_PERFORMER_MATH"] if all(isinstance(i, (int, np.integer)) for i in mask22) else y22_clf

            # Actually let's simplify: use the index-based approach
            country22_idx = df22[df22["CNT"] == country].index
            country18_idx = df18[df18["CNT"] == country].index

            if len(country22_idx) < 100 or len(country18_idx) < 100:
                continue

            x_c22 = x22_imp.loc[country22_idx]
            y_c22 = y22_clf.loc[country22_idx]
            x_c18 = x18_imp.loc[country18_idx]
            y_c18 = y18_clf.loc[country18_idx]

            auc_22on18_c = roc_auc_score(y_c18, model22.predict_proba(x_c18)[:, 1]) if y_c18.nunique() > 1 else None
            auc_18on22_c = roc_auc_score(y_c22, model18.predict_proba(x_c22)[:, 1]) if y_c22.nunique() > 1 else None

            if auc_22on18_c is not None and auc_18on22_c is not None:
                transfer_rows.append({
                    "country": country,
                    "auc_2018_model_on_2022": round(auc_18on22_c, 4),
                    "auc_2022_model_on_2018": round(auc_22on18_c, 4),
                    "transfer_gap": round(auc_22on18_c - auc_18on22_c, 4),
                })

        if transfer_rows:
            transfer_df = pd.DataFrame(transfer_rows)
            transfer_path = tables_dir / "cross_wave_transfer_by_country.csv"
            transfer_df.to_csv(transfer_path, index=False)
            print(f"    Saved: {transfer_path}")
            print(f"    Mean transfer AUC gap: {transfer_df['transfer_gap'].mean():.3f}")

    # Step 7: Visualization
    print("\n[7] Generating temporal comparison visualizations...")
    import matplotlib.pyplot as plt

    # Feature importance change plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = imp_merged.nsmallest(15, "rank_2018")
    x_pos = range(len(top_features))
    ax.barh([p + 0.2 for p in x_pos], top_features["shap_importance_2018"], 0.4,
            label="PISA 2018", color="#1f77b4", alpha=0.8)
    ax.barh([p - 0.2 for p in x_pos], top_features["shap_importance_2022"], 0.4,
            label="PISA 2022", color="#ff7f0e", alpha=0.8)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title(f"Feature Importance Stability: PISA 2018 vs 2022\n"
                 f"(Spearman's ρ = {rho:.3f}, Kendall's τ = {tau:.3f})")
    ax.legend()
    fig.tight_layout()
    vis_path = figures_dir / "temporal_shap_stability.png"
    fig.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {vis_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Feature importance stability: ρ = {rho:.3f}")
    print(f"  Cross-wave transfer: 2018→2022 AUC = {auc_18on22:.3f}")
    print(f"  Cross-wave transfer: 2022→2018 AUC = {auc_22on18:.3f}")
    if rho >= 0.75:
        print("  Interpretation: Strong inter-wave feature importance stability")
        print("    → XAI insights are broadly stable across PISA waves")
    elif rho >= 0.50:
        print("  Interpretation: Moderate inter-wave feature importance stability")
        print("    → Core predictors consistent, but some shifts in relative importance")
    else:
        print("  Interpretation: Weak inter-wave feature importance stability")
        print("    → Predictive patterns may be wave-specific; caution in temporal generalization")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
