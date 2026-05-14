#!/usr/bin/env python3
"""Explanation stability analysis across demographic subgroups.

Computes SHAP feature importance rankings for gender, immigrant background,
and ESCS quintile subgroups, then measures ranking consistency via Spearman's ρ.
Reference: Tiukhova et al. (2025), Decision Support Systems.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def subgroup_shap_importance(
    estimator, x_all, y_all, subgroup_mask, feat_names, rng, n_sample=2000
):
    """Compute SHAP-based feature importance for a subgroup."""
    import numpy as np
    import pandas as pd

    indices = np.where(subgroup_mask)[0]
    if len(indices) == 0:
        return None
    n = min(n_sample, len(indices))
    idx = rng.choice(indices, size=n, replace=False)
    x_sub = x_all[idx]

    try:
        import shap
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(x_sub)
        except Exception:
            explainer = shap.Explainer(estimator, x_sub)
            shap_vals = explainer(x_sub, check_additivity=False).values
    except Exception:
        return None

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    imp = pd.DataFrame({
        "feature": feat_names[:shap_vals.shape[1]],
        "importance": np.abs(shap_vals).mean(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = range(1, len(imp) + 1)
    return imp


def ranking_consistency(rankings: dict) -> "pd.DataFrame":
    """Compute pairwise Spearman's ρ between subgroup rankings."""
    from scipy.stats import spearmanr
    import pandas as pd

    groups = sorted(rankings.keys())
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = rankings[groups[i]].set_index("feature")["rank"]
            b = rankings[groups[j]].set_index("feature")["rank"]
            common = a.index.intersection(b.index)
            if len(common) < 5:
                continue
            rho, p = spearmanr(a.loc[common], b.loc[common])
            rows.append({
                "group_a": groups[i], "group_b": groups[j],
                "n_common_features": len(common),
                "spearman_r": round(rho, 4), "p_value": round(p, 6),
            })
    return pd.DataFrame(rows)


def main() -> int:
    require_package("joblib", "pip install joblib")
    require_package("numpy", "pip install -r requirements.txt")
    import joblib
    import numpy as np
    import pandas as pd

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    model_path = model_dir / f"classification_{best['best_classification_model']}.joblib"
    if not model_path.exists():
        model_path = model_dir / "classification_lightgbm.joblib"
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))
    model = joblib.load(model_path)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    x = df[features]

    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    x_num = x.copy()
    for c in cat_cols:
        x_num[c] = x_num[c].cat.codes.astype("int8")

    # Get transformed features and estimator
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocess")
        estimator = model.named_steps.get("model")
        if preprocessor is not None:
            x_transformed = preprocessor.transform(x_num)
            if hasattr(x_transformed, "toarray"):
                x_transformed = x_transformed.toarray()
            try:
                feat_names = np.array(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = np.array([f"f{i}" for i in range(x_transformed.shape[1])])
        else:
            estimator = model
            x_transformed = x_num.values
            feat_names = np.array(features)
    else:
        estimator = model
        x_transformed = x_num.values
        feat_names = np.array(features)

    rng = np.random.default_rng(config["sample"]["random_state"])
    y = df["LOW_PERFORMER_MATH"].values
    all_rows = []

    # ── 1. Gender stability ─────────────────────────────────────
    print("Computing gender stability", flush=True)
    gender_rankings = {}
    g_cons = pd.DataFrame()
    for gender_val, gender_label in [(1, "Female"), (2, "Male")]:
        if "ST004D01T" in df.columns:
            mask = df["ST004D01T"].values == gender_val
        else:
            continue
        imp = subgroup_shap_importance(
            estimator, x_transformed, y, mask, feat_names, rng
        )
        if imp is not None:
            gender_rankings[gender_label] = imp
            imp["group"] = gender_label
            all_rows.append(imp)

    if len(gender_rankings) >= 2:
        g_cons = ranking_consistency(gender_rankings)
        g_cons["dimension"] = "gender"
        print(g_cons.to_string(index=False))

    # ── 2. Immigrant background stability ────────────────────────
    print("Computing immigrant background stability", flush=True)
    imm_rankings = {}
    imm_cons = pd.DataFrame()
    for imm_val, imm_label in [(1, "Native"), (2, "Second-Gen"), (3, "First-Gen")]:
        if "IMMIG" in df.columns:
            mask = df["IMMIG"].values == imm_val
        else:
            continue
        imp = subgroup_shap_importance(
            estimator, x_transformed, y, mask, feat_names, rng
        )
        if imp is not None:
            imm_rankings[imm_label] = imp
            imp["group"] = imm_label
            all_rows.append(imp)

    imm_cons = pd.DataFrame()
    if len(imm_rankings) >= 2:
        imm_cons = ranking_consistency(imm_rankings)
        imm_cons["dimension"] = "immigrant"
        print(imm_cons.to_string(index=False))

    # ── 3. ESCS quintile stability ───────────────────────────────
    print("Computing ESCS quintile stability", flush=True)
    escs_rankings = {}
    escs_cons = pd.DataFrame()
    if "ESCS" in df.columns:
        escs_q = pd.qcut(
            pd.Series(df["ESCS"].values).rank(method="first"), 5,
            labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
        )
        for q_label in escs_q.cat.categories:
            mask = escs_q.values == q_label
            imp = subgroup_shap_importance(
                estimator, x_transformed, y, mask, feat_names, rng
            )
            if imp is not None:
                escs_rankings[str(q_label)] = imp
                imp["group"] = str(q_label)
                all_rows.append(imp)

    escs_cons = pd.DataFrame()
    if len(escs_rankings) >= 2:
        escs_cons = ranking_consistency(escs_rankings)
        escs_cons["dimension"] = "ESCS"
        print(escs_cons.to_string(index=False))

    # ── Save outputs ─────────────────────────────────────────────
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(tables_dir / "explanation_stability_detail.csv", index=False)

    parts = []
    if len(g_cons) > 0: parts.append(g_cons)
    if len(imm_cons) > 0: parts.append(imm_cons)
    if len(escs_cons) > 0: parts.append(escs_cons)
    cons_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cons_df.to_csv(tables_dir / "explanation_stability.csv", index=False)

    # ── Plot ─────────────────────────────────────────────────────
    require_package("matplotlib", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (title, rankings) in zip(
        axes,
        [("Gender", gender_rankings), ("Immigrant", imm_rankings), ("ESCS", escs_rankings)]
    ):
        if len(rankings) < 2:
            ax.set_title(f"{title}: insufficient data")
            continue
        # Get top-10 features from the first group
        first_group = sorted(rankings.keys())[0]
        top_feats = rankings[first_group].head(10)["feature"].tolist()
        x_pos = range(len(top_feats))
        w = 0.8 / len(rankings)
        for i, (grp, imp_df) in enumerate(rankings.items()):
            ranks = []
            for f in top_feats:
                match = imp_df[imp_df["feature"] == f]
                ranks.append(match["rank"].values[0] if len(match) > 0 else 30)
            ax.barh([p + i * w for p in x_pos], ranks, w, label=grp, alpha=0.8)
        ax.set_yticks([p + w * (len(rankings) - 1) / 2 for p in x_pos])
        ax.set_yticklabels(top_feats, fontsize=8)
        ax.set_xlabel("SHAP Rank (lower = more important)")
        ax.set_title(f"Feature Importance Stability: {title}")
        ax.legend(fontsize=7)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(figures_dir / "explanation_stability.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nExplanation stability analysis complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
