#!/usr/bin/env python3
"""Fairness evaluation for the best classification model.

Computes formal algorithmic fairness metrics across demographic subgroups
(gender, immigrant background, ESCS quintiles) and their intersections.
Extends the existing subgroup descriptive evaluation with formal fairness
metrics inspired by Idowu (2024), Verger et al. (2024), and Gándara et al.
(2024).

Metrics computed:
    - Equalized Odds Difference (TPR and FPR gaps)
    - Demographic Parity Difference
    - ABROCA (Absolute Between-ROC Area)
    - Intersectional subgroup analysis

References:
    Idowu, J.A. (2024). Debiasing Education Algorithms. IJAIED, 34, 1510-1540.
    Verger et al. (2024). MADD Metric. JEDM, 16(1), 365-409.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.evaluation import classification_metrics
from pisa_xai.io import load_table, require_package


def equalized_odds_difference(
    y_true,
    y_pred_proba,
    group_mask,
    threshold: float = 0.50,
    sample_weight=None,
) -> Dict[str, float]:
    """Compute TPR and FPR differences between privileged and unprivileged groups.

    Returns a dict with tpr_gap and fpr_gap. Smaller absolute values → fairer.
    When sample_weight is provided (e.g., normalized W_FSTUWT), gaps are
    population-weighted; otherwise unweighted.
    """
    require_package("numpy", "pip install -r requirements.txt")
    import numpy as np

    y_pred = (y_pred_proba >= threshold).astype(int)
    y_true_arr = np.asarray(y_true)
    group_arr = np.asarray(group_mask)
    priv = group_arr
    unpriv = ~group_arr
    if sample_weight is None:
        sample_weight = np.ones(len(y_true_arr))
    w = np.asarray(sample_weight)

    def _tpr(mask):
        tp = (w * (y_pred == 1) * (y_true_arr == 1) * mask).sum()
        pos = (w * (y_true_arr == 1) * mask).sum()
        return tp / pos if pos > 0 else float("nan")

    def _fpr(mask):
        fp = (w * (y_pred == 1) * (y_true_arr == 0) * mask).sum()
        neg = (w * (y_true_arr == 0) * mask).sum()
        return fp / neg if neg > 0 else float("nan")

    return {
        "tpr_privileged": _tpr(priv),
        "tpr_unprivileged": _tpr(unpriv),
        "tpr_gap": _tpr(priv) - _tpr(unpriv),
        "fpr_privileged": _fpr(priv),
        "fpr_unprivileged": _fpr(unpriv),
        "fpr_gap": _fpr(priv) - _fpr(unpriv),
    }


def demographic_parity_difference(
    y_pred_proba,
    group_mask,
    threshold: float = 0.50,
    sample_weight=None,
) -> float:
    """Difference in positive prediction rate between groups.

    Weighted by sample_weight (e.g., normalized W_FSTUWT) when provided.
    """
    import numpy as np

    y_pred = (y_pred_proba >= threshold).astype(int)
    if sample_weight is None:
        sample_weight = np.ones(len(y_pred))
    w = np.asarray(sample_weight)
    priv_rate = (w * y_pred * np.asarray(group_mask)).sum() / (w * np.asarray(group_mask)).sum()
    unpriv_rate = (w * y_pred * ~np.asarray(group_mask)).sum() / (w * ~np.asarray(group_mask)).sum()
    return float(priv_rate - unpriv_rate)


def abroca(
    y_true,
    y_pred_proba,
    group_a_mask,
    group_b_mask,
    sample_weight=None,
) -> Optional[float]:
    """Absolute Between-ROC Area (ABROCA) — measures AUC-based unfairness.

    ABROCA computes the area between ROC curves of two groups.
    When sample_weight is provided, the ROC curves are population-weighted.
    """
    require_package("numpy", "pip install -r requirements.txt")
    import numpy as np

    try:
        from sklearn.metrics import roc_curve
    except Exception:
        return None

    def _roc_points(mask):
        if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
            return None
        if sample_weight is not None:
            fpr, tpr, _ = roc_curve(
                y_true[mask], y_pred_proba[mask], sample_weight=sample_weight[mask]
            )
        else:
            fpr, tpr, _ = roc_curve(y_true[mask], y_pred_proba[mask])
        return fpr, tpr

    pts_a = _roc_points(group_a_mask)
    pts_b = _roc_points(group_b_mask)
    if pts_a is None or pts_b is None:
        return None

    fpr_common = np.linspace(0, 1, 101)
    tpr_a = np.interp(fpr_common, pts_a[0], pts_a[1])
    tpr_b = np.interp(fpr_common, pts_b[0], pts_b[1])

    abroca_val = float(np.trapz(np.abs(tpr_a - tpr_b), fpr_common))
    return abroca_val


def subgroup_performance(
    y_true,
    y_pred,
    y_pred_proba,
    group_series,
    sample_weight,
) -> List[dict]:
    """Compute classification metrics per group."""
    rows = []
    for label, mask in group_series.groupby(group_series, dropna=True).groups.items():
        mask_arr = group_series.isin([label]).values
        n = mask_arr.sum()
        if n < 10 or len(np.unique(y_true[mask_arr])) < 2:
            continue
        metrics = classification_metrics(
            y_true[mask_arr],
            y_pred_proba[mask_arr],
            sample_weight=sample_weight[mask_arr] if sample_weight is not None else None,
        )
        metrics["group_value"] = str(label)
        metrics["n"] = int(n)
        rows.append(metrics)
    return rows


def intersectional_subgroups(
    df,
    protected_vars: List[str],
    min_group_size: int = 100,
) -> List[dict]:
    """Create intersectional subgroups and compute model performance per subgroup."""
    rows = []
    for var_i, var_j in [(protected_vars[i], protected_vars[j])
                         for i in range(len(protected_vars))
                         for j in range(i + 1, len(protected_vars))]:
        for vi in df[var_i].dropna().unique():
            for vj in df[var_j].dropna().unique():
                mask = (df[var_i] == vi) & (df[var_j] == vj)
                n = mask.sum()
                if n < min_group_size or df.loc[mask, "LOW_PERFORMER_MATH"].nunique() < 2:
                    continue
                w_raw = df.loc[mask, "W_FSTUWT"]
                sw = (w_raw / w_raw.mean()).values if w_raw.sum() > 0 else None
                metrics = classification_metrics(
                    df.loc[mask, "LOW_PERFORMER_MATH"],
                    df.loc[mask, "best_classification_score"],
                    sample_weight=sw,
                )
                metrics["intersection"] = f"{var_i}={vi} & {var_j}={vj}"
                metrics["var_i"] = var_i
                metrics["var_j"] = var_j
                metrics["n"] = int(n)
                rows.append(metrics)
    return rows


def main() -> int:
    require_package("pandas", "pip install -r requirements.txt")
    require_package("numpy", "pip install -r requirements.txt")
    import pandas as pd
    import numpy as np

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best = json.loads((model_dir / "best_model_summary.json").read_text(encoding="utf-8"))
    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")

    predictions_file = processed_dir / "predictions" / "holdout_predictions.parquet"
    if predictions_file.exists():
        pred_df = load_table(predictions_file)
    else:
        pred_df = df.copy()
        pred_df["best_classification_score"] = 0.5
        print("Warning: holdout predictions not found; using placeholder scores")

    y_true = pred_df["LOW_PERFORMER_MATH"].values
    y_pred_proba = pred_df["best_classification_score"].values
    weight = pred_df.get("W_FSTUWT", None)
    if weight is not None:
        weight = weight / weight.mean()

    print("=" * 70)
    print("Algorithmic Fairness Evaluation")
    print("=" * 70)

    fairness_rows = []

    # ── 1. Gender fairness (ST004D01T) ──
    if "ST004D01T" in pred_df.columns:
        gender = pred_df["ST004D01T"]
        for female_val in ["Female", "female", "Female ", "1"]:
            if female_val in gender.unique():
                female_mask = (gender == female_val).values
                eod = equalized_odds_difference(y_true, y_pred_proba, female_mask, sample_weight=weight)
                dem_parity = demographic_parity_difference(y_pred_proba, female_mask, sample_weight=weight)
                abr = abroca(y_true, y_pred_proba, female_mask, ~female_mask, sample_weight=weight)
                fairness_rows.append({
                    "attribute": "gender",
                    "privileged": "female",
                    **eod,
                    "demographic_parity_diff": dem_parity,
                    "abroca": abr,
                })
                print(f"Gender — EOD TPR gap: {eod['tpr_gap']:.4f}, "
                      f"Demographic Parity: {dem_parity:.4f}, ABROCA: {abr}")
                break

    # ── 2. Immigrant background fairness (IMMIG) ──
    # [Change] Fix IMMIG label-matching bug: PISA 2022 labels are
    # 'Native student' / 'First-Generation student' / 'Second-Generation student',
    # so the previous hardcoded list ["Native", "native", "1"] never matched.
    # Original: for native_val in ["Native", "native", "1"]:
    if "IMMIG" in pred_df.columns:
        immig = pred_df["IMMIG"]
        native_vals = [
            v for v in immig.dropna().unique()
            if isinstance(v, str) and v.strip().lower().startswith("native")
        ]
        for native_val in native_vals:
            if native_val in immig.unique():
                native_mask = (immig == native_val).values
                eod = equalized_odds_difference(y_true, y_pred_proba, native_mask, sample_weight=weight)
                dem_parity = demographic_parity_difference(y_pred_proba, native_mask, sample_weight=weight)
                abr = abroca(y_true, y_pred_proba, native_mask, ~native_mask, sample_weight=weight)
                fairness_rows.append({
                    "attribute": "immigrant_background",
                    "privileged": "native",
                    **eod,
                    "demographic_parity_diff": dem_parity,
                    "abroca": abr,
                })
                print(f"Immigrant BG — EOD TPR gap: {eod['tpr_gap']:.4f}, "
                      f"Demographic Parity: {dem_parity:.4f}, ABROCA: {abr}")
                break

    # ── 3. ESCS quintile fairness ──
    if "ESCS" in pred_df.columns:
        escs = pred_df["ESCS"].copy()
        valid = escs.notna()
        escs_q = pd.Series("missing", index=pred_df.index)
        escs_q[valid] = pd.qcut(escs[valid].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        pred_df_temp = pred_df.copy()
        pred_df_temp["ESCS_QUARTILE"] = escs_q

        q1_mask = (escs_q == "Q1").values
        q4_mask = (escs_q == "Q4").values
        if q1_mask.sum() >= 10 and q4_mask.sum() >= 10:
            eod = equalized_odds_difference(y_true, y_pred_proba, q4_mask, sample_weight=weight)
            dem_parity = demographic_parity_difference(y_pred_proba, q4_mask, sample_weight=weight)
            abr = abroca(y_true, y_pred_proba, q4_mask, q1_mask, sample_weight=weight)
            fairness_rows.append({
                "attribute": "escs_quartile",
                "privileged": "Q4 (highest)",
                "unprivileged": "Q1 (lowest)",
                **eod,
                "demographic_parity_diff": dem_parity,
                "abroca": abr,
            })
            print(f"ESCS Q4 vs Q1 — EOD TPR gap: {eod['tpr_gap']:.4f}, "
                  f"Demographic Parity: {dem_parity:.4f}, ABROCA: {abr}")

    # ── 4. Intersectional fairness ──
    intersectional_vars = []
    for var_name in ["ST004D01T", "IMMIG", "ESCS_QUARTILE"]:
        if var_name in pred_df_temp.columns:
            intersectional_vars.append(var_name)
        elif var_name == "ESCS_QUARTILE" and "ESCS_QUARTILE" in pred_df_temp.columns:
            intersectional_vars.append(var_name)

    if len(intersectional_vars) >= 2:
        print(f"\nIntersectional fairness: {intersectional_vars}")
        intersectional = intersectional_subgroups(
            pred_df_temp, intersectional_vars, min_group_size=200
        )
        inter_df = pd.DataFrame(intersectional)
        inter_path = tables_dir / "intersectional_fairness.csv"
        inter_df.to_csv(inter_path, index=False)
        print(f"Saved intersectional fairness: {inter_path} ({len(intersectional)} subgroups)")

        # Identify most disadvantaged subgroups
        if not inter_df.empty:
            worst = inter_df.sort_values("auc").head(10)
            print("Least-served intersectional subgroups (by AUC):")
            for _, row in worst.iterrows():
                print(f"  {row['intersection']}: AUC={row['auc']:.4f}, "
                      f"n={int(row['n'])}")

    # ── 5. Fairness visualization ──
    if fairness_rows:
        fairness_df = pd.DataFrame(fairness_rows)
        fairness_df.to_csv(tables_dir / "fairness_metrics.csv", index=False)
        print(f"\nSaved fairness metrics: {tables_dir / 'fairness_metrics.csv'}")

        require_package("matplotlib", "pip install -r requirements.txt")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        attr_labels = [r["attribute"] for r in fairness_rows]
        tpr_gaps = [abs(r["tpr_gap"]) for r in fairness_rows]
        fpr_gaps = [abs(r["fpr_gap"]) for r in fairness_rows]
        dem_pars = [abs(r["demographic_parity_diff"]) for r in fairness_rows]
        abrocas = [r.get("abroca", 0) or 0 for r in fairness_rows]

        x_pos = range(len(attr_labels))
        width = 0.35

        axes[0].bar([p - width / 2 for p in x_pos], tpr_gaps, width, label="|TPR Gap|", color="#1f77b4")
        axes[0].bar([p + width / 2 for p in x_pos], fpr_gaps, width, label="|FPR Gap|", color="#ff7f0e")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(attr_labels, rotation=15)
        axes[0].set_ylabel("Gap (absolute)")
        axes[0].set_title("Equalized Odds Gaps (lower = fairer)")
        axes[0].legend()
        axes[0].axhline(y=0.05, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

        axes[1].bar(attr_labels, dem_pars, color="#2ca02c")
        axes[1].set_ylabel("|Demographic Parity Diff|")
        axes[1].set_title("Demographic Parity (lower = fairer)")
        axes[1].axhline(y=0.05, color="gray", linestyle="--", alpha=0.5)

        axes[2].bar(attr_labels, abrocas, color="#d62728")
        axes[2].set_ylabel("ABROCA")
        axes[2].set_title("ABROCA (lower = fairer)")
        axes[2].axhline(y=0.02, color="gray", linestyle="--", alpha=0.5, label="Threshold")

        fig.suptitle("Algorithmic Fairness Evaluation", fontsize=13, y=1.02)
        fig.tight_layout()
        fair_viz_path = figures_dir / "fairness_metrics_overview.png"
        fig.savefig(fair_viz_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved fairness visualization: {fair_viz_path}")

    print("\nFairness evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
