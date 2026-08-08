#!/usr/bin/env python3
r"""Bootstrap confidence intervals for headline and intersectional metrics.

Reproduces the 95% bootstrap CIs reported in the TLT manuscript:
  - weighted AUC and RMSE on the holdout (seed 20260510, 1000 resamples)
  - intersectional AUC gap (low-SES non-native vs high-SES native)

Reads holdout_predictions.csv produced by 03_train_models.py and writes
reports/tables/bootstrap_ci_summary.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from pisa_xai.config import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from pisa_xai.io import require_package


def weighted_auc(y, s, w):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s, sample_weight=w)


def weighted_rmse(y_true, y_pred, w):
    return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))


def bootstrap_ci(sample_fn, n_boot=1000, seed=20260510):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        try:
            stats.append(sample_fn(rng))
        except Exception:
            continue
    arr = np.asarray(stats)
    arr = arr[~np.isnan(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), float(arr.mean())


def main() -> int:
    require_package("pandas", "pip install pandas")
    require_package("numpy", "pip install numpy")

    config = load_config(DEFAULT_CONFIG_PATH)
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    holdout_path = tables_dir / "holdout_predictions.csv"
    if not holdout_path.exists():
        raise SystemExit(f"holdout predictions not found: {holdout_path}")
    df = pd.read_csv(holdout_path)
    df = df.dropna(subset=["best_classification_score", "LOW_PERFORMER_MATH", "sample_weight"])
    s = df["best_classification_score"].values
    y = df["LOW_PERFORMER_MATH"].values
    w = df["sample_weight"].values
    y_reg = df["MATH_PV_MEAN"].values
    y_reg_pred = df["best_regression_prediction"].values
    n = len(df)

    # AUC (weighted)
    auc_lo, auc_hi, auc_mean = bootstrap_ci(lambda rng: _auc_sample(df, rng, y, s, w))
    # RMSE (weighted)
    rmse_lo, rmse_hi, rmse_mean = bootstrap_ci(lambda rng: _rmse_sample(df, rng, y_reg, y_reg_pred, w))

    # Intersectional AUC gap
    df = df.copy()
    df["escs_q"] = pd.qcut(df["ESCS"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    df["non_native"] = df["IMMIG"].isin(["First-Generation student", "Second-Generation student"])
    lo_mask = (df["escs_q"] == "Q1") & df["non_native"]
    hi_mask = (df["escs_q"] == "Q4") & (~df["non_native"])
    lo_idx = df.index[lo_mask].to_numpy()
    hi_idx = df.index[hi_mask].to_numpy()

    def _gap_sample(rng):
        b_lo = df.loc[rng.choice(lo_idx, size=len(lo_idx), replace=True)]
        b_hi = df.loc[rng.choice(hi_idx, size=len(hi_idx), replace=True)]
        a_lo = weighted_auc(b_lo["LOW_PERFORMER_MATH"], b_lo["best_classification_score"], b_lo["sample_weight"])
        a_hi = weighted_auc(b_hi["LOW_PERFORMER_MATH"], b_hi["best_classification_score"], b_hi["sample_weight"])
        return a_lo - a_hi

    gap_lo, gap_hi, gap_mean = bootstrap_ci(_gap_sample)

    summary = {
        "n_holdout": int(n),
        "auc_point": float(weighted_auc(y, s, w)),
        "auc_ci_lower": round(auc_lo, 4),
        "auc_ci_upper": round(auc_hi, 4),
        "rmse_point": float(weighted_rmse(y_reg, y_reg_pred, w)),
        "rmse_ci_lower": round(rmse_lo, 2),
        "rmse_ci_upper": round(rmse_hi, 2),
        "intersectional_gap_point": float(weighted_auc(y[lo_mask], s[lo_mask], w[lo_mask])
                                         - weighted_auc(y[hi_mask], s[hi_mask], w[hi_mask])),
        "intersectional_gap_ci_lower": round(gap_lo, 4),
        "intersectional_gap_ci_upper": round(gap_hi, 4),
        "seed": 20260510,
        "n_bootstrap": 1000,
    }
    out_path = tables_dir / "bootstrap_ci_summary.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")
    return 0


def _auc_sample(df, rng, y, s, w):
    idx = rng.choice(np.arange(len(df)), size=len(df), replace=True)
    return weighted_auc(y[idx], s[idx], w[idx])


def _rmse_sample(df, rng, y_reg, y_reg_pred, w):
    idx = rng.choice(np.arange(len(df)), size=len(df), replace=True)
    return weighted_rmse(y_reg[idx], y_reg_pred[idx], w[idx])


if __name__ == "__main__":
    sys.exit(main())
