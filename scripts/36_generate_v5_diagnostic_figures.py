#!/usr/bin/env python3
"""Generate deterministic v5 data-derived diagnostic figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MANUSCRIPT = ROOT / "eaai_submission" / "manuscript"


def pooled(task: str, group: str, metric: str) -> pd.Series:
    table = pd.read_csv(TABLES / "v5_pv_pooled_metrics.csv")
    row = table[
        (table["estimand"] == "population")
        & (table["task"] == task)
        & (table["group"] == group)
        & (table["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one pooled row for {task}/{group}/{metric}")
    return row.iloc[0]


def save_both(fig, name: str, dpi: int = 300) -> None:
    (FIGURES / "pdf").mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / "pdf" / name
    manuscript = MANUSCRIPT / name
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(manuscript, bbox_inches="tight", dpi=dpi)


def generate_calibration_curves() -> None:
    predictions = pd.read_parquet(ROOT / "data" / "interim" / "v5_pv_specific_holdout_predictions.parquet")
    frame = pd.read_parquet(ROOT / "data" / "processed" / "pisa2022_math_model_frame.parquet")
    x_values = np.linspace(0.05, 0.95, 10)
    predicted_by_pv = []
    observed_by_pv = []
    for pv, group in predictions.groupby("pv", sort=True):
        score = group["classification_probability"].to_numpy(dtype=float)
        outcome = group["y_classification"].to_numpy(dtype=int)
        weights = frame.loc[group["row_index"].astype(int), "W_FSTUWT"].to_numpy(dtype=float)
        weights = weights / weights.mean()
        bin_index = np.minimum((score * 10).astype(int), 9)
        predicted = []
        observed = []
        for index in range(10):
            mask = bin_index == index
            if not mask.any():
                predicted.append(np.nan)
                observed.append(np.nan)
            else:
                predicted.append(np.average(score[mask], weights=weights[mask]))
                observed.append(np.average(outcome[mask], weights=weights[mask]))
        predicted_by_pv.append(predicted)
        observed_by_pv.append(observed)
    predicted_matrix = np.asarray(predicted_by_pv, dtype=float)
    observed_matrix = np.asarray(observed_by_pv, dtype=float)
    fig, ax = plt.subplots(figsize=(4.5, 3.4), constrained_layout=True)
    mean_predicted = np.nanmean(predicted_matrix, axis=0)
    mean_observed = np.nanmean(observed_matrix, axis=0)
    ax.plot(x_values, x_values, linestyle="--", color="#666666", linewidth=1, label="Perfect calibration")
    ax.plot(mean_predicted, mean_observed, marker="o", color="#0072B2", linewidth=2, label="PV-pooled XGBoost")
    ax.fill_between(
        mean_predicted,
        np.nanmin(observed_matrix, axis=0),
        np.nanmax(observed_matrix, axis=0),
        color="#0072B2",
        alpha=0.12,
        linewidth=0,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed imputed-target rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    save_both(fig, "calibration_curves.pdf")
    plt.close(fig)


def main() -> int:
    generate_calibration_curves()
    labels = ["Global", "Low-ESCS\nnon-native", "High-ESCS\nnative"]
    groups = ["global", "low_ses_non_native", "high_ses_native"]
    colors = ["#0072B2", "#D55E00", "#009E73"]

    auc_rows = [pooled("classification", group, "auc") for group in groups]
    ece_rows = [pooled("classification", group, "ece") for group in groups]
    slope_rows = [pooled("classification", group, "calibration_slope") for group in groups]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    x = np.arange(len(labels))
    auc_values = np.array([row["estimate"] for row in auc_rows])
    auc_err = np.array(
        [
            [row["estimate"] - row["ci_lower"] for row in auc_rows],
            [row["ci_upper"] - row["estimate"] for row in auc_rows],
        ]
    )
    axes[0].bar(x, auc_values, yerr=auc_err, capsize=3, color=colors, edgecolor="white")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.65, 0.95)
    axes[0].set_title("AUC diagnostic")
    axes[0].grid(axis="y", alpha=0.2)

    ece_values = np.array([row["estimate"] for row in ece_rows])
    ece_err = np.array(
        [
            [row["estimate"] - row["ci_lower"] for row in ece_rows],
            [row["ci_upper"] - row["estimate"] for row in ece_rows],
        ]
    )
    axes[1].bar(x, ece_values, yerr=ece_err, capsize=3, color=colors, edgecolor="white")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Expected calibration error")
    axes[1].set_ylim(0, 0.20)
    axes[1].set_title("Calibration diagnostic")
    axes[1].grid(axis="y", alpha=0.2)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
    save_both(fig, "intersectional_heatmap.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].bar(x, ece_values, color=colors, edgecolor="white")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("ECE")
    axes[0].set_title("PV-pooled ECE")
    axes[0].set_ylim(0, 0.14)
    axes[1].bar(x, [row["estimate"] for row in slope_rows], color=colors, edgecolor="white")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Calibration slope")
    axes[1].set_title("PV-pooled slope")
    axes[1].set_ylim(0, 1.2)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.grid(axis="y", alpha=0.2)
    save_both(fig, "calibration_parity_figure.png", dpi=300)
    plt.close(fig)
    print("Saved v5 diagnostic figures to manuscript and reports/figures/pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
