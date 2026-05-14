#!/usr/bin/env python3
"""UMAP visualization and Digital Poverty Index analysis.

Phase F: UMAP projection of student feature space, colored by performance/ICT.
Phase E: Construct Digital Poverty Index from ICT resource indicators.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.config import load_config, resolve_project_path
from pisa_xai.io import load_table, require_package


def compute_digital_poverty_index(df):
    """Composite Digital Poverty Index from ICT resource indicators.

    Combines ICTRES (ICT resources at home), ICTHOME (ICT availability at home),
    and ICTSCH (ICT availability at school) into a single normalized index.
    Lower values = greater digital poverty.
    """
    import numpy as np

    ict_cols = []
    for c in ["ICTRES", "ICTHOME", "ICTSCH"]:
        if c in df.columns:
            ict_cols.append(c)

    if not ict_cols:
        return None

    # Z-score each component and average
    z_scores = np.zeros(len(df))
    for c in ict_cols:
        col = df[c]
        if str(col.dtype) == "category":
            col = col.cat.codes.astype("float64")
        mean_val = col.mean()
        std_val = col.std()
        if std_val > 0:
            z_scores += (col - mean_val) / std_val
    dpi = z_scores / len(ict_cols)

    # Correlation with ESCS
    if "ESCS" in df.columns:
        escs = df["ESCS"]
        if str(escs.dtype) == "category":
            escs = escs.cat.codes.astype("float64")
        dpi_corr = np.corrcoef(dpi, escs.fillna(escs.median()))[0, 1]
    else:
        dpi_corr = float("nan")

    return {"index": dpi, "corr_with_escs": dpi_corr, "components": ict_cols}


def main() -> int:
    require_package("umap-learn", "pip install umap-learn")
    require_package("matplotlib", "pip install -r requirements.txt")
    require_package("numpy", "pip install -r requirements.txt")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import umap

    config = load_config()
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    model_dir = processed_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(processed_dir / "pisa2022_math_model_frame.parquet")
    features = json.loads((model_dir / "features.json").read_text(encoding="utf-8"))

    # Convert category columns
    x = df[features].copy()
    cat_cols = [c for c in x.columns if str(x[c].dtype) == "category"]
    for c in cat_cols:
        x[c] = x[c].cat.codes.astype("float64")

    # Fill missing with median
    x = x.fillna(x.median())

    # ── Phase E: Digital Poverty Index ───────────────────────────
    print("Computing Digital Poverty Index", flush=True)
    dpi_result = compute_digital_poverty_index(df)
    if dpi_result is not None:
        df["DPI"] = dpi_result["index"]
        print(f"DPI-ESCS correlation: {dpi_result['corr_with_escs']:.4f}", flush=True)
        print(f"DPI components: {dpi_result['components']}", flush=True)

        # DPI quintile analysis
        df["DPI_QUINTILE"] = pd.qcut(
            pd.Series(dpi_result["index"]).rank(method="first"), 5, labels=["Q1 (lowest ICT)", "Q2", "Q3", "Q4", "Q5 (highest ICT)"]
        )
        dpi_stats = (
            df.groupby("DPI_QUINTILE", observed=True)
            .agg(
                n_students=("DPI", "count"),
                mean_math=("MATH_PV_MEAN", "mean"),
                low_performer_rate=("LOW_PERFORMER_MATH", "mean"),
                mean_escs=("ESCS", "mean"),
            )
            .reset_index()
        )
        dpi_stats.to_csv(tables_dir / "digital_poverty_index.csv", index=False)
        print(dpi_stats.to_string(index=False))

        # Plot DPI vs Math
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sample = df.sample(min(10000, len(df)), random_state=config["sample"]["random_state"])
        ax = axes[0]
        ax.scatter(sample["DPI"], sample["MATH_PV_MEAN"], alpha=0.1, s=1)
        ax.set_xlabel("Digital Poverty Index (higher = more ICT resources)")
        ax.set_ylabel("Mathematics Score")
        ax.set_title("Digital Resources vs. Mathematics Achievement")
        ax.axhline(y=420.07, color="red", linestyle="--", alpha=0.5, label="Low-performer cutoff")
        ax.legend()

        ax2 = axes[1]
        dpi_stats.plot(x="DPI_QUINTILE", y=["low_performer_rate"], kind="bar", ax=ax2, legend=False)
        ax2.set_xlabel("DPI Quintile")
        ax2.set_ylabel("Low Performer Rate")
        ax2.set_title("Low Performer Rate by Digital Poverty Quintile")
        plt.tight_layout()
        plt.savefig(figures_dir / "digital_poverty_index.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ── Phase F: UMAP visualization ─────────────────────────────
    print("Computing UMAP projection", flush=True)
    n_umap = min(20000, len(x))
    x_sample = x.sample(n_umap, random_state=config["sample"]["random_state"])
    idx_sample = x_sample.index

    reducer = umap.UMAP(n_components=2, random_state=config["sample"]["random_state"], n_jobs=-1)
    embedding = reducer.fit_transform(x_sample.values)

    # Plot colored by low-performer status
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    labels = df.loc[idx_sample, "LOW_PERFORMER_MATH"].values
    for lbl, color, name in [(0, "#2ca02c", "Non-Low-Performer"), (1, "#d62728", "Low-Performer")]:
        mask = labels == lbl
        axes2[0].scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=name, alpha=0.3, s=1)
    axes2[0].set_title("UMAP Projection by Performance Status")
    axes2[0].legend(markerscale=8)

    # Plot colored by ICT resources
    if "ICTRES" in df.columns:
        ict_vals = df.loc[idx_sample, "ICTRES"]
        if str(ict_vals.dtype) == "category":
            ict_vals = ict_vals.cat.codes
        sc = axes2[1].scatter(embedding[:, 0], embedding[:, 1], c=ict_vals, cmap="viridis", alpha=0.3, s=1)
        axes2[1].set_title("UMAP Projection by ICT Resources")
        plt.colorbar(sc, ax=axes2[1], label="ICT Resources")

    # Plot colored by DPI
    if dpi_result is not None:
        dpi_vals = dpi_result["index"][idx_sample]
        sc2 = axes2[2].scatter(embedding[:, 0], embedding[:, 1], c=dpi_vals, cmap="plasma", alpha=0.3, s=1)
        axes2[2].set_title("UMAP Projection by Digital Poverty Index")
        plt.colorbar(sc2, ax=axes2[2], label="DPI")

    plt.tight_layout()
    plt.savefig(figures_dir / "umap_projections.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("UMAP and DPI outputs saved", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
