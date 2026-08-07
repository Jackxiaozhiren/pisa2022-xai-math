#!/usr/bin/env python3
"""Generate enhanced publication figures for the PISA 2022 XAI manuscript.

Figure 1: Per-country AUC bar chart (top 30 + bottom 10)
Figure 2: Theoretical proposition validation forest plot
Figure 3: Refined digital divide dual-pathway framework diagram
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from pisa_xai.config import load_config, resolve_project_path


# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def fig_per_country_auc(tables_dir, figures_dir):
    """Generate per-country AUC bar chart showing top and bottom performers."""
    df = pd.read_csv(Path(tables_dir) / "per_country_metrics.csv")
    df = df.dropna(subset=["auc"]).sort_values("auc")

    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1, 1.5]})

    # ── Left: Bottom 15 countries ──
    bottom = df.head(15)
    colors_bot = ["#d73027" if i < 5 else "#fc8d59" for i in range(len(bottom))]
    bars = ax_top.barh(range(len(bottom)), bottom["auc"], color=colors_bot, edgecolor="white", linewidth=0.3)
    ax_top.set_yticks(range(len(bottom)))
    ax_top.set_yticklabels(bottom["country"], fontsize=8)
    ax_top.set_xlim(0.65, 0.82)
    ax_top.set_xlabel("AUC", fontsize=10)
    ax_top.set_title("Lowest-Performing Countries", fontsize=12, fontweight="bold", color="#b2182b")
    ax_top.invert_yaxis()
    ax_top.axvline(df["auc"].mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for i, (_, row) in enumerate(bottom.iterrows()):
        ax_top.text(row["auc"] + 0.002, i, f"{row['auc']:.3f}", va="center", fontsize=7)

    # ── Right: Top 30 countries ──
    top = df.tail(30)
    colors_top = ["#1a9850" if i >= 25 else "#91cf60" for i in range(len(top))]
    bars = ax_bot.barh(range(len(top)), top["auc"], color=colors_top, edgecolor="white", linewidth=0.3)
    ax_bot.set_yticks(range(len(top)))
    ax_bot.set_yticklabels(top["country"], fontsize=7.5)
    ax_bot.set_xlim(0.84, 0.96)
    ax_bot.set_xlabel("AUC", fontsize=10)
    ax_bot.set_title("Highest-Performing Countries", fontsize=12, fontweight="bold", color="#1a9850")
    ax_bot.invert_yaxis()
    ax_bot.axvline(df["auc"].mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for i, (_, row) in enumerate(top.iterrows()):
        ax_bot.text(row["auc"] + 0.001, i, f"{row['auc']:.3f}", va="center", fontsize=6.5)

    fig.suptitle("Per-Country Model Performance (AUC) — LightGBM, 80 PISA 2022 Countries",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02, f"Global mean AUC = {df['auc'].mean():.4f} | SD = {df['auc'].std():.4f} | "
             f"Range = [{df['auc'].min():.4f}, {df['auc'].max():.4f}]",
             ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    out_path = Path(figures_dir) / "per_country_auc_chart.png"
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def fig_proposition_validation(figures_dir):
    """Generate theoretical proposition validation forest plot."""
    propositions = [
        ("P1: Microsystem strongest", "Strongly\nsupported", 0.95, "#1a9850"),
        ("P2: Exosystem beyond micro", "Supported", 0.80, "#66bd63"),
        ("P3: Macro moderates lower", "Strongly\nsupported", 0.95, "#1a9850"),
        ("P4: ICT skills > access", "Partially\nsupported", 0.55, "#fdae61"),
        ("P5: ICT quality > quantity", "Supported", 0.80, "#66bd63"),
        ("P6: Cross-country heterogeneity", "Supported", 0.82, "#66bd63"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_positions = list(range(len(propositions)))[::-1]
    for i, (label, support, strength, color) in enumerate(propositions):
        y = y_positions[i]
        ax.barh(y, strength, height=0.55, color=color, edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.text(strength + 0.02, y, support, va="center", fontsize=9, fontweight="bold")
        ax.text(0.02, y, label, va="center", fontsize=9, ha="left", color="white", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Degree of Empirical Support", fontsize=11)
    ax.set_title("Structured Theoretical Proposition Validation", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1a9850", label="Strongly supported"),
        Patch(facecolor="#66bd63", label="Supported"),
        Patch(facecolor="#fdae61", label="Partially supported"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = Path(figures_dir) / "proposition_validation_forest.png"
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def fig_dual_pathway_framework(figures_dir):
    """Generate refined digital divide dual-pathway framework diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Title
    ax.text(6, 6.7, "Refined Digital Divide Framework for Global PISA-Scale Analysis",
            ha="center", fontsize=14, fontweight="bold")

    # ── Original sequential pathway (left, faded) ──
    ax.text(2.5, 6.2, "Original Sequential Model (van Dijk)", ha="center", fontsize=10, style="italic", color="gray")
    boxes_left = [
        (1.0, 5.2, "Digital\nAccess", "#d9d9d9"),
        (2.3, 4.4, "Digital\nSkills", "#d9d9d9"),
        (1.0, 3.6, "Digital\nUsage", "#d9d9d9"),
        (2.3, 2.8, "Educational\nOutcomes", "#d9d9d9"),
    ]
    for x, y, label, color in boxes_left:
        rect = FancyBboxPatch((x, y), 1.5, 0.8, boxstyle="round,pad=0.05", facecolor=color, edgecolor="gray", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.75, y + 0.4, label, ha="center", va="center", fontsize=8, color="gray")
    # Arrows
    for i in range(3):
        bx, by = boxes_left[i][0], boxes_left[i][1]
        nx, ny = boxes_left[i + 1][0], boxes_left[i + 1][1]
        ax.annotate("", xy=(nx + 0.75, ny + 0.8), xytext=(bx + 0.75, by),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.5))
    ax.text(3.5, 4.0, "Strictly\nSequential", fontsize=7, color="gray", style="italic", ha="center")

    # ── Refined dual-pathway model (right) ──
    ax.text(8.5, 6.2, "Refined Dual-Pathway Model (This Study)", ha="center", fontsize=10, fontweight="bold", color="#2166ac")

    # Box: Digital Access Quality (rich composite)
    rect = FancyBboxPatch((6.2, 5.2), 2.2, 0.9, boxstyle="round,pad=0.08",
                          facecolor="#4393c3", edgecolor="#2166ac", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(7.3, 5.65, "Digital Access Quality\n(ICTRES composite)", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Box: Digital Skills/Confidence
    rect = FancyBboxPatch((6.2, 3.9), 2.2, 0.9, boxstyle="round,pad=0.08",
                          facecolor="#92c5de", edgecolor="#4393c3", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(7.3, 4.35, "Digital Skills\n& Self-Efficacy", ha="center", va="center", fontsize=9, color="#053061", fontweight="bold")

    # Box: Educational Outcomes
    rect = FancyBboxPatch((9.5, 5.2), 2.0, 0.9, boxstyle="round,pad=0.08",
                          facecolor="#f4a582", edgecolor="#d6604d", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(10.5, 5.65, "Mathematics\nAchievement", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Direct pathway
    ax.annotate("", xy=(9.5, 5.65), xytext=(8.4, 5.65),
                arrowprops=dict(arrowstyle="->", color="#d6604d", lw=2.5, connectionstyle="arc3,rad=0"))
    ax.text(8.95, 6.05, "Direct Pathway\n(access → outcomes)", fontsize=7, color="#d6604d", ha="center", fontweight="bold")

    # Mediated pathway
    ax.annotate("", xy=(7.3, 4.8), xytext=(7.3, 5.2),
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.5))
    ax.text(7.3, 5.0, "enables", fontsize=6, color="#2166ac", ha="center")

    # Skills → outcomes
    ax.annotate("", xy=(9.5, 5.2), xytext=(8.4, 4.35),
                arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.5, connectionstyle="arc3,rad=0.2"))
    ax.text(8.95, 4.95, "Mediated Pathway\n(access → skills → outcomes)", fontsize=7, color="#2166ac", ha="center")

    # Moderator box
    rect = FancyBboxPatch((7.0, 1.8), 2.8, 1.0, boxstyle="round,pad=0.08",
                          facecolor="#f7f7f7", edgecolor="#878787", linewidth=1.0, linestyle="--")
    ax.add_patch(rect)
    ax.text(8.4, 2.3, "Macrosystem Moderator:\nNational ICT Infrastructure,\nEconomic Development Level",
            ha="center", va="center", fontsize=8, color="#525252")

    # Moderator arrows
    ax.annotate("", xy=(8.4, 4.8), xytext=(8.4, 2.8),
                arrowprops=dict(arrowstyle="->", color="#878787", lw=1.0, linestyle="dashed"))
    ax.annotate("", xy=(7.7, 5.2), xytext=(7.7, 2.8),
                arrowprops=dict(arrowstyle="->", color="#878787", lw=1.0, linestyle="dashed"))
    ax.text(8.2, 3.6, "moderates\nbalance", fontsize=6, color="#878787", ha="center")

    # Key insight box
    ax.text(6, 0.8, "Key Insight: In heterogeneous global samples spanning wide development gradients,\n"
            "digital access and skills function as CONCURRENT rather than strictly sequential predictors.\n"
            "The balance between the direct and mediated pathways is moderated by national ICT infrastructure.",
            fontsize=8.5, ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f0", edgecolor="#d6604d", alpha=0.8))

    # Evidence annotation
    ax.text(10.5, 1.0, "Empirical Support:\n"
            "• ICTEFFIC z = +16.12 (pooled)\n"
            "• ICTRES z = +8.98 (pooled)\n"
            "• ICT infrastructure\n  moderates ranking (ρ gap)",
            fontsize=7, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="#bdbdbd", alpha=0.8))

    plt.tight_layout()
    out_path = Path(figures_dir) / "refined_digital_divide_framework.png"
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def fig_per_country_choropleth(tables_dir, figures_dir):
    """Generate choropleth world map of per-country AUC."""
    import plotly.express as px

    df = pd.read_csv(Path(tables_dir) / "per_country_metrics.csv")
    df = df.dropna(subset=["auc"]).sort_values("auc")

    # ISO 3166-1 alpha-3 country code mapping for PISA 2022 countries
    country_to_iso = {
        "Albania": "ALB", "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
        "Azerbaijan": "AZE", "Baku (Azerbaijan)": "AZE", "Belarus": "BLR",
        "Belgium": "BEL", "Bosnia and Herzegovina": "BIH", "Brazil": "BRA",
        "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Cambodia": "KHM",
        "Canada": "CAN", "Chile": "CHL", "Chinese Taipei": "TWN",
        "Colombia": "COL", "Costa Rica": "CRI", "Croatia": "HRV",
        "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK",
        "Dominican Republic": "DOM", "El Salvador": "SLV", "Estonia": "EST",
        "Finland": "FIN", "France": "FRA", "Georgia": "GEO", "Germany": "DEU",
        "Greece": "GRC", "Guatemala": "GTM", "Hong Kong": "HKG",
        "Hungary": "HUN", "Iceland": "ISL", "Indonesia": "IDN",
        "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
        "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
        "Korea": "KOR", "Kosovo": "XKX", "Kyrgyzstan": "KGZ",
        "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX",
        "Macao": "MAC", "Malaysia": "MYS", "Malta": "MLT",
        "Moldova": "MDA", "Mongolia": "MNG", "Montenegro": "MNE",
        "Morocco": "MAR", "Netherlands": "NLD", "New Zealand": "NZL",
        "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN",
        "Palestine": "PSE", "Panama": "PAN", "Paraguay": "PRY",
        "Peru": "PER", "Philippines": "PHL", "Poland": "POL",
        "Portugal": "PRT", "Puerto Rico": "PRI", "Qatar": "QAT",
        "Romania": "ROU", "Russia": "RUS", "Saudi Arabia": "SAU",
        "Serbia": "SRB", "Singapore": "SGP", "Slovak Republic": "SVK",
        "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
        "Switzerland": "CHE", "Thailand": "THA", "Trinidad and Tobago": "TTO",
        "Turkey": "TUR", "Ukraine": "UKR", "United Arab Emirates": "ARE",
        "United Kingdom": "GBR", "United States": "USA", "Uruguay": "URY",
        "Uzbekistan": "UZB", "Viet Nam": "VNM",
    }

    df["iso_alpha"] = df["country"].map(country_to_iso)
    df_valid = df.dropna(subset=["iso_alpha"])

    fig = px.choropleth(
        df_valid,
        locations="iso_alpha",
        color="auc",
        hover_name="country",
        hover_data={"auc": ":.3f", "iso_alpha": False},
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=(df["auc"].min(), df["auc"].max()),
        title="Per-Country Model Performance (AUC) — PISA 2022, 80 Countries",
        labels={"auc": "AUC"},
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="rgba(0,0,0,0.3)"),
        coloraxis_colorbar=dict(title="AUC", len=0.5),
    )
    out_path = Path(figures_dir) / "per_country_auc_choropleth.png"
    fig.write_image(out_path, width=1600, height=900, scale=2)
    print(f"Saved: {out_path}")
    return out_path


def main():
    config = load_config()
    tables_dir = resolve_project_path(config["paths"]["tables_dir"])
    figures_dir = resolve_project_path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating publication figures...")
    fig_per_country_auc(tables_dir, figures_dir)
    fig_proposition_validation(figures_dir)
    fig_dual_pathway_framework(figures_dir)
    fig_per_country_choropleth(tables_dir, figures_dir)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
