#!/usr/bin/env python3
"""Generate a structured benchmark table comparing the current study
with published PISA-ML studies from SCI/SSCI journals (2024-2025).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pisa_xai.io import require_package


def main() -> int:
    require_package("pandas", "pip install -r requirements.txt")
    import pandas as pd

    benchmark_data = [
        {
            "study": "Current study",
            "journal": "(target journal)",
            "year": 2025,
            "n_students": 613744,
            "pisa_wave": "2022",
            "countries": 80,
            "models": "XGBoost, LightGBM, HGB, RF, Stacking",
            "best_auc": 0.903,
            "best_r2": 0.681,
            "xai_methods": "SHAP, Permutation, LIME, ALE",
            "fairness": "Descriptive subgroups + formal metrics",
            "country_scope": "Global (80 countries)",
            "theory": "Ecological systems + Digital divide",
        },
        {
            "study": "Gómez-Talal et al.",
            "journal": "IEEE Access",
            "year": 2025,
            "n_students": "~30,000",
            "pisa_wave": "2022",
            "countries": 1,
            "models": "Stacking meta-model (8 models)",
            "best_auc": 0.977,
            "best_r2": "—",
            "xai_methods": "SHAP + UMAP dashboard",
            "fairness": "Not evaluated",
            "country_scope": "Spain only",
            "theory": "Not specified",
        },
        {
            "study": "Huang et al.",
            "journal": "J. of Intelligence (SSCI)",
            "year": 2024,
            "n_students": 34968,
            "pisa_wave": "2022",
            "countries": 6,
            "models": "XGBoost",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "SHAP",
            "fairness": "Not evaluated",
            "country_scope": "East Asia (6 systems)",
            "theory": "Not specified",
        },
        {
            "study": "Cheung et al.",
            "journal": "Brit. J. Educ. Psychol. (SSCI)",
            "year": 2024,
            "n_students": 147658,
            "pisa_wave": "2022",
            "countries": 79,
            "models": "Random Forest",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "SHAP",
            "fairness": "Gender subgroup",
            "country_scope": "Global (79 countries)",
            "theory": "Academic resilience theory",
        },
        {
            "study": "Alvarez-Garcia et al.",
            "journal": "Computers & Education (SSCI, Q1)",
            "year": 2024,
            "n_students": 30800,
            "pisa_wave": "2022",
            "countries": 1,
            "models": "XGBoost + clustering",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "SHAP (local + global)",
            "fairness": "Not evaluated",
            "country_scope": "Spain only",
            "theory": "Not specified",
        },
        {
            "study": "Öz & Bulut",
            "journal": "Educ. & Inf. Technol. (SSCI)",
            "year": 2025,
            "n_students": 613744,
            "pisa_wave": "2022",
            "countries": 80,
            "models": "Stacking ensemble",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "Not specified",
            "fairness": "Not evaluated",
            "country_scope": "Global (80 countries)",
            "theory": "Not specified",
        },
        {
            "study": "Khine et al.",
            "journal": "Education Sciences (SSCI)",
            "year": 2024,
            "n_students": 13437,
            "pisa_wave": "2022",
            "countries": 1,
            "models": "XGBoost, SVM, KNN, DT",
            "best_auc": "—",
            "best_r2": 0.42,
            "xai_methods": "SHAP",
            "fairness": "Not evaluated",
            "country_scope": "Australia only",
            "theory": "Not specified",
        },
        {
            "study": "Zhu et al.",
            "journal": "Int. J. Educ. Res. (SSCI)",
            "year": 2025,
            "n_students": "~4,500",
            "pisa_wave": "2022",
            "countries": 1,
            "models": "XGBoost, LightGBM, RF",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "SHAP",
            "fairness": "Not evaluated",
            "country_scope": "U.S. only",
            "theory": "Not specified",
        },
        {
            "study": "Soares",
            "journal": "J. Psychoeduc. Assess.",
            "year": 2024,
            "n_students": "~20,000",
            "pisa_wave": "2018",
            "countries": 1,
            "models": "RF + multilevel logistic",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "RF importance + multilevel OR",
            "fairness": "Not evaluated",
            "country_scope": "Brazil only",
            "theory": "Not specified",
        },
        {
            "study": "Liu et al.",
            "journal": "iScience (SCIE, IF=4.6)",
            "year": 2024,
            "n_students": 12058,
            "pisa_wave": "2018",
            "countries": 1,
            "models": "RE-EM regression trees + RF",
            "best_auc": "—",
            "best_r2": "—",
            "xai_methods": "RE-EM tree structure",
            "fairness": "Not evaluated",
            "country_scope": "China (4 provinces)",
            "theory": "Not specified",
        },
    ]

    df = pd.DataFrame(benchmark_data)

    output_dir = Path(__file__).resolve().parents[1] / "reports" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "literature_benchmark_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved benchmark table: {output_path}")
    print(f"\n{df[['study', 'journal', 'n_students', 'countries', 'xai_methods', 'fairness', 'theory']].to_string(index=False)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
