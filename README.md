# Explainable and Fair Machine Learning for Educational Analytics

Reproducible analysis pipeline for the paper:

**Explainable and Fair Machine Learning for Educational Analytics: Interpreting PISA 2022 Mathematics Achievement across 80 Countries**

Target journal: *IEEE Transactions on Learning Technologies* (TLT).

## Overview

This repository implements a fully reproducible analytical framework that combines multi-method explainable AI (XAI) with formal algorithmic-fairness evaluation on large-scale international assessment data. It predicts PISA 2022 mathematics achievement for 613,744 students across 80 countries and economies using theory-driven feature organization (Bronfenbrenner's ecological systems theory + van Dijk's ICT taxonomy), tuned XGBoost/LightGBM ensembles, four XAI methods (SHAP, permutation importance, ALE, LIME) with rank-correlation convergence validation, and intersectional fairness auditing (Equalized Odds, Demographic Parity, ABROCA).

## Reproducibility

- **Fixed random seed**: `20260510` (set in `configs/project.json` and used across all scripts)
- **Deterministic explanation samples**: SHAP 5,000 rows; permutation 10,000 rows × 5 repeats; ALE 5,000 rows / 20 bins; LIME 500 instances
- **Runtime**: ~12 hours on a single workstation
- **Weights**: final student weights `W_FSTUWT` (mean-normalized); 80 BRR replicate weights for descriptive estimates; 10 mathematics plausible values

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the downloaded PISA 2022 public-use files in `data/raw/`:

- Student questionnaire data (e.g., `CY08MSP_STU_QQQ.sav` / `.parquet`)
- School questionnaire data (e.g., `CY08MSP_SCH_QQQ.sav` / `.parquet`)

Official source: [OECD PISA 2022 Database](https://www.oecd.org/en/data/datasets/pisa-2022-database.html).

This repository does **not** redistribute OECD raw data files.

## Run Order

```bash
python scripts/00_check_inputs.py      # verify inputs
python scripts/01_prepare_data.py      # prepare analysis frame
python scripts/02_describe_sample.py   # descriptive statistics (BRR + PV)
python scripts/03_train_models.py      # train + tune models, holdout predictions
python scripts/04_explain_models.py    # SHAP / permutation / LIME explanations
python scripts/06_robustness_checks.py # 8 robustness checks + calibration
python scripts/05_build_tables.py      # build result tables
python scripts/08_generate_latex_tables.py
python scripts/09_visualizations.py    # publication figures
python scripts/10_counterfactual_xai.py
python scripts/11_umap_dpi.py
python scripts/12_multi_xai_comparison.py
python scripts/13_explanation_stability.py
python scripts/14_ale_analysis.py
python scripts/15_fairness_evaluation.py   # formal fairness metrics + intersectional audit
python scripts/23_per_country_analysis.py  # per-country performance
python scripts/24_mice_robustness.py
python scripts/26_ebm_baseline.py          # glass-box baseline
python scripts/27_knowledge_ablation.py
python scripts/28_kfold_cv.py              # supplementary 5-fold CV
python scripts/29_xai_convergence_verify.py # XAI convergence recomputation (verified rho values)
python scripts/30_headline_intersectional.py # merged headline intersectional subgroup audit
```

## Key Results

- Tuned XGBoost: AUC = 0.903 (95% bootstrap CI [0.898, 0.907]), RMSE = 54.10 (R² = 0.681), 23% RMSE reduction over the default-parameter ridge baseline
- XAI convergence (recomputed): cross-model SHAP ρ = 0.99; SHAP vs permutation 0.81; SHAP vs ALE 0.63; LIME diverges (ρ = 0.01) under feature correlation
- Fairness: SES is the largest concern (ABROCA = 0.027); low-SES immigrant-background students most underserved (AUC 0.779 vs 0.880, gap 0.101, n = 2,495 in holdout)
- Country-group holdout AUC = 0.847 — context-specific validation required before deployment

## License

MIT.
