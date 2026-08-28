# Explainable and Fairness-Audited Machine Learning for Large-Scale Educational Assessment

Reproducible analysis pipeline for the paper:

**Explainable and Fairness-Audited Machine Learning for Large-Scale Educational Assessment: Multi-Method XAI and a Calibration-Parity Audit on PISA 2022 across 80 Countries**

## Overview

This repository implements a reproducible analytical framework combining multi-method explainable AI (XAI) with algorithmic-fairness evaluation on PISA 2022 mathematics data. The workflow covers 613,744 students across 80 countries and economies and incorporates PISA plausible values, survey weighting, model comparison, explanation methods, robustness checks, and subgroup/fairness audits.

## Current manuscript-active route

The manuscript-active Route A analysis uses ten mathematics plausible values, 80 Fay--BRR replicate weights, population-versus-SENWT sensitivity, a matched additive EBM, and a whole-school unseen-institution cold-start stress test.

Current manuscript-active headline values:

- Primary PV-pooled XGBoost: AUC 0.8865, Brier 0.1375, RMSE 59.82, R² 0.6346.
- Matched EBM: AUC 0.8689, Brier 0.1465, RMSE 66.15, R² 0.5531.
- Unseen-school secondary validation: AUC 0.8865, Brier 0.1358, RMSE 61.02, R² 0.6219, evaluated on 4,326 held-out schools across 80 countries.
- The intersectional C1 signal is descriptive: design-aware intervals cross zero and SENWT sensitivity attenuates or reverses point contrasts.

The corresponding scripts, aggregate outputs, manifests, and scientific methodology records are in `scripts/33_*`, `scripts/34_*`, `scripts/36_*`, `scripts/37_*`, `src/pisa_xai/v5_survey.py`, `reports/tables/v5_*`, and `docs/v5_eaai/`.

Historical baseline results are intentionally separated from the active manuscript path; see `docs/LEGACY_BASELINE.md`.

## Reproducibility

- **Fixed random seed:** `20260510` (`configs/project.json`).
- **Survey design:** 10 mathematics plausible values and 80 Fay--BRR replicate weights for manuscript-active Route A analyses.
- **Explanation samples:** SHAP 5,000 rows; permutation 10,000 rows × 5 repeats; ALE 5,000 rows / 20 bins; LIME 500 instances for the legacy/full XAI workflow where applicable.
- **Runtime:** approximately 12 hours for the broader workflow on a single workstation; manuscript-active subsets vary by route and cached artifacts.
- **Environment:** `requirements.txt` / `pyproject.toml` are compatibility specifications, not an exact historical lock. The final archival release must export the exact environment used for the manuscript run.

See `docs/REPRODUCIBILITY.md` for the archival-release checklist.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Obtain PISA 2022 public-use files from the OECD PISA 2022 Database and place the required files in `data/raw/`, for example:

- student questionnaire data (`CY08MSP_STU_QQQ.sav` or an equivalent converted form);
- school questionnaire data (`CY08MSP_SCH_QQQ.sav` or an equivalent converted form).

This repository does **not** redistribute OECD raw data, row-level predictions, or fitted model artifacts. See `PUBLIC_RELEASE_MANIFEST.md` and `manuscript/data_availability.md` for the public-data boundary.

## Main run order

```bash
python scripts/00_check_inputs.py
python scripts/01_prepare_data.py
python scripts/02_describe_sample.py
python scripts/03_train_models.py
python scripts/04_explain_models.py
python scripts/06_robustness_checks.py
python scripts/05_build_tables.py
python scripts/08_generate_latex_tables.py
python scripts/09_visualizations.py
python scripts/10_counterfactual_xai.py
python scripts/11_umap_dpi.py
python scripts/12_multi_xai_comparison.py
python scripts/13_explanation_stability.py
python scripts/14_ale_analysis.py
python scripts/15_fairness_evaluation.py
python scripts/23_per_country_analysis.py
python scripts/24_mice_robustness.py
python scripts/26_ebm_baseline.py
python scripts/27_knowledge_ablation.py
python scripts/28_kfold_cv.py
python scripts/29_xai_convergence_verify.py
python scripts/30_headline_intersectional.py
```

Route A manuscript-specific scripts are documented under `docs/v5_eaai/` and in their script headers.

## Repository structure

- `src/` — reusable analysis modules;
- `scripts/` — numbered end-to-end analysis steps;
- `configs/` — frozen project configuration and seeds;
- `tests/` — unit/regression tests;
- `reports/tables/` — aggregate result tables and manifests;
- `reports/figures/` — publication figures;
- `manuscript/` — manuscript and reproducibility-support source files;
- `docs/` — scientific protocol, source manifests, and methodology records.

## Citation

Use GitHub's **Cite this repository** function generated from `CITATION.cff` for the software/reproducibility repository. After formal article publication, the CFF metadata can be updated with the article as the preferred citation.

## License

MIT for project-authored code and repository documentation. Third-party PISA data and external source materials remain subject to their original terms.
