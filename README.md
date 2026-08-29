# PISA 2022 Model-Level Verification and Validation

Reproducibility repository for the EAAI manuscript:

**A Reproducible Model-Level Verification and Validation Protocol for Predictive AI in High-Impact Educational Assessment: Evidence from PISA 2022**

## Current manuscript-active route

The manuscript-active analysis is Route A: ten mathematics plausible-value-specific models, normalized student weights, 80 Fay--BRR replicate weights, population-versus-senate sensitivity, a matched additive Explainable Boosting Machine (EBM), and an unseen-school cold-start stress test. The analysis is explicitly model-level: it does not treat PISA plausible values as individual student scores and does not claim deployment or intervention effectiveness.

Current manuscript-active headline values:

- Population-pooled XGBoost: AUC **0.8865**, Brier **0.1375**, RMSE **59.82**, R² **0.6346**.
- Matched additive EBM: AUC **0.8689**, Brier **0.1465**, RMSE **66.15**, R² **0.5531**.
- Unseen-school cold-start: AUC **0.8865**, Brier **0.1358**, RMSE **61.02**, R² **0.6219**, over 4,326 held-out schools across 80 countries/economies.
- The low-ESCS non-native C1 contrast is a **design-sensitive descriptive diagnostic**: its AUC and calibration-slope intervals cross zero, and senate weighting attenuates the point contrasts.

The authoritative active-result register is `docs/v5_eaai/EAAI_v5_07_active_result_register.md`. Historical values such as AUC 0.903, Brier 0.126, RMSE 54.10, and R² 0.681 are retained only as legacy evidence; see `docs/LEGACY_BASELINE.md`.

## Canonical scientific sources

- `docs/v5_eaai/` — preregistered Route A protocol, validity audit, active-result register, claim/evidence crosswalk, and cold-start validation records.
- `scripts/33_*`, `scripts/34_*`, `scripts/36_*`, `scripts/37_*`, and `src/pisa_xai/v5_survey.py` — manuscript-active analysis code.
- `reports/tables/v5_*` — aggregate manuscript-active outputs and manifests.
- `manuscript/` — public data/reproducibility statements and manuscript-boundary documentation.
- The exact 2026-08-25 EAAI submission source package is maintained as a versioned release artifact rather than mixing Editorial Manager staging files into the repository tree.

## Reproducibility boundary

- Fixed random seed: `20260510` (`configs/project.json`).
- Survey design: 10 mathematics plausible values and 80 Fay--BRR replicate weights for Route A.
- The v5 uncertainty intervals quantify fixed-model evaluation uncertainty from replicate weights plus between-PV imputation variance; they are not full training uncertainty or individual inference.
- `requirements.txt` / `pyproject.toml` are compatibility specifications. `docs/ENVIRONMENT_OBSERVED_2026-08-25.txt` records package versions observed in the manuscript workstation environment from the archived `.dist-info` inventory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q src scripts tests
pytest -q
```

The independent audit of the 2026-08-25 research bundle passed **12/12 v5 survey tests**.

## Data

Obtain PISA 2022 public-use files directly from the OECD PISA 2022 Database and place the required student/school files in `data/raw/`. This repository does **not** redistribute OECD raw data, row-level holdout predictions, or fitted model artifacts. See `PUBLIC_RELEASE_MANIFEST.md` and `manuscript/data_availability.md`.

## Analysis routes

The repository intentionally preserves older analysis scripts and aggregate outputs because they document the research history. They are not silently promoted into the active manuscript claim set. For the current paper, start with:

```text
scripts/33_pisa_pv_replicate_weight_audit.py
scripts/34_controlled_ebm_baseline.py
scripts/36_generate_v5_diagnostic_figures.py
scripts/37_institution_cold_start_validation.py
src/pisa_xai/v5_survey.py
```

Earlier scripts `00_*` through `32_*` support the legacy/full XAI workflow and historical sensitivity analyses.

## Citation and license

Use GitHub's **Cite this repository** metadata from `CITATION.cff`. Project-authored code and repository documentation are MIT licensed. OECD data and other third-party materials remain subject to their original terms.
