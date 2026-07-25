# PISA 2022 Knowledge-Informed Expert System for Educational Assessment

This repository implements the reproducible pipeline for the paper:

**A Knowledge-Informed Expert System for Educational Assessment: Multi-Method Explainable AI with Formal Fairness Evaluation on Global-Scale Student Achievement Data**

Target journal: *Expert Systems with Applications* (ESWA).

## What Is Already Implemented

- Project structure for data, scripts, reports, documentation, and manuscript files.
- PISA-aware constants for student weights, replicate weights, plausible values, and low-performer threshold.
- Reusable Python modules for:
  - locating and loading PISA files;
  - computing BRR standard errors;
  - combining plausible-value estimates;
  - selecting candidate variables;
  - building baseline and machine-learning models;
  - evaluating regression and classification tasks;
  - producing SHAP or permutation-importance explanations.
- Script sequence from input checks to model interpretation.
- Literature matrix, variable plan, manuscript skeleton, cover letter, highlights, and data availability text.

## What You Need To Provide

Place the downloaded PISA 2022 public use files in `data/raw/`.

Recommended files:

- Student questionnaire data file, usually named like `CY08MSP_STU_QQQ.sav`, `.sas7bdat`, `.csv`, or `.parquet`.
- School questionnaire data file, usually named like `CY08MSP_SCH_QQQ.sav`, `.sas7bdat`, `.csv`, or `.parquet`.
- OECD codebook and technical report PDF, optional but useful for final methods writing.

Official source: [OECD PISA 2022 Database](https://www.oecd.org/en/data/datasets/pisa-2022-database.html).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Current machine note: the base Python environment does not have the required analysis packages installed yet.

## Run Order

```bash
python scripts/00_check_inputs.py
python scripts/01_prepare_data.py
python scripts/02_describe_sample.py
python scripts/03_train_models.py
python scripts/04_explain_models.py
python scripts/06_robustness_checks.py
python scripts/05_build_tables.py
```

The scripts fail early with clear messages if required PISA files or Python packages are missing.

For the manuscript draft, keep `05_build_tables.py` last so the generated artifact index includes every table and figure.

## Default Analysis Decisions

- Continuous outcome: mean of `PV1MATH` through `PV10MATH` for model training convenience; plausible-value pooling is retained for descriptive/statistical estimates.
- Classification outcome: low-performing student, default threshold `PV*MATH < 420.07`, matching the lower bound of PISA mathematics Level 2.
- Weight: `W_FSTUWT`.
- Replicate weights: `W_FSTURWT1` through `W_FSTURWT80`.
- Main target journal: **Expert Systems with Applications** (ESWA).

## Important Statistical Guardrails

- Do not describe predictors as causes unless a separate causal design is added.
- Report PISA weights and plausible values in the methods.
- Use replicate weights for standard errors in descriptive and regression-style claims.
- Use machine learning metrics for prediction and SHAP for interpretation, while explicitly distinguishing prediction importance from statistical significance.

## Local Validation

Pure Python utility tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Full pipeline validation requires the dependencies and PISA data files.
