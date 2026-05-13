# Reproducibility Statement

## Analysis Environment

- Project root: `/Users/jackson/论文/pisa2022-xai-math/workspace`.
- Python environment: `.venv` in the project root.
- Main command convention: run scripts from the project root with `.venv/bin/python`.
- Current validation commands:
  - `.venv/bin/python -m py_compile src/pisa_xai/*.py scripts/*.py`
  - `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'`

## Input Data

The raw PISA 2022 files are not redistribution artifacts. They must be obtained by readers from the OECD PISA 2022 Database and placed under `data/raw/`.

Locally verified inputs are documented in `docs/sources/pisa/download_manifest.csv`:

- `data/raw/STU_QQQ_SPSS.zip`
- `data/raw/CY08MSP_STU_QQQ.SAV`
- `data/raw/SCH_QQQ_SPSS.zip`
- `data/raw/CY08MSP_SCH_QQQ.SAV`
- `docs/sources/pisa/CY08MSP_CODEBOOK_27thJune24.xlsx`
- OECD PISA 2022 technical and results report PDFs under `docs/sources/pisa/reports/`

## Reproduction Order

Run the scripts in this order:

```bash
.venv/bin/python scripts/00_check_inputs.py
.venv/bin/python scripts/01_prepare_data.py
.venv/bin/python scripts/02_describe_sample.py
.venv/bin/python scripts/03_train_models.py
.venv/bin/python scripts/04_explain_models.py
.venv/bin/python scripts/06_robustness_checks.py
.venv/bin/python scripts/05_build_tables.py
```

`scripts/05_build_tables.py` should remain last so `manuscript/generated_tables_index.md` captures all generated tables and figures.

## Current Reproducibility Targets

- Processed frame: `data/processed/pisa2022_math_model_frame.parquet`.
- Main feature set: 33 predictors listed in `data/processed/feature_sets.json`.
- Main model split: random state `20260510`, 80/20 split, `490,995` training rows and `122,749` holdout rows.
- Headline model results: LightGBM classification AUC about `0.8904`; LightGBM regression RMSE about `57.61`.

## Public Release Boundary

The public repository should include source code, configuration, manuscript source, non-restricted generated tables/figures, and source manifests. It should not include OECD raw data files, extracted raw `.SAV` files, model artifacts containing raw-row-derived fitted state if the release policy is uncertain, local caches, personal metadata, or credentials.
