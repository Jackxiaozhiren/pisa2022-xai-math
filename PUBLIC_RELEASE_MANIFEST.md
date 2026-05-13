# Public Release Manifest

Repository target: `Jackxiaozhiren/pisa2022-xai-math`

This package is the cleaned public-release copy of the PISA 2022 XAI mathematics project. It is intended for a GitHub repository containing reproducible code, manuscript source, aggregate outputs, and source manifests.

## Included

- Analysis code: `src/`, `scripts/`
- Configuration and tests: `configs/`, `tests/`, `pyproject.toml`, `requirements.txt`
- Manuscript source and submission support files: `manuscript/`
- Aggregate tables and figures: `reports/tables/`, `reports/figures/`
- Research protocol, variable plan, literature matrix, and source manifests: `docs/`

## Excluded

- OECD raw data files under `data/raw/`
- Extracted OECD `.SAV`, `.sas7bdat`, `.zip`, or equivalent raw public-use files
- Row-level prediction output: `reports/tables/holdout_predictions.csv`
- Fitted model artifacts such as `.joblib`
- Local virtual environments, caches, and build outputs
- Downloaded source PDFs/HTML pages that are not needed for the public code release

## Data Boundary

Readers should obtain PISA 2022 public-use files directly from the OECD PISA 2022 Database and place them in `data/raw/` before running the full pipeline. This repository should not redistribute OECD raw files.

## License Note

Code and repository documentation are released under the MIT License. Third-party data and source materials remain subject to their original terms.
