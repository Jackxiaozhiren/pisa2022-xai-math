# Public Release Manifest

Repository target: `Jackxiaozhiren/pisa2022-xai-math`

This package is the cleaned public-release copy of the PISA 2022 XAI mathematics project. It is intended for a GitHub repository containing reproducible code, manuscript source, aggregate outputs, and source manifests.

## Included

- Analysis code: `src/`, `scripts/` (legacy scripts plus v5 Route A scripts 33, 34, 36 and 37)
- Configuration and tests: `configs/`, `tests/`, `pyproject.toml`, `requirements.txt`
- Manuscript source and submission support files: `manuscript/` (main TeX, supplementary TeX, Markdown source, BibTeX, 10 LaTeX tables, 5 supplementary LaTeX tables, declarations, checklist, QA reports)
- Aggregate tables: `reports/tables/` (legacy aggregate tables plus non-restricted `v5_*.csv` and `v5_*.json` manifests)
- Figures: `reports/figures/` (30 PNG files: 7 main manuscript, 6 supplementary, 17 SHAP dependence plots)
- Research protocol, variable plan, literature matrix, and source manifests: `docs/`

## Excluded

- OECD raw data files under `data/raw/`
- Extracted OECD `.SAV`, `.sas7bdat`, `.zip`, or equivalent raw public-use files
- Row-level prediction output: `reports/tables/holdout_predictions.csv`
- Fitted model artifacts such as `.joblib`
- Local virtual environments, caches, and build outputs
- Downloaded source PDFs/HTML pages that are not needed for the public code release
- Editorial Manager title pages, cover letters, declarations, anonymous PDFs and upload staging folders

## EAAI v5 update (2026-08-25)

The repository now includes the v5 survey-analysis module and tests, Route A
analysis scripts, aggregate PV/replicate/EBM/institution-cold-start results,
manifests, and the v5 methodology/review/positioning records under
`docs/v5_eaai/`. The institution-cold-start result is a same-cycle
unseen-school stress test, not external institution or deployment validation.

## Data Boundary

Readers should obtain PISA 2022 public-use files directly from the OECD PISA 2022 Database and place them in `data/raw/` before running the full pipeline. This repository should not redistribute OECD raw files.

## License Note

Code and repository documentation are released under the MIT License. Third-party data and source materials remain subject to their original terms.
