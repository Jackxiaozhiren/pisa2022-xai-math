# Public Release Manifest

Repository target: `Jackxiaozhiren/pisa2022-xai-math`

This repository is the cleaned public research-companion copy of the PISA 2022 XAI mathematics project. It is intended to contain reproducible code, manuscript/scientific source material, aggregate outputs, and source manifests—not private submission workflow artifacts.

## Included

- Analysis code: `src/`, `scripts/` (legacy workflow plus manuscript-active Route A scripts)
- Configuration and tests: `configs/`, `tests/`, `pyproject.toml`, `requirements.txt`
- Manuscript/scientific source: manuscript text, supplementary material, BibTeX, generated-table mappings, data/reproducibility/ethics statements
- Aggregate tables and manifests: `reports/tables/`
- Publication figures: `reports/figures/`
- Scientific protocol, variable plan, literature/source manifests, estimand/validity audits, Route A protocol/results, and active-result register: `docs/`

## Excluded

- OECD raw data files under `data/raw/`
- Extracted OECD `.SAV`, `.sas7bdat`, `.zip`, or equivalent raw public-use files
- Row-level prediction output such as `reports/tables/holdout_predictions.csv`
- Fitted model artifacts such as `.joblib`
- Local virtual environments, caches, and build outputs
- Downloaded source PDFs/HTML pages not required for the public research package
- Cover letters, upload staging material, prior-journal archives, acceptance forecasts, editorial/meta-review notes, internal reviewer simulations, and other submission-management artifacts

## Manuscript-active Route A material

The public scientific package includes the v5 survey-analysis module and tests, Route A analysis scripts, aggregate PV/replicate/EBM/institution-cold-start results, manifests, and scientific methodology/validity records. The institution-cold-start result is a same-cycle unseen-school stress test, not external institution or deployment validation.

## Data boundary

Readers should obtain PISA 2022 public-use files directly from the OECD PISA 2022 Database and place them in `data/raw/` before running the full pipeline. This repository does not redistribute OECD raw files.

## Reproducibility boundary

`requirements.txt` and `pyproject.toml` define compatible installation ranges. The final archival manuscript release must additionally include an exact environment export and checksum manifest generated from the verified manuscript run; package versions should not be guessed after the fact. See `docs/REPRODUCIBILITY.md`.

## License

Project-authored code and repository documentation are released under the MIT License. Third-party data and source materials remain subject to their original terms.
