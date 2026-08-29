# Public Release Manifest

Repository target: `Jackxiaozhiren/pisa2022-xai-math`

This repository is the cleaned research-companion package for the EAAI manuscript **A Reproducible Model-Level Verification and Validation Protocol for Predictive AI in High-Impact Educational Assessment: Evidence from PISA 2022**.

## Manuscript-active material

Included in the active reproducibility boundary:

- Route A scripts: `scripts/33_*`, `scripts/34_*`, `scripts/36_*`, `scripts/37_*`;
- survey/PV utilities: `src/pisa_xai/v5_survey.py` and `tests/test_v5_survey.py`;
- active aggregate outputs/manifests: `reports/tables/v5_*`;
- active scientific protocol and claim register: `docs/v5_eaai/`;
- configuration, tests, dependency specifications, citation metadata, and source manifests;
- public manuscript-boundary/data/reproducibility documentation under `manuscript/`.

The exact EAAI submission source package is versioned as a release artifact. Editorial Manager staging material is not mixed into the repository tree.

## Historical material

Earlier scripts, figures, and aggregate outputs are retained to preserve research chronology and to support legacy fitted-model diagnostics used explicitly in the paper. They are **not** the source of the active performance claims. `docs/LEGACY_BASELINE.md` records the legacy boundary.

## Excluded from the public release

- OECD raw public-use files (`data/raw/`) and extracted `.SAV`, `.sas7bdat`, `.zip`, or equivalent raw files;
- row-level predictions / student-level exports and large intermediate parquet files;
- fitted `.joblib` models and large model/checkpoint binaries;
- local virtual environments, caches, LaTeX build files, and notebook/checkpoint caches;
- local absolute paths and machine-specific session traces;
- cover letters, title pages, Editorial Manager staging/checklists, acceptance forecasts, reviewer simulations, EIC/meta-review notes, prompts, and submission-management material;
- downloaded third-party PDFs/HTML pages that are not required to reproduce the public code.

## Scientific boundary

The primary Route A results are PV-pooled model-evaluation quantities. The classification indicator is constructed separately for each plausible value for model evaluation; it is not an individual proficiency label. The unseen-school analysis is a same-cycle cold-start stress test, not external institutional or deployment validation.

## Environment boundary

`requirements.txt` and `pyproject.toml` define compatible installations. `docs/ENVIRONMENT_OBSERVED_2026-08-25.txt` records versions observed from the archived manuscript environment's installed package metadata. This is an environment evidence snapshot, not a claim that every transitive dependency was required by every analysis route.

## License

MIT applies to project-authored code and repository documentation. Third-party data and source materials remain subject to their original terms.
