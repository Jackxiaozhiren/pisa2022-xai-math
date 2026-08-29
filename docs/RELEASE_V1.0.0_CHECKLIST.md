# v1.0.0 release checklist

- [x] Active v5 scripts/results/docs match the 2026-08-25 audit package by Git blob hash for the checked manuscript-active files.
- [x] 12/12 v5 survey tests pass in the independent audit environment.
- [x] Current manuscript values use Route A (AUC 0.8865; Brier 0.1375; RMSE 59.82; R² 0.6346).
- [x] Legacy AUC 0.903 / RMSE 54.10 values are classified as historical only.
- [x] Public local-path leakage removed from the reproducibility statement.
- [x] Exact submission-source package audited; the source-only `reports/v5_` path typo was corrected to `reports/tables/v5_` without changing scientific results.
- [x] Stale TLT manuscript sources removed from the active repository tree; history remains in Git.
- [x] Public release excludes raw OECD files, row-level predictions, fitted models, venv/caches, and submission-management material.
- [ ] CI passes on the release-candidate branch.
- [ ] Freeze release date and set `CITATION.cff` to `version: 1.0.0`.
- [ ] Generate and commit deterministic release SHA-256 manifest.
- [ ] Re-run CI on that exact commit.
- [ ] Merge and publish GitHub release/tag `v1.0.0`.
- [ ] Attach the corrected 2026-08-25 LaTeX/source package to the release.
