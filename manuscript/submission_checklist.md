# EAIT Submission Checklist

## Manuscript Readiness

- [x] Original empirical study positioned for *Education and Information Technologies*.
- [x] Abstract kept within the journal target range of 150-250 words.
- [x] Keywords limited to six terms.
- [x] Main text uses name-year citation markers compatible with Pandoc/Springer workflows.
- [x] Data availability statement avoids redistributing OECD raw data.
- [x] AI-assisted work statement included.
- [x] Ethics/public secondary-data statement drafted.
- [x] Competing interests statement drafted.
- [x] Author name, affiliation, ORCID ID, funding status, corresponding author, and single-author order confirmed by the author.
- [ ] Institution-specific ethics/public-data exemption wording confirmed by the submitting institution or supervisor.
- [x] Citation keys in `manuscript/manuscript.md` match `manuscript/references.bib`; Zotero local API status and reconciliation limitation are recorded in `manuscript/reference_audit.md`.
- [x] Local source manifest and Hugging Face/arXiv evidence audit completed for core XAI/PISA references in `manuscript/reference_audit.md`.

## Analysis Readiness

- [x] Unit tests pass in `.venv`.
- [x] Main result tables regenerate from scripts.
- [x] Weighted descriptive standard errors generated in `reports/tables/weighted_descriptive_se.csv`.
- [x] Calibration diagnostics generated in `reports/tables/calibration_metrics.csv`.
- [x] Country-group holdout robustness generated in `reports/tables/country_group_holdout_metrics.csv`.
- [x] Subgroup metrics include AUC, Brier, precision, recall, and F1.
- [x] Full pipeline rerun checked immediately before submission.
- [x] Key manuscript numbers spot-checked against `reports/tables`.

## Files for Submission

- [x] Main manuscript: `manuscript/manuscript.md`.
- [x] References: `manuscript/references.bib`.
- [x] Title page draft: `manuscript/title_page.md`.
- [x] Cover letter draft: `manuscript/cover_letter.md`.
- [x] Highlights: `manuscript/highlights.md`.
- [x] Data availability: `manuscript/data_availability.md`.
- [x] Author contributions: `manuscript/author_contributions.md`.
- [x] Ethics statement: `manuscript/ethics_statement.md`.
- [x] Competing interests: `manuscript/competing_interests.md`.
- [x] Springer LaTeX source exported from the Markdown manuscript: `manuscript/springer_submission.tex`.
- [x] Figure files exist, exceed 2,000 px width for current PNG outputs, and are captioned in the manuscript source.
- [x] Official Springer-template PDF rendered and visually checked for author metadata, caption placement, and figure readability. Status recorded in `manuscript/format_qa.md`.
- [x] Pre-submission QA report created in `manuscript/pre_submission_qa_report.md`.
- [x] Clean public-release package prepared for `Jackxiaozhiren/pisa2022-xai-math`.
