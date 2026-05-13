# Format QA Ledger

## Figure Checks

Status: pass for source image dimensions.

- `reports/figures/classification_lightgbm_shap_summary.png`: `2370x2822`.
- `reports/figures/regression_lightgbm_shap_summary.png`: `2370x2822`.
- `reports/figures/digital_feature_importance.png`: `2070x870`.

All figure files exceed 2,000 px width and are referenced with captions in both `manuscript/manuscript.md` and `manuscript/springer_submission.tex`.

## PDF Compilation

Status: pass with the official Springer Nature template class supplied by the
author.

Completed command:

```bash
cd /Users/jackson/论文/pisa2022-xai-math/workspace/manuscript
/Users/jackson/.codex/plugins/cache/openai-bundled/latex-tectonic/0.1.1/bin/tectonic --outdir build springer_submission.tex
```

Output:

- `manuscript/build/springer_submission.pdf`
- `manuscript/build/springer_submission.log`
- `manuscript/build/springer_submission.bbl`
- `manuscript/build/springer_submission.blg`
- `14` rendered pages

The active `manuscript/sn-jnl.cls` is copied from
`manuscript/sn-article-template/sn-jnl.cls`. The obsolete local proof-only shim
has been removed.

The final kept-log proof run has no fatal errors, no undefined citation
warnings, and no BibTeX metadata warnings. Remaining TeX warnings are non-fatal
underfull page/line warnings and one small bibliography overfull line caused by
a long reference/URL.

## PDF Visual Rendering

Status: pass for local proof inspection with the available renderer.

`pdftoppm` was not available in the shell, so the proof PDF was rendered with
macOS PDFKit into:

- `manuscript/build/rendered_pages/page-01.png` through `page-14.png`
- `manuscript/build/rendered_pages/contact_sheet.png`

Visual inspection notes:

- The official-template title page renders the single author, affiliation, and
  corresponding-author email.
- Citations render as author-year entries on the inspected pages; no visible
  citation placeholder leakage was found.
- The three analysis figures render with captions and readable labels.
- The SHAP summary figures occupy full proof pages. This is acceptable for
  readability in the official-template proof.
- Tectonic reported non-fatal overfull/underfull box warnings, concentrated in
  page balancing and bibliography line breaking. No fatal layout error or
  undefined-reference warning was observed in the final proof log.

## DOCX Visual Rendering

Status: not required for the current LaTeX-first submission package.

If a Word manuscript is required, create DOCX from the Markdown source and render it with LibreOffice/`soffice` using the Documents skill. `soffice`/LibreOffice was not available in the shell during planning, so DOCX visual QA cannot be claimed until that renderer is installed.
