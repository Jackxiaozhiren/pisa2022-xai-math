# Reference Audit

## Citation-Key Consistency

Status: pass.

All citation keys used in `manuscript/manuscript.md` are present in `manuscript/references.bib`.

Keys checked:

- `Angelopoulos_Bates_2021`
- `Breiman_2001`
- `Chen_Guestrin_2016`
- `Fernandez_Delgado_2014`
- `Hastie_Tibshirani_Friedman_2009`
- `He_Hu_Garcia_2008`
- `Ifenthaler_Yau_2020`
- `James_Witten_Hastie_Tibshirani_2021`
- `Joksimovic_2020`
- `Ke_2017`
- `Lundberg_Lee_2017`
- `Molnar_2022`
- `NCES_2022_cut_scores`
- `OECD_2022_database`
- `OECD_2022_results_vol1`
- `OECD_2022_results_vol2`
- `OECD_2022_results_vol5`
- `OECD_2022_technical`
- `PISA_2018_XAI_math`
- `PISA_2022_XAI_low_performers`
- `PISA_2022_XAI_resilience`
- `Powers_2011`
- `Romero_Ventura_2020`
- `Rudin_2019`
- `Steyerberg_2010`
- `Susnjak_2022`

## Zotero Reconciliation

Status: blocked by local Zotero Desktop API authorization, with limitation recorded.

The Zotero helper checked `http://127.0.0.1:23119` after Zotero Desktop was opened. Zotero Connector responded successfully, but the local API returned `403 Forbidden`. The enable helper backed up and edited the Zotero preferences file, but the API still returned `403` in this session. Therefore, no Zotero export was used to overwrite `manuscript/references.bib`. The existing BibTeX remains the source of truth for this submission package until Zotero Desktop allows local API access or the library is exported manually from Zotero Desktop.

## Source Manifest Coverage

Source coverage is tracked in `docs/sources/references/source_manifest.csv`.

- OECD PISA database, technical report, and results volumes are downloaded or locally documented from official OECD sources.
- Core model-method sources include downloaded PDFs for Lundberg and Lee 2017, Chen and Guestrin 2016, Ke 2017, Powers 2011, James et al. 2021, Fernandez-Delgado et al. 2014, and related open-access sources.
- Several publisher-controlled references remain `metadata_only` or `needs_manual_access`; these should not be described as locally downloaded full text.
- The target-journal pages are strategy references, not manuscript citations.

## Hugging Face / arXiv Audit

Status: pass for locally archived arXiv metadata; Hugging Face paper-page indexing is not available for these three PISA candidates.

The XAI/PISA precedent papers in the manuscript are tracked through arXiv identifiers in the local manifest:

- `PISA_2018_XAI_math` - arXiv `2508.16747`, local XML metadata and PDF recorded.
- `PISA_2022_XAI_low_performers` - arXiv `2509.24508`, local XML metadata and PDF recorded.
- `PISA_2022_XAI_resilience` - arXiv `2509.24830`, local XML metadata and PDF recorded.
- `Susnjak_2022` - Hugging Face paper route recorded in the literature matrix; keep in the manuscript only as explainable learning analytics context.

Live Hugging Face API calls for `2508.16747`, `2509.24508`, and `2509.24830` returned "Paper not found", so Hugging Face should not be cited as the source for those PISA precedent papers. The arXiv pages and local arXiv XML/PDF archives are the appropriate evidence source.

## Submission Guidance

Do not add broad generic machine-learning citations unless they support a claim already present in the manuscript. If adding references before submission, prioritize recent educational technology, PISA 2022, learning analytics, or XAI-in-education evidence that directly sharpens the current contribution.

## Local BibTeX QA

Status: pass after local cleanup.

The local Tectonic/BibTeX proof run initially warned that `Molnar_2022` had no
publisher field. The reference was normalized as an independently published
online book, consistent with the archived source page and public catalog
metadata. After switching to the official Springer `sn-mathphys-ay` style, OECD
title bracing and URL underscores were also normalized. The final proof
bibliography no longer has BibTeX metadata warnings.
