# Public Release Audit

## Release Status

Status: local public-release package prepared.

The local workspace contains OECD raw data and large derived artifacts. The public release should be the cleaned export at `/Users/jackson/论文/pisa2022-xai-math/public_release/pisa2022-xai-math`, not this full working directory as-is.

## Files That Must Not Be Publicly Redistributed

- `data/raw/STU_QQQ_SPSS.zip`
- `data/raw/CY08MSP_STU_QQQ.SAV`
- `data/raw/SCH_QQQ_SPSS.zip`
- `data/raw/CY08MSP_SCH_QQQ.SAV`

These files are locally available for reproduction but must be obtained from OECD by readers.

## Files That Need Policy Review Before Public Release

- `data/processed/pisa2022_math_model_frame.parquet`
- `data/processed/models/*.joblib`
- `reports/tables/holdout_predictions.csv`

These are derived from the public-use data but may still contain row-level or fitted-state information. Prefer excluding them from a public repository unless the author and journal policy explicitly permit release.

## Files Suitable for Public Release

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `docs/research_protocol.md`
- `docs/variable_plan.md`
- `docs/literature_matrix.csv`
- `docs/sources/pisa/download_manifest.csv`
- `docs/sources/references/source_manifest.csv`
- `manuscript/`
- Aggregate result tables and figures under `reports/`, except row-level prediction files that require policy review.

## Secret Scan

Status: pass for the checked text surfaces.

A repository text scan excluding `.venv`, raw/processed data, downloaded
HTML/PDF assets, build artifacts, and the audit files themselves found no
obvious API keys, service-role keys, passwords, tokens, or private-key blocks.
No `.env` files were found in the release-relevant project tree.

## Optional Deployment Boundary

If a static results site is later created, use only aggregate tables and figures. Cloudflare Pages can host a static site. Supabase or Neon should be used only for aggregate, non-row-level result browsing if a database-backed interface becomes necessary.

## Prepared GitHub Package

Repository target: `Jackxiaozhiren/pisa2022-xai-math`.

Prepared local package:

- `/Users/jackson/论文/pisa2022-xai-math/public_release/pisa2022-xai-math`
- `/Users/jackson/论文/pisa2022-xai-math/public_release/pisa2022-xai-math-public-release.zip`

The package includes MIT license text, a public-release manifest, source code,
configuration, tests, manuscript source, `19` public aggregate CSV tables, `30`
figures (7 main manuscript, 6 supplementary, 17 SHAP dependence plots), and source manifests. A release-boundary scan found no `.SAV`, `.zip`,
`.joblib`, `holdout_predictions.csv`, `.venv`, old proof-only class directory,
or manuscript build directory in the prepared package.
