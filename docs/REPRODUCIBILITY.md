# Reproducibility and Archival Release

## Scientific reproduction boundary

This repository contains the analysis code, configuration, aggregate outputs, tests, manuscript source, and scientific protocol records required to understand and reproduce the workflow, subject to the OECD PISA data-distribution boundary.

Raw PISA public-use files are not redistributed. Users must obtain them from the OECD and place the required inputs under `data/raw/` as documented by the project.

## Environment policy

`requirements.txt` and `pyproject.toml` currently specify compatible dependency ranges. They are not an exact historical environment lock.

For the final manuscript archival release, export the exact environment from the workstation/container used for the verified final run. Do not guess or reconstruct package versions later. Record at minimum:

- Python version;
- operating system and architecture;
- exact package versions;
- Git commit SHA;
- project seed and manuscript-active configuration;
- relevant OECD file identifiers/metadata and input checksums where permitted;
- checksums for manuscript-active aggregate outputs and figures.

## Manuscript verification gate

Before tagging the manuscript reproducibility release:

1. run the unit/regression test suite in a clean environment;
2. execute the manuscript-active Route A pipeline from the documented inputs;
3. compare regenerated active-result tables with `docs/v5_eaai/EAAI_v5_07_active_result_register.md` and the submitted manuscript;
4. verify all manuscript-active figures and tables map to released artifacts;
5. ensure legacy results are not promoted into the active result path;
6. generate a release checksum manifest;
7. freeze a semantic version tag and archive that exact tag in a DOI-issuing repository.

## CI scope

GitHub Actions provides an automated code/test gate. It does not download or redistribute the OECD raw dataset and is not intended to execute the full multi-hour manuscript pipeline on every commit.
