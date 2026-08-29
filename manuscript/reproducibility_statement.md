# Reproducibility Statement

## Current manuscript-active analysis

The EAAI Route A analysis evaluates ten mathematics plausible values separately, uses normalized `W_FSTUWT` student weights, propagates sampling uncertainty with 80 Fay--BRR replicate weights, and reports a population-versus-senate sensitivity for the cross-country estimand.

Primary active values are recorded in `docs/v5_eaai/EAAI_v5_07_active_result_register.md`:

- XGBoost: AUC 0.88652, Brier 0.13754, RMSE 59.82284, R² 0.63458.
- Matched additive EBM: AUC 0.86891, Brier 0.1465, RMSE 66.15436, R² 0.55314.
- Unseen-school cold-start: AUC 0.88652, Brier 0.13584, RMSE 61.02284, R² 0.62187.

Legacy values such as AUC 0.903 and RMSE 54.10 are historical baseline results and are not manuscript-active.

## Environment

- Random seed: `20260510` (`configs/project.json`).
- Compatibility dependencies: `requirements.txt` / `pyproject.toml`.
- Observed manuscript-workstation package versions: `docs/ENVIRONMENT_OBSERVED_2026-08-25.txt`.
- The v5 manifest records core versions including Python 3.9.6, pandas 2.3.3, NumPy 2.0.2, scikit-learn 1.6.1, XGBoost 2.1.4, and pyreadstat 1.2.9.

No local absolute filesystem path is part of the public reproducibility contract. Run commands from the repository root.

## Public verification

```bash
python -m compileall -q src scripts tests
pytest -q
```

The independently audited 2026-08-25 research bundle passed 12/12 `test_v5_survey.py` tests.

## Data boundary

Raw PISA 2022 files are not redistribution artifacts. Obtain them from the OECD PISA 2022 Database and place the required inputs under `data/raw/`. The public repository contains aggregate outputs and scientific manifests rather than student-level predictions.

## Inference boundary

The Fay--BRR intervals in the active analysis describe fixed-model evaluation uncertainty plus between-PV imputation variance. They are not full model-training uncertainty and are not individual-level inference. The unseen-school analysis is a same-cycle cold-start stress test, not external institutional validation or deployment evidence.
