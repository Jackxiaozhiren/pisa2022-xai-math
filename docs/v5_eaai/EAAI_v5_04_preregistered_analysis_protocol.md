# EAAI v5 Phase C：预注册 Route A 分析协议

**Registered before v5 analysis execution:** 2026-08-23  
**Data:** existing public PISA 2022 source file and existing processed model frame only.  
**No new data, no test-set tuning, no result-dependent endpoint selection.**

## A. Estimand and boundary

1. **Unit of analysis:** model-evaluation observations in a fixed PISA student holdout; no observation is interpreted as an individual student's true score/status.
2. **Population-weighted estimand:** PISA student-population weighting with `W_FSTUWT`.
3. **Equal-country estimand:** official `SENWT`, with each country/economy contributing an equal target total.
4. **Application:** a model-level pre-use verification/validation stress test of assessment analytics.
5. **Forbidden inference:** individual prediction, intervention, policy, institutional deployment, causal effect, or full training-process uncertainty.

## B. Data identity and fixed split

- Use the legacy model frame `data/processed/pisa2022_math_model_frame.parquet` (613,744 rows).
- Join only `CNT`, `CNTSTUID`, `CNTSCHID`, `W_FSTUWT`, and `SENWT` from the original PISA student SAV. The processed frame stores human-readable country names whereas the raw SAV stores three-letter codes, so the verified stable join key is `CNTSTUID` alone; it is unique in both files. `CNTSCHID` and `W_FSTUWT` must agree post-join, which provides independent country/school and sampling-weight linkage checks.
- Require unique keys, unchanged row count, non-negative/finite full/replicate/senate weights, ten non-missing math PVs, and all 80 replicate-weight columns.
- Reconstruct the frozen legacy 80/20 split with seed `20260510`, test size `0.20`, and the frozen legacy stratification target. The split is not re-optimized per PV.
- Use exactly the existing 33 features. Convert categorical fields deterministically to existing category codes; do not add/remove predictors or tune from test results.

## C. PV-specific models and metrics

For each `PVvMATH`, v=1…10:

1. Regression outcome: `PVvMATH`.
2. Classification outcome: `I(PVvMATH < 420.07)`; it is an imputed model-evaluation target, not a person-level label.
3. Train a fresh XGBoost regressor and classifier on the fixed training split using the frozen legacy tuned hyperparameters, seed `20260510`, 33 predictors, and normalized `W_FSTUWT` sample weights. No Optuna/tuning occurs in v5.
4. On the holdout, calculate population- and senate-weighted RMSE, R², AUC, Brier, 10-bin ECE, calibration slope, and core subgroup/intersectional metrics.
5. Define C1 core contrasts, per PV, as: `AUC(low-SES non-native) - AUC(high-SES native)`, `ECE(intersection) - ECE(global)`, and `slope(intersection) - slope(global)`. Exact subgroup definitions reproduce the legacy `ESCS` quartile and `IMMIG` grouping policy and are saved in the manifest.

## D. Replicate-weight variance and PV pooling

For every PV-specific, population-weighted core metric:

1. Calculate the full-sample estimate under `W_FSTUWT`.
2. Recalculate with each `W_FSTURWT1`…`W_FSTURWT80`.
3. Calculate sampling variance as `0.05 × Σ_r (T_r - T_full)^2`.
4. Pool ten PV-specific estimates: mean estimate; mean sampling variance; between-PV variance; total variance = mean sampling variance + `(1 + 1/10)` × between-PV variance.
5. Use normal-approximation 95% intervals only where the metric is finite in all required replicates. Mark cases with invalid replicates explicitly rather than filling values.

This is **conditional-on-fitted-model evaluation uncertainty**. It does not re-fit every model under each replicate weight and is not full training uncertainty.

## E. Senate-weight sensitivity

Recalculate holdout metrics and core C1 contrasts under `SENWT` for every PV-specific model. Pool point estimates across PVs. Do not attach the population BRR variance to senate results unless corresponding senate replicate weights are verified; describe this as an estimand sensitivity, not a full senate design-variance analysis.

## F. Controlled EBM baseline

- Use the same fixed split, 33 encoded predictors, per-PV outcome, normalized `W_FSTUWT`, no test tuning, and the full training data.
- Use the already present global `python3` InterpretML 0.7.8 environment; do not install any dependency.
- Use an additive EBM (`interactions=0`) with fixed pre-registered capacity/early-stopping settings. Record model version, configuration, wall time, and peak process RSS.

### Resource amendment recorded 2026-08-24

The initial `max_rounds=5000`, `early_stopping_rounds=100` configuration was started on the full data but stopped after approximately 11 minutes during the first PV model without completing. The controlled comparison is amended to `max_rounds=1000`, `early_stopping_rounds=50`, `outer_bags=1`, `interactions=0`, `max_bins=256`, and `min_samples_leaf=10`. This is a pre-registered computational cap, not a sample-size reduction or test-set tuning: all 490,995 training rows, 33 predictors, fixed split, PV route, and normalized `W_FSTUWT` remain unchanged. The interrupted run is invalid and produces no manuscript evidence.
- Report AUC, Brier, RMSE, R², ECE, slope for every PV and the pooled comparison; do not call it controlled if any matching constraint fails.

## G. Synthetic/unit checks and output contract

Tests must cover: key join, PV/replicate discovery, weight validity, weighted metric sanity, extreme calibration probabilities, single-class subgroup handling, Fay-BRR variance, PV pooling, seed determinism, output schemas, and immutable legacy paths.

Required candidate outputs:

- `reports/tables/v5_pv_specific_metrics.csv`
- `reports/tables/v5_replicate_weight_uncertainty.csv`
- `reports/tables/v5_pv_pooled_metrics.csv`
- `reports/tables/v5_population_vs_senate_weights.csv`
- `reports/tables/v5_intersectional_design_aware_ci.csv`
- `reports/tables/v5_analysis_failures.csv`
- `reports/tables/v5_analysis_manifest.json`
- `reports/tables/v5_controlled_ebm_baseline.csv`

## H. Pre-specified stops

The six conditions in `EAAI_v5_03_methodology_branch_decision.md` are binding. Smoke/debug output validates code only; it cannot become a table, figure, manuscript number, or promotion candidate.
