# Legacy Baseline Results

This document preserves historical baseline values for provenance. These values are **not manuscript-active** under the current survey-aware Route A analysis and should not be cited as the current headline results.

- Tuned XGBoost: AUC = 0.903 (95% bootstrap CI [0.898, 0.907]), RMSE = 54.10 (R² = 0.681), 23% RMSE reduction over the default-parameter ridge baseline.
- XAI convergence (recomputed): cross-model SHAP ρ = 0.99; SHAP vs permutation 0.81; SHAP vs ALE 0.63; LIME diverges (ρ = 0.01) under feature correlation.
- Fairness: SES was the largest historical concern (ABROCA = 0.027); low-SES immigrant-background students had AUC 0.779 vs 0.880 (gap 0.101, n = 2,495 in the historical holdout).
- Country-group holdout AUC = 0.847.

These results came from the earlier row-wise plausible-value-mean workflow and related legacy evaluation choices. The current manuscript-active results are defined by the Route A active-result register and the current top-level README.
