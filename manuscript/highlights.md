# Highlights

- Expert system integrating hierarchical feature organization, ICT taxonomy, multi-method XAI, and fairness evaluation within a reproducible single-workstation pipeline.
- Uses PISA 2022 public-use data from 613,744 students in 80 countries/economies with full survey-weight integration and Bayesian hyperparameter optimization.
- Tuned XGBoost inference engine achieved strongest overall weighted holdout performance: regression RMSE = 54.10, R² = 0.681; classification AUC = 0.903, Brier = 0.126 — a 23% improvement over linear baselines.
- ICT resources and ICT self-efficacy were the most model-important digital-learning predictors, second only to home possessions and mathematics self-efficacy — consistent with the digital divide framework's emphasis on skills over mere access.
- SHAP interaction analysis revealed that ICT self-efficacy's predictive effect strengthens at higher levels of ICT resources, supporting the compounding-inequality mechanism predicted by digital divide theory.
- Seven robustness checks cover calibration (ECE = 0.011), subgroups (gender, ESCS, immigrant background), complete-case sensitivity, plausible-value label stability, country fixed effects, and country-group holdout — demonstrating meaningful cross-country heterogeneity.
- SHAP, permutation importance, and SHAP interaction values are explicitly interpreted as fitted-model explanations, not causal effects.
- Reproducible pipeline with fixed random seed, publicly available analysis code, and a theoretically-grounded variable organization framework adaptable to future PISA cycles.
