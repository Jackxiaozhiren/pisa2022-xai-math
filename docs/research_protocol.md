# Research Protocol

## Working Title

Explainable Machine Learning for Predicting Mathematics Literacy and Low-Performing Students: Evidence from PISA 2022

## Target Contribution

This study evaluates whether explainable machine learning can improve the prediction and interpretation of PISA 2022 mathematics outcomes while respecting the statistical structure of large-scale educational assessments.

The intended contribution is threefold:

1. Provide a reproducible PISA 2022 workflow that combines complex survey conventions with machine-learning prediction.
2. Compare traditional statistical models and machine-learning models for mathematics literacy and low-performing student identification.
3. Interpret key student, family, school, and digital-learning predictors using SHAP/permutation importance and compare them with regression-style findings.

## Research Questions

1. Which student, family, school, and digital-learning factors are most predictive of PISA 2022 mathematics literacy?
2. Do machine-learning models materially outperform traditional statistical baselines?
3. Are machine-learning explanations consistent with educational theory and traditional regression results?
4. Are predictive performance and explanations stable across gender, socioeconomic status, and immigrant-background groups?

## Outcomes

- Continuous outcome: `MATH_PV_MEAN`, computed from `PV1MATH` through `PV10MATH` for model training.
- Descriptive/statistical estimates: pooled across plausible values when reporting inferential quantities.
- Classification outcome: `LOW_PERFORMER_MATH`, defined as mathematics score below `420.07`, the lower bound of PISA mathematics proficiency Level 2.

## Statistical Design

- Use `W_FSTUWT` as the final student weight.
- Retain `W_FSTURWT1` through `W_FSTURWT80` for replicate-weight standard errors.
- Use BRR standard errors for weighted descriptive claims.
- Use plausible-value pooling for estimates that are treated as population parameters.
- Keep model performance and model explanation separate from causal interpretation.

## Modeling Design

Baseline models:

- Ridge regression for mathematics score.
- Elastic Net regression for mathematics score.
- Logistic regression for low-performing student classification.

Machine-learning models:

- Random Forest.
- Histogram Gradient Boosting.
- XGBoost when installed.
- LightGBM when installed.
- CatBoost when installed.

Primary metrics:

- Regression: RMSE, MAE, R².
- Classification: AUC, average precision, F1, precision, recall, Brier score.

Interpretability:

- SHAP summary plot for the best supported tree-based model.
- Permutation importance as a model-agnostic fallback and robustness check.
- Group-specific comparisons for gender, ESCS quantiles, and immigrant background where variables are available.

## Robustness Checks

Minimum required checks before submission:

- Repeat classification with an alternative low-performer threshold.
- Compare complete-case results against imputed-feature results.
- Run models on at least one country/region subset or OECD-only subset.
- Compare SHAP/permutation top predictors across at least two model families.

## Reporting Rules

- Use "predictive of" and "associated with"; avoid causal wording.
- Treat SHAP values as model explanations, not causal effects.
- Report unavailable configured variables transparently.
- Report whether school questionnaire data were available and merged.
