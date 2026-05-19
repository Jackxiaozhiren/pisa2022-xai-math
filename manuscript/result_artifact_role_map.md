# Result Artifact Role Map

This file classifies generated result artifacts for submission packaging. It keeps the main manuscript focused while preserving enough audit material for reviewers and replication.

## Main Manuscript Tables

- `reports/tables/sample_summary.csv` - Table 1, processed global sample.
- `reports/tables/weighted_descriptive_se.csv` - Table 2, BRR and plausible-value descriptive estimates.
- `reports/tables/model_metrics.csv` - Tables 3 and 4, regression and classification performance.
- `reports/tables/variable_audit.csv` - Table 5, feature availability, missingness, and model-use decision.
- `reports/tables/classification_threshold_sensitivity.csv` - Table 6, default and optimized threshold behavior.
- `reports/tables/calibration_metrics.csv` and `reports/tables/calibration_bins.csv` - Table 7, calibration diagnostics.
- `reports/tables/subgroup_holdout_metrics.csv` - Table 8, gender, immigrant-background, and ESCS subgroup performance.
- `reports/tables/oecd_holdout_metrics.csv`, `reports/tables/country_fixed_effects_sensitivity.csv`, and `reports/tables/country_group_holdout_metrics.csv` - Table 9, country-context robustness.

## Main Manuscript Figures

- `reports/figures/conceptual_framework.png` - Figure 1, conceptual framework integrating ecological systems and digital divide theories.
- `reports/figures/methodology_flowchart.png` - Figure 2, methodology flowchart summarizing the analytical pipeline.
- `reports/figures/classification_lightgbm_shap_summary.png` - Figure 3, low-performer classification SHAP summary.
- `reports/figures/regression_lightgbm_shap_summary.png` - Figure 4, mathematics-score regression SHAP summary.
- `reports/figures/calibration_curves.png` - Figure 5, calibration curves for the best classification model.
- `reports/figures/country_heterogeneity.png` - Figure 6, country heterogeneity visualization.
- `reports/figures/digital_feature_importance.png` - Figure 7, digital-feature permutation importance.

## Supplementary Tables

- `reports/tables/alternative_low_performer_labels.csv` - plausible-value label sensitivity.
- `reports/tables/complete_case_sensitivity.csv` - complete-case population shift.
- `reports/tables/classification_lightgbm_permutation_importance.csv` - global classification importance ranking.
- `reports/tables/regression_lightgbm_permutation_importance.csv` - global regression importance ranking.
- `reports/tables/digital_feature_importance.csv` - numeric source for Figure 3.
- `reports/tables/sample_descriptives_by_country.csv` - country/economy descriptive checks.
- `reports/tables/subgroup_descriptives.csv` - subgroup descriptive checks.

## Reviewer/Replication Audit Files

- `reports/tables/explanation_artifacts.csv` - explanation artifact manifest.
- `reports/tables/holdout_predictions.csv` - holdout predictions for reproducibility checks; do not include in the main manuscript because of size.
- `data/processed/prepare_data_report.json` - processed-frame construction report.
- `data/processed/feature_sets.json` - main and extended feature sets.
- `data/processed/models/best_model_summary.json` and `data/processed/models/split_summary.json` - model-selection and split metadata.

## Submission Decision

The manuscript should cite only the main tables and figures. Supplementary and reviewer files should be retained for peer review, replication, and response-to-review work.
