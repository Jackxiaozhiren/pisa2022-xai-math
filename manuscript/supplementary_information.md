# Supplementary Information

## S1. Extended Methods

### S1.1 Full Feature Audit

The variable audit (Table 5 in the main manuscript) reports all 40 candidate features considered for the main model. This supplementary section provides additional detail on the feature audit process: each variable was checked against the PISA 2022 codebook (CY08MSP_CODEBOOK_27thJune24.xlsx), tested for availability in the processed student and school questionnaire files, and audited for missing-data rates on the full processed frame (n = 613,744). Variables with missingness exceeding 50% were excluded from the main model; variables with missingness between 50% and 80% were retained only for extended or robustness use.

The 33 main-model features span five conceptual groups:
- **Student background** (7): ST004D01T, GRADE, AGE, ESCS, IMMIG, HISEI, PAREDINT
- **Family resources and support** (3): HOMEPOS, FAMSUP, FAMCON
- **Mathematics attitudes and dispositions** (4): ANXMAT, MATHEFF, MATHEF21, MATHPERS
- **School climate, teaching, and resources** (10): BELONG, BULLIED, FEELSAFE, SCHRISK, DISCLIM, TEACHSUP, COGACRCO, COGACMCO, STUBEHA, TEACHBEHA, EDUSHORT, STAFFSHORT
- **Digital learning and ICT** (7): ICTRES, ICTHOME, ICTSCH, ICTEFFIC, ICTINFO, ICTSUBJ, STUDYHMW

Note that the digital-learning group had the highest missingness rates (ranging from 6.3% for STUDYHMW to 49.2% for ICTEFFIC), reflecting the fact that the ICT questionnaire module was not administered in all participating countries/economies. Gradient boosting models handle these missing values natively through their split-finding mechanism, which is a key reason they were selected as the primary model family.

### S1.2 Model Architecture and Hyperparameters

All models were implemented using scikit-learn 1.5+ with LightGBM and XGBoost as optional dependencies. The preprocessing pipeline applied:
1. Median imputation for numeric features
2. Most-frequent imputation for categorical features
3. Standard scaling for numeric features
4. One-hot encoding for categorical features (with rare-category grouping, min_frequency=25)

Detailed hyperparameters for each model:

| Model | Key Hyperparameters |
|-------|-------------------|
| Ridge | alpha=1.0 |
| Elastic Net | alpha=0.01, l1_ratio=0.2, max_iter=3000 |
| Logistic (L2) | C=1.0, max_iter=3000, class_weight='balanced' |
| Random Forest (reg) | n_estimators=120, min_samples_leaf=75, max_features='sqrt', max_samples=0.7 |
| Random Forest (cls) | n_estimators=120, min_samples_leaf=75, max_features='sqrt', max_samples=0.7, class_weight='balanced_subsample' |
| HistGradientBoosting (both) | max_iter=250, learning_rate=0.06, l2_regularization=0.05 |
| XGBoost (both) | n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85 |
| LightGBM (both) | n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=80 |

All models used random_state=20260510 for reproducibility. Hyperparameters were chosen based on established defaults for large-sample tabular educational data, informed by prior PISA-ML studies and the scikit-learn, XGBoost, and LightGBM documentation. No systematic hyperparameter tuning (grid search or Bayesian optimization) was performed, as the computational cost would be substantial with 613,744 observations and the primary goal was comparative model evaluation rather than maximizing a single metric.

### S1.3 Split and Evaluation Details

- **Random seed**: 20260510
- **Test proportion**: 20%
- **Stratification**: By low-performer status (model label)
- **Train size**: 490,995 students
- **Test size**: 122,749 students
- **Sample weights**: W_FSTUWT normalized to mean 1
- **Weight application**: Used in model fitting where supported (all models except those without sample_weight support) and in all holdout metric computations

The robustness sample (120,000 rows) used a separate stratified draw for the country fixed-effects and country-group holdout checks to ensure computational tractability.

### S1.4 Explanation Protocol

- **SHAP**: TreeSHAP algorithm on a deterministic 5,000-row sample; beeswarm summary plots with top 20 features
- **Permutation importance**: 5 repeats on a deterministic 10,000-row sample; scoring = 'roc_auc' (classification) or 'neg_root_mean_squared_error' (regression)
- **Digital-feature importance**: Subset extraction from full permutation importance table, isolating ICT-prefixed features plus STUDYHMW

---

## S2. Supplementary Tables

### S2.1 Alternative Low-Performer Label Stability

Each of the 10 plausible values (PV1MATH through PV10MATH) was used independently to define the low-performer label (threshold = 420.07). The low-performer rate range was 45.24%–45.54%, with a standard deviation of approximately 0.03 percentage points, confirming near-perfect label stability across plausible values.

### S2.2 Complete-Case Population Shift

Only 6.47% of students (39,708 of 613,744) had complete data on all 33 main-model features. The low-performer rate in the complete-case subset was 37.9%, compared with 45.5% in the full unweighted sample — a difference of 7.6 percentage points. This confirms non-random missingness: students with more complete questionnaire data tend to have higher mathematics achievement.

### S2.3 Country-Level Descriptive Statistics

The full sample_descriptives_by_country.csv file provides per-country/economy weighted mathematics means, weighted low-performer rates, unweighted low-performer rates, and sample sizes for all 80 participating countries/economies. Weighted low-performer rates range from approximately 7.6% (Singapore) to 93.6% (Dominican Republic), reflecting the substantial cross-national variation that motivates the country-context robustness checks.

---

## S3. Supplementary Figures

The following supplementary figures are available in the analysis repository:

| Figure | Description |
|--------|-------------|
| `reports/figures/classification_lightgbm_shap_summary.png` | Full SHAP beeswarm summary for the classification LightGBM model (2,370 × 2,822 px) |
| `reports/figures/regression_lightgbm_shap_summary.png` | Full SHAP beeswarm summary for the regression LightGBM model (2,370 × 2,822 px) |
| `reports/figures/digital_feature_importance.png` | Digital-feature permutation importance comparison (2,070 × 870 px) |

---

## S4. Computational Reproducibility

All analyses were conducted in Python 3.11+ with pinned dependency versions specified in `requirements.txt`. The analysis pipeline can be reproduced by:

1. Downloading the PISA 2022 student and school questionnaire SPSS files from the OECD PISA 2022 Database
2. Placing them in `data/raw/`
3. Running scripts 00–08 in sequence:
   ```
   python scripts/00_check_inputs.py
   python scripts/01_prepare_data.py
   python scripts/02_describe_sample.py
   python scripts/03_train_models.py
   python scripts/04_explain_models.py
   python scripts/05_build_tables.py
   python scripts/06_robustness_checks.py
   python scripts/07_export_latex.py
   python scripts/08_generate_latex_tables.py
   ```
4. Compiling the LaTeX manuscript with a Springer Nature-compatible LaTeX distribution

The random seed (20260510) is fixed across all scripts. All table values and figure renderings are deterministic given the same input data and seed.

## S5. Repository and Data Access

- **Analysis code**: https://github.com/Jackxiaozhiren/pisa2022-xai-math
- **Raw data**: Available from OECD PISA 2022 Database (not redistributed in the repository)
- **Derived outputs**: Tables, figures, and fitted model metadata included in repository
- **Contact**: 241734106@m.gduf.edu.cn
