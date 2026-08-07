# REFORMS Compliance Checklist

**Paper:** Explainable Machine Learning for Predicting Mathematics Literacy and Low-Performing Students: Evidence from PISA 2022

**Reference:** Kapoor, S., et al. (2024). REFORMS: Consensus-based Recommendations for Machine-learning-based Science. *Science Advances*, 10(18), eadk3452.

**Assessment date:** 2026-05-14

**Status key:** ✅ Compliant &vert; ⚠️ Partial &vert; ❌ Not addressed &vert; N/A Not applicable

---

## 1. Study Goals and Claims

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 1.1 | State whether the study is predictive, explanatory, or causal | ✅ | Abstract states "predictive"; Section 2.3 clarifies non-causal framework; "predictive patterns rather than causal claims" repeated throughout |
| 1.2 | Match claims to study design and limitations | ✅ | Section 7 Limitations explicitly addresses causal inference limits; Discussion cross-references Limitations |
| 1.3 | Avoid speculative or overgeneralized claims | ✅ | Conclusion qualifies "first to combine" with "among studies using the full global PISA 2022 sample"; benchmark comparison includes caveats |
| 1.4 | Report negative results and null findings | ✅ | LIME divergence from SHAP reported openly; limited cross-country generalization documented; SMOTE and class-weight results reported |

---

## 2. Computational Reproducibility

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 2.1 | Share code and data where possible | ✅ | Public release package prepared; code at `public_release/pisa2022-xai-math/`; all analysis scripts numbered for pipeline reproducibility |
| 2.2 | Document software versions and environment | ⚠️ | `requirements.txt` exists but no explicit version pinning (e.g., `pip freeze` output); could add `environment.yml` or `Pipfile.lock` |
| 2.3 | Use fixed random seeds | ✅ | Fixed seed `20260510` used throughout all scripts; documented in Section 4.4 |
| 2.4 | Document computational resources used | ⚠️ | Hardware specs and runtime not documented. Recommend adding to Section 4.4 or Supplementary |
| 2.5 | Provide documentation for code usage | ✅ | README exists; scripts have header docstrings; pipeline is numbered sequentially |

---

## 3. Data Quality

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 3.1 | Describe data source and collection process | ✅ | Section 4.1 details OECD PISA 2022 Database; cites OECD documentation |
| 3.2 | Report sample size and characteristics | ✅ | N=613,744 from 80 countries; Table 1 summarizes demographics; weighted statistics reported |
| 3.3 | Document missing data patterns | ✅ | Section 4.3 variable audit; complete-case sensitivity (6.47% complete); per-variable missingness documented |
| 3.4 | Discuss potential selection biases | ✅ | Section 7 Limitations discusses complete-case bias; missingness mechanisms noted |

---

## 4. Data Preprocessing

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 4.1 | Document all preprocessing steps | ✅ | Section 4.3 describes imputation (median/mode), encoding (one-hot), variable exclusion criteria |
| 4.2 | Justify preprocessing choices | ✅ | Missingness >50% exclusion threshold justified; imputation choice documented |
| 4.3 | Report whether preprocessing is applied before or after train-test split | ✅ | Preprocessing pipeline uses frozen imputation from training data; reproducible variable audit before modeling |

---

## 5. Modeling Choices

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 5.1 | Justify model selection | ✅ | Tree-based ensembles chosen for nonlinear capture; Section 4.4 cites Breiman (2001), Fernández-Delgado (2014) |
| 5.2 | Compare against appropriate baselines | ✅ | Ridge, elastic net, logistic regression baselines; stacking ensemble; MLP deep learning baseline added |
| 5.3 | Report hyperparameter tuning procedure | ✅ | Optuna Bayesian optimization, 50 trials; search space documented (n_estimators, learning rate, depth, leaves, subsample, regularization); Table 10 |
| 5.4 | Report both default and tuned performance | ✅ | Default LightGBM AUC=0.890 vs. tuned XGBoost AUC=0.903; Table 3-4 report both |

---

## 6. Data Leakage

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 6.1 | Ensure no train-test contamination | ✅ | Single 80/20 stratified split before any modeling; all imputation fit on train only |
| 6.2 | Handle temporal/spatial dependencies | ✅ | Country-group holdout addresses spatial nesting; temporal limitations noted in Limitations |
| 6.3 | Avoid using outcome information in feature construction | ✅ | Features derived from questionnaire items, not outcome; PV mean used as outcome only |

---

## 7. Metrics and Evaluation

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 7.1 | Use multiple evaluation metrics | ✅ | 6 classification metrics (AUC, AP, F1, precision, recall, Brier) + calibration; 3 regression metrics (RMSE, MAE, R²) |
| 7.2 | Report uncertainty estimates | ✅ | Standard errors from BRR replicate weights for descriptive statistics; calibration CIs |
| 7.3 | Use appropriate metrics for imbalanced data | ✅ | AUC + F1 + precision + recall + threshold sensitivity explicitly chosen for imbalance |
| 7.4 | Avoid over-reliance on a single metric | ✅ | Multi-metric approach explicitly justified in Section 4.4 |

---

## 8. Generalization

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 8.1 | Evaluate on held-out test data | ✅ | 80/20 stratified holdout; no test data used for tuning |
| 8.2 | Test generalization across subgroups/datasets | ✅ | 7 robustness checks: OECD holdout, gender, immigrant, ESCS, country FE, country-group holdout, PV label stability |
| 8.3 | Discuss limits of generalizability | ✅ | Section 7: country-group holdout performance decline (AUC 0.903→0.847) documented as generalizability limit |
| 8.4 | Report performance across relevant subgroups | ✅ | Subgroup AUC reported (gender 0.901/0.904, immigrant 0.902→0.847, ESCS 0.871→0.877); explanation stability (ρ=0.72–0.99) |

---

## 9. Fairness, Bias, and Ethics

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 9.1 | Evaluate model fairness across demographic groups | ✅ | Formal fairness metrics (Equalized Odds, Demographic Parity, ABROCA) computed; Section 5.7 |
| 9.2 | Consider intersectional fairness | ✅ | Intersectional subgroup analysis (ESCS × gender, ESCS × immigrant); least-served subgroup identified |
| 9.3 | Discuss ethical implications of model use | ✅ | Section 6.5 addresses deficit-labeling concern; Section 7 discusses deployment safeguards |
| 9.4 | Report limitations of fairness evaluation | ✅ | "Fairness metrics remain descriptive... do not constitute causal evidence of algorithmic discrimination"; single-wave cross-sectional limitation noted |

---

## 10. Reporting and Transparency

| # | REFORMS Item | Status | Evidence / Notes |
|---|---|---|---|
| 10.1 | Report all methodological details needed for replication | ✅ | Full pipeline available; config files, feature lists, model specs, and seed reported |
| 10.2 | Disclose use of AI-assisted tools | ⚠️ | Cover letter mentions AI-assisted tools for programming/drafting. Could add explicit statement in manuscript (e.g., in Acknowledgments or Methods) |
| 10.3 | Provide structured abstract | ✅ | Abstract restructured with Background/Objectives/Methods/Results/Conclusions headers |
| 10.4 | Share model artifacts where possible | ⚠️ | Best model `.joblib` files exist but not in public release (excluded with raw data). Could provide model performance prediction interface instead |

---

## Summary

| Category | Compliant | Partial | Not Addressed |
|---|---|---|---|
| 1. Study Goals and Claims (4) | 4 | 0 | 0 |
| 2. Computational Reproducibility (5) | 3 | 2 | 0 |
| 3. Data Quality (4) | 4 | 0 | 0 |
| 4. Data Preprocessing (3) | 3 | 0 | 0 |
| 5. Modeling Choices (4) | 4 | 0 | 0 |
| 6. Data Leakage (3) | 3 | 0 | 0 |
| 7. Metrics and Evaluation (4) | 4 | 0 | 0 |
| 8. Generalization (4) | 4 | 0 | 0 |
| 9. Fairness, Bias, and Ethics (4) | 4 | 0 | 0 |
| 10. Reporting and Transparency (4) | 2 | 2 | 0 |
| **TOTAL** | **35/39** | **4/39** | **0/39** |

**Overall compliance:** 90% (35/39 fully compliant, 4 partial, 0 not addressed)

---

## Recommended Actions (Low Effort)

1. **[2.2]** Add `environment.yml` or `pip freeze > requirements-locked.txt` with exact version pins.
2. **[2.4]** Add a sentence to Section 4.4: "Models were trained on [hardware specs]; total wall-clock time for the full pipeline was approximately [X] hours."
3. **[10.2]** Add to Acknowledgments: "AI-assisted tools were used for programming support, analysis organization, and drafting. All intellectual decisions, theoretical interpretations, and statistical judgments remain the author's responsibility."
4. **[10.4]** Consider adding a note: "Trained model artifacts are available from the author upon reasonable request, subject to OECD data use agreement constraints."
