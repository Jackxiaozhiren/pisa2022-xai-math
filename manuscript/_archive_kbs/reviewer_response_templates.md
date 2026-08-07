# Reviewer Response Templates

This file anticipates common reviewer questions and outlines evidence-based responses. Update with specific reviewer comments after receiving the decision letter.

---

## Q1: Why not use causal inference methods?

**Likelihood**: HIGH. Any prediction study using observational PISA data will face this question.

**Response strategy**: The manuscript already distinguishes prediction from causation in the abstract, introduction (Section 1), XAI section (2.3), discussion (5), and limitations (6). The key point: machine learning is positioned as an "educational-statistical interpretation tool" for identifying fitted-model patterns, not for estimating treatment effects. The robustness checks (country fixed effects, country-group holdout) demonstrate sensitivity to context, which further underscores that these are predictive associations, not causal laws. If reviewers request additional language, we can add a short paragraph explicitly comparing the predictive framework to the potential outcomes framework and clarifying why causal identification would require a different design (instrumental variables, regression discontinuity, or within-country panel data).

---

## Q2: LightGBM and HistGradientBoosting have nearly identical performance. Why feature LightGBM?

**Likelihood**: MODERATE.

**Response strategy**: The manuscript already notes that both models are very close (RMSE: 57.61 vs 57.69, AUC: 0.890 vs 0.889). The choice to feature LightGBM is pragmatic — its native missing-value handling via histogram-based splitting is well-documented and it is widely used in recent PISA-ML studies. If the reviewer prefers, we can present both models equally and use HGB as a sensitivity check rather than a secondary model. The conclusion that tree-based ensembles outperform linear baselines does not depend on which gradient boosting implementation is selected.

---

## Q3: How robust are results to the missing-data handling strategy?

**Likelihood**: MODERATE.

**Response strategy**: We already have the complete-case sensitivity analysis (Section 4.5) showing that only 6.47% of students have complete data and that the complete-case population is substantially different (37.9% vs 45.5% low-performer rate). We also report a variable-level missingness audit (Table 5). If the reviewer requests additional sensitivity checks, we can:
- Run the main models with multiple imputation (MICE) on a smaller robustness sample
- Compare LightGBM's native missing-value handling against median/mode imputation
- Report missingness patterns by country to assess whether missingness is systematic at the country level

---

## Q4: What about external validation on an independent dataset?

**Likelihood**: LOW-MODERATE.

**Response strategy**: The country-group holdout check (trained on 64 countries, tested on 16 held-out countries) serves as a form of external validation — the held-out countries represent different education systems with potentially different predictor-outcome relationships. The performance drop (AUC: 0.890 → 0.847, R²: 0.638 → 0.502) is documented and discussed. True external validation on an independent dataset (e.g., TIMSS, national assessments) would require a different study design. We can acknowledge this as a limitation and future direction.

---

## Q5: Are SHAP explanations reliable for correlated predictors?

**Likelihood**: MODERATE.

**Response strategy**: SHAP values can be affected by predictor correlation because TreeSHAP conditions on feature values to break dependencies. The manuscript reports both SHAP and permutation importance, which uses a different (marginal) approach to feature importance. The convergence between SHAP and permutation importance rankings provides confidence in the overall importance ordering. If the reviewer requests this, we can add a predictor correlation matrix and note which SHAP values may be influenced by correlations (e.g., HOMEPOS and ESCS are conceptually related).

---

## Q6: Why not include country as a feature in the main model?

**Likelihood**: LOW-MODERATE.

**Response strategy**: Including country as a predictor (80+ dummy variables) would substantially increase model complexity and risk overfitting to country-specific patterns. The main model is designed to capture student-level, family-level, school-level, and digital-learning associations that are broadly informative across countries. The country fixed-effects sensitivity check (Section 4.5) demonstrates that adding country does improve performance, which we interpret as evidence for country-context dependence rather than as an argument for including 80 country dummies in the main model. The country-group holdout check is a more demanding test of cross-country generalization.

---

## Q7: The low-performer rate (53.9%) suggests this is not an imbalanced classification problem. Is the problem well-posed?

**Likelihood**: LOW.

**Response strategy**: The global weighted low-performer rate is indeed near 50%, which makes the problem approximately balanced globally. However, the rate varies substantially across subgroups (e.g., ESCS Q1: 95.3% low-performer rate, ESCS Q5: 26.2%). Classification performance metrics are reported at multiple thresholds (default, Youden's J, Max-F1) to account for this. The primary contribution is predictive, not diagnostic — the goal is understanding which factors are model-important, not deploying a screening tool.

---

## Q8: How do these findings relate to specific educational technology interventions?

**Likelihood**: LOW-MODERATE.

**Response strategy**: The manuscript explicitly avoids making prescriptive claims about interventions. However, the finding that ICT resources and ICT self-efficacy are more model-important than simple ICT availability/use has implications for educational technology design: ensuring access is necessary but insufficient; building students' confidence and purposeful engagement with technology appears more predictively relevant. If the reviewer requests this, we can add a paragraph in the Discussion about design implications while maintaining the predictive-not-causal framing.

---

## Q9: Concern about single-author study

**Likelihood**: LOW.

**Response strategy**: The author contributions statement lists all roles as performed by the single author, which is transparent and follows ICMJE guidelines. The AI-assisted work statement acknowledges tool support for programming and drafting. The computational reproducibility of the full pipeline (fixed random seed, publicly available code) provides verification that does not depend on author count.

---

## Q10: Reference formatting or journal scope concerns

**Likelihood**: LOW.

**Response strategy**: References follow Springer name-year conventions with the sn-mathphys-ay BibTeX style. The reference list has been strengthened with 10+ recent publications from EAIT and related journals to demonstrate engagement with the journal's scope. If specific formatting issues are flagged, they can be addressed in revision using the journal's official style guide.
