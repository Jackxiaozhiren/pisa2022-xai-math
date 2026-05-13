# Explainable Machine Learning for Predicting Mathematics Literacy and Low-Performing Students: Evidence from PISA 2022

## Abstract

Mathematics literacy remains a central concern after the educational disruptions surrounding COVID-19, and large-scale assessments now include rich information about students' digital learning contexts. This study uses public-use PISA 2022 data from 613,744 students in 80 countries and economies to compare traditional statistical baselines with explainable machine-learning models for predicting mathematics literacy and low-performing student status. The workflow combines final student weights, mathematics plausible values, replicate-weight descriptive estimates, reproducible variable auditing, and holdout evaluation across student, family, school, and digital-learning predictors. LightGBM achieved the strongest main holdout performance, with weighted RMSE = 57.61 for mathematics scores and AUC = 0.890 for low-performer classification. SHAP and permutation importance indicated that home possessions, mathematics self-efficacy, family connection, grade, ICT resources, ICT self-efficacy, cognitive activation, and school behavior/resource indicators were consistently model-important. Robustness checks using OECD holdout evaluation, subgroup evaluation, complete-case sensitivity, plausible-value labels, calibration diagnostics, country fixed effects, and country-group holdout supported the broad predictive pattern while showing meaningful country-context dependence. The findings suggest that explainable machine learning can add predictive and diagnostic value to PISA-based educational technology research when explanations are interpreted as fitted-model patterns rather than causal effects.

Keywords: PISA 2022; explainable artificial intelligence; mathematics literacy; educational technology; learning analytics; digital learning

## 1. Introduction

Mathematics literacy is a core indicator of students' readiness to reason quantitatively, solve real-world problems, and participate in knowledge-based societies. PISA 2022 is a timely setting for examining this outcome because mathematics was the major assessment domain and the assessment followed a period of severe disruption to schooling and digital learning arrangements [@OECD_2022_results_vol1; @OECD_2022_results_vol2]. The public-use files also include student, family, school, and ICT-related questionnaire variables, which makes it possible to examine mathematics performance together with students' learning resources and technology contexts [@OECD_2022_database].

Educational data mining and learning analytics research increasingly uses machine learning to predict achievement and identify students who may need support [@Romero_Ventura_2020; @Ifenthaler_Yau_2020]. This work is useful because tree-based and ensemble models can capture nonlinear predictive structure that is difficult to represent manually [@Breiman_2001; @Fernandez_Delgado_2014]. However, prediction studies using large-scale assessment data face two additional obligations. First, PISA estimates should acknowledge plausible values, final student weights, and replicate weights [@OECD_2022_technical]. Second, model explanations should not be interpreted as evidence of what would happen if a family, school, or technology condition were changed.

This study addresses those obligations by positioning machine learning as an educational-statistical interpretation tool, not as a causal design. The substantive contribution is an education-technology analysis of digital-learning variables - especially ICT resources, ICT self-efficacy, ICT availability/use, subject-related ICT use, and study-related indicators - alongside broader student, family, and school context. The methodological contribution is a reproducible PISA 2022 workflow that reports model performance, explanations, missingness, calibration, subgroup performance, and country-context sensitivity.

The study asks four research questions:

1. Which student, family, school, and digital-learning factors are most predictive of PISA 2022 mathematics literacy?
2. Do machine-learning models materially outperform traditional statistical baselines?
3. Are machine-learning explanations consistent with educational theory and traditional regression-style expectations?
4. Are predictive performance and explanations stable across gender, socioeconomic status, immigrant-background groups, and country-scope sensitivity checks?

## 2. Literature Review

### 2.1 Mathematics Literacy and Low Performance in PISA

PISA defines mathematics literacy as students' capacity to formulate, employ, and interpret mathematics in a range of contexts. The PISA 2022 reports document substantial cross-national variation in mathematics performance and connect performance differences to learning conditions, student dispositions, and education-system disruption [@OECD_2022_results_vol1; @OECD_2022_results_vol5]. In this study, low-performing students are defined as students below the lower bound of PISA mathematics proficiency Level 2. The cut score of 420.07 is treated as a policy-relevant threshold for prediction and comparison, not as a clinical diagnosis of individual ability [@NCES_2022_cut_scores].

### 2.2 Educational Data Mining, Learning Analytics, and Prediction

Achievement prediction studies commonly compare linear models, regularized regression, random forests, gradient boosting, and other machine-learning approaches [@Hastie_Tibshirani_Friedman_2009; @James_Witten_Hastie_Tibshirani_2021]. Traditional baselines remain important in educational research because they are transparent and familiar. Gradient boosting models, including XGBoost and LightGBM, can improve predictive accuracy by modeling nonlinearities and interactions, but they require careful validation and interpretation [@Chen_Guestrin_2016; @Ke_2017].

For large-scale assessment data, a model leaderboard is not enough. Predictive gains must be judged alongside sampling design, missingness, cross-national comparability, calibration, and subgroup performance. Low-performing student prediction is also an imbalanced-classification problem, so discrimination metrics such as AUC should be read together with precision, recall, F1, Brier score, and threshold sensitivity [@He_Hu_Garcia_2008; @Powers_2011; @Steyerberg_2010].

### 2.3 Explainable AI in Education

SHAP and permutation importance are widely used to summarize how predictors contribute to fitted machine-learning models [@Lundberg_Lee_2017; @Molnar_2022]. SHAP provides local and global additive explanations, while permutation importance evaluates how much a model's performance declines when a predictor is shuffled. These tools can identify model-important variables, but they remain descriptive of a fitted predictive system.

This distinction matters in education. A variable can be highly predictive because it reflects prior opportunity, measurement design, country context, or structural inequality. Responsible learning analytics work warns against using predictive outputs as one-size-fits-all prescriptions [@Joksimovic_2020]. Interpretable and explainable models can support diagnosis and human review, but high-stakes decisions still require caution, domain expertise, and explicit separation between prediction and causation [@Rudin_2019; @Susnjak_2022].

### 2.4 Digital Learning and Educational Technology Context

PISA 2022 includes indicators related to ICT resources, ICT self-efficacy, ICT availability or use at home and school, digital information behavior, subject-related ICT use, and homework/study indicators [@OECD_2022_results_vol2]. These variables are central to an educational technology contribution because they distinguish between access, confidence, learning use, and possible distraction rather than treating "technology" as a single exposure.

The ICT variables also pose methodological challenges. Digital-learning constructs often have substantial missingness and can vary by questionnaire form or country. Accordingly, this study uses a reproducible variable audit, keeps high-missingness variables out of the main model unless justified, and interprets retained digital predictors as predictive signals rather than policy levers.

### 2.5 Research Gap

Recent PISA and XAI studies show growing interest in interpretable models for achievement, low-performing students, and academic resilience [@PISA_2018_XAI_math; @PISA_2022_XAI_low_performers; @PISA_2022_XAI_resilience]. Fewer studies combine global PISA 2022 mathematics outcomes, complex assessment-data conventions, multiple model families, digital-learning predictors, calibration checks, subgroup evaluation, and explicit safeguards against causal overinterpretation. This study fills that gap with a reproducible workflow and an EAIT-oriented contribution to educational technology research.

## 3. Methods

### 3.1 Data and Sample

The study uses public-use PISA 2022 student and school questionnaire data obtained from the OECD PISA 2022 Database [@OECD_2022_database]. The processed analysis frame contains 613,744 students from 80 countries and economies. The school questionnaire was merged by country/economy and school identifier. Four school-questionnaire variables entered the processed model frame: student behavior hindering learning (`STUBEHA`), teacher behavior hindering learning (`TEACHBEHA`), educational material shortage (`EDUSHORT`), and staff shortage (`STAFFSHORT`).

The weighted global mathematics mean was 424.29. Using BRR replicate weights and plausible-value pooling, the standard error for the mathematics mean was 0.72, with a 95% confidence interval from 422.88 to 425.70. The weighted low-performer rate based on plausible-value pooling was 53.30% (SE = 0.35 percentage points), while the modeling label based on the row-wise plausible-value mean produced a weighted rate of 53.90% (SE = 0.35 percentage points). These estimates are reported to distinguish population descriptive quantities from modeling outcomes.

### 3.2 Outcomes

The continuous modeling outcome is mathematics literacy, represented by the row-wise mean of `PV1MATH` through `PV10MATH`. For descriptive reporting, plausible values are pooled using multiple-imputation logic and BRR replicate-weight sampling variance [@OECD_2022_technical]. The binary outcome is low-performing student status, defined as mathematics performance below 420.07, the lower bound of PISA mathematics proficiency Level 2 [@NCES_2022_cut_scores].

### 3.3 Predictors and Variable Audit

Candidate predictors covered student background, family resources, mathematics attitudes, school climate, school resources, and digital-learning indicators. A reproducible variable audit was conducted before modeling. Of the configured predictors, 36 were available in the processed data. The main model retained 33 predictors with missingness at or below 50%. `PQSCHOOL` and `PASCHPOL` were excluded from the main model because both exceeded 85% missingness. `ICTDISTR` was retained only for extended or robustness use because its missingness was 56.78%. `PERFEED`, `LEARNRES`, and `DISTICT` were configured but unavailable in the processed public-use files.

Missing predictor values in the modeling pipeline were handled by median imputation for numeric variables and most-frequent imputation for categorical variables. Categorical variables were one-hot encoded with rare-category grouping. The same frozen main feature set was used for regression and classification tasks.

### 3.4 Models and Evaluation

Traditional baselines included ridge regression, elastic net regression, and L2-regularized logistic regression. Machine-learning models included random forest, histogram gradient boosting, XGBoost, and LightGBM. The primary split used 80% of the processed data for training and 20% for holdout evaluation with random seed 20260510. The split was stratified by low-performing student status. Final student weights were normalized to mean 1 and used where estimators supported sample weights; weighted metrics were reported on the holdout set.

Regression performance was evaluated using RMSE, MAE, and R-squared. Classification performance was evaluated using AUC, average precision, F1, precision, recall, and Brier score. Classification metrics were reported at the default threshold of 0.50, with additional threshold sensitivity using Youden's J and maximum-F1 thresholds. Calibration was summarized using Brier score, calibration intercept, calibration slope, expected calibration error, and decile-style calibration bins [@Steyerberg_2010].

### 3.5 Explainability

The best-supported tree-based models were interpreted using SHAP summary plots and permutation importance. SHAP summaries were generated on a deterministic 5,000-row explanation sample. Permutation importance was computed on a deterministic 10,000-row explanation sample with five repeats. A separate digital-feature importance output was generated to support the educational technology interpretation.

### 3.6 Robustness Checks

Six robustness checks were implemented. First, the global best model was evaluated on the OECD holdout subset. Second, holdout performance was compared across gender, immigrant background, and ESCS quintiles. Third, complete-case sensitivity was summarized for the frozen main feature set. Fourth, low-performing labels were recomputed using each mathematics plausible value. Fifth, a country fixed-effect sensitivity check compared lightweight models with and without country/economy as an additional categorical predictor on a stratified 120,000-row robustness sample. Sixth, a country-group holdout split trained models on one set of countries and evaluated them on held-out countries to assess cross-country generalization.

## 4. Results

### 4.1 Sample and Variable Audit

Table 1 summarizes the processed global sample. The final analytic frame included 613,744 students from 80 countries and economies. The weighted mathematics mean was 424.29, and the weighted low-performer rate for the main modeling label was 53.90%. Table 2 reports BRR and plausible-value descriptive standard errors. The main feature set contained 33 predictors. The variable audit confirmed that high-missingness parent-school relationship variables should not be used in the main model, while several digital-learning indicators were retained despite moderate missingness because of their theoretical relevance.

Country-level descriptive statistics showed substantial cross-national variation. For example, the weighted low-performer rate was below 25% in several higher-performing systems and above 70% in several lower-performing systems. This variation motivated both the country fixed-effect sensitivity check and the country-group holdout check.

### 4.2 Model Performance

Table 3 reports weighted holdout performance for the regression task. LightGBM produced the strongest regression performance, with RMSE = 57.61, MAE = 45.12, and R-squared = 0.638. Histogram gradient boosting was very close, with RMSE = 57.69 and R-squared = 0.637. XGBoost performed below LightGBM but above random forest and the linear baselines. Ridge and elastic net had RMSE values around 70.13, indicating that nonlinear tree-based models captured substantially more predictive structure than the linear baselines.

Table 4 reports weighted holdout performance for the low-performer classification task. LightGBM again performed best, with AUC = 0.890, average precision = 0.903, F1 = 0.818, precision = 0.811, recall = 0.826, and Brier score = 0.134 at the default 0.50 threshold. Histogram gradient boosting was nearly identical, with AUC = 0.889. Logistic regression had lower but still meaningful performance, with AUC = 0.840.

Calibration diagnostics supported the main classification model but suggested modest imperfect calibration. The LightGBM weighted mean predicted probability was 0.538, close to the observed weighted low-performer rate of 0.539 in the holdout set. The calibration intercept was -0.001, the calibration slope was 1.062, and the expected calibration error across 10 probability bins was 0.011.

### 4.3 Global Explanations

Permutation importance for the LightGBM classification model identified home possessions (`HOMEPOS`) as the strongest predictor, followed by mathematics self-efficacy (`MATHEFF`), family connection (`FAMCON`), grade, ICT resources (`ICTRES`), cognitive activation, ICT self-efficacy (`ICTEFFIC`), student behavior hindering learning (`STUBEHA`), parental occupational status (`HISEI`), and mathematics anxiety (`ANXMAT`). The same broad pattern appeared in the regression model, where `HOMEPOS` and `MATHEFF` had the largest importance values by a wide margin.

The SHAP summaries in Figure 1 and Figure 2 support the same interpretation: the fitted models relied most strongly on family resources, mathematics-related self-beliefs, family connection, grade, selected digital-learning variables, and school-context indicators. These are model explanations, not evidence that changing any single predictor would cause a corresponding change in mathematics performance.

![Figure 1. SHAP summary for the LightGBM low-performer classification model. The plot summarizes fitted-model feature contributions in the deterministic explanation sample and should be interpreted as predictive model explanation, not causal evidence.](../reports/figures/classification_lightgbm_shap_summary.png)

![Figure 2. SHAP summary for the LightGBM mathematics-score regression model. The plot summarizes fitted-model feature contributions in the deterministic explanation sample and should be interpreted as predictive model explanation, not causal evidence.](../reports/figures/regression_lightgbm_shap_summary.png)

### 4.4 Digital-Learning Predictors

The digital-feature importance output showed that `ICTRES` and `ICTEFFIC` were the most important digital-learning predictors in both tasks. For classification, `ICTRES` had permutation importance of 0.0072 and `ICTEFFIC` had importance of 0.0062. For regression, `ICTRES` and `ICTEFFIC` were also the highest-ranked digital variables, followed by subject-related ICT use (`ICTSUBJ`) and homework/study time (`STUDYHMW`). In contrast, ICT information behavior, home ICT, and school ICT availability/use had smaller model importance values.

This pattern suggests that the model used digital-learning predictors most strongly when they captured resource access and students' confidence with ICT rather than availability alone. Because these variables are observational and partly missing, the interpretation should remain predictive and descriptive.

![Figure 3. Digital-feature permutation importance for the best LightGBM models. ICT resources and ICT self-efficacy were the strongest digital-learning predictors in both tasks.](../reports/figures/digital_feature_importance.png)

### 4.5 Robustness, Subgroups, and Country Context

Subgroup holdout performance was broadly stable by gender. LightGBM AUC was 0.887 for female students and 0.894 for male students. Performance differed more by immigrant background: AUC was 0.889 for native students, 0.859 for first-generation immigrant students, and 0.834 for second-generation immigrant students. Across ESCS quintiles, AUC ranged from 0.853 to 0.875, while F1 decreased in higher-ESCS groups because the low-performer prevalence was lower.

The OECD holdout evaluation produced AUC = 0.865 and regression RMSE = 61.27, lower than the global holdout metrics but still indicating meaningful predictive performance. Complete-case sensitivity showed that only 6.47% of students had complete data on all 33 main predictors. The complete-case low-performer rate was 37.88%, compared with 45.54% in the full unweighted sample, confirming that complete-case analysis would substantially change the analyzed population.

The country fixed-effect sensitivity check showed that adding country/economy improved lightweight model performance on the robustness sample: regression RMSE improved from 70.88 to 64.85, and classification AUC improved from 0.843 to 0.873. The country-group holdout check was more demanding: training on 64 countries and evaluating on 16 held-out countries produced regression RMSE = 65.23 and classification AUC = 0.847. Together, these checks show that country context contributes meaningfully to prediction and that cross-country generalization is weaker than random holdout performance.

## 5. Discussion

This study found that explainable machine-learning models, especially LightGBM and histogram gradient boosting, materially outperformed traditional linear baselines in both mathematics-score prediction and low-performer classification. The predictive gain was not trivial: for regression, LightGBM reduced RMSE from about 70.13 for the linear baselines to 57.61; for classification, LightGBM improved AUC from 0.840 for logistic regression to 0.890.

The explanation results were educationally plausible. Home possessions and family socioeconomic resources were highly model-important, consistent with the broader PISA finding that achievement reflects unequal learning opportunities and home resources [@OECD_2022_results_vol1]. Mathematics self-efficacy and mathematics anxiety were also important, supporting the relevance of affective and motivational constructs [@OECD_2022_results_vol5]. School behavior, teacher behavior, cognitive activation, and resource shortage indicators contributed additional predictive information.

For educational technology research, the key finding is that digital-learning predictors were not merely peripheral. ICT resources and ICT self-efficacy were consistently among the more important predictors, while simpler availability/use indicators were less influential. This does not imply that providing devices or raising self-efficacy would by itself improve mathematics achievement. Rather, the predictive pattern suggests that PISA-based educational technology research should distinguish access, confidence, learning use, and distraction instead of treating digital technology as a single construct.

The robustness checks qualify the interpretation. OECD holdout and country-group holdout performance remained meaningful but lower than global random holdout performance. Country fixed effects improved lightweight models, showing that global prediction partly reflects cross-national context. Subgroup differences in AUC were moderate, but F1 varied with subgroup base rates, especially across ESCS quintiles. These results support reporting multiple metrics rather than relying on a single aggregate score.

## 6. Limitations

This study is cross-sectional and observational. The findings identify predictive patterns and model-important variables, not causal effects. No result should be read as evidence that changing ICT resources, ICT self-efficacy, school behavior, or family support would necessarily change mathematics performance.

Machine-learning workflows do not perfectly align with complex survey estimation. This study uses student weights in model fitting and holdout metrics where supported, and it reports BRR and plausible-value descriptive estimates, but prediction metrics remain model-evaluation quantities rather than population parameter estimates. Future work could extend this workflow with uncertainty-aware prediction, such as conformal prediction, while retaining the distinction between predictive uncertainty and causal uncertainty [@Angelopoulos_Bates_2021].

Missingness is also important. Several digital and family variables had moderate or high missingness, and complete-case analysis would describe a substantially different subset of students. The imputed-feature main model is therefore preferred for prediction, but the interpretation of variables with substantial missingness should remain cautious.

Finally, country-level context matters. Adding country fixed effects improved robustness-sample performance, while country-group holdout reduced performance compared with random holdout. Global results should therefore be read as broad predictive patterns across PISA 2022 participants, not as country-neutral laws.

## 7. Conclusion

Using a global PISA 2022 sample, this study shows that explainable machine learning can improve prediction of mathematics literacy and low-performing student status while producing interpretable patterns aligned with educational theory. LightGBM achieved the best overall performance, and model explanations highlighted family resources, mathematics self-beliefs, family connection, grade, ICT resources, ICT self-efficacy, and school behavior/resource indicators. The results support the value of combining educational technology variables with broader student and school context in PISA-based prediction studies, provided that explanations are kept separate from causal claims.

## Data Availability

The data are publicly available from the OECD PISA 2022 Database. Analysis code and non-restricted derived outputs will be made available in a public repository subject to OECD data-use terms and journal policy. The repository should not redistribute OECD raw data files.

## Ethics Statement

This study uses publicly available, de-identified secondary data from the OECD PISA 2022 Database. No new human participants were recruited by the author, and no individual-level identifiable data were accessed. Institution-specific public-data or secondary-data exemption wording remains to be confirmed before submission.

## AI-Assisted Work Statement

AI-assisted coding and drafting tools were used to support programming, reproducible analysis organization, and manuscript preparation. All intellectual decisions, interpretation, verification, and final manuscript approval remain the responsibility of the human author.

## Generated Tables and Figures

- Table 1: Sample summary, `reports/tables/sample_summary.csv`.
- Table 2: Weighted descriptive estimates with BRR and plausible-value standard errors, `reports/tables/weighted_descriptive_se.csv`.
- Table 3: Regression model metrics, `reports/tables/model_metrics.csv`.
- Table 4: Classification model metrics, `reports/tables/model_metrics.csv`.
- Table 5: Variable audit, `reports/tables/variable_audit.csv`.
- Table 6: Threshold sensitivity, `reports/tables/classification_threshold_sensitivity.csv`.
- Table 7: Calibration diagnostics, `reports/tables/calibration_metrics.csv` and `reports/tables/calibration_bins.csv`.
- Table 8: Subgroup holdout metrics, `reports/tables/subgroup_holdout_metrics.csv`.
- Table 9: Country-context robustness, `reports/tables/oecd_holdout_metrics.csv`, `reports/tables/country_fixed_effects_sensitivity.csv`, and `reports/tables/country_group_holdout_metrics.csv`.
- Figure 1: Classification SHAP summary, `reports/figures/classification_lightgbm_shap_summary.png`.
- Figure 2: Regression SHAP summary, `reports/figures/regression_lightgbm_shap_summary.png`.
- Figure 3: Digital-feature importance, `reports/figures/digital_feature_importance.png`.
