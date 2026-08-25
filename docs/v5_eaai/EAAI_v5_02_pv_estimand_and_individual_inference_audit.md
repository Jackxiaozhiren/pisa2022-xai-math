# EAAI v5 Phase B：PISA PV、estimand 与个人推断硬门禁

**审计日期：** 2026-08-23  
**结论：** `P1/HIGH` 已确认；Route A 可在不收集新数据的前提下执行，但旧的个人风险叙事和 row-wise PV mean 不能晋升为 v5 active evidence。

| Claim / issue | Current manuscript or code | Live OECD evidence | Conflict | Safe Route A interpretation | Required analysis / text action |
|---|---|---|---|---|---|
| Ten-PV construction | `scripts/01_prepare_data.py` creates `MATH_PV_MEAN`; Methods calls it a standard secondary-analysis convention. | OECD says compute a statistic separately per PV and average the statistics; explicitly says not to average PVs at student level. | Direct conflict. | A PV-specific scoring-model evaluation is a model-level sensitivity/validation procedure, not an estimate of an individual score. | Train/evaluate per PV; pool estimates and variances. Remove the old convention claim. |
| Low-performer target | `LOW_PERFORMER_MATH = I(mean(PV1...PV10)<420.07)`. | OECD says PVs are not individual point estimates and should not support individual inference. | Direct conflict if interpreted as a student's true state. | For each PV, `I(PVv<420.07)` is an imputed outcome used only to audit model behavior over the population/subgroups. | Never call any row a true low-performing student; eliminate per-student deployment/action examples. |
| Individual application | Abstract/Introduction/Discussion refer to low-performer risk scores, teachers/counselors, human review, intervention/dashboard. | OECD PISA design is for population/group statistics, not optimal individual statistics. | Direct conflict for actual student decision use. | Assessment-pipeline pre-use verification: test whether a predictive modeling workflow has acceptable population/subgroup/cross-country behavior before it is ever considered elsewhere. | Retain no teacher, counselor, individual score, intervention, or policy-effect language. |
| Estimand | Legacy reported values use normalized `W_FSTUWT`. | Final student weights represent the multi-country student population; large countries contribute more. | No conflict if clearly named; hidden estimand is a reporting risk. | Population-weighted estimand. | Report it explicitly. |
| Eighty-economy narrative | Legacy text repeatedly describes an 80-economy audit without equal-country sensitivity. | OECD identifies `SENWT` for equal country/economy contribution. | Incomplete estimand sensitivity. | Both population-weighted and equal-country estimands are relevant to an “80 economies” audit claim. | Join official `SENWT` by stable IDs and report both. |
| Sampling uncertainty | Legacy C1 CIs use a student-level bootstrap. | PISA 2022 provides 80 Fay-BRR replicate weights and the formula `0.05 Σ(rep - full)^2`. | Not an official design-consistent estimator. | Fixed-model evaluation uncertainty under the PISA sampling design. | Run all 80 replicate estimates; do not call this training uncertainty. |
| PV uncertainty | Legacy robustness checks only vary PV labels around a mean-target model. | OECD requires repeated per-PV estimates and combination of sampling/imputation variance. | Incomplete. | Full Route A uses PV-specific models under frozen settings, then pools estimates. | Preserve old sensitivity as historical only. |
| Predictor conditioning/circularity | The 33 predictors are student/school questionnaire variables, while the technical report says nearly all BQ variables enter the PISA population model that produces PVs. | The exact conditioned-variable list was not verified field-by-field in this phase; population-model inclusion is documented. | Potential circularity/leakage risk, not yet a proof for every variable. | Treat all model results as conditional fitted-model/pipeline diagnostics, not independent educational prediction or causal evidence. | Add an explicit limitation and avoid causal/ability-prediction claims; a feature-set redesign would require a new research-design decision. |
| Cross-context and institutional validation | Legacy country-group holdout is a 64/16-country split. | No OECD conflict; it is not an institutional intervention validation. | Interpretation risk. | Cross-country transfer stress test only. | Keep as a boundary; do not call it real-world deployment validation. |

## Hard classification of inferential units

| Unit / statement | Route A status |
|---|---|
| Named individual student's proficiency, true low-performance status, score threshold, intervention eligibility | Prohibited |
| Population/subpopulation performance or calibration statistic pooled over ten PVs | Permitted when weights/variance are reported |
| Fixed-model metric uncertainty from 80 replicate weights | Permitted; must be labelled conditional on fitted model |
| Full model-training uncertainty | Not estimated by the minimum Route A replicate procedure |
| Country-group held-out performance | Permitted as cross-context transfer boundary |
| Institutional deployment, teacher action, user utility, policy impact | Not evaluated; prohibited claim |

## Phase-B status

The original paper cannot remain submission-ready without a v5 correction. Route A is scientifically executable using existing data, but it does **not** remove the conditional-PV/predictor circularity limitation. Eliminating that limitation through a redesigned outcome/predictor architecture would be a separate research-design choice and is not silently undertaken here.
