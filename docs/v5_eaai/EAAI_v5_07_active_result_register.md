# EAAI v5 Active Result Register

## MANUSCRIPT_ACTIVE_RESULT

| Result | Active value | Source | Interpretation |
|---|---:|---|---|
| Population-pooled XGBoost AUC | 0.88652 | `v5_pv_pooled_metrics.csv` | PV-pooled fixed-split model evaluation |
| Population-pooled XGBoost Brier | 0.13754 | `v5_pv_pooled_metrics.csv` | Lower is better; conditional on fitted models |
| Population-pooled XGBoost RMSE | 59.82284 | `v5_pv_pooled_metrics.csv` | PV-pooled regression evaluation |
| Population-pooled XGBoost R² | 0.63458 | `v5_pv_pooled_metrics.csv` | PV-pooled regression evaluation |
| Population-pooled C1 AUC contrast | -0.07849; CI [-0.16175, 0.00477] | `v5_intersectional_design_aware_ci.csv` | Descriptive; not robustly different from zero |
| Population-pooled C1 ECE contrast | 0.08094; CI [0.01997, 0.14191] | `v5_intersectional_design_aware_ci.csv` | Positive descriptive calibration signal |
| Population-pooled C1 slope contrast | -0.21908; CI [-0.47327, 0.03512] | `v5_intersectional_design_aware_ci.csv` | Descriptive; interval crosses zero |
| Matched EBM AUC / RMSE / R² | 0.86891 / 66.15436 / 0.55314 | `v5_controlled_ebm_baseline.csv` | Descriptive glass-box comparator |
| Matched EBM ECE / slope | 0.00835 / 1.02996 | `v5_controlled_ebm_baseline.csv` | Calibration point comparison; no universal superiority claim |
| Institution cold-start XGBoost AUC / Brier | 0.88652 / 0.13584; AUC CI [0.87751, 0.89553] | `v5_institution_cold_start_pooled_metrics.csv` | Secondary unseen-school validation; not external institutional validation |
| Institution cold-start RMSE / R² | 61.02284 / 0.62187; intervals [60.01758, 62.02810] / [0.60227, 0.64148] | `v5_institution_cold_start_pooled_metrics.csv` | Generalization boundary; weaker regression performance than random-student split |
| Institution cold-start C1 contrasts | AUC -0.02386 [-0.08524, 0.03753]; ECE 0.04249 [-0.01951, 0.10448]; slope -0.03884 [-0.28599, 0.20831] | `v5_institution_cold_start_intersectional_ci.csv` | All intervals cross zero; secondary diagnostic only |

## LEGACY_VERIFIED_BASELINE (retained, not active)

Legacy AUC 0.903, Brier 0.126, RMSE 54.10, R² 0.681, old C1 values, old country holdout, and old XAI figures remain historical artifacts only. They are not silently deleted and are not used in the revised active claim set.
