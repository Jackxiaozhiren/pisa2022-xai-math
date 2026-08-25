# EAAI v5 Active Claim--Evidence Crosswalk

| Active claim | Evidence | Boundary | Status |
|---|---|---|---|
| Route A is a reproducible model-level verification/validation protocol | `scripts/33_pisa_pv_replicate_weight_audit.py`, `EAAI_v5_04_preregistered_analysis_protocol.md`, full manifest | No individual inference, intervention or deployment | PROMOTED_BOUNDED |
| PV-pooled XGBoost candidate performance | `reports/tables/v5_pv_pooled_metrics.csv`, `table_03_regression.tex`, `table_04_classification.tex` | Conditional on fitted models; pooled across ten PVs | PROMOTED_BOUNDED |
| Official design-aware uncertainty is visible | `v5_replicate_weight_uncertainty.csv`, full manifest, OECD source | Fixed-model Fay-BRR uncertainty, not full training uncertainty | PROMOTED_BOUNDED |
| C1 intersection is a descriptive diagnostic signal | `v5_intersectional_design_aware_ci.csv`, `v5_population_vs_senate_weights.csv`, `table_calibration_parity.tex` | AUC/slope intervals cross zero; senate attenuates | PROMOTED_WITH_DOWNGRADE |
| Additive EBM is a matched glass-box reference | `v5_controlled_ebm_baseline.csv`, `v5_controlled_ebm_manifest.json`, `EAAI_v5_05b_ebm_decision.md` | Descriptive comparison; no paired covariance or deployment claim | PROMOTED_DESCRIPTIVE |
| Unseen-school institution cold-start validation | `EAAI_v5_12_institution_cold_start_results_and_promotion.md`, `v5_institution_cold_start_manifest.json`, pooled/CI CSVs | Same-cycle school-boundary stress test; not external institution, user or deployment validation | PROMOTED_SECONDARY_BOUNDED |
| Multi-method XAI values are fitted-model diagnostics | `reports/tables/xai_convergence_verified.csv`, legacy figures, Supplementary S6 | Not PV-pooled active performance; no causal/human-utility claim | PROMOTED_AS_LEGACY_DIAGNOSTIC |
| Cross-country transfer is a boundary | legacy `country_group_holdout_metrics.csv`, revised table 09 | Legacy target; not institutional external validation | PROMOTED_AS_HISTORICAL_BOUNDARY |
