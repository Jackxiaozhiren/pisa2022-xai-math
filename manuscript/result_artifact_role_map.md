# Result Artifact Role Map

## A. Manuscript-active evidence

- `reports/tables/v5_pv_pooled_metrics.csv` — primary XGBoost PV-pooled performance.
- `reports/tables/v5_intersectional_design_aware_ci.csv` — design/PV uncertainty for the C1 diagnostic.
- `reports/tables/v5_population_vs_senate_weights.csv` — estimand sensitivity.
- `reports/tables/v5_controlled_ebm_baseline.csv` — matched additive EBM comparator.
- `reports/tables/v5_institution_cold_start_*` — same-cycle unseen-school stress test.
- `docs/v5_eaai/EAAI_v5_07_active_result_register.md` — authoritative active-value register.
- `docs/v5_eaai/EAAI_v5_07_claim_evidence_crosswalk.md` — claim/evidence mapping.

## B. Legacy diagnostic evidence

The historical model/XAI/calibration/country-holdout outputs are retained for chronology and for legacy fitted-model diagnostics that are explicitly labeled as such in the manuscript. They are not active Route A performance evidence.

## C. Publicly excluded evidence

Do not publish row-level predictions, raw OECD files, fitted model binaries, private/local paths, virtual environments, session traces, prompts, acceptance forecasts, reviewer simulations, or Editorial Manager staging material.

## Publication rule

When a number appears in the active manuscript, its controlling source must be either a `v5_*` aggregate output or the active-result register/crosswalk. Legacy values may appear only when explicitly identified as historical diagnostics or baselines.
