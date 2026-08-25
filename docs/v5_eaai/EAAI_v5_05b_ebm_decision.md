# EAAI v5 Controlled EBM Decision

**Run:** full-data candidate completed 2026-08-24 13:18 CST  
**Manifest:** `reports/tables/v5_controlled_ebm_manifest.json`  
**Configuration:** InterpretML 0.7.8; additive EBM (`interactions=0`), 33 encoded predictors, fixed split, all 490,995 training rows, normalized `W_FSTUWT`, ten PV-specific outcomes, max rounds 1,000 and early stopping 50.

## Pooled population-weighted comparison

| Metric | Route A XGBoost candidate | Controlled EBM candidate | Direction |
|---|---:|---:|---|
| AUC | 0.88652 (95% design/PV interval [0.87989, 0.89315]) | 0.86891 (interval [0.86258, 0.87524]) | XGBoost higher point estimate |
| Brier | 0.13754 | 0.14654 | XGBoost lower error |
| RMSE | 59.82284 (interval [59.13633, 60.50934]) | 66.15436 (interval [65.45194, 66.85677]) | XGBoost lower error |
| R² | 0.63458 (interval [0.62486, 0.64431]) | 0.55314 (interval [0.54172, 0.56457]) | XGBoost higher |
| ECE | 0.03253 | 0.00835 | EBM lower point error |
| Calibration slope | 0.80510 | 1.02996 | EBM closer to ideal slope |

## Decision

The controlled EBM does not reverse the primary model-performance conclusion. XGBoost remains the stronger discriminative/regression model under the matched full-data/PV/weight protocol, while the additive EBM is a useful glass-box comparator with better pooled calibration point estimates. The comparison is descriptive: the current implementation does not estimate a paired covariance for the difference, so no formal “significant superiority” claim is promoted.

The manuscript may state that the matched EBM comparison makes the performance-versus-glass-box trade-off explicit. It must not claim that post-hoc XAI is universally necessary, that EBM is inferior on every property, or that this comparison validates deployment utility.

The earlier interrupted `max_rounds=5000` run is invalid and is not used. The full `max_rounds=1000` run is the only EBM candidate eligible for promotion review.
