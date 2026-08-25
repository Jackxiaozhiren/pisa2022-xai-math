# EAAI v5 Route A+：Institution Cold-Start Results and Promotion Decision

**运行日期：** 2026-08-25  
**输入协议：** `EAAI_v5_12_institution_cold_start_protocol.md`  
**状态：** `MANUSCRIPT_ACTIVE_SECONDARY_VALIDATION`  
**边界：** 同一 PISA 2022 公共样本内的 unseen-school stress test；不是外部机构部署、用户研究或干预验证。

## 1. 执行证据

- 613,744 rows；490,338 train / 123,406 held-out rows。
- 17,303 train schools / 4,326 held-out schools；80 countries/economies；school overlap = 0。
- 10 PV-specific XGBoost regression/classification fits；all 80 replicate weights；`failure_count=0`。
- 33 frozen predictors；same legacy-tuned hyperparameters；population and SENWT point sensitivity retained。
- Runtime 298.12 seconds under Python 3.9.6, pandas 2.3.3, NumPy 2.0.2, scikit-learn 1.6.1, XGBoost 2.1.4, pyreadstat 1.2.9.

Manifest: `reports/tables/v5_institution_cold_start_manifest.json`  
Outputs: `reports/tables/v5_institution_cold_start_*.csv`, `data/interim/v5_institution_cold_start_predictions.parquet`.

## 2. Population-pooled unseen-school performance

| Metric | Primary random-student Route A | Institution cold-start | Cold-start 95% design/PV interval | Interpretation |
|---|---:|---:|---:|---|
| AUC | 0.8865 | 0.8865 | [0.8775, 0.8955] | Discrimination is stable under unseen-school split |
| Brier | 0.1375 | 0.1358 | [0.1297, 0.1419] | Slightly lower point error; descriptive |
| RMSE | 59.82 | 61.02 | [60.02, 62.03] | Regression error increases for unseen schools |
| $R^2$ | 0.6346 | 0.6219 | [0.6023, 0.6415] | Generalization is weaker than random-student split |

The result supports a bounded engineering statement: the classification discrimination level is not destroyed by holding out complete schools, while continuous-score prediction degrades. It is a cold-start generalization boundary, not proof of institutional deployment or external validity.

## 3. C1 and estimand sensitivity

| Contrast | Estimate | 95% design/PV interval | Senate point estimate |
|---|---:|---:|---:|
| AUC (low-ESCS non-native minus reference) | -0.0239 | [-0.0852, 0.0375] | 0.0028 |
| ECE (intersection minus global) | 0.0425 | [-0.0195, 0.1045] | 0.0186 |
| Calibration slope (intersection minus global) | -0.0388 | [-0.2860, 0.2083] | 0.0302 |

All three design-aware intervals cross zero, and the senate point estimates remove or reverse the population point direction. This is consistent with the primary Route A stop decision and is not evidence of subgroup equality; it is evidence that the C1 signal is not stable under an institution-boundary split.

## 4. Promotion decision

**PROMOTED AS SECONDARY VALIDATION ONLY.** The result is added to the manuscript as an institution cold-start stress test because it directly tests unseen-school generalization with the same PV/replicate-weight discipline. It does **not** replace primary Route A metrics, does not revive C1, and does not close the external institution/user/deployment gap.

The full result is publication-eligible only for the bounded claims above. Any later institution-level outcome, user study, intervention or external partner validation remains a new research design requiring separate data and ethics decisions.

