# EAAI v5 Phase D：Route A 完整结果与有效性审计

**运行日期：** 2026-08-23  
**运行状态：** `FULL-DATA RESULT` 完成；初始 `MANUSCRIPT-ELIGIBLE RESULT` 因注册停止条件暂停，后经作者于 2026-08-24 明确授权转为保守 Route A，有限结果已晋升为 active，强 C1 仍停止。  
**代码：** `scripts/33_pisa_pv_replicate_weight_audit.py`  
**分析 manifest：** `reports/tables/v5_analysis_manifest.json`

## 1. 完整执行证据

- 613,744 行；490,995 train / 122,749 fixed holdout；33 个冻结特征。
- PV1MATH--PV10MATH 均完成；每个 PV 都训练独立的加权 XGBoost 回归/分类模型。
- `W_FSTURWT1`--`W_FSTURWT80` 均完成固定模型 replicate 评估；`failure_count=0`。
- 原始 PISA SAV 的 `SENWT` 经唯一 `CNTSTUID` 连接，并用 `CNTSCHID` 与 `W_FSTUWT` 交叉核对。
- Fay-BRR sampling variance 使用 `0.05 * sum((replicate-full)^2)`；PV 之间按预注册公式合并 sampling/imputation variance。
- 所有新输出均使用 `v5_` 前缀；没有覆盖 legacy CSV、JSON、模型、PDF 或正文。

## 2. Legacy 与 Route A 候选结果

| Metric | `LEGACY_VERIFIED_BASELINE` | Route A candidate | Difference / interpretation |
|---|---:|---:|---|
| Global AUC | 0.903 | 0.88652 | -0.01648; headline weaker |
| Global Brier | 0.126 | 0.13754 | +0.01154; worse probability error |
| Global RMSE | 54.10 | 59.82284 | +5.72284; worse |
| Global R² | 0.681 | 0.63458 | -0.04642; weaker |
| Low-SES non-native AUC | 0.779 | 0.77703 | Similar point estimate |
| High-SES native AUC | 0.880 | 0.85552 | Lower reference estimate |
| C1 AUC contrast | -0.101 | -0.07849, 95% CI [-0.16175, 0.00477] | CI crosses zero |
| C1 ECE contrast | +0.114 legacy point contrast | +0.08094, 95% CI [0.01997, 0.14191] | Positive contrast remains |
| C1 slope contrast | -0.385 legacy point contrast | -0.21908, 95% CI [-0.47327, 0.03512] | CI crosses zero |

The full candidate result is not a correction of the old number; it is a different, design-aware PV analysis layer. The legacy layer remains preserved for traceability and is not silently substituted.

## 3. Binding stop conditions triggered

| Stop condition | Evidence | Status |
|---|---|---|
| C1 AUC interval must preserve a negative direction and exclude zero | Population pooled contrast -0.07849; CI upper bound +0.00477 | **TRIGGERED** |
| C1 slope contrast must preserve a negative direction and exclude zero | Population pooled contrast -0.21908; CI upper bound +0.03512 | **TRIGGERED** |
| Population and senate estimands must not materially reverse/attenuate the conclusion | AUC contrast -0.07849 population vs -0.01294 senate; ECE +0.08094 vs +0.02598; slope -0.21908 vs -0.02988 | **TRIGGERED AS MATERIAL SENSITIVITY** |
| Main headline must not be materially weakened without human decision | AUC 0.903→0.88652; RMSE 54.10→59.82284; R² 0.681→0.63458 | **TRIGGERED** |
| Replicate failure threshold (>4/80) | `reports/tables/v5_analysis_failures.csv` has header only | NOT TRIGGERED |
| Key join/row drift | Manifest reports full rows and verified join | NOT TRIGGERED |

## 4. Interpretation boundary

The result supports a narrower statement: the low-SES/non-native contrast remains a negative point estimate and the ECE contrast remains positive under population-weighted PV pooling, but the complete C1 claim is not design-robust because its AUC and slope intervals cross zero and senate weighting attenuates all three contrasts. This is an inferential weakening, not evidence that the subgroup is equal or that the legacy result was fabricated.

The candidate global performance is materially weaker than the legacy headline. Because the registered stop rule says to pause when C1 no longer holds or the headline materially changes, these values cannot be promoted into the manuscript without a human decision about the scientific claim and route.

## 5. Initial stop-state actions (superseded by continuation decision)

- At the initial stop point, controlled EBM was **not run**. Running it before resolving the user-authorized scientific branch would have created additional evidence without a route decision.
- At the initial stop point, no canonical TeX, supplementary source, cover letter, Highlights, declarations, graphical abstract, PDF, source ZIP or final package was modified.
- At the initial stop point, no acceptance probability was re-estimated from the stopped Route A state.

These statements describe the pre-continuation checkpoint only. The continuation decision below authorized the bounded EBM, manuscript revision, package rebuild and v5 probability forecast.

## 6. Initial stop-state required human decision (resolved)

The next action cannot be chosen safely by local execution alone. The author must decide whether to:

1. accept a narrower Route A paper in which C1 is reported as a descriptive, non-robust sensitivity finding and the weaker pooled performance becomes active;
2. abandon Route A as the main paper and retain only a transparent methodology limitation/diagnostic appendix; or
3. authorize a genuinely different research design/feature/outcome strategy (which is outside the current pre-authorization and may require new data, research questions, or expert psychometric review).

At that initial checkpoint, all v5 candidate results remained unpromoted and the EAAI package remained pre-v5/conditionally ready only as a historical state. The author subsequently authorized the conservative Route A continuation; the decision is recorded below and the active register/final gate supersede this initial stop-state description.

## Continuation decision recorded 2026-08-24

The author authorized the recommended conservative Route A continuation. The active revision therefore downgrades C1 to a descriptive, design-sensitive diagnostic, uses the weaker PV-pooled XGBoost performance values, and adds the full-data matched EBM as a descriptive glass-box comparator. The legacy result layer remains preserved and is not silently restored. The controlled EBM did not reverse the model-performance conclusion: XGBoost remains higher in pooled AUC/RMSE/$R^2$, while EBM has better pooled ECE and calibration slope. See `EAAI_v5_05b_ebm_decision.md` and the active-result register.
