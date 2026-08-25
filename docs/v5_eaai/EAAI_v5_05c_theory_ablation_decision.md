# EAAI v5 Phase F：Theory-group ablation 决策

**决策日期：** 2026-08-24  
**状态：** `DECISION_COMPLETE_NOT_RUN`  
**结果层：** 不产生新的 publication result；不改变 `MANUSCRIPT_ACTIVE_RESULT`

## 1. 审计结论

当前理论组织有两个用途：

1. 作为 33 个 PISA questionnaire predictors 的结构化 inventory，帮助读者理解 individual、interaction、institutional、contextual 层级以及 ICT access--skills--usage 分类；
2. 作为 legacy fitted-model SHAP/permutation/ALE/LIME 结果的解释索引，用于描述哪些变量组在历史模型中占据较多 attribution。

现有代码和主稿没有把这些理论分组编码为新的输入变换、正则项、交互约束、模型结构或训练目标。当前 active Route A 贡献是 PV/replicate-weight/estimand-aware verification protocol，而不是理论驱动模型或理论因果检验。因此，按组汇总 attribution 只能称为 `group-level interpretation`，不能称为 theory ablation、theory validation 或 theory-induced performance improvement。

## 2. 为什么本轮不运行消融

**NOT RUN BY DESIGN。** 运行 leave-one-theory-group-out、group-only models 或完整的 PV-specific variant set 会引入新的研究分支；它们不是验证当前 active claim 所必需的最小分析，而且可能重新定义理论贡献、模型选择和研究问题。当前稿件已经将理论定位降级为 interpretation aid，并在 Limitations 中明确没有 theory-group ablation。

不运行的决定不是因为缺少一个“好看”的结果，也不是用旧的 SHAP 汇总替代消融；而是为了避免在 C1 已触发停止、active 贡献已收窄后再加入未经注册的理论比较。

## 3. 不得声称的内容

- 不得声称理论组织改善 AUC、Brier、RMSE、R²、ECE 或 C1；
- 不得声称 flat feature list 无法获得相同的分组汇总；
- 不得把 legacy SHAP/permutation/ALE/LIME 分组图当作 PV-pooled active evidence；
- 不得把理论层级解释为因果层级、干预杠杆或机构决策依据。

## 4. 当前证据与文稿边界

| 证据 | 当前处理 |
|---|---|
| `eaai_submission/manuscript/eaai_manuscript.tex` 的 predictor inventory 与 theory table | 保留为变量组织和解释辅助；不作为新模型结构证据 |
| legacy XAI 表/图与 `xai_convergence_verified.csv` | 保留为 fitted-model historical diagnostic；不晋升为 PV-specific XAI 结果 |
| Route A XGBoost/EBM metrics | 不依赖理论分组消融；active 结果来自冻结 33-feature matched protocol |
| Limitations | 明确记录 no theory-group ablation、no causal validation 和 no human/user study |

## 5. 若未来必须运行，最低可接受设计

未来若将理论机制重新提升为中央研究问题，必须单独预注册：leave-one-group-out、group-only 和预先固定的 full-feature comparator；固定 split、预处理、超参数、10 PV、80 replicate weights、population/SENWT estimands；每个 variant 预先绑定机制问题，并报告 ΔAUC、ΔBrier、ΔRMSE、ΔR²、Δintersectional ECE 及其不确定性。该路线需要新的科学决定，不属于当前投稿包的未完成任务。

**最终决定：** 当前稿件不运行 theory-group ablation；保留理论组织的描述性、可审计边界，并不把它包装成已验证的理论贡献。

