# EAAI v5 条件化接受概率重估

**预测日期：** 2026-08-25  
**稿件状态：** Route A+（unseen-school secondary validation）；本地门禁 `CONDITIONALLY READY`  
**预测性质：** 结构化专家判断，不是频率学估计、EAAI 官方统计或录用保证。

## 1. 先验与证据边界

**VERIFIED PUBLIC FACT.** 本次刷新检索了 EAAI 官方期刊页和 Guide for Authors。公开页面说明 EAAI 关注 AI 方法在真实工程应用中的新颖方面，并要求使用公共数据进行验证；页面还显示投稿到首个决定、审稿决定和录用的流程时长指标，但没有提供可靠的官方 acceptance-rate prior：

> **No reliable official EAAI acceptance-rate prior was found.**

来源：

- [EAAI journal page](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
- [EAAI Guide for Authors](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors)

**VERIFIED PUBLIC FACT.** OECD 当前 PISA 分析指导要求对 plausible values 逐一分析并合并结果，使用 80 个 replicate weights/Fay 方法，并警告 PISA 不适合直接作个体层面统计；这些规则已用于当前 Route A。来源：[OECD PISA database analysis guidance](https://www.oecd.org/en/about/programmes/pisa/how-to-prepare-and-analyse-the-pisa-database.html)。

**LOCAL EVIDENCE.** 当前匿名主稿为 36 页，匿名补充件为 5 页；十个 PV、80 个 replicate、SENWT sensitivity、matched controlled EBM 和 unseen-school cold-start validation 均有 manifest/CSV/代码链；C1 的 AUC 与 calibration-slope 区间跨零，senate point contrast attenuates or reverses；九角色独立审稿及 2026-08-25 latest-neighbor addendum 共同认为“方法学明显改善，但 scope/应用解释仍决定送审”。详见 `EAAI_v5_06_latest_literature_and_positioning.md` 和 `EAAI_v5_11_enhancement_value_assessment.md`。

**INFERENCE.** 区间至少保留约 10 个百分点宽度，以反映 EAAI 编辑口径、审稿人相关性、PISA predictor-conditioning 和外部系统状态的不确定性。不能把本地 LaTeX 通过等同于送审或录用。

## 2. 情景定义

### Scenario 0 — PV/个体推断冲突未关闭（历史对照）

这是 v4 风格稿件的反事实基线：row-wise PV mean、个体低分者/早期预警叙事和未充分设计权重不确定性仍在。它不是当前投稿包，也不是建议路线。

### Scenario 1 — 当前 Route A-conservative（实际可投稿状态）

保留 model-level pre-use verification/validation 的研究问题；使用 10 PV、80 Fay--BRR replicate、population/SENWT sensitivity 和 matched EBM；把 C1 降级为 design-sensitive descriptive diagnostic；不声称个体诊断、机构部署、干预效果、因果解释或新算法。

### Scenario 2 — 未来真正适合个体推断的设计（假设情景）

需要新的外部/纵向结果、独立于 PV 生成模型的预测目标、机构或用户验证及相应伦理/数据授权。这些工作本轮没有执行，不能作为当前稿件的证据或承诺；该情景只用于说明潜在上限。

## 3. 概率区间

| Scenario | `P(send to external review)` | `P(eventual acceptance \| sent to review)` | `P(eventual acceptance at EAAI)` | 中央判断的下一决策分布* | Confidence |
|---|---:|---:|---:|---|---|
| 0. 未关闭 PV/个体冲突（反事实） | 10--25% | 10--25% | 2--12% | Desk reject 70%; Reject after review 17%; Major revision 8%; Minor revision 4%; Accept 1% | 中低 |
| 1. 当前 Route A+（实际，含 unseen-school secondary validation 与 2026-08-25 新近邻） | 28--48% | 24--44% | 7--21% | Desk reject 52%; Reject after review 20%; Major revision 17%; Minor revision 8%; Accept 3% | 中等偏低 |
| 2. 新数据/机构验证/新设计（假设） | 45--65% | 35--55% | 16--36% | Desk reject 30%; Reject after review 20%; Major revision 25%; Minor revision 15%; Accept 10% | 低（未执行） |

\* “下一决策分布”是一次投稿在编辑初筛或审稿后的可观察决定类别的中央结构化判断，五项合计 100%；它不是已校准的期刊频率。总体录用区间按送审与送审后条件录用的乘积作主计算，同时保留相关性、情景不确定性和区间四舍五入造成的宽度。

### 情景 1 的解释

- `P(send)` 只作小幅上移判断：unseen-school secondary validation 提供了更接近工程验证的证据，但仍是同一 PISA 周期的公开样本，不是外部机构或部署验证；最新近邻仍压缩 novelty margin。EAAI 的真实工程应用口径仍构成高敏感 desk risk。
- `P(accept | review)` 受益于 PV/replicate 纠偏、可复现 manifest、主动降级 C1、EBM comparator、匿名包完整性和 unseen-school AUC stability；但 RMSE/R² 在 cold-start 下变差、C1 全部跨零，且 predictor conditioning、fixed-model replicate uncertainty、legacy XAI 未按 PV 重跑和无用户验证仍是限制。
- 总体区间 7--21% 是对上述两层区间的保守乘积/结构性扩展，不应读成精确百分比。

## 4. 最敏感变量与预期价值最高的行动

**最敏感变量（P(send) 与总体概率）：** 编辑是否接受“高影响评估数据的 model-level pre-use verification”作为 EAAI 所需的真实工程应用。PV 修复不能消除这一范围风险。

**影响 `P(accept | review)` 的主要变量：**

1. 审稿人是否把 bounded estimand、C1 停止规则和不确定性诚实披露视为贡献，而非把它们误读为学生诊断；
2. 是否保持 active/legacy 分层，不把历史 SHAP/transfer 图表当作 PV-pooled active evidence；
3. 已确认的 BibTeX 空页警告、AI/code disclosure、graphical-abstract 许可和作者事实能否在 EM 文件映射中保持一致；
4. 是否由作者在 EM 中检查合并 PDF 和文件类型映射。

**预期价值最高的安全行动：** 保持当前保守范围，将已确认的事实完整映射到 EM，并在上传后检查 merged PDF；不要在没有新数据/设计的情况下恢复个体/部署 headline。若必须显著提升 EAAI 适配度，最有效但超出本轮授权的路径是新的机构/用户验证设计，而不是继续润色现有 PISA 指标。

## 5. 结论

**RECOMMENDATION.** 作者事实核对已完成，当前 Route A+ 稿件可以进入 Editorial Manager 外部动作阶段；门禁应保持 `CONDITIONALLY READY`，不能标记 `READY FOR EDITORIAL MANAGER` 或任何 `ACCEPTANCE READY` 等价标签。当前最诚实的成功定义是：让编辑能够清楚看到一个可复现、设计感知、包含 unseen-school cold-start 边界且会主动停止过强结论的 model-level verification artifact；不是保证送审或录用。
