# EAAI v5 可选增强的价值、可执行性与人工决策边界

**评估日期：** 2026-08-25  
**当前状态：** Route A-conservative；`CONDITIONALLY READY`  
**当前概率：** `P(send)=28--48%`；`P(accept | sent)=24--44%`；`P(overall)=7--21%`

## 1. 已经完成、并已计入当前概率的高价值增强

| 增强 | 当前证据 | 对接受概率的作用 |
|---|---|---|
| 10 PV + 80 Fay--BRR + SENWT | 完整 manifest、零 replicate failures、active register | 关闭最严重的方法学/个体推断风险；避免更低的 v4 反事实情景 |
| Matched controlled EBM | 同一数据、split、33 features、weights、PV route；InterpretML 0.7.8 | 提供可比 glass-box comparator；对 `P(accept\|review)` 是有限正向，不是新算法贡献 |
| C1 停止和降级 | AUC/slope CI 跨零，senate attenuates | 提高诚信/统计可信度，但牺牲原始 headline；不能把它表述成稳健 fairness 结果 |
| 2026-08-25 近邻刷新 | 新增 teacher-judgment early-warning 与 exam-score verification 全文近邻，已进 BibTeX/Related Work | 提高文献诚实性和审稿准备度，同时压缩 novelty margin；因此当前概率已下调而非机械上调 |
| 36/5 匿名包、ZIP 独立编译、12 tests | 当前 final gate 与 source parity | 降低可避免的 desk/technical rejection；不能消除 scope 风险 |
| 同周期 unseen-school cold-start | 17,303 train schools / 4,326 held-out schools；80 countries；10 PV/80 replicate；0 failures | 直接增加机构边界的 model-level verification evidence；但不是外部机构或用户验证，且 RMSE/R² 变差、C1 仍不稳 | **已完成并晋升 secondary-only** | `EAAI_v5_12_institution_cold_start_results_and_promotion.md` |

这些项目已完成，不应再把“继续润色”误认为会产生固定的概率提升。

## 2. 仍可考虑的增强

| 可选增强 | 潜在收益 | 现实代价/风险 | 本轮能否直接完成 | 建议 |
|---|---|---|---|---|
| 外部机构/用户/工程流程验证 | 对 EAAI real-world application fit 的提升最大，可能把当前问题从“离线 stress test”推进到应用验证 | 需要新数据、机构合作、伦理/治理、用户或干预设计；结果可能不支持原结论 | **不能**，需作者批准新研究设计与数据权限 | 若目标是显著提高 EAAI 适配度，这是最有效路径；不是投稿前小修 |
| 独立于 PV 生成模型的 outcome/predictor 设计 | 直接缓解 predictor--PV conditioning/circularity | 需要新的 outcome、测量或外部/纵向数据；会改变 estimand/RQ | **不能**，触发 v5 人工决策边界 | 适合下一篇或重大重设计，不应在当前包中暗加 |
| 十个 PV-specific XAI 重跑 | 可加强 RQ3 的 active evidence，减少 legacy-XAI limitation | 当前 `lime` 依赖缺失；完整 SHAP/permutation/ALE/LIME 方案需新代码、资源和预注册；解释排名可能发散并降低结论 | **部分可做，但不能安全地声称完整四方法重跑** | 只有作者明确接受新增 XAI 分支并确认依赖/范围后再启动；当前 manuscript 已透明降级，边际收益不确定 |
| Theory-group ablation | 若理论是中央机制，可提供 ΔAUC/ΔRMSE 等证据 | 当前理论只是解释框架；消融会增加低价值分支并可能改变 RQ | **不建议当前做**；决定已记录为 `NOT RUN BY DESIGN` | 保持现状最诚实；不把分组 SHAP 汇总称为 ablation |
| 再做文字/图形润色 | 可能改善可读性 | 不解决 scope、PV conditioning 或无部署证据；过度润色会增加漂移 | **可以，但预期收益低** | 当前包已收敛，不建议为概率而继续改写 |
| EM 文件映射与 merged-PDF 检查 | 防止实际投递中的技术/身份错误 | 只能在 Editorial Manager 中完成 | **不能代做** | 必须做，但这是投递门禁，不是科学增强 |

## 3. 结论性建议

**RECOMMENDATION.** 当前稿件已加入同周期 unseen-school secondary validation；最合理的动作是以 Route A+ 包投递，而不是在没有新数据/新设计的情况下继续添加低收益分析。当前概率仍需重新根据冷启动结果和 reviewer impact 校准；真正的外部机构/用户验证仍是更高成本路径。

若作者愿意承担新研究设计，优先顺序是：

1. 独立 outcome/predictor 与 psychometric 设计；
2. 真实机构/用户/工程流程验证；
3. 在新设计稳定后再考虑 PV-specific XAI 或 theory ablation。

第 1、2 项不能由当前公开 PISA 文件和现有授权安全完成；它们不是“补一张图/再跑一次模型”，而是新的研究项目。第 3 项可以由我在作者明确授权后准备，但不能承诺会提高概率，也不能在 `lime` 缺失时冒充完整四方法验证。

**当前可执行结论：** 同周期机构冷启动验证已完成并通过审计；投递之外的其他安全本地任务已完成。真正能进一步显著提升 EAAI application fit 的外部机构/用户验证仍需要作者决定是否启动新数据/新研究设计。

外部验证的可直接执行设计已准备在 `EAAI_v5_13_external_institution_validation_design.md`；它不包含虚构数据或当前概率提升。
