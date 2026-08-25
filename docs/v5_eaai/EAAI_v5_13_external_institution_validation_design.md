# EAAI v5 外部机构/工程验证设计（供合作方与伦理审批使用）

**设计日期：** 2026-08-25  
**状态：** `DESIGN_READY_NOT_EXECUTED`  
**用途：** 为后续真实机构/工程流程验证准备可审计方案；不属于当前 PISA 证据，不得写成已完成验证。

## 1. 最小可接受验证对象

合作方必须提供一个真实的 assessment-analytics 或其他高影响预测流程，并明确：

- 实际使用场景、机构责任人和模型进入本地使用前的决策节点；
- 预测目标和独立观测结果（不能只用同一 PV 生成模型的再表达）；
- institution ID、时间戳、训练/验证/部署前窗口和数据漂移记录；
- protected/group variables、缺失机制、数据授权、伦理审批或豁免依据；
- 模型输出、阈值、校准方案、人工复核/申诉路径和禁止的个体用途。

没有真实机构责任、独立结果和时间/机构边界的公开数据复现，只能称为 stress test，不能称为 external institutional validation。

## 2. 预注册估计量

首要 estimand 是“对未见 institution 的模型效能与审计稳定性”，而不是学生个体准确率：

1. leave-one-institution-out 或 train-institutions/test-unseen-institutions 的 AUC、Brier、RMSE、$R^2$；
2. 机构/群体分层 ECE、calibration slope、AUC contrast 和预先固定的 subgroup definitions；
3. 若数据来自复杂抽样，逐 PV、replicate weights、Fay--BRR 和 population/equal-contribution estimands；否则记录独立结果的测量不确定性与 institution-cluster bootstrap；
4. time-forward validation：训练窗口不能读取未来结果或后续干预信息；
5. sensitivity：threshold、missingness、institution size、country/system and class-imbalance 的预注册敏感性。

所有 primary/secondary metrics 必须在分析前冻结方向、区间、最小有效 institution 数、失败定义和晋升规则。不得根据结果挑选有利 institution 或 subgroup。

## 3. 安全与伦理边界

- 不输出或评估 named-student alerts、教师惩罚、招生/分流、资源剥夺或自动干预；
- 先做 model-level pre-use verification，再决定是否值得 local governance review；
- 明确 human oversight、申诉/纠错、数据保留、访问控制、模型撤回条件和利益相关者责任；
- 任何个体层面用途都需要独立伦理、法律、隐私和公平影响评估；当前作者不能代合作机构批准这些事项。

## 4. 停止与结果解释

出现 institution leakage、结果目标循环、严重缺失、外部结果无法独立核验、关键 subgroup 只有极少 institution、或外部验证反转主结论时，停止晋升并报告失败/不稳定边界。成功也只能支持“在该合作机构/时间窗口内通过预先定义的模型级验证”，不能外推到所有学校、国家或学生。

## 5. 与当前 Route A+ 的关系

当前 unseen-school PISA 分支已提供同周期 cold-start secondary evidence，但不是本设计的外部验证。若未来合作机构数据到位：

1. 保留当前 Route A primary metrics 和 legacy baseline；
2. 将合作机构结果作为独立 validation layer，不覆盖 PISA 结果；
3. 由作者、机构负责人和伦理/治理负责人共同决定是否改变研究问题、作者贡献、声明和投递策略；
4. 若结果与当前模型不一致，按预注册停止规则报告，而不是为了接受率重调模型。

**执行状态：** 设计已准备；数据、机构授权、伦理决定和真实工程流程尚未提供，因此本文件不产生 acceptance-probability uplift 证据。

