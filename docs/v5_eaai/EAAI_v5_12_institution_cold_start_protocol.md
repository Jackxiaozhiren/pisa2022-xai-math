# EAAI v5 Route A+：Institution Cold-Start Validation Protocol

**协议日期：** 2026-08-25  
**状态：** `FULL_DATA_CANDIDATE_AUDITED`  
**授权：** 用户已明确接受为提升 EAAI 适配度而增加机构/工程验证路径。  
**边界：** 这是同一 PISA 2022 公共样本内的 unseen-school validation stress test，不是外部机构部署、用户研究或干预验证。

## 1. 研究问题与工程解释

**RQ-A+.** 当模型在完全未见过的学校上进行预测时，PV-pooled performance、subgroup calibration/discrimination 和 cross-country estimand sensitivity 是否仍能被审计？

工程解释是“模型在进入另一所学校前的 cold-start verification”。学校作为 institution unit；所有同一 `CNT`–`CNTSCHID` 组合的学生必须完全属于 train 或 test，防止同一学校的学生信息泄漏到两侧。

本分支不能声称：真实外部机构泛化、机构部署成功、教师/用户效用、干预效果、政策效果或个人学生诊断。它只提供比随机学生切分更严格的同周期 institution-boundary stress test。

## 2. 数据身份与固定设计

- 输入：现有 `data/processed/pisa2022_math_model_frame.parquet` 与官方 raw SAV 的 `SENWT` join；不下载新数据。
- 样本：613,744 行、80 个国家/经济体、21,629 个学校（预检实测）。
- Features：冻结的 33 个 Route A predictors；不添加 `CNTSCHID` 或学校 ID。
- Outcome：每个 `PV1MATH`--`PV10MATH` 逐一建立 regression outcome；classification 使用 `I[PV_vMATH < 420.07]`，仅为 imputed model-evaluation target，不是个人 proficiency label。
- Split：在每个国家/经济体内对唯一学校 ID 做确定性 80% train / 20% test 划分；每个国家至少保留一所 train 和一所 test 学校。学校排序后使用 `seed=20260510 + country_rank*1009` 的独立 RNG，避免 Python hash 漂移。
- Weights：训练和测试使用 normalized `W_FSTUWT`；测试使用全部 80 个 `W_FSTURWT` 做 Fay--BRR fixed-model evaluation；另报告 `SENWT` point-estimand sensitivity。
- Models：冻结 legacy-tuned XGBoost hyperparameters；每个 PV 独立训练 regression/classification model。EBM 不在此分支重复训练，除非该分支通过后另行授权。

## 3. 估计量与输出

对每个 PV 输出 population 与 senate point estimates；对 population test estimates 使用 80 replicates 计算 `0.05 * sum((T_r-T_full)^2)`，再用十 PV pooling 合并 sampling/imputation variance。输出 global regression/classification、low-ESCS/non-native intersection、high-ESCS/native reference、C1 contrasts 和 failures。

输出必须全部使用新前缀，不能覆盖 Route A 或 legacy 文件：

- `reports/tables/v5_institution_cold_start_pv_specific_metrics.csv`
- `reports/tables/v5_institution_cold_start_replicate_uncertainty.csv`
- `reports/tables/v5_institution_cold_start_pooled_metrics.csv`
- `reports/tables/v5_institution_cold_start_senate_sensitivity.csv`
- `reports/tables/v5_institution_cold_start_intersectional_ci.csv`
- `reports/tables/v5_institution_cold_start_failures.csv`
- `reports/tables/v5_institution_cold_start_manifest.json`
- `data/interim/v5_institution_cold_start_predictions.parquet`

## 4. 停止与晋升规则

立即停止并保留 candidate-only 的条件：

1. school join/split 有重复、缺失、train-test school overlap 或国家缺失；
2. PV/replicate join 失败，任何 PV 缺失，或超过 4/80 replicate failure；
3. classification target 在 train/test 或核心 subgroup 中无法计算；
4. 结果只支持个人、教师、机构部署或干预结论；
5. institution holdout 结果与 Route A active result materially reverse 而无法在同一 estimand 下解释。

即使全部通过，也只能先晋升为 `MANUSCRIPT_ACTIVE_SECONDARY_VALIDATION`；不得覆盖 primary Route A metrics，不得把它写成 external validation。若结果不稳定，保留为 negative boundary evidence，不隐藏。

## 5. 资源与回归要求

- 先运行 synthetic/unit split checks 和 smoke path；smoke 结果不得进入论文。
- 全量运行前检查预计内存、学校/国家计数和分割审计表。
- 全量完成后运行 Python syntax、11 个既有 v5 tests、schema、source/legacy immutability、PDF/package parity；若 active claim 或正文改变，必须重新编译匿名包并更新独立审稿与 acceptance forecast。

**预注册决定：** 运行该分支可检验 unseen-institution boundary，但不会被宣传为真实机构部署；是否晋升到正文取决于结果、审稿影响和 EAAI scope 解释，而不是预先假定结果会提高接受率。
