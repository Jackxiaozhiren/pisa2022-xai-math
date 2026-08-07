# Peer Review Report — KBS Simulation

**Paper:** A Knowledge-Based System for Educational Assessment: Knowledge Engineering, Multi-Method Explainable AI, and Fairness-Aware Decision Support on PISA 2022 Data
**Target Journal:** Knowledge-Based Systems (KBS)
**Date:** 2026-07-28
**Review Context:** First KBS review simulation (independent from prior ESWA rounds)

## Scoring Overview

| Dimension | Score (1-10) | Justification |
|-----------|:---:|---------------|
| Relevance to KBS | **7** | Knowledge engineering framing is structurally present in Title, Abstract, Introduction, §4.7, Discussion, and Conclusion. However, the Results section — the most-read section — deploys KBS terminology zero times, which an editor scanning for "fit with KBS" will notice. |
| Knowledge Engineering | **7** | Five-layer architecture (Knowledge Acquisition → Representation → Inference → Extraction → Decision Support) is clean and defensible. The P1-P6 proposition validation protocol is a genuine knowledge engineering contribution. However, the central claim — that the hierarchical feature organization and ICT taxonomy improve explanation quality — lacks ablation evidence (acknowledged but not demonstrated). |
| Methodology Rigor | **8** | Eight robustness checks, formal fairness metrics with intersectional analysis, multi-method XAI with rank correlation, bootstrap CIs, supplementary k-fold CV, fixed seeds, reproducible pipeline. Methodologically thorough. Single-split primary evaluation is mitigated by strong supplementary evidence. |
| Originality/Novelty | **7** | Individual components (XGBoost, SHAP, fairness) are well-established. The integration as a KBS architecture with structured proposition validation, ICT taxonomy, and multi-method XAI convergence analysis represents genuine system-level novelty. Not a breakthough but a solid systems contribution. |
| Significance & Impact | **7** | Template is reusable for PISA 2025, TIMSS, PIRLS. Single-workstation reproducibility lowers adoption barriers. Cross-context generalization limits (AUC 0.903→0.847) are transparently documented, though this also limits practical deployability. |
| Quality of Presentation | **8** | Clean academic writing, well-structured, consistent terminology. Figures properly labeled. Supplementary materials comprehensive. No "expert system" residuals. References current (32% from 2025-2026). Minor: abstract word count at ~200, well within KBS ≤300 limit. |

## Major Concerns

### Major Concern 1: KBS Narrative Depth Insufficient in Results Section

**Section:** Experimental Results (§5, `kbs_results_body.tex`)

**Problem:** The KBS framing is concentrated in the structural sections — Title, Abstract, Introduction, §4.7, Discussion, and Conclusion. The Results section (the largest section at ~134 lines of substantive text) uses KBS terminology exactly **once**: in §5.1 ("The knowledge acquisition layer processed..."). The remaining ~130 lines discuss "models," "predictions," "feature importance," and "metrics" using standard ML vocabulary indistinguishable from a conventional educational data mining paper. A KBS editor or reviewer scanning the Results section for evidence of knowledge-based system design will find none.

**Why this matters:** KBS reviewers consider knowledge engineering contribution as a high-weight criterion. If the Results section reads as a standard ML analysis, the KBS framing in the structural sections appears as a strategic relabeling rather than a genuine reconceptualization. This is the same structural weakness that led to the ESWA desk reject (same framing-in-structural-sections-but-not-in-Results pattern), now reappearing in the KBS context.

**Suggestion:** Add 3-4 KBS-anchoring sentences distributed across Results subsections. In §5.3 (Global feature importance), frame SHAP/permutation findings as "the knowledge extraction layer's output." In §5.4 (ICT feature analysis), reference "the knowledge representation layer's ICT taxonomy." In §5.7 (Fairness evaluation), describe the intersectional analysis as "the decision-support layer's fairness audit." These are small additions (1 sentence each) that maintain the KBS frame without distorting technical content.

> **[Chinese]** 问题：KBS框架集中于结构性章节（标题、摘要、引言、§4.7、讨论、结论），但Results章节（最大章节，约134行实质性文本）仅使用KBS术语一次（§5.1）。剩余约130行使用标准ML词汇。KBS审稿人阅读Results章节时将看不到任何知识系统设计的证据。为什么重要：KBS将知识工程贡献作为高权重评审标准。如果Results章节读起来像标准ML分析，结构性章节中的KBS框架就表现为策略性重标签而非真正的概念重构。这是导致ESWA desk reject的同一结构性缺陷在KBS上下文中的重现。建议：在Results各子节中分散添加3-4句KBS锚定语句，保持KBS框架的持续可见性。

### Major Concern 2: KBS Journal Citations Insufficient

**Section:** References (`references.bib`)

**Problem:** Of 56 references, only **3** (5.4%) are from Knowledge-Based Systems: Caro-Martinez et al. (2024, iSee platform), Zhang et al. (2025, fairness-aware feature selection), and Pei (2025, F3Fair federated learning). KBS expects 10-15% citations from the target journal to demonstrate intellectual engagement with the journal's community. At 5.4%, the paper appears to cite KBS papers only perfunctorily.

**Why this matters:** A KBS editor may interpret low KBS citation density as weak community engagement. The paper's core themes (knowledge engineering, XAI for knowledge extraction, fairness-aware decision support) are well-represented in recent KBS publications; additional citations would strengthen the "fit with KBS" argument.

**Suggestion:** Add 3-5 KBS citations in relevant contexts. Topics to explore: (1) KBS papers on knowledge engineering methodology for tabular data; (2) KBS papers on XAI integration in decision-support systems; (3) KBS papers on educational or social-science applications of knowledge-based systems. Each added citation needs a sentence explaining its connection to the present work, not just a parenthetical insertion.

> **[Chinese]** 问题：56条参考文献中仅3条（5.4%）来自Knowledge-Based Systems期刊。KBS期望10-15%的目标期刊引用比例以展示与期刊学术社区的智力互动。为什么重要：低KBS引用密度可能被KBS编辑解读为薄弱的学术社区参与度。论文的核心主题（知识工程、用于知识提取的XAI、公平意识决策支持）在近期的KBS论文中均有充分表现。建议：增加3-5条KBS引用，每条附带一句关系说明。

### Major Concern 3: Anonymous Version Not Actually Anonymous

**Section:** `kbs_manuscript_anonymous.tex`

**Problem:** The file labeled `_anonymous` is a direct copy of `kbs_manuscript.tex` and contains the author's full name (Zhiren Xiao), institutional affiliation (Guangdong University of Finance), email address, and ORCID. KBS operates double-blind peer review; an editor who spots this will view it as a submission error.

**Why this matters:** This is a submission-system issue, not a scientific one, but it could delay editorial processing or create a negative first impression.

**Suggestion:** Strip all author-identifying information from `kbs_manuscript_anonymous.tex`: remove the `\author{}`, `\ead{}`, `\address{}`, `\cortext{}` blocks from the `frontmatter` environment; remove the CRediT statement's author name; remove any GitHub repository link that includes the author's GitHub handle (verify the repo URL); verify that PDF metadata is clean.

> **[Chinese]** 问题：标记为双盲的匿名版本文件包含作者全名、机构、邮箱和ORCID。KBS采用双盲审稿。为什么重要：这不会影响科学内容，但可能导致编辑处理延迟或造成负面第一印象。建议：从匿名版本中移除所有作者身份识别信息，包括frontmatter中的作者块、CRediT声明和GitHub仓库链接。

## Minor Concerns

### Minor Concern 1: LIME Discussion Still Disproportionate

**Section:** Results §5.6 (ALE analysis) / `kbs_results_body.tex:119-121`

**Problem:** The LIME divergence discussion occupies a dedicated subsubsection in Results plus references to Supplementary Table S5. LIME is explicitly positioned as a "supplementary comparison" and the paper's own findings demonstrate its unreliability. The space devoted to a method the paper recommends against using could be redirected to reinforce KBS terminology in the Results section.

**Why this matters:** With the KBS framing needing deeper penetration in Results, every paragraph competes for space. The current LIME allocation (a full subsubsection in main text) is generous for a negative methodological finding.

**Suggestion:** Compress the LIME discussion in main text to 2-3 sentences (keep the Spearman's ρ < 0.03 finding and the practical recommendation), move detailed LIME analysis to Supplementary Materials, and use the freed space for KBS-anchoring language.

> **[Chinese]** 问题：LIME分歧讨论占据主文一个完整子节加补充表引用。LIME被明确定位为"补充比较"且论文自身证明其不可靠。建议：将主文中LIME压缩至2-3句，将详细分析移至补充材料，利用腾出的空间强化KBS锚定语言。

### Minor Concern 2: No Discussion of 50-Trial HPO Design Rationale

**Section:** §4.3 (Model training)

**Problem:** The paper states "50 trials per model-task" for Bayesian HPO via Optuna without explaining why 50 — was this based on convergence diagnostics, computational budget, or convention? This matters for reproducibility: a reader attempting to replicate with a different computational budget doesn't know whether 50 was a stopping point based on observed diminishing returns or a fixed budget allocation.

**Why this matters:** KBS values system design rationale. A one-sentence justification would improve reproducibility guidance.

**Suggestion:** Add one sentence: "50 trials was chosen based on preliminary convergence diagnostics showing diminishing returns beyond 40 trials; the full search space is documented in Supplementary Table S7."

> **[Chinese]** 问题：论文提到贝叶斯超参优化的"50 trials per model-task"但未说明原因。为什么重要：KBS重视系统设计理由。一句理由说明即可。建议：添加一句说明50 trials的选择基于初步收敛诊断。

### Minor Concern 3: KBS-Specific Terminology One Instance of "KBS" as Acronym

**Section:** Related Work (§3.2), line: "Within KBS, Zhang et al."

**Problem:** The acronym "KBS" is used once to mean "Knowledge-Based Systems journal" (in "Within KBS, Zhang et al.") and elsewhere to mean "knowledge-based system" (the paper's own architecture). This dual usage could confuse readers.

**Why this matters:** Minor — most readers will disambiguate from context. But a reader scanning quickly might momentarily misread "Within KBS" as the paper's own system rather than the journal.

**Suggestion:** In §3.2, replace "Within KBS" with "Within the Knowledge-Based Systems literature" or "In recent KBS publications" to remove ambiguity.

> **[Chinese]** 问题："KBS"缩写在一处指代期刊（"Within KBS, Zhang et al."），其余指代论文自身的知识系统。建议：在§3.2中将"Within KBS"改为"Within the Knowledge-Based Systems literature"以消除歧义。

### Minor Concern 4: Submission Checklist Still References ESWA

**Section:** `SUBMISSION_CHECKLIST.md`

**Problem:** The submission checklist says `eswa_manuscript_anonymous.pdf` should be uploaded. Also lists 62 pages (ESWA) — the KBS PDF has been recompiled and may differ in page count.

**Why this matters:** Cosmetic — the checklist is a local reference file, not submitted. But it indicates the migration from ESWA to KBS is not fully clean.

**Suggestion:** Update the checklist to reference `kbs_manuscript_anonymous.pdf` and verify current page count.

> **[Chinese]** 问题：投稿检查清单仍引用eswa_manuscript_anonymous.pdf。建议：更新为kbs_manuscript_anonymous.pdf。

## Questions for Authors

1. **On the knowledge framework ablation gap:** The Limitations section acknowledges the absence of a formal ablation comparing organized vs. flat feature representations. Is this something you plan to run before submitting to KBS, or is it deferred to future work as stated? A KBS reviewer may view this as the most significant empirical gap given the paper's central claim about knowledge engineering value.
   > **[Chinese]** 关于知识框架消融缺口：在提交KBS之前是否计划运行这个已计划的消融实验？鉴于论文关于知识工程价值的中心声明，KBS审稿人可能将其视为最显著的经验性缺口。

2. **On the country-group holdout:** The drop from AUC 0.903 to 0.847 in the country-group holdout is substantial. Do you recommend a minimum AUC threshold below which the system should not be deployed in a given country, or is per-country validation data always required? A brief deployment-recommendation sentence in the Conclusion would strengthen the decision-support framing.
   > **[Chinese]** 关于国家组留出验证：AUC从0.903降至0.847是实质性的。是否建议最低AUC阈值，还是始终要求各国验证数据？

3. **On KBS scope:** The paper positions its contribution as a KBS but acknowledges that the knowledge representation is declarative (feature-type labels) rather than procedural (no inference rules or causal graph). Do you consider this sufficient for a KBS contribution, or is there a path to encoding richer knowledge structures (e.g., rule extraction from the trained ensemble) in future work?
   > **[Chinese]** 关于KBS范围：论文的贡献定位为KBS，但承认知识表示是声明性的（特征类型标签）而非过程性的（无推理规则或因果图）。这足以作为KBS贡献吗？

## Verdict

**Recommendation:** Major Revision

The paper has genuine knowledge engineering contributions — the five-layer architecture, the P1-P6 proposition validation protocol, the ICT taxonomy — and is methodologically sound. However, three issues require attention before submission: (1) the KBS narrative must penetrate the Results section substantively (same issue that caused the ESWA desk reject); (2) KBS journal citations at 5.4% are below the expected 10-15% threshold; (3) the anonymous version is not actually anonymous. These are all fixable without new experiments. If addressed, the paper would be a strong fit for KBS.

> **[Chinese]** 建议：大修。论文具有真正的知识工程贡献（五层架构、P1-P6命题验证协议、ICT分类法）且方法论可靠。但三个问题需在投稿前处理：(1) KBS叙事必须实质性渗透Results章节（导致ESWA desk reject的同一问题）；(2) KBS期刊引用比例5.4%低于预期的10-15%；(3) 匿名版本并未匿名。这些问题都不需要新实验。如解决，该论文将是KBS的强匹配投稿。
