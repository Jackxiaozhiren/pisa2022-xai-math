# EAAI v5 九角色独立终审

**终审日期：** 2026-08-25（含最新近邻 addendum）  
**输入：** canonical/anonymous TeX and PDF, supplementary files, v5 manifests/tables, current official-rule refresh, recent-literature monitor, source ZIP.  
**审稿原则：** 先列 concerns，再给分数；旧 v4 PASS 不继承；所有判断标出证据边界。

## Reviewer 1 — EIC / Desk Editor

**One-sentence summary:** The manuscript now presents a bounded, model-level verification protocol for predictive AI in assessment data, with official PV/replicate-weight treatment and a deliberately weakened intersectional claim.

**Strengths:** The title/abstract distinguish AI artifact from application; the current results state that C1 intervals cross zero; the public-data protocol, matched EBM, and anonymous package are inspectable.

**Major concerns:**

1. EAAI requires novel AI aspects used for a real-world engineering application. The manuscript supplies a consequential assessment setting and a reproducible audit artifact, but no institution, user, deployment constraint, or intervention evaluation. This remains a desk-screen interpretation risk.
2. The contribution is a verification contract and evidence-boundary protocol, not a new algorithm. Its acceptability depends on whether EAAI accepts model-level pre-use verification as an engineering application.
3. EM merged-review-PDF behavior, author facts, AI disclosure facts, and graphical-abstract permission remain unverified.

**Scores:** Novelty 6/10; methodology 7/10; validity 7/10; application significance 6/10; writing 8/10; presentation 8/10; reproducibility 8/10.

**Recommendation:** Borderline send to external review / Major Revision if sent.  
**Confidence:** 4/5.  
**Score-change condition:** Accept stance rises if the editor accepts the bounded model-level application; it falls to desk reject if existing deployment or algorithmic novelty is required.

## Reviewer 2 — AI/ML Methodology

**One-sentence summary:** The PV-specific weighted XGBoost/EBM comparison is substantially more controlled than v4, but the method contribution is protocol-level and the XAI evidence remains legacy fitted-model evidence.

**Strengths:** Section 3.3 and Supplementary S1 freeze the split, predictors, PV route, weights and replicate formula; the matched EBM uses full data and the same outcome route; the paper avoids claiming causal explanations.

**Major concerns:**

1. The XGBoost hyperparameters are inherited from a legacy tuning record rather than re-tuned or nested within each PV; this is reproducible but selection-conditional.
2. Replicate weights evaluate predictions conditional on fitted models; the paper correctly does not call this full model-training uncertainty, but readers may still overinterpret the intervals.
3. Multi-method XAI correlations were not recomputed for the ten PV-specific models. The active text correctly labels them legacy diagnostics, reducing the strength of RQ3.
4. No theory-group ablation was run; the theory organization remains an interpretation scaffold.

**Scores:** Novelty 6/10; methodology 7/10; validity 7/10; application significance 6/10; writing 8/10; presentation 8/10; reproducibility 8/10.

**Recommendation:** Major Revision if reviewed.  
**Confidence:** 4/5.  
**Repair condition:** A future full refit-under-replicate analysis or PV-specific XAI rerun could raise methodology/evidence, but neither should be claimed now.

## Reviewer 3 — PISA / Psychometrics

**One-sentence summary:** The revision correctly removes the student-level PV mean and individual inference, but the predictor-conditioning relationship between questionnaire variables and PV generation remains a central psychometric limitation.

**Strengths:** Section 3.2 and Supplementary S1 state ten-PV pooling, 80 replicate weights, SENWT sensitivity, and the prohibition on individual PV interpretation; the classification target is explicitly called an imputed model-evaluation target.

**Major concerns:**

1. The paper uses background questionnaire predictors that may participate in PISA population models generating PVs. The limitation is disclosed, but the analysis cannot be read as independent prediction of latent proficiency.
2. The 420.07 proficiency boundary remains a benchmark for an imputed target, not a diagnostically valid individual threshold.
3. Senate results are point-estimand sensitivities without separately verified senate replicate weights; this must stay clearly separated from design variance.

**Scores:** Novelty 6/10; methodology 6/10; validity 6/10; application significance 6/10; writing 8/10; presentation 7/10; reproducibility 8/10.

**Recommendation:** Major Revision, with a credible path if the bounded estimand language is retained.  
**Confidence:** 5/5.  
**Repair condition:** A psychometric expert review or a future predictor/outcome redesign would be the strongest improvement; it is not silently assumed here.

## Reviewer 4 — Statistics / Complex-Survey

**One-sentence summary:** The active numerical layer is materially improved by PV pooling and Fay-BRR replicates, and the stopped C1 claim is now reported honestly.

**Strengths:** The full manifest records zero join/replicate failures; C1 AUC and slope intervals cross zero; senate weighting attenuates the point contrasts; fixed-model uncertainty is not mislabeled as training uncertainty.

**Major concerns:**

1. AUC and calibration intervals use normal approximations and can extend beyond metric bounds; the paper should keep them as approximate design/PV intervals rather than exact finite-sample coverage guarantees.
2. Replicate-weight evaluation does not refit each model under each replicate, so model-selection and training uncertainty remain outside scope.
3. The subgroup ECE uses ten equal-width bins and the slope uses weighted logistic calibration; sensitivity to binning and separation is not fully explored.
4. The legacy tables retain historical student-level diagnostics. Their captions now mark them historical, but reviewers may still confuse them with active evidence.

**Scores:** Novelty 6/10; methodology 7/10; validity 7/10; application significance 6/10; writing 7/10; presentation 7/10; reproducibility 8/10.

**Recommendation:** Major Revision / send only with the present caveats visible.  
**Confidence:** 5/5.

## Reviewer 5 — XAI / Fairness / Calibration

**One-sentence summary:** The paper’s strongest contribution is now the audit’s ability to downgrade a compelling intersectional point estimate when design and estimand sensitivity weaken it.

**Strengths:** The C1 point estimates, intervals, and senate sensitivity are co-reported; the EBM demonstrates a calibration/performance trade-off; no universal ABROCA or ECE cutoff is claimed.

**Major concerns:**

1. Legacy SHAP/permutation/ALE/LIME correlations are not evidence for the PV-pooled active model and should remain visibly secondary.
2. The paper does not implement the formal strong-calibration multi-coverage test cited in prior work; the text now avoids claiming it does.
3. The EBM/XGBoost calibration comparison lacks paired covariance and should remain descriptive.
4. No analyst or user study establishes explanation utility.

**Scores:** Novelty 7/10; methodology 7/10; validity 7/10; application significance 6/10; writing 8/10; presentation 8/10; reproducibility 8/10.

**Recommendation:** Major Revision with a credible bounded-audit path.  
**Confidence:** 4/5.

## Reviewer 6 — Educational Assessment Domain

**One-sentence summary:** The revision now treats PISA as a model-audit stress test rather than a student-diagnosis instrument, which is methodologically responsible but narrows practical impact.

**Strengths:** Concurrent self-efficacy and home-resource predictors are explicitly noncausal; Section 6 rejects intervention-lever interpretation; the 80-economy scope and transfer boundary are useful for assessment analytics.

**Major concerns:**

1. The protocol’s practical audience is an assessment-model analyst, not a teacher or institution; this weakens the original early-warning significance claim.
2. The proficiency boundary is not an individual decision threshold.
3. Construct comparability, PV conditioning, and institutional transfer remain unresolved.

**Scores:** Novelty 6/10; methodology 6/10; validity 6/10; application significance 6/10; writing 8/10; presentation 7/10; reproducibility 7/10.

**Recommendation:** Major Revision if the venue accepts the model-level application category.  
**Confidence:** 4/5.

## Reviewer 7 — Devil’s Advocate

**One-sentence summary:** The strongest rejection argument is that this is a careful assembly of known models and audit metrics without a deployment artifact or algorithmic novelty.

**Reject-grade concerns:**

1. EAAI may regard a public, offline assessment stress test as educational analytics rather than a real-world engineering application.
2. Recent 2026 work now covers continuous standardized-score fairness, executable governance, multi-group mitigation, and deployment-oriented early-warning evaluation.
3. The contribution can be read as a checklist unless the verification contract and its negative/uncertain C1 result are treated as the central engineering artifact.

**Counter-evidence:** The revision does not hide the weakened result; it adds official survey-design treatment, a full-data matched EBM, a reproducible manifest, and explicit non-deployment boundaries.

**Scores:** Novelty 5/10; methodology 6/10; validity 6/10; application significance 5/10; writing 7/10; presentation 7/10; reproducibility 7/10.

**Recommendation:** Borderline / desk-reject risk remains high.  
**Confidence:** 4/5.

## Reviewer 8 — Writing / Visual

**One-sentence summary:** The revised title, abstract, figures and tables now share the conservative verification spine and are readable in the regenerated PDFs.

**Strengths:** Anonymous main PDF is 36 pages and supplementary is 5 pages; figure captions distinguish active PV-pooled diagnostics from legacy XAI/transfer figures; graphical abstract is 3315×1353 px at 300 dpi.

**Minor concerns:**

1. The manuscript still contains a substantial legacy XAI/transfer layer, which requires careful reader labeling.
2. Some tables are dense, especially the model-level verification map.
3. Final EM conversion remains unobserved.

**Scores:** Novelty 6/10; methodology 7/10; validity 7/10; application significance 6/10; writing 8/10; presentation 8/10; reproducibility 8/10.

**Recommendation:** Minor-to-Major Revision depending on scope decision.  
**Confidence:** 5/5 for local render, 3/5 for EM rendering.

## Reviewer 9 — Integrity / Reproducibility

**One-sentence summary:** The current package has a traceable candidate-result chain, but author facts and AI/graphical-abstract disclosures remain outside local verification.

**Strengths:** Full XGBoost and EBM manifests, v5 tables, source ZIP, anonymous TeX/PDF, fixed seed, and zero replicate failures are present; no legacy artifact was overwritten.

**Major concerns:**

1. Two bibliography records retain empty-pages warnings (`Lundberg_Lee_2017`, `Ke_2017`); the author reviewed and accepted the DOI/venue metadata warnings on 2026-08-24.
2. The author confirmed that the AI disclosure accurately describes the actual code/analysis assistance used in this revision; Elsevier’s current policy remains relevant to any later factual change.
3. The graphical-abstract tool/license and EM permission were confirmed by the author; EM file-type behavior remains external.
4. The EM merged review PDF and external similarity/integrity systems are unavailable locally.

**Scores:** Novelty 6/10; methodology 7/10; validity 7/10; application significance 6/10; writing 7/10; presentation 7/10; reproducibility 8/10.

**Recommendation:** Conditionally ready for author/EM checks; not a guarantee of review or acceptance.  
**Confidence:** 5/5 for local artifacts, 2/5 for external systems.

## 2026-08-25 latest-neighbor addendum

The live refresh added two full-text Frontiers papers with closer conceptual overlap than the earlier monitor: Xia & Chen's teacher-judgment/institutional-fairness early-warning framework (SPHERE, teacher workflow and ablation) and Olaniyan et al.'s explainable fairness-aware exam-score verification framework (ASAP 2.0, rubric/peer consistency and human-in-the-loop audit). They do not invalidate the PISA PV/replicate analysis or require a new current design, but they reduce the novelty margin and strengthen the following reviewer interpretation:

- **EIC/desk:** the model-level verification artifact must be distinguished from teacher decision support, score verification, human utility and deployment; scope remains the dominant desk-screen variable.
- **AI/ML and XAI:** the contribution is a design-aware verification contract and negative/uncertain result, not a new explainability, fairness, calibration or ablation method.
- **Educational/application:** the absence of institutional/user validation is more salient, not less; no individual or teacher-facing claim may return.
- **Devil's advocate:** the “careful audit assembly” rejection argument is stronger, so acceptance probability should be lowered rather than hidden by adding citations.

The local package gate remains unchanged after the citation refresh: current anonymous main 36 pages, supplementary 5 pages, valid source ZIP, and no final-log errors/undefined references/overfull boxes.

## 2026-08-25 institution cold-start addendum

The authorized Route A+ branch held out complete schools within each country/economy and evaluated 4,326 unseen schools across all 80 countries using ten PVs and 80 replicates. AUC remained 0.8865 [0.8775, 0.8955], while RMSE increased to 61.02 and $R^2$ decreased to 0.6219; all three C1 intervals crossed zero and senate point estimates removed or reversed the population directions. The panel therefore treats this as a useful secondary unseen-institution verification boundary, not external institutional validation or a new robust fairness result. It modestly improves the application narrative while preserving the major scope limitation.
