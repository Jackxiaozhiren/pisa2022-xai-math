# GitHub Methodology Crosswalk: PISA BRR + Plausible Values

**Purpose:** Compare the current Python implementation with established R-based PISA analysis tools to verify methodological consistency.

**Date:** 2026-05-14

---

## 1. Reference Implementations

| Tool | Language | Repository | Key Functionality |
|------|----------|-----------|-------------------|
| `brr` | R | `github.com/debrouwere/brr` | BRR weight handling, plausible values, poststratification |
| `Rrepest` | R (CRAN) | `github.com/cran/Rrepest` | Multi-ILSA framework with BRR/jackknife + PV pooling |
| `learningtower` | R (CRAN) | `github.com/kevinwang09/learningtower` | Harmonized PISA 2000-2022 data |
| `pisa_xai` | Python | Current project | BRR + PV + weights + XAI pipeline |

---

## 2. Methodological Crosswalk

### 2.1 Descriptive Statistics with Plausible Values

| Operation | `Rrepest` approach | Current `pisa_xai` approach | Consistency |
|-----------|-------------------|----------------------------|-------------|
| PV pooling | Rubin's rules: mean = mean(PV_means), var = var_within + (1+1/m)*var_between | Row-wise PV mean for modeling; Rubin's rules for descriptives | ✅ Consistent for descriptives; row-wise mean documented as modeling simplification |
| BRR SE | `repest()` with `svyrep.design()` and BRR weights | Manual BRR loop with `scipy.stats` | ✅ Equivalent algorithm |
| Weight normalization | Senate weights / student weights normalized to effective sample size | Student weights normalized to mean = 1 | ✅ Conceptually equivalent |

### 2.2 BRR Variance Estimation

`brr` package does:
```r
# R pseudocode
design <- svrepdesign(weights = ~W_FSTUWT, repweights = "W_FSTR[0-9]+", data = pisa)
mean_est <- svymean(~PV1MATH, design)
```

Current `pisa_xai` does:
```python
# Python equivalent
weights = df["W_FSTUWT"]
for r in range(1, 81):
    rep_weight = df[f"W_FSTR{r}"]
    # compute estimate with replicate weight
    ...
se = sqrt(sum((estimates - theta_full)^2) / 80)
```

**Verdict:** Algorithmically equivalent. Both implement the standard BRR formula:
$$SE(\hat{\theta}) = \sqrt{\frac{1}{R}\sum_{r=1}^{R}(\hat{\theta}_r - \hat{\theta})^2}$$

### 2.3 Model Fitting with Survey Weights

| Framework | Weight handling | Regularization | XAI integration |
|-----------|----------------|----------------|-----------------|
| `Rrepest::repGlm()` | BRR-aware GLM fitting | No built-in ML tuning | No |
| `brr` | Arbitrary fitter + BRR loop | User-defined | No |
| `pisa_xai` | `sample_weight` in sklearn | Optuna Bayesian tuning | SHAP + ALE + LIME |

`pisa_xai` extends beyond Rrepest/brr by integrating survey weights into modern ML pipelines with XAI. The trade-off is that `pisa_xai` does not run the full model within a BRR loop (computationally prohibitive for 80 BRR weights × 50 Optuna trials × 4 models), which is acknowledged in Limitations.

---

## 3. Verified Consistency Points

- ✅ BRR standard error formula matches across implementations
- ✅ Plausible value pooling follows OECD technical guidelines
- ✅ Student weight normalization approach documented and consistent
- ✅ Country-level aggregation logic matches OECD reporting conventions
- ⚠️ Model-level BRR uncertainty (for ML coefficients/SHAP values) not computed (computational constraint)

---

## 4. Recommendations

1. **Short-term:** Add a footnote in Section 4.1 citing `brr` and `Rrepest` as methodological references
2. **Medium-term:** Run a spot-check: compute BRR SE for the weighted math mean using both `pisa_xai` (Python) and `Rrepest` (R) on a common subset, report correlation
3. **Long-term:** If computational resources permit, implement model-level BRR for at least the best-performing model to quantify ML coefficient/SHAP uncertainty
