# Addressing the "False Discovery" of Heterogeneity: Optimism-Corrected R² in Meta-Regression with the metaoverfit R Package

## Authors
**Mahmood Ahmad**¹², **Laiba Khan**³
¹ Royal Free London NHS Foundation Trust, London, UK
² Tahir Heart Institute, Rabwah, Pakistan
³ Independent Researcher, London, UK

**Corresponding author:** Mahmood Ahmad (mahmood.ahmad2@nhs.net)

## Abstract
Meta-regression is widely used in systematic reviews to explore sources of heterogeneity between studies. However, the reliability of the proportion of heterogeneity explained (apparent $R^2_{het}$) is often compromised when the number of studies ($k$) is small relative to the number of moderators ($p$). In such cases, the apparent $R^2_{het}$ can be substantially inflated due to overfitting, leading to "false discoveries" of moderator effects. We present `metaoverfit`, an R package designed to detect and correct for overfitting in meta-regression. The package implements optimism-correction techniques through cross-validation (leave-one-out and $k$-fold) and bootstrap methods specifically tailored for random-effects meta-regression models. We validated `metaoverfit` using a comprehensive simulation study (12,000 iterations) and an empirical analysis of 501 Cochrane meta-analyses. Our results demonstrate that $R^2_{het}$ optimism is severe when the $k/p$ ratio is less than 10. `metaoverfit` provides researchers with automated risk assessments, optimism-corrected $R^2$ estimates, and bootstrap confidence intervals, ensuring more robust and reproducible evidence synthesis.

## Introduction
Heterogeneity is a fundamental challenge in meta-analysis. Meta-regression, extending the random-effects model to include study-level covariates, is the primary tool for explaining this variance. The performance of these models is typically summarized by $R^2_{het}$ (the proportion of between-study variance, $	au^2$, explained by the moderators). 

Despite its popularity, meta-regression is prone to overfitting. When $k$ is small, moderators can appear to explain a large portion of heterogeneity simply by chance. This "optimism" in $R^2_{het}$ leads to overconfident clinical conclusions. While optimism correction is standard in primary clinical prediction modeling (e.g., Harrell's $R^2$), it is rarely applied in the meta-analytic context. The `metaoverfit` package fills this gap by providing a unified framework for cross-validated and optimism-corrected heterogeneity analysis.

## Methods
The `metaoverfit` package provides a suite of tools for robust heterogeneity assessment:

- **Apparent and Adjusted $R^2_{het}$:** Calculates the standard and small-sample adjusted $R^2_{het}$ using the `metafor` framework.
- **Cross-Validation (CV):** Implements leave-one-out (LOO-CV) and $k$-fold cross-validation to estimate the "out-of-sample" $R^2_{het}$. This provides a more realistic estimate of how well the moderators would explain heterogeneity in new, unseen studies.
- **Bootstrap Optimism Correction:** Uses the bootstrap-correction method (Harrell et al., 1996) to calculate the difference between the apparent performance on bootstrap samples and the performance on the original dataset, yielding a "corrected" $R^2_{het}$.
- **Overfitting Risk Categorization:** Automatically classifies the risk of overfitting (Low, Moderate, Severe, Extreme) based on the $k/p$ ratio and the observed optimism.
- **Sample Size Guidance:** Provides evidence-based recommendations for the minimum $k$ required to achieve a target level of $R^2$ stability.

## Implementation
`metaoverfit` is built on the industry-standard `metafor` package. Its primary user-facing function, `check_overfitting()`, integrates CV and bootstrapping into a single diagnostic report. The package supports both matrix and formula interfaces, ensuring compatibility with standard R workflows. Parallel processing via the `foreach` and `doParallel` packages is supported to handle the computational load of bootstrap iterations.

## Results: Validation and Performance
### Simulation Study
Across 12,000 simulations, we found that:
1. When the $k/p$ ratio is $< 5$, the apparent $R^2_{het}$ can exceed 50% even when no true relationship exists (Null models).
2. The `metaoverfit` corrected $R^2$ successfully reduced this inflation, maintaining Type I error rates near 5%.
3. Stability in $R^2$ estimates required a minimum $k/p$ ratio of 10 for low-complexity models.

### Empirical Validation (Cochrane Data)
Analysis of 501 real-world meta-analyses showed that over 30% of published meta-regressions had "Severe" or "Extreme" overfitting risk, with a median optimism of 18%. `metaoverfit` identified several cases where an apparent $R^2_{het}$ of >80% collapsed to <20% after optimism correction.

## Discussion
The `metaoverfit` package provides a critical safeguard for evidence synthesis. By exposing the "optimism" in heterogeneity explanations, it encourages more cautious and realistic interpretations of meta-regression results. We recommend that researchers report both the apparent and optimism-corrected $R^2_{het}$ in all published meta-analyses, particularly when $k < 20$ or $k/p < 10$.

## Availability and Requirements
- **Project name:** metaoverfit
- **Programming language:** R (≥ 3.5.0)
- **Dependencies:** metafor, stats, foreach, doParallel
- **License:** GPL-3
- **URL:** https://github.com/mahmood726-cyber/metaoverfit

## References

[1] Harrell FE, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat Med. 1996;15(4):361-387. https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4

[2] Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis. Stat Med. 2002;21(11):1539-1558. https://doi.org/10.1002/sim.1186

[3] Viechtbauer W. Conducting meta-analyses in R with the metafor package. J Stat Softw. 2010;36(3):1-48. https://doi.org/10.18637/jss.v036.i03

[4] Thompson SG, Higgins JPT. How should meta-regression analyses be undertaken and interpreted? Stat Med. 2002;21(11):1559-1573. https://doi.org/10.1002/sim.1187

[5] Knapp G, Hartung J. Improved tests for a random effects meta-regression with a single covariate. Stat Med. 2003;22(17):2693-2710. https://doi.org/10.1002/sim.1482

[6] Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman & Hall; 1993. https://doi.org/10.1007/978-1-4899-4541-9

[7] R Core Team. R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing; 2024. https://www.R-project.org/

[8] Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. Chichester: John Wiley & Sons; 2009. https://doi.org/10.1002/9780470743386

[9] Lopez-Lopez JA, Page MJ, Lipsey MW, Higgins JPT. Dealing with effect size multiplicity in systematic reviews and meta-analyses. Res Synth Methods. 2018;9(3):336-351. https://doi.org/10.1002/jrsm.1310

[10] IntHout J, Ioannidis JP, Borm GF. The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. BMC Med Res Methodol. 2014;14:25. https://doi.org/10.1186/1471-2288-14-25
