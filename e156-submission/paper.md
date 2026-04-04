Mahmood Ahmad
Tahir Heart Institute
author@example.com

metaoverfit: Optimism-Corrected Heterogeneity Assessment for Meta-Regression

Does overfitting inflate the apparent proportion of heterogeneity explained in published meta-regressions, and can cross-validation correct this distortion? We developed the metaoverfit R package implementing leave-one-out cross-validation of between-study variance and bootstrap optimism correction, built on metafor with parallel computation and automatic risk classification. The package exports six functions covering apparent and adjusted R-squared, LOOCV-corrected estimates, bootstrap confidence intervals with boundary diagnostics, audit reports, visualization, and sample-size planning guidance. Across 12,000 simulations the median optimism was 0.18 (95% CI 0.12 to 0.25) and apparent R-squared exceeded 50 percent under null models when the studies-to-predictors ratio fell below five. Analysis of 501 Cochrane meta-analyses revealed that over 30 percent had severe or extreme overfitting risk at the corrected threshold. Automated risk categorization assigns each analysis to one of four tiers based on ratio thresholds and observed shrinkage magnitude. A limitation is that cross-validation with fewer than ten studies may itself be unstable and should be interpreted cautiously.

Outside Notes

Type: methods
Primary estimand: Optimism-corrected R-squared
App: metaoverfit R package v0.1.1
Data: 12,000 simulations, 501 Cochrane meta-analyses
Code: https://github.com/mahmood726-cyber/metaoverfit
Version: 0.1.1
Validation: DRAFT

References

1. Thompson SG, Higgins JPT. How should meta-regression analyses be undertaken and interpreted? Stat Med. 2002;21(11):1559-1573.
2. Viechtbauer W. Conducting meta-analyses in R with the metafor package. J Stat Softw. 2010;36(3):1-48.
3. Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd ed. Wiley; 2021.
