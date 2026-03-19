# metaoverfit: Detecting Overfitting in Meta-Regression

**Optimism-corrected R²_het estimation through cross-validation and bootstrap methods for meta-regression analysis.**

[![R-CMD-check](https://github.com/mahmood726-cyber/metaoverfit/workflows/R-CMD-check/badge.svg)](https://github.com/mahmood726-cyber/metaoverfit/actions)

## Overview

Meta-regression is commonly used in meta-analysis to explore sources of heterogeneity. However, with small numbers of studies (k) relative to parameters (p), apparent R²_het estimates can be substantially inflated. The **metaoverfit** package provides:

- **Cross-validation** (leave-one-out and k-fold) for optimism-corrected R²_het
- **Bootstrap confidence intervals** for uncertainty quantification
- **Overfitting risk assessment** with sample size guidance
- **Diagnostic plots** for visual assessment

## Installation

```r
# Install from GitHub (once available)
# devtools::install_github("mahmood726-cyber/metaoverfit")

# Or install from local directory
devtools::install("path/to/metaoverfit")
```

## Quick Start

```r
library(metaoverfit)
library(metafor)

# Example data
k <- 30
yi <- rnorm(k, 0, sqrt(0.1))
vi <- runif(k, 0.01, 0.1)
mods <- cbind(1, rnorm(k))

# Check for overfitting
check <- check_overfitting(yi, vi, mods)
print(check)
```

## Main Functions

| Function | Purpose |
|----------|---------|
| `r2het()` | Calculate apparent and adjusted R²_het |
| `r2het_cv()` | Cross-validated (optimism-corrected) R²_het |
| `r2het_boot()` | Bootstrap confidence intervals for R²_het |
| `check_overfitting()` | Complete overfitting risk assessment |
| `sample_size_recommendation()` | Sample size guidance for planning |
| `plot_overfitting()` | Diagnostic visualization |

## Example Output

```
============================================================
META-REGRESSION OVERFITTING ASSESSMENT
============================================================

Sample Size:
  k = 30 studies
  p = 2 parameters
  k/p ratio = 15

Risk Assessment:
  Category: Low
  Expected optimism: <10%
  Actual optimism: 8.5%
  CV convergence: 100%

R^2 for Heterogeneity:
  Apparent R^2_het: 45.2%
  Corrected R^2_het: 36.7%
  95% CI: [12.3%, 61.5%]

RECOMMENDATION:
  Acceptable sample size - still report optimism correction
============================================================
```

## Understanding Warnings

The package provides informative warnings:

- **"Fewer than 10 studies"**: R²_het estimates are less stable
- **"τ² = 0 (at boundary)"**: Heterogeneity estimate at boundary condition
- **"Too few studies for stable CV"**: k ≤ p+1, CV not possible

These are **expected and informative**—they indicate important limitations.

## Validation

The package has been validated with:

1. **Unit Tests**: 20+ test cases covering statistical validity
2. **Cochrane Data**: 501 real meta-analyses from Pairwise70
3. **Simulation Study**: 12,000 simulations across 12 scenarios
4. **Edge Cases**: Small k, low k/p ratios, boundary conditions

See `validation/VALIDATION_SUMMARY.md` for complete details.

## Best Practices

| Situation | Recommendation |
|-----------|----------------|
| k < 10 | Do not conduct meta-regression |
| k/p < 10 | Report optimism correction required |
| k/p ≥ 15 | Results more reliable |
| τ² = 0 | Consider whether heterogeneity is present |

**Always report both apparent and corrected R²_het** when conducting meta-regression.

## Citation

If you use this package, please cite:

```
Ahmad M, Khan L, Claude AI (2026). metaoverfit: Optimism-Corrected
Heterogeneity in Meta-Regression. R package version 0.1.0.
```

## References

- Harrell FE Jr, Lee KL, Mark DB (1996). Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Stat Med*, 15(4):361-387.

- Higgins JP, Thompson SG (2002). Quantifying heterogeneity in a meta-analysis. *Stat Med*, 21(11):1539-1558.

- Viechtbauer W (2010). Conducting meta-analyses in R with the metafor package. *J Stat Softw*, 36:1-48.

## License

GPL-3

## Authors

- Mahmood Ahmad [aut, cre]
- Laiba Khan [aut]

## Contact

For questions or issues, please open a GitHub issue or contact:
 Mahmood Ahmad (mahmood.ahmad2@nhs.net)
