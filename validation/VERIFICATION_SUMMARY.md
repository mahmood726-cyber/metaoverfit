# metaoverfit Validation Scripts - Verification Summary

**Date:** 2026-01-15
**Verified by:** Claude (AI Assistant)

---

## Overview

This document verifies that all validation scripts for the metaoverfit package are correctly structured and ready to run.

---

## Package Structure Verification

### ✅ Core Package Files

| File | Status | Notes |
|------|--------|-------|
| `DESCRIPTION` | ✅ Valid | Proper package metadata, dependencies listed |
| `NAMESPACE` | ✅ Valid | All exports and imports correctly specified |
| `R/metaoverfit.R` | ✅ Valid | Single consolidated R source file |

### ✅ Package Exports (from NAMESPACE)

```
Exported Functions:
- check_overfitting()
- plot_overfitting()
- r2het()
- r2het_boot()
- r2het_cv()
- sample_size_recommendation()

S3 Methods:
- print.metaoverfit
- print.sample_size_rec
```

### ✅ Dependencies

**Required:**
- R >= 3.5.0
- metafor >= 3.0.0
- stats
- utils

**Suggested:**
- ggplot2 (for plotting)
- testthat (for unit tests)
- knitr/rmarkdown (for vignettes)
- metadat (for example datasets)

---

## Validation Scripts Verification

### 1. Unit Tests (`tests/testthat/test-basic.R`)

**Status:** ✅ Syntactically Valid

**Test Coverage:**
- ✅ Basic structure tests
- ✅ Statistical validity tests (R² bounds, monotonicity)
- ✅ Edge case tests (NULL mods, k ≤ p+1)
- ✅ Input validation tests
- ✅ Formula interface tests
- ✅ Sample size recommendation tests
- ✅ Print method tests
- ✅ Class inheritance tests

**Expected Test Count:** 20+ tests

**Run Command:**
```r
# In R console
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
devtools::test()
# Or
testthat::test_dir("tests/testthat")
```

---

### 2. Demo Validation Script (`validation/demo_validation.R`)

**Status:** ✅ Syntactically Valid

**Examples Included:**
1. Basic R²_het calculation
2. Cross-validation (LOO)
3. Bootstrap confidence intervals
4. Complete overfitting check
5. Sample size recommendations
6. Formula interface
7. Small k overfitting demonstration
8. K-fold cross-validation
9. Validation sanity checks
10. Optional ggplot2 plots

**Run Command:**
```r
# In R console
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
source("validation/demo_validation.R")
```

**Expected Output:**
```
=== Example 1: Basic R²_het Calculation ===
Apparent R²_het: XX.XX%
Adjusted R²_het: XX.XX%
τ²_null: X.XXXX
τ²_full: X.XXXX
I²: XX.XX%

=== Example 2: Cross-Validation (LOO) ===
...

=== VALIDATION CHECKS ===
Check 1 - Corrected R² <= Apparent R²: PASS
Check 2 - Optimism >= 0: PASS
Check 3 - R² in [0, 1]: PASS
Check 4 - CI ordered correctly: PASS
Check 5 - Convergence > 50%: PASS

Overall: ALL TESTS PASSED
```

---

### 3. Quick Simulation Test (`validation/simulation_quick_test.R`)

**Status:** ✅ Syntactically Valid

**Configuration:**
- Scenarios: 8
- Iterations per scenario: 100
- Total simulations: 800
- Estimated runtime: 5-10 minutes

**Test Scenarios:**
| Scenario | k | p | k/p | tau² | beta_mod | Focus |
|----------|---|---|-----|------|----------|-------|
| 1 | 5 | 2 | 2.5 | 0.05 | 0.3 | Small k |
| 2 | 10 | 2 | 5 | 0.05 | 0.3 | k/p = 5 |
| 3 | 15 | 2 | 7.5 | 0.05 | 0.3 | Moderate |
| 4 | 20 | 2 | 10 | 0 | 0.3 | tau² = 0 |
| 5 | 10 | 3 | 3.3 | 0.05 | 0 | Null + low k/p |
| 6 | 10 | 2 | 5 | 0.1 | 0.3 | High tau² |
| 7 | 30 | 2 | 15 | 0.05 | 0.3 | Adequate |
| 8 | 50 | 2 | 25 | 0.05 | 0.3 | Large k |

**Run Command:**
```r
# In R console
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
source("validation/simulation_quick_test.R")
```

**Expected Output:**
- CSV file: `validation/simulation_results/quick_test_results.csv`
- Console summary with sanity checks

---

### 4. Full Simulation Study (`validation/simulation_study.R`)

**Status:** ✅ Syntactically Valid

**Configuration:**
- Scenarios: 12
- Iterations per scenario: 1000
- Total simulations: 12,000
- Estimated runtime: 30-60 minutes

**Additional Features:**
- More extensive scenario coverage
- Type I error analysis
- Power calculations
- Bootstrap CI coverage assessment
- Comprehensive visualizations

**Run Command:**
```r
# In R console
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
source("validation/simulation_study.R")
```

---

### 5. Cochrane Data Validation (`validation/validate_with_cochrane_data.R`)

**Status:** ✅ Syntactically Valid

**Requirements:**
- Access to Cochrane datasets at:
  `C:/Users/user/OneDrive - NHS/Documents/Pairwise70/analysis/output/cleaned_rds/`
- Packages: data.table, ggplot2

**Features:**
- Automatic data extraction from diverse formats
- Processes all 501 datasets
- Full validation pipeline
- Comprehensive summary statistics
- Visualizations

**Run Command:**
```r
# In R console
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
source("validation/validate_with_cochrane_data.R")
```

---

## Pre-Flight Checklist

Before running validation scripts, ensure:

- [ ] R is installed (version >= 3.5.0)
- [ ] Required packages are installed:
  ```r
  install.packages(c("metafor", "testthat", "data.table", "ggplot2"))
  ```
- [ ] Working directory is set to metaoverfit package root
- [ ] Package is loaded or installed:
  ```r
  # For development
  devtools::load_all(".")

  # Or install
  devtools::install(".")
  ```

---

## Validation Results Location

All results will be saved to:
```
metaoverfit/validation/simulation_results/
├── quick_test_results.csv           (from quick test)
├── simulation_full_results.csv      (from full study)
├── simulation_summary_by_scenario.csv
├── simulation_summary.txt
├── optimism_vs_ratio.png
├── apparent_vs_corrected.png
├── optimism_by_k.png
└── convergence_vs_ratio.png
```

For Cochrane validation:
```
metaoverfit/validation/cochrane_results/
├── validation_results.csv
├── validation_summary.txt
├── risk_distribution.png
├── optimism_vs_ratio.png
└── apparent_vs_corrected.png
```

---

## Troubleshooting

### Issue: "Error: could not find function 'r2het'"

**Solution:** Load the package first
```r
devtools::load_all(".")
# or
library(metaoverfit)
```

### Issue: "Error: package 'metafor' not available"

**Solution:** Install dependencies
```r
install.packages("metafor")
```

### Issue: "Warning: package not found"

**Solution:** Use `devtools::install()` to install with dependencies
```r
devtools::install(dep = TRUE)
```

### Issue: Cochrane validation fails

**Solution:** Check data path is correct
```r
file.exists("C:/Users/user/OneDrive - NHS/Documents/Pairwise70/analysis/output/cleaned_rds/")
```

---

## Summary

| Script | Syntax | Ready to Run | Output |
|--------|--------|--------------|--------|
| Unit Tests | ✅ Valid | ✅ Yes | Console |
| Demo Validation | ✅ Valid | ✅ Yes | Console + Plots |
| Quick Simulation | ✅ Valid | ✅ Yes | CSV + Summary |
| Full Simulation | ✅ Valid | ✅ Yes | CSV + Plots + Summary |
| Cochrane Validation | ✅ Valid | ⚠️ Requires data access | CSV + Plots |

**Overall Status:** ✅ All validation scripts are syntactically valid and ready to run.

---

## Next Steps

1. **Quick Start (5 minutes):**
   ```r
   source("validation/demo_validation.R")
   ```

2. **Unit Tests (1 minute):**
   ```r
   devtools::test()
   ```

3. **Quick Simulation (5-10 minutes):**
   ```r
   source("validation/simulation_quick_test.R")
   ```

4. **Full Validation (30-60 minutes):**
   ```r
   source("validation/simulation_study.R")
   ```

---

**Verification Complete:** All scripts verified and ready for execution.
