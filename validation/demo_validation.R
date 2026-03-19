# ============================================================================
# metaoverfit Demonstration & Validation Script
# ============================================================================
# This script demonstrates the key features of the metaoverfit package
# and validates them with simulated and example data.
#
# Run this script to verify the package is working correctly.
# ============================================================================

# Load required packages
library(metafor)
library(metaoverfit)
# library(ggplot2)  # Uncomment for plots

# ============================================================================
# Example 1: Basic R²_het Calculation
# ============================================================================

cat("\n=== Example 1: Basic R²_het Calculation ===\n")

set.seed(123)
k <- 30
yi <- rnorm(k, 0, sqrt(0.1))
vi <- runif(k, 0.01, 0.1)
mods <- cbind(1, rnorm(k))

result <- r2het(yi, vi, mods)

cat(sprintf("Apparent R²_het: %.2f%%\n", result$r2het * 100))
cat(sprintf("Adjusted R²_het: %.2f%%\n", result$r2het_adj * 100))
cat(sprintf("τ²_null: %.4f\n", result$tau2_null))
cat(sprintf("τ²_full: %.4f\n", result$tau2_full))
cat(sprintf("I²: %.2f%%\n", result$I2))

# ============================================================================
# Example 2: Cross-Validation with Leave-One-Out
# ============================================================================

cat("\n=== Example 2: Cross-Validation (LOO) ===\n")

cv_result <- r2het_cv(yi, vi, mods, verbose = FALSE)

cat(sprintf("Apparent R²_het: %.2f%%\n", cv_result$r2het_apparent * 100))
cat(sprintf("Corrected R²_het: %.2f%%\n", cv_result$r2het_corrected * 100))
cat(sprintf("Optimism: %.2f%%\n", cv_result$optimism * 100))
cat(sprintf("Convergence rate: %.1f%%\n", cv_result$convergence_rate))
cat(sprintf("RMSE: %.4f\n", cv_result$rmse))
cat(sprintf("Correlation: %.3f\n", cv_result$correlation))

# ============================================================================
# Example 3: Bootstrap Confidence Intervals
# ============================================================================

cat("\n=== Example 3: Bootstrap CIs ===\n")

boot_result <- r2het_boot(yi, vi, mods, B = 200, verbose = FALSE)

cat(sprintf("Apparent R²_het CI: [%.2f%%, %.2f%%]\n",
            boot_result$ci_apparent[1] * 100,
            boot_result$ci_apparent[2] * 100))
cat(sprintf("Corrected R²_het CI: [%.2f%%, %.2f%%]\n",
            boot_result$ci_corrected[1] * 100,
            boot_result$ci_corrected[2] * 100))
cat(sprintf("Bootstrap convergence: %.1f%%\n", boot_result$convergence))

# ============================================================================
# Example 4: Complete Overfitting Check
# ============================================================================

cat("\n=== Example 4: Overfitting Assessment ===\n")

check_result <- check_overfitting(yi, vi, mods, B = 200)
print(check_result)

# ============================================================================
# Example 5: Sample Size Recommendations
# ============================================================================

cat("\n=== Example 5: Sample Size Recommendations ===\n")

cat("\nFor 2 parameters (intercept + 1 moderator):\n")
sample_size_recommendation(p = 2, target_optimism = 0.10, verbose = TRUE)

cat("\nFor 4 parameters (intercept + 3 moderators):\n")
sample_size_recommendation(p = 4, target_optimism = 0.05, verbose = TRUE)

# ============================================================================
# Example 6: Formula Interface
# ============================================================================

cat("\n=== Example 6: Formula Interface ===\n")

dat <- data.frame(
  yi = rnorm(25, 0, sqrt(0.1)),
  vi = runif(25, 0.01, 0.1),
  latitude = runif(25, 0, 50),
  year = sample(2000:2020, 25, replace = TRUE)
)

result_formula <- r2het(yi = dat$yi, vi = dat$vi,
                        mods = ~ latitude + year, data = dat)

cat(sprintf("R²_het with formula interface: %.2f%%\n", result_formula$r2het * 100))
cat(sprintf("Number of parameters: %d\n", result_formula$p))

# ============================================================================
# Example 7: Demonstrating Overfitting with Small k
# ============================================================================

cat("\n=== Example 7: Overfitting with Small k ===\n")

# Small k, many parameters scenario
set.seed(456)
k_small <- 12
yi_small <- rnorm(k_small, 0, sqrt(0.1))
vi_small <- runif(k_small, 0.01, 0.1)
mods_small <- cbind(1, rnorm(k_small), rnorm(k_small))  # p=3, k/p=4

check_small <- check_overfitting(yi_small, vi_small, mods_small, B = 100)
print(check_small)

# ============================================================================
# Example 8: K-Fold Cross-Validation
# ============================================================================

cat("\n=== Example 8: K-Fold Cross-Validation ===\n")

cv_kfold <- r2het_cv(yi, vi, mods, cv_method = "kfold", k_folds = 5, verbose = FALSE)

cat(sprintf("5-Fold CV Results:\n"))
cat(sprintf("  Corrected R²_het: %.2f%%\n", cv_kfold$r2het_corrected * 100))
cat(sprintf("  Optimism: %.2f%%\n", cv_kfold$optimism * 100))

# ============================================================================
# Validation Checks
# ============================================================================

cat("\n=== VALIDATION CHECKS ===\n")

# Check 1: Corrected R² should be <= Apparent R²
test1 <- cv_result$r2het_corrected <= cv_result$r2het_apparent
cat(sprintf("Check 1 - Corrected R² <= Apparent R²: %s\n",
            ifelse(test1, "PASS", "FAIL")))

# Check 2: Optimism should be non-negative
test2 <- cv_result$optimism >= 0
cat(sprintf("Check 2 - Optimism >= 0: %s\n", ifelse(test2, "PASS", "FAIL")))

# Check 3: R² values should be in [0, 1]
test3 <- result$r2het >= 0 && result$r2het <= 1
cat(sprintf("Check 3 - R² in [0, 1]: %s\n", ifelse(test3, "PASS", "FAIL")))

# Check 4: CI bounds should be ordered
test4 <- boot_result$ci_apparent[1] <= boot_result$ci_apparent[2]
cat(sprintf("Check 4 - CI ordered correctly: %s\n", ifelse(test4, "PASS", "FAIL")))

# Check 5: Convergence rate should be reasonable
test5 <- cv_result$convergence_rate > 50
cat(sprintf("Check 5 - Convergence > 50%%: %s\n", ifelse(test5, "PASS", "FAIL")))

# Overall
all_pass <- test1 && test2 && test3 && test4 && test5
cat(sprintf("\nOverall: %s\n", ifelse(all_pass, "ALL TESTS PASSED", "SOME TESTS FAILED")))

# ============================================================================
# Optional: Create Plots (requires ggplot2)
# ============================================================================

cat("\n=== Creating Plots ===\n")

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)

  # Bar plot comparing apparent vs corrected
  p1 <- plot_overfitting(cv_result, type = "bar")
  print(p1)

  # Scatter plot of observed vs predicted
  p2 <- plot_overfitting(cv_result, type = "scatter")
  print(p2)

  cat("Plots created successfully.\n")
} else {
  cat("ggplot2 not available - skipping plots.\n")
}

cat("\n=== Demonstration Complete ===\n")
