# ============================================================================
# metaoverfit Simulation Study - 1000 Iterations
# ============================================================================
# This script conducts a comprehensive simulation study to validate the
# metaoverfit package, with special focus on edge cases.
#
# Scenarios:
# 1. Small k (k = 5, 10, 15, 20)
# 2. k/p ratios (k/p = 2, 3, 5, 10, 15)
# 3. Boundary conditions (tau² = 0, tau² near 0)
# 4. No moderator effect (null scenario)
# 5. Strong moderator effect
#
# Evaluations:
# - Type I error control (false positive rate)
# - Power to detect genuine moderators
# - Coverage of bootstrap CIs
# - Bias in R²_het estimation
# - Convergence rates
#
# Author: Mahmood Ahmad & Claude
# Date: 2026-01-15
# ============================================================================

# Load required packages
library(metafor)
library(data.table)
library(ggplot2)

# Load metaoverfit from source (not installed as a package)
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
devtools::load_all(".")

# Set seed for reproducibility
set.seed(12345)

# ============================================================================
# Configuration
# ============================================================================

# Number of simulations per scenario
N_SIM <- 1000

# Output directory
OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/simulation_results"

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Progress reporting
PROGRESS_INTERVAL <- 100  # Print progress every N simulations

# ============================================================================
# Simulation Scenarios
# ============================================================================

# Define simulation scenarios
scenarios <- expand.grid(
  scenario_id = 1:12,
  k = c(5, 10, 15, 20, 30, 50),
  p = c(2, 3),
  tau2 = c(0, 0.05, 0.1),
  beta_mod = c(0, 0.3),  # 0 = no moderator effect (null), 0.3 = moderate effect
  stringsAsFactors = FALSE
)

# Filter to create meaningful edge case scenarios
scenarios <- scenarios[
  # Small k edge cases
  (scenarios$k == 5 & scenarios$p == 2) |
  (scenarios$k == 10 & scenarios$p == 2) |
  # k/p ratio edge cases
  (scenarios$k / scenarios$p <= 3) |
  (scenarios$k / scenarios$p >= 10) |
  # Boundary conditions
  scenarios$tau2 == 0 |
  # Null scenarios (for Type I error)
  scenarios$beta_mod == 0
  , ]

# Remove duplicates and limit to 12 key scenarios
scenarios <- scenarios[!duplicated(scenarios[, c("k", "p", "tau2", "beta_mod")]), ]
scenarios <- scenarios[1:min(12, nrow(scenarios)), ]
scenarios$scenario_id <- 1:nrow(scenarios)

cat("Simulation Scenarios:\n")
print(scenarios[, c("scenario_id", "k", "p", "tau2", "beta_mod")])
cat("\n")

# ============================================================================
# Helper Functions
# ============================================================================

#' Generate simulated meta-analysis data
#'
#' @param k Number of studies
#' @param tau2 True heterogeneity variance
#' @param beta_mod True moderator coefficient
#' @param p Number of parameters (including intercept)
#' @return List with yi, vi, mods, true_values
generate_sim_data <- function(k, tau2, beta_mod, p) {
  # Sampling variances (moderate variation)
  vi <- runif(k, 0.01, 0.1)

  # Generate moderator(s)
  if (p == 2) {
    # Single moderator
    mods <- cbind(1, rnorm(k, 0, 1))
    beta <- c(0, beta_mod)  # Intercept = 0, moderator effect = beta_mod
  } else {
    # Two moderators
    mods <- cbind(1, rnorm(k, 0, 1), rnorm(k, 0, 1))
    beta <- c(0, beta_mod, 0)  # Only first moderator has effect
  }

  # Generate random effects
  ui <- rnorm(k, 0, sqrt(tau2))

  # True effect sizes (fixed + random + moderator effect)
  yi <- mods %*% beta + ui

  # Add sampling error
  yi <- yi + rnorm(k, 0, sqrt(vi))

  list(
    yi = as.vector(yi),
    vi = vi,
    mods = mods,
    true_tau2 = tau2,
    true_beta_mod = beta_mod,
    true_r2het = if (tau2 > 0 && beta_mod != 0) 0.5 else 0  # Approximate
  )
}

#' Run single simulation iteration
#'
#' @param sim_data Simulated data from generate_sim_data()
#' @return Data frame with simulation results
run_single_sim <- function(sim_data) {
  result <- tryCatch({
    # Run overfitting check
    check <- suppressWarnings(
      check_overfitting(
        yi = sim_data$yi,
        vi = sim_data$vi,
        mods = sim_data$mods,
        B = 100  # Reduced for speed
      )
    )

    # Run bootstrap for CI
    boot <- suppressWarnings(
      r2het_boot(
        yi = sim_data$yi,
        vi = sim_data$vi,
        mods = sim_data$mods,
        B = 100
      )
    )

    # Run CV
    cv <- suppressWarnings(
      r2het_cv(
        yi = sim_data$yi,
        vi = sim_data$vi,
        mods = sim_data$mods
      )
    )

    data.frame(
      # Input parameters
      k = length(sim_data$yi),
      p = ncol(sim_data$mods),
      tau2_true = sim_data$true_tau2,
      beta_mod_true = sim_data$true_beta_mod,

      # R² estimates
      r2_apparent = check$r2het_apparent / 100,
      r2_corrected = check$r2het_corrected / 100,
      optimism = check$actual_optimism / 100,

      # CI coverage (will be evaluated later)
      ci_lower = boot$ci_corrected[1] / 100,
      ci_upper = boot$ci_corrected[2] / 100,

      # Convergence
      convergence = check$convergence_rate,
      cv_convergence = cv$convergence_rate,

      # Risk
      risk_category = check$risk_category,
      k_per_p = check$k_per_p,

      # Status
      status = "success"
    )

  }, error = function(e) {
    data.frame(
      k = length(sim_data$yi),
      p = ncol(sim_data$mods),
      tau2_true = sim_data$true_tau2,
      beta_mod_true = sim_data$true_beta_mod,
      r2_apparent = NA_real_,
      r2_corrected = NA_real_,
      optimism = NA_real_,
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      convergence = NA_real_,
      cv_convergence = NA_real_,
      risk_category = NA_character_,
      k_per_p = NA_real_,
      status = paste("error:", substr(e$message, 1, 30))
    )
  })

  result
}

# ============================================================================
# Main Simulation Loop
# ============================================================================

cat("\n========================================\n")
cat("METAOVERFIT SIMULATION STUDY\n")
cat("========================================\n\n")
cat(sprintf("Total scenarios: %d\n", nrow(scenarios)))
cat(sprintf("Simulations per scenario: %d\n", N_SIM))
cat(sprintf("Total simulations: %d\n\n", nrow(scenarios) * N_SIM))

# Initialize results storage
all_results <- list()

# Run simulations for each scenario
for (s in 1:nrow(scenarios)) {
  scenario <- scenarios[s, ]

  cat(sprintf("\n--- Scenario %d: k=%d, p=%d, tau²=%.3f, beta=%.2f ---\n",
              scenario$scenario_id, scenario$k, scenario$p,
              scenario$tau2, scenario$beta_mod))

  # Pre-allocate results data frame
  scenario_results <- vector("list", N_SIM)

  # Run simulations
  for (i in 1:N_SIM) {
    # Generate data
    sim_data <- generate_sim_data(
      k = scenario$k,
      tau2 = scenario$tau2,
      beta_mod = scenario$beta_mod,
      p = scenario$p
    )

    # Run simulation
    scenario_results[[i]] <- run_single_sim(sim_data)

    # Progress indicator
    if (i %% PROGRESS_INTERVAL == 0) {
      cat(sprintf("  Completed %d/%d simulations\n", i, N_SIM))
    }
  }

  # Combine results
  scenario_df <- rbindlist(scenario_results)
  scenario_df$scenario_id <- scenario$scenario_id

  # Store results
  all_results[[scenario$scenario_id]] <- scenario_df

  # Quick summary
  success_rate <- mean(scenario_df$status == "success") * 100
  mean_optimism <- mean(scenario_df$optimism, na.rm = TRUE) * 100
  cat(sprintf("  Success rate: %.1f%%\n", success_rate))
  cat(sprintf("  Mean optimism: %.2f%%\n", mean_optimism))
}

# Combine all results
final_results <- rbindlist(all_results)

cat("\n========================================\n")
cat("SIMULATION COMPLETE\n")
cat("========================================\n\n")

# ============================================================================
# Analysis of Results
# ============================================================================

cat("=== ANALYSIS OF SIMULATION RESULTS ===\n\n")

# Overall success rate
overall_success <- mean(final_results$status == "success") * 100
cat(sprintf("Overall success rate: %.1f%%\n", overall_success))

# Filter successful results for analysis
success_results <- final_results[final_results$status == "success", ]

if (nrow(success_results) > 0) {

  # 1. R²_het Statistics
  cat("\n--- R²_het Statistics ---\n")
  r2_summary <- success_results[, .(
    mean_apparent = mean(r2_apparent, na.rm = TRUE),
    mean_corrected = mean(r2_corrected, na.rm = TRUE),
    mean_optimism = mean(optimism, na.rm = TRUE),
    sd_apparent = sd(r2_apparent, na.rm = TRUE),
    sd_corrected = sd(r2_corrected, na.rm = TRUE)
  ), by = .(scenario_id, k, p, k_per_p)]
  print(r2_summary)

  # 2. Sanity Checks
  cat("\n--- Sanity Checks ---\n")

  # Check: Corrected R² <= Apparent R²
  check1 <- mean(success_results$r2_corrected <= success_results$r2_apparent, na.rm = TRUE) * 100
  cat(sprintf("Corrected R² <= Apparent R²: %.1f%% of simulations\n", check1))

  # Check: Optimism >= 0
  check2 <- mean(success_results$optimism >= 0, na.rm = TRUE) * 100
  cat(sprintf("Optimism >= 0: %.1f%% of simulations\n", check2))

  # Check: R² in [0, 1]
  check3 <- mean(success_results$r2_apparent >= 0 & success_results$r2_apparent <= 1, na.rm = TRUE) * 100
  cat(sprintf("Apparent R² in [0, 1]: %.1f%% of simulations\n", check3))

  check4 <- mean(success_results$r2_corrected >= 0 & success_results$r2_corrected <= 1, na.rm = TRUE) * 100
  cat(sprintf("Corrected R² in [0, 1]: %.1f%% of simulations\n", check4))

  # 3. Convergence Analysis
  cat("\n--- Convergence Analysis ---\n")
  conv_summary <- success_results[, .(
    mean_convergence = mean(convergence, na.rm = TRUE),
    min_convergence = min(convergence, na.rm = TRUE)
  ), by = .(scenario_id, k, p)]
  print(conv_summary)

  # 4. Edge Case Analysis: Small k
  cat("\n--- Edge Case: Small k (k <= 10) ---\n")
  small_k <- success_results[k <= 10, ]
  if (nrow(small_k) > 0) {
    cat(sprintf("Simulations with k <= 10: %d\n", nrow(small_k)))
    cat(sprintf("Mean optimism: %.3f (SD: %.3f)\n",
                mean(small_k$optimism, na.rm = TRUE),
                sd(small_k$optimism, na.rm = TRUE)))
    cat(sprintf("Convergence rate: %.1f%%\n",
                mean(small_k$convergence, na.rm = TRUE)))
  }

  # 5. Edge Case: Low k/p ratio
  cat("\n--- Edge Case: Low k/p Ratio (< 5) ---\n")
  low_ratio <- success_results[k_per_p < 5, ]
  if (nrow(low_ratio) > 0) {
    cat(sprintf("Simulations with k/p < 5: %d\n", nrow(low_ratio)))
    cat(sprintf("Mean optimism: %.3f (SD: %.3f)\n",
                mean(low_ratio$optimism, na.rm = TRUE),
                sd(low_ratio$optimism, na.rm = TRUE)))
    risk_dist <- table(low_ratio$risk_category)
    cat("Risk category distribution:\n")
    print(risk_dist)
  }

  # 6. Type I Error Analysis (Null scenarios: beta_mod = 0)
  cat("\n--- Type I Error Analysis (Null Scenarios) ---\n")
  null_results <- success_results[beta_mod_true == 0, ]
  if (nrow(null_results) > 0) {
    # Count false positives (apparent R² > 0.10 when true effect = 0)
    false_positive_rate <- mean(null_results$r2_apparent > 0.10, na.rm = TRUE) * 100
    cat(sprintf("False positive rate (apparent R² > 10%%): %.1f%%\n", false_positive_rate))

    # Corrected R² false positives
    false_positive_corr <- mean(null_results$r2_corrected > 0.10, na.rm = TRUE) * 100
    cat(sprintf("False positive rate (corrected R² > 10%%): %.1f%%\n", false_positive_corr))
  }

  # 7. Bootstrap CI Coverage (when true R² ≈ 0)
  cat("\n--- Bootstrap CI Coverage (Null Scenarios) ---\n")
  if (nrow(null_results) > 0 && "ci_lower" %in% names(null_results)) {
    # CI contains 0
    ci_contains_zero <- null_results[!is.na(ci_lower) & !is.na(ci_upper), ]
    ci_contains_zero$contains_zero <- (ci_contains_zero$ci_lower <= 0) &
                                       (ci_contains_zero$ci_upper >= 0)
    coverage_rate <- mean(ci_contains_zero$contains_zero, na.rm = TRUE) * 100
    cat(sprintf("CI contains 0 (null scenarios): %.1f%%\n", coverage_rate))
  }

  # 8. Power Analysis (Alternative scenarios: beta_mod > 0)
  cat("\n--- Power Analysis (Alternative Scenarios) ---\n")
  alt_results <- success_results[beta_mod_true > 0, ]
  if (nrow(alt_results) > 0) {
    # Power: proportion detecting meaningful R²
    power_apparent <- mean(alt_results$r2_apparent > 0.10, na.rm = TRUE) * 100
    power_corrected <- mean(alt_results$r2_corrected > 0.10, na.rm = TRUE) * 100
    cat(sprintf("Power (apparent R² > 10%%): %.1f%%\n", power_apparent))
    cat(sprintf("Power (corrected R² > 10%%): %.1f%%\n", power_corrected))
  }
}

# ============================================================================
# Save Results
# ============================================================================

cat("\n=== SAVING RESULTS ===\n")

# Save full results
output_file <- file.path(OUTPUT_DIR, "simulation_full_results.csv")
write.csv(final_results, output_file, row.names = FALSE)
cat(sprintf("Full results saved to: %s\n", output_file))

# Save summary by scenario
summary_file <- file.path(OUTPUT_DIR, "simulation_summary_by_scenario.csv")
write.csv(r2_summary, summary_file, row.names = FALSE)
cat(sprintf("Scenario summary saved to: %s\n", summary_file))

# Save text summary
text_file <- file.path(OUTPUT_DIR, "simulation_summary.txt")
sink(text_file)
cat("========================================\n")
cat("METAOVERFIT SIMULATION STUDY SUMMARY\n")
cat("========================================\n\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat(sprintf("Total simulations: %d\n", nrow(final_results)))
cat(sprintf("Successful simulations: %d (%.1f%%)\n\n",
            nrow(success_results), overall_success))

cat("Sanity Checks:\n")
cat(sprintf("  Corrected R² <= Apparent R²: %.1f%%\n", check1))
cat(sprintf("  Optimism >= 0: %.1f%%\n", check2))
cat(sprintf("  R² in valid range: %.1f%%\n", check3))

if (exists("false_positive_rate")) {
  cat(sprintf("\nType I Error Rate: %.1f%%\n", false_positive_rate))
}
if (exists("coverage_rate")) {
  cat(sprintf("CI Coverage (null): %.1f%%\n", coverage_rate))
}
if (exists("power_corrected")) {
  cat(sprintf("Power (alternative): %.1f%%\n", power_corrected))
}
sink()

cat(sprintf("Text summary saved to: %s\n", text_file))

# ============================================================================
# Create Visualizations
# ============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

if (nrow(success_results) > 0) {

  # 1. Optimism by k/p ratio
  p1 <- ggplot(success_results, aes(x = k_per_p, y = optimism * 100)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "red") +
    labs(title = "Optimism vs Studies-to-Parameters Ratio",
         x = "k/p Ratio", y = "Optimism (%)") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "optimism_vs_ratio.png"), p1, width = 6, height = 4)

  # 2. Apparent vs Corrected R²
  p2 <- ggplot(success_results, aes(x = r2_apparent * 100, y = r2_corrected * 100)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Apparent vs Corrected R²_het",
         x = "Apparent R²_het (%)", y = "Corrected R²_het (%)") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "apparent_vs_corrected.png"), p2, width = 6, height = 4)

  # 3. Optimism distribution by k
  p3 <- ggplot(success_results, aes(x = factor(k), y = optimism * 100)) +
    geom_boxplot() +
    labs(title = "Optimism Distribution by Number of Studies",
         x = "k (Number of Studies)", y = "Optimism (%)") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "optimism_by_k.png"), p3, width = 6, height = 4)

  # 4. Convergence rate by k/p ratio
  p4 <- ggplot(success_results, aes(x = k_per_p, y = convergence)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "blue") +
    labs(title = "Convergence Rate vs Studies-to-Parameters Ratio",
         x = "k/p Ratio", y = "Convergence Rate (%)") +
    theme_minimal() +
    ylim(0, 100)

  ggsave(file.path(OUTPUT_DIR, "convergence_vs_ratio.png"), p4, width = 6, height = 4)

  cat("Visualizations saved to:", OUTPUT_DIR, "\n")
}

cat("\n========================================\n")
cat("SIMULATION STUDY COMPLETE\n")
cat("========================================\n")
