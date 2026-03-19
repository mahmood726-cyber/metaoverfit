# ============================================================================
# metaoverfit Quick Simulation Test - 100 Iterations (for testing)
# ============================================================================
# This is a fast version of the simulation study for quick validation.
# Runs 100 iterations instead of 1000.
#
# Run this first to verify the code works before running the full study.
# ============================================================================

# Set library path for data.table
.libPaths("C:/Users/user/AppData/Local/R/win-library/4.5")

# Load required packages
library(metafor)
library(data.table)

# Load metaoverfit from source (not installed as a package)
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
devtools::load_all(".")

# Set seed
set.seed(12345)

# Quick configuration
N_SIM <- 100  # Quick test version
OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/simulation_results"

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Test Scenarios (Edge Cases)
# ============================================================================

test_scenarios <- data.frame(
  scenario_id = 1:8,
  k = c(5, 10, 15, 20, 10, 10, 30, 50),
  p = c(2, 2, 2, 2, 3, 2, 2, 2),
  tau2 = c(0.05, 0.05, 0.05, 0, 0.05, 0.1, 0.05, 0.05),
  beta_mod = c(0.3, 0.3, 0.3, 0.3, 0, 0.3, 0.3, 0.3)  # 0 = null scenario
)

cat("Quick Test Scenarios:\n")
print(test_scenarios)

# ============================================================================
# Helper Functions
# ============================================================================

generate_sim_data <- function(k, tau2, beta_mod, p) {
  vi <- runif(k, 0.01, 0.1)

  if (p == 2) {
    mods <- cbind(1, rnorm(k, 0, 1))
    beta <- c(0, beta_mod)
  } else {
    mods <- cbind(1, rnorm(k, 0, 1), rnorm(k, 0, 1))
    beta <- c(0, beta_mod, 0)
  }

  ui <- rnorm(k, 0, sqrt(tau2))
  yi <- mods %*% beta + ui + rnorm(k, 0, sqrt(vi))

  list(yi = as.vector(yi), vi = vi, mods = mods,
       true_tau2 = tau2, true_beta_mod = beta_mod)
}

run_single_sim <- function(sim_data) {
  tryCatch({
    check <- suppressWarnings(check_overfitting(
      yi = sim_data$yi, vi = sim_data$vi, mods = sim_data$mods,
      B = 50
    ))

    data.frame(
      k = length(sim_data$yi),
      p = ncol(sim_data$mods),
      tau2_true = sim_data$true_tau2,
      beta_mod_true = sim_data$true_beta_mod,
      r2_apparent = check$r2het_apparent / 100,
      r2_corrected = check$r2het_corrected / 100,
      optimism = check$actual_optimism / 100,
      convergence = check$convergence_rate,
      risk_category = check$risk_category,
      k_per_p = check$k_per_p,
      status = "success"
    )
  }, error = function(e) {
    data.frame(
      k = length(sim_data$yi), p = ncol(sim_data$mods),
      tau2_true = sim_data$true_tau2, beta_mod_true = sim_data$true_beta_mod,
      r2_apparent = NA_real_, r2_corrected = NA_real_, optimism = NA_real_,
      convergence = NA_real_, risk_category = NA_character_,
      k_per_p = NA_real_, status = "error"
    )
  })
}

# ============================================================================
# Run Quick Test
# ============================================================================

cat("\n========================================\n")
cat("METAOVERFIT QUICK SIMULATION TEST\n")
cat("========================================\n\n")
cat(sprintf("Scenarios: %d\n", nrow(test_scenarios)))
cat(sprintf("Iterations per scenario: %d\n", N_SIM))
cat(sprintf("Total simulations: %d\n\n", nrow(test_scenarios) * N_SIM))

all_results <- list()

for (s in 1:nrow(test_scenarios)) {
  scenario <- test_scenarios[s, ]
  cat(sprintf("Scenario %d: k=%d, p=%d, tau²=%.2f, beta=%.1f\n",
              s, scenario$k, scenario$p, scenario$tau2, scenario$beta_mod))

  scenario_results <- vector("list", N_SIM)

  for (i in 1:N_SIM) {
    sim_data <- generate_sim_data(scenario$k, scenario$tau2,
                                   scenario$beta_mod, scenario$p)
    scenario_results[[i]] <- run_single_sim(sim_data)
  }

  scenario_df <- rbindlist(scenario_results)
  scenario_df$scenario_id <- s
  all_results[[s]] <- scenario_df
}

final_results <- rbindlist(all_results)

# ============================================================================
# Quick Summary
# ============================================================================

cat("\n=== QUICK TEST RESULTS ===\n\n")

success_rate <- mean(final_results$status == "success") * 100
cat(sprintf("Success rate: %.1f%%\n\n", success_rate))

success_results <- final_results[final_results$status == "success", ]

if (nrow(success_results) > 0) {
  # Sanity checks
  cat("Sanity Checks:\n")
  cat(sprintf("  Corrected R² <= Apparent R²: %.1f%%\n",
              mean(success_results$r2_corrected <= success_results$r2_apparent, na.rm = TRUE) * 100))
  cat(sprintf("  Optimism >= 0: %.1f%%\n",
              mean(success_results$optimism >= 0, na.rm = TRUE) * 100))
  cat(sprintf("  R² in [0, 1]: %.1f%%\n\n",
              mean(success_results$r2_apparent >= 0 & success_results$r2_apparent <= 1, na.rm = TRUE) * 100))

  # Summary by scenario
  cat("Results by Scenario:\n")
  summary_by_scen <- success_results[, .(
    n_sims = .N,
    mean_optimism = mean(optimism, na.rm = TRUE) * 100,
    mean_convergence = mean(convergence, na.rm = TRUE)
  ), by = .(scenario_id, k, p, k_per_p, beta_mod_true)]
  print(summary_by_scen)

  # Edge case: Small k
  cat("\nEdge Case (k <= 10):\n")
  small_k <- success_results[k <= 10, ]
  if (nrow(small_k) > 0) {
    cat(sprintf("  Mean optimism: %.2f%% (SD: %.2f%%)\n",
                mean(small_k$optimism, na.rm = TRUE) * 100,
                sd(small_k$optimism, na.rm = TRUE) * 100))
  }

  # Edge case: Null scenarios (Type I error)
  cat("\nNull Scenarios (Type I Error):\n")
  null_results <- success_results[beta_mod_true == 0, ]
  if (nrow(null_results) > 0) {
    cat(sprintf("  False positive rate: %.1f%%\n",
                mean(null_results$r2_apparent > 0.10, na.rm = TRUE) * 100))
  }
}

# Save results
output_file <- file.path(OUTPUT_DIR, "quick_test_results.csv")
write.csv(final_results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_file))

cat("\nQuick test complete! If results look good, run the full simulation study.\n")
