# ============================================================================
# metaoverfit Simulation Study - Parallel Execution
# ============================================================================
# This script runs the simulation study using the installed metaoverfit package
# with parallel processing enabled.

.libPaths(c("C:/Users/user/AppData/Local/R/win-library/4.5", .libPaths()))
print(.libPaths())

# Load required packages
library(metafor)
library(data.table)
library(ggplot2)
library(metaoverfit) # Load installed package

# Set seed for reproducibility
set.seed(12345)

# ============================================================================
# Configuration
# ============================================================================

# Number of simulations per scenario
# Using 100 for demonstration purposes in this session. 
# For full validation, set to 1000.
N_SIM <- 100 

# Output directory
OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/simulation_results_parallel"

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Progress reporting
PROGRESS_INTERVAL <- 10

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

cat("Simulation Scenarios:
")
print(scenarios[, c("scenario_id", "k", "p", "tau2", "beta_mod")])
cat("
")

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
  list(
    yi = as.vector(yi),
    vi = vi,
    mods = mods,
    true_tau2 = tau2,
    true_beta_mod = beta_mod,
    true_r2het = if (tau2 > 0 && beta_mod != 0) 0.5 else 0
  )
}

run_single_sim <- function(sim_data) {
  result <- tryCatch({
    # Run overfitting check with PARALLEL = TRUE
    check <- suppressWarnings(
      check_overfitting(
        yi = sim_data$yi,
        vi = sim_data$vi,
        mods = sim_data$mods,
        B = 100,  # Bootstrap samples
        parallel = TRUE # Enable parallel processing
      )
    )

    # Note: we don't need to run r2het_boot separately as check_overfitting does it.
    
    # Run CV (optional, if we want raw CV details not in check output, but check has most)
    # keeping it simple using check output
    
    data.frame(
      k = length(sim_data$yi),
      p = ncol(sim_data$mods),
      tau2_true = sim_data$true_tau2,
      beta_mod_true = sim_data$true_beta_mod,
      r2_apparent = check$r2het_apparent / 100,
      r2_corrected = check$r2het_corrected / 100,
      optimism = check$actual_optimism / 100,
      ci_lower = check$ci_corrected[1] / 100,
      ci_upper = check$ci_corrected[2] / 100,
      convergence = check$convergence_rate,
      risk_category = check$risk_category,
      k_per_p = check$k_per_p,
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

cat("
========================================
")
cat("METAOVERFIT PARALLEL SIMULATION STUDY
")
cat("========================================

")
cat(sprintf("Total scenarios: %d
", nrow(scenarios)))
cat(sprintf("Simulations per scenario: %d
", N_SIM))
cat(sprintf("Total simulations: %d

", nrow(scenarios) * N_SIM))

all_results <- list()

for (s in 1:nrow(scenarios)) {
  scenario <- scenarios[s, ]
  cat(sprintf("
--- Scenario %d: k=%d, p=%d, tau²=%.3f, beta=%.2f ---
",
              scenario$scenario_id, scenario$k, scenario$p,
              scenario$tau2, scenario$beta_mod))

  scenario_results <- vector("list", N_SIM)

  for (i in 1:N_SIM) {
    sim_data <- generate_sim_data(
      k = scenario$k,
      tau2 = scenario$tau2,
      beta_mod = scenario$beta_mod,
      p = scenario$p
    )
    scenario_results[[i]] <- run_single_sim(sim_data)
    if (i %% PROGRESS_INTERVAL == 0) cat(".")
  }
  cat("
")

  scenario_df <- rbindlist(scenario_results)
  scenario_df$scenario_id <- scenario$scenario_id
  all_results[[scenario$scenario_id]] <- scenario_df
}

final_results <- rbindlist(all_results)

# Save results
output_file <- file.path(OUTPUT_DIR, "simulation_parallel_results.csv")
write.csv(final_results, output_file, row.names = FALSE)
cat(sprintf("
Full results saved to: %s
", output_file))

cat("
Simulation completed successfully.
")
