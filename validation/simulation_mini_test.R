# ============================================================================
# metaoverfit Mini Simulation Test - 5 Iterations (for testing)
# ============================================================================

# Set library path
.libPaths("C:/Users/user/AppData/Local/R/win-library/4.5")

# Load required packages
library(metafor)
library(data.table)

# Load metaoverfit from source
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
devtools::load_all(".")

# Set seed
set.seed(12345)

# Mini configuration
N_SIM <- 5  
OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/simulation_results"

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Test Scenarios
test_scenarios <- data.frame(
  scenario_id = 1:2,
  k = c(10, 30),
  p = c(2, 2),
  tau2 = c(0.05, 0.05),
  beta_mod = c(0.3, 0.3)
)

cat("Mini Test Scenarios:\n")
print(test_scenarios)

# Helper Functions
generate_sim_data <- function(k, tau2, beta_mod, p) {
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k, 0, 1))
  beta <- c(0, beta_mod)
  ui <- rnorm(k, 0, sqrt(tau2))
  yi <- mods %*% beta + ui + rnorm(k, 0, sqrt(vi))
  list(yi = as.vector(yi), vi = vi, mods = mods,
       true_tau2 = tau2, true_beta_mod = beta_mod)
}

run_single_sim <- function(sim_data) {
  tryCatch({
    check <- suppressWarnings(check_overfitting(
      yi = sim_data$yi, vi = sim_data$vi, mods = sim_data$mods,
      B = 10
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

all_results <- list()

for (s in 1:nrow(test_scenarios)) {
  scenario <- test_scenarios[s, ]
  cat(sprintf("Scenario %d: k=%d, p=%d\n", s, scenario$k, scenario$p))
  scenario_results <- vector("list", N_SIM)
  for (i in 1:N_SIM) {
    sim_data <- generate_sim_data(scenario$k, scenario$tau2,
                                   scenario$beta_mod, scenario$p)
    scenario_results[[i]] <- run_single_sim(sim_data)
  }
  scenario_df <- rbindlist(scenario_results)
  all_results[[s]] <- scenario_df
}

final_results <- rbindlist(all_results)
cat("\nSuccess rate: ", mean(final_results$status == "success") * 100, "%\n")
print(final_results)
