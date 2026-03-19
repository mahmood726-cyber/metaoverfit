# Generate summary from completed simulation results
library(data.table)

OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/simulation_results"

# Load results
results <- fread(file.path(OUTPUT_DIR, "simulation_full_results.csv"))

cat("=== SIMULATION STUDY RESULTS ===\n\n")
cat(sprintf("Total simulations: %d\n\n", nrow(results)))

# Success rate
success_rate <- mean(results$status == "success") * 100
cat(sprintf("Success rate: %.1f%%\n\n", success_rate))

success_results <- results[status == "success", ]

if (nrow(success_results) > 0) {
  # Sanity checks
  cat("=== SANITY CHECKS ===\n")
  cat(sprintf("Corrected R² <= Apparent R²: %.1f%%\n",
              mean(success_results$r2_corrected <= success_results$r2_apparent, na.rm = TRUE) * 100))
  cat(sprintf("Optimism >= 0: %.1f%%\n",
              mean(success_results$optimism >= 0, na.rm = TRUE) * 100))
  cat(sprintf("R² in [0, 1]: %.1f%%\n\n",
              mean(success_results$r2_apparent >= 0 & success_results$r2_apparent <= 1, na.rm = TRUE) * 100))

  # Summary by scenario
  cat("=== RESULTS BY SCENARIO ===\n")
  summary_by_scen <- success_results[, .(
    n_sims = .N,
    mean_r2_apparent = mean(r2_apparent, na.rm = TRUE) * 100,
    mean_r2_corrected = mean(r2_corrected, na.rm = TRUE) * 100,
    mean_optimism = mean(optimism, na.rm = TRUE) * 100,
    mean_convergence = mean(convergence, na.rm = TRUE),
    sd_optimism = sd(optimism, na.rm = TRUE) * 100
  ), by = .(scenario_id, k, p, k_per_p, tau2_true, beta_mod_true)]
  print(summary_by_scen)

  # Small k analysis
  cat("\n=== SMALL K ANALYSIS ===\n")
  small_k <- success_results[k < 10, ]
  if (nrow(small_k) > 0) {
    cat(sprintf("k < 10 (n=%d): Mean optimism = %.2f%% (SD: %.2f%%)\n",
                nrow(small_k),
                mean(small_k$optimism, na.rm = TRUE) * 100,
                sd(small_k$optimism, na.rm = TRUE) * 100))
  }

  # k/p ratio analysis
  cat("\n=== K/P RATIO ANALYSIS ===\n")
  low_ratio <- success_results[k_per_p < 5, ]
  med_ratio <- success_results[k_per_p >= 5 & k_per_p < 10, ]
  high_ratio <- success_results[k_per_p >= 10, ]
  cat(sprintf("k/p < 5 (n=%d): Mean optimism = %.2f%%\n",
              nrow(low_ratio), mean(low_ratio$optimism, na.rm = TRUE) * 100))
  cat(sprintf("5 <= k/p < 10 (n=%d): Mean optimism = %.2f%%\n",
              nrow(med_ratio), mean(med_ratio$optimism, na.rm = TRUE) * 100))
  cat(sprintf("k/p >= 10 (n=%d): Mean optimism = %.2f%%\n",
              nrow(high_ratio), mean(high_ratio$optimism, na.rm = TRUE) * 100))

  # Null scenarios (Type I error)
  cat("\n=== NULL SCENARIOS (Type I Error) ===\n")
  null_results <- success_results[beta_mod_true == 0, ]
  if (nrow(null_results) > 0) {
    false_positive <- mean(null_results$r2_apparent > 0.10, na.rm = TRUE) * 100
    cat(sprintf("False positive rate (apparent R² > 10%%): %.1f%%\n", false_positive))
  }

  # Alternative scenarios (Power)
  cat("\n=== ALTERNATIVE SCENARIOS (Power) ===\n")
  alt_results <- success_results[beta_mod_true > 0, ]
  if (nrow(alt_results) > 0) {
    power <- mean(alt_results$r2_corrected > 0.10, na.rm = TRUE) * 100
    cat(sprintf("Power (corrected R² > 10%%): %.1f%%\n", power))
  }
}

# Save summary
summary_file <- file.path(OUTPUT_DIR, "simulation_summary_by_scenario.csv")
write.csv(summary_by_scen, summary_file, row.names = FALSE)
cat(sprintf("\nSummary saved to: %s\n", summary_file))

cat("\n=== SIMULATION COMPLETE ===\n")
