# ============================================================================
# metaoverfit Package Validation with 501 Cochrane Datasets
# ============================================================================
# This script validates the metaoverfit package using real Cochrane meta-analysis
# data. It performs comprehensive validation including:
# 1. Cross-validation optimism correction
# 2. Bootstrap confidence intervals
# 3. Overfitting risk assessment
# 4. Statistical sanity checks
#
# Author: Mahmood Ahmad & Claude
# Date: 2026-01-15
# ============================================================================

# Suppress warnings for cleaner output (uncomment for debugging)
# options(warn = 1)

# Load required packages
library(metafor)
library(data.table)
library(ggplot2)

# Load metaoverfit package (assuming it's installed)
# devtools::load_all("path/to/metaoverfit")
library(metaoverfit)

# ============================================================================
# Configuration
# ============================================================================

# Path to Cochrane datasets
DATA_DIR <- "C:/Users/user/OneDrive - NHS/Documents/Pairwise70/analysis/output/cleaned_rds"

# Output directory
OUTPUT_DIR <- "C:/Users/user/OneDrive - NHS/Documents/metaoverfit/validation/results"

# Bootstrap samples for CI (increase for final validation)
B_BOOTSTRAP <- 200

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Helper Functions
# ============================================================================

#' Extract meta-analysis data from Cochrane dataset
#'
#' @param dat The loaded dataset
#' @param outcome_col Column name for the outcome identifier
#' @return A list with yi, vi, and potential moderators
extract_ma_data <- function(dat, outcome_col = NULL) {
  # Standard Cochrane column names
  yi_col <- NULL
  vi_col <- NULL

  # Find effect size column (log risk ratio or similar)
  possible_yi <- c("yi", "log_risk_ratio", "log_odds_ratio", "log_hazard_ratio",
                   "effect_size", "es", "md", "smd")
  for (col in possible_yi) {
    if (col %in% names(dat)) {
      yi_col <- col
      break
    }
  }

  # Find variance column
  possible_vi <- c("vi", "variance", "se", "sei")
  for (col in possible_vi) {
    if (col %in% names(dat)) {
      if (col == "se" || col == "sei") {
        # Convert SE to variance
        dat$vi_computed <- dat[[col]]^2
        vi_col <- "vi_computed"
      } else {
        vi_col <- col
      }
      break
    }
  }

  # If no yi found, try to compute from 2x2 table data
  if (is.null(yi_col)) {
    if (all(c("a", "c", "b", "d") %in% names(dat))) {
      # Compute log risk ratio
      dat$yi_computed <- metafor::escalc(
        measure = "RR",
        ai = a, bi = b,
        ci = c, di = d,
        data = dat
      )$yi
      dat$vi_computed <- metafor::escalc(
        measure = "RR",
        ai = a, bi = b,
        ci = c, di = d,
        data = dat
      )$vi
      yi_col <- "yi_computed"
      vi_col <- "vi_computed"
    }
  }

  if (is.null(yi_col) || is.null(vi_col)) {
    return(NULL)
  }

  # Remove rows with missing values
  complete_rows <- complete.cases(dat[[yi_col]], dat[[vi_col]])
  dat_clean <- dat[complete_rows, ]

  # Identify potential moderators (numeric columns with variation)
  moderator_cols <- character(0)
  numeric_cols <- names(dat_clean)[sapply(dat_clean, is.numeric)]

  for (col in numeric_cols) {
    if (col %in% c(yi_col, vi_col, "yi_computed", "vi_computed")) next
    if (length(unique(dat_clean[[col]])) > 2 && var(dat_clean[[col]], na.rm = TRUE) > 0) {
      moderator_cols <- c(moderator_cols, col)
    }
  }

  list(
    yi = dat_clean[[yi_col]],
    vi = dat_clean[[vi_col]],
    moderators = moderator_cols,
    data = dat_clean,
    n_studies = nrow(dat_clean)
  )
}

#' Run validation on a single dataset
#'
#' @param file_path Path to the RDS file
#' @param file_name Name of the file
#' @return A data frame with validation results
validate_single_dataset <- function(file_path, file_name) {
  cat(sprintf("Processing: %s\n", file_name))

  result <- tryCatch({
    # Load data
    dat <- readRDS(file_path)

    # Extract meta-analysis data
    ma_data <- extract_ma_data(dat)

    if (is.null(ma_data) || ma_data$n_studies < 5) {
      return(data.frame(
        file_name = file_name,
        n_studies = ifelse(is.null(ma_data), 0, ma_data$n_studies),
        status = "skipped",
        k_per_p = NA,
        risk_category = NA,
        r2het_apparent = NA,
        r2het_corrected = NA,
        optimism = NA,
        convergence_rate = NA,
        ci_lower = NA,
        ci_upper = NA
      ))
    }

    yi <- ma_data$yi
    vi <- ma_data$vi
    k <- length(yi)

    # Select first available moderator for meta-regression
    mods <- NULL
    if (length(ma_data$moderators) > 0) {
      mod_col <- ma_data$moderators[1]
      mods <- cbind(1, ma_data$data[[mod_col]])
      p <- ncol(mods)
    } else {
      # No suitable moderator - use sample size as moderator if available
      if ("n1" %in% names(ma_data$data) || "n2" %in% names(ma_data$data)) {
        if ("n1" %in% names(ma_data$data) && "n2" %in% names(ma_data$data)) {
          total_n <- ma_data$data$n1 + ma_data$data$n2
          mods <- cbind(1, log(total_n))
          p <- 2
        }
      } else {
        p <- 1
      }
    }

    if (is.null(mods) || k <= p + 1) {
      return(data.frame(
        file_name = file_name,
        n_studies = k,
        status = "insufficient_data",
        k_per_p = NA,
        risk_category = NA,
        r2het_apparent = NA,
        r2het_corrected = NA,
        optimism = NA,
        convergence_rate = NA,
        ci_lower = NA,
        ci_upper = NA
      ))
    }

    # Run overfitting check
    check_result <- suppressWarnings(
      check_overfitting(
        yi = yi,
        vi = vi,
        mods = mods,
        B = B_BOOTSTRAP
      )
    )

    # Get bootstrap CIs
    boot_result <- suppressWarnings(
      r2het_boot(
        yi = yi,
        vi = vi,
        mods = mods,
        B = B_BOOTSTRAP,
        verbose = FALSE
      )
    )

    data.frame(
      file_name = file_name,
      n_studies = k,
      status = "success",
      k_per_p = check_result$k_per_p,
      risk_category = check_result$risk_category,
      r2het_apparent = check_result$r2het_apparent,
      r2het_corrected = check_result$r2het_corrected,
      optimism = check_result$actual_optimism,
      convergence_rate = check_result$convergence_rate,
      ci_lower = boot_result$ci_corrected[1],
      ci_upper = boot_result$ci_corrected[2]
    )

  }, error = function(e) {
    data.frame(
      file_name = file_name,
      n_studies = NA,
      status = paste("error:", substr(e$message, 1, 50)),
      k_per_p = NA,
      risk_category = NA,
      r2het_apparent = NA,
      r2het_corrected = NA,
      optimism = NA,
      convergence_rate = NA,
      ci_lower = NA,
      ci_upper = NA
    )
  })

  result
}

# ============================================================================
# Main Validation Process
# ============================================================================

cat("========================================\n")
cat("metaoverfit Validation with Cochrane Data\n")
cat("========================================\n\n")

# Get all RDS files
rds_files <- list.files(DATA_DIR, pattern = "\\.rds$", full.names = TRUE)

cat(sprintf("Found %d datasets\n\n", length(rds_files)))

# Initialize results storage
all_results <- data.frame()

# Process each dataset
for (i in seq_along(rds_files)) {
  file_path <- rds_files[i]
  file_name <- basename(file_path)

  result <- validate_single_dataset(file_path, file_name)
  all_results <- rbind(all_results, result)

  # Progress indicator
  if (i %% 50 == 0) {
    cat(sprintf("Processed %d/%d datasets\n", i, length(rds_files)))
  }
}

cat(sprintf("\nCompleted: %d datasets\n", nrow(all_results)))

# ============================================================================
# Summary Statistics
# ============================================================================

cat("\n========================================\n")
cat("VALIDATION SUMMARY\n")
cat("========================================\n\n")

# Status breakdown
status_counts <- table(all_results$status)
cat("Status Breakdown:\n")
print(status_counts)
cat("\n")

# Successful analyses
success_results <- all_results[all_results$status == "success", ]

if (nrow(success_results) > 0) {
  cat(sprintf("Successfully analyzed: %d datasets\n\n", nrow(success_results)))

  # Risk category distribution
  risk_dist <- table(success_results$risk_category)
  cat("Risk Category Distribution:\n")
  print(risk_dist)
  cat("\n")

  # Statistics
  cat("R²_het Statistics:\n")
  cat(sprintf("  Apparent R²_het:  Mean=%.2f%%, Median=%.2f%%, Range=[%.2f%%, %.2f%%]\n",
              mean(success_results$r2het_apparent, na.rm = TRUE),
              median(success_results$r2het_apparent, na.rm = TRUE),
              min(success_results$r2het_apparent, na.rm = TRUE),
              max(success_results$r2het_apparent, na.rm = TRUE)))

  cat(sprintf("  Corrected R²_het:  Mean=%.2f%%, Median=%.2f%%, Range=[%.2f%%, %.2f%%]\n",
              mean(success_results$r2het_corrected, na.rm = TRUE),
              median(success_results$r2het_corrected, na.rm = TRUE),
              min(success_results$r2het_corrected, na.rm = TRUE),
              max(success_results$r2het_corrected, na.rm = TRUE)))

  cat(sprintf("  Optimism:          Mean=%.2f%%, Median=%.2f%%, Range=[%.2f%%, %.2f%%]\n\n",
              mean(success_results$optimism, na.rm = TRUE),
              median(success_results$optimism, na.rm = TRUE),
              min(success_results$optimism, na.rm = TRUE),
              max(success_results$optimism, na.rm = TRUE)))

  # Convergence
  cat(sprintf("Convergence Rate: Mean=%.1f%%, Median=%.1f%%\n\n",
              mean(success_results$convergence_rate, na.rm = TRUE),
              median(success_results$convergence_rate, na.rm = TRUE)))

  # Sanity checks
  cat("Sanity Checks:\n")
  cat(sprintf("  Corrected R² <= Apparent R²: %d/%d (%.1f%%)\n",
              sum(success_results$r2het_corrected <= success_results$r2het_apparent, na.rm = TRUE),
              nrow(success_results),
              100 * mean(success_results$r2het_corrected <= success_results$r2het_apparent, na.rm = TRUE)))

  cat(sprintf("  Optimism >= 0: %d/%d (%.1f%%)\n",
              sum(success_results$optimism >= 0, na.rm = TRUE),
              nrow(success_results),
              100 * mean(success_results$optimism >= 0, na.rm = TRUE)))

  cat(sprintf("  R² values in [0, 100]: %d/%d (%.1f%%)\n",
              sum(success_results$r2het_apparent >= 0 & success_results$r2het_apparent <= 100, na.rm = TRUE),
              nrow(success_results),
              100 * mean(success_results$r2het_apparent >= 0 & success_results$r2het_apparent <= 100, na.rm = TRUE)))

  # CI coverage
  valid_ci <- success_results[!is.na(success_results$ci_lower) & !is.na(success_results$ci_upper), ]
  if (nrow(valid_ci) > 0) {
    cat(sprintf("  Valid CIs: %d/%d (%.1f%%)\n",
                nrow(valid_ci),
                nrow(success_results),
                100 * nrow(valid_ci) / nrow(success_results)))
  }
}

# ============================================================================
# Save Results
# ============================================================================

# Save full results
output_file <- file.path(OUTPUT_DIR, "validation_results.csv")
write.csv(all_results, output_file, row.names = FALSE)
cat(sprintf("\nFull results saved to: %s\n", output_file))

# Save summary
summary_file <- file.path(OUTPUT_DIR, "validation_summary.txt")
sink(summary_file)
cat("========================================\n")
cat("metaoverfit Package Validation Summary\n")
cat("========================================\n\n")
cat(sprintf("Date: %s\n", Sys.Date()))
cat(sprintf("Datasets analyzed: %d\n", length(rds_files)))
cat(sprintf("Successful analyses: %d\n", nrow(success_results)))
cat("\n--- Risk Category Distribution ---\n")
print(risk_dist)
sink()

cat(sprintf("Summary saved to: %s\n", summary_file))

# ============================================================================
# Create Visualizations
# ============================================================================

if (nrow(success_results) > 0) {
  # Risk category distribution plot
  p1 <- ggplot(success_results, aes(x = risk_category)) +
    geom_bar(fill = "steelblue") +
    labs(title = "Overfitting Risk Category Distribution",
         x = "Risk Category", y = "Count") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "risk_distribution.png"), p1, width = 6, height = 4)

  # Optimism vs k/p ratio
  p2 <- ggplot(success_results, aes(x = k_per_p, y = optimism, color = risk_category)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE, color = "black") +
    labs(title = "Optimism vs Studies-to-Parameters Ratio",
         x = "k/p ratio", y = "Optimism (%)", color = "Risk") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "optimism_vs_ratio.png"), p2, width = 6, height = 4)

  # Apparent vs Corrected R²
  p3 <- ggplot(success_results, aes(x = r2het_apparent, y = r2het_corrected)) +
    geom_point(alpha = 0.6, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Apparent vs Corrected R²_het",
         x = "Apparent R²_het (%)", y = "Corrected R²_het (%)") +
    theme_minimal()

  ggsave(file.path(OUTPUT_DIR, "apparent_vs_corrected.png"), p3, width = 6, height = 4)

  cat("Visualizations saved to:", OUTPUT_DIR, "\n")
}

cat("\n========================================\n")
cat("VALIDATION COMPLETE\n")
cat("========================================\n")
