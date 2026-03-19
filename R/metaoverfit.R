#' metaoverfit: Optimism-Corrected Heterogeneity in Meta-Regression
#'
#' Tools for detecting and correcting overfitting in meta-regression analyses
#' through cross-validation and bootstrap methods.
#'
#' @keywords internal
"_PACKAGE"

# Imports needed by this package (generate proper NAMESPACE)
#' @importFrom stats coef cor model.matrix quantile sd
#' @importFrom utils txtProgressBar setTxtProgressBar
NULL

# Silence NOTES about non-standard evaluation column names used in ggplot2
if (getRversion() >= "2.15.1") {
  utils::globalVariables(c("Type", "R2het", "Predicted", "Observed"))
}

# ==============================
# Core functions
# ==============================

#' Calculate Apparent R-squared for Heterogeneity
#'
#' @param yi Vector of effect sizes.
#' @param vi Vector of sampling variances.
#' @param mods Model matrix or formula for moderators. If a matrix, include intercept if desired.
#' @param data Optional data frame (required if `mods` is a formula).
#' @param method Heterogeneity estimator (default: "REML").
#'
#' @return A list with class "r2het" containing:
#' \itemize{
#'   \item \code{r2het} Apparent R^2_het (numeric).
#'   \item \code{r2het_adj} Adjusted R^2_het with small-sample penalty (numeric).
#'   \item \code{tau2_null} Estimated tau^2 from null model (numeric).
#'   \item \code{tau2_full} Estimated tau^2 from full model (numeric).
#'   \item \code{I2} I^2 from the null model (numeric).
#'   \item \code{k} Number of studies (integer).
#'   \item \code{p} Number of parameters (integer).
#'   \item \code{model_null} \code{metafor::rma} fit for null model.
#'   \item \code{model_full} \code{metafor::rma} fit for full model.
#' }
#' @export
#'
#' @examples
#' k <- 30
#' yi <- rnorm(k, 0, sqrt(0.1))
#' vi <- runif(k, 0.01, 0.1)
#' mods <- cbind(1, rnorm(k))
#' r2het(yi, vi, mods)
r2het <- function(yi, vi, mods = NULL, data = NULL, method = "REML") {
  if (length(yi) != length(vi)) {
    stop("yi and vi must have the same length")
  }

  k <- length(yi)

  # Validate sample size
  if (k < 5) {
    warning("Fewer than 5 studies - R²_het estimates are highly unreliable")
  } else if (k < 10) {
    warning("Fewer than 10 studies - R²_het estimates should be interpreted with caution")
  }

  # Fit null model
  fit_null <- metafor::rma(yi = yi, vi = vi, data = data, method = method)
  tau2_null <- fit_null$tau2
  I2 <- fit_null$I2

  # Warn if tau² is at boundary
  if (tau2_null == 0) {
    warning("Null model τ² = 0 (heterogeneity estimate at boundary); R²_het may be unreliable")
  }

  # If no moderators, return trivial
  if (is.null(mods)) {
    return(list(
      r2het = 0,
      r2het_adj = 0,
      tau2_null = tau2_null,
      tau2_full = tau2_null,
      I2 = I2,
      k = length(yi),
      p = 1,
      model_null = fit_null,
      model_full = fit_null
    ))
  }

  # Fit full model (formula vs matrix)
  if (inherits(mods, "formula")) {
    fit_full <- metafor::rma(yi = yi, vi = vi, mods = mods, data = data, method = method)
    p <- length(coef(fit_full))
  } else {
    mods <- as.matrix(mods)
    fit_full <- metafor::rma(yi = yi, vi = vi, mods = ~ mods - 1, data = data, method = method)
    p <- ncol(mods)
  }

  tau2_full <- fit_full$tau2

  # Warn if tau² is at boundary in full model
  if (tau2_full < 1e-10 && tau2_null > 1e-10) {
    warning("Full model τ² = 0; may indicate overfitting or insufficient heterogeneity")
  }

  # Apparent R^2_het
  r2het_val <- if (tau2_null > 0) max(0, 1 - tau2_full / tau2_null) else 0

  # Adjusted R^2_het
  r2het_adj <- if (tau2_null > 0 && k > p) max(0, 1 - (tau2_full / tau2_null) * ((k - 1) / (k - p))) else 0

  result <- list(
    r2het = r2het_val,
    r2het_adj = r2het_adj,
    tau2_null = tau2_null,
    tau2_full = tau2_full,
    I2 = I2,
    k = k,
    p = p,
    model_null = fit_null,
    model_full = fit_full
  )
  class(result) <- c("r2het", "list")
  result
}

#' Cross-Validated R-squared for Heterogeneity
#'
#' @param yi Vector of effect sizes.
#' @param vi Vector of sampling variances.
#' @param mods Model matrix or formula for moderators.
#' @param data Optional data frame.
#' @param method Heterogeneity estimator (default: "REML").
#' @param cv_method "loo" for leave-one-out or "kfold" for k-fold CV.
#' @param k_folds Number of folds for k-fold CV (default: 5).
#' @param verbose Show progress bar.
#'
#' @return A list with class "r2het_cv" containing:
#' \itemize{
#'   \item \code{r2het_apparent} Apparent R^2_het (numeric).
#'   \item \code{r2het_corrected} Cross-validated R^2_het (numeric).
#'   \item \code{optimism} Difference between apparent and corrected (numeric).
#'   \item \code{tau2_null} Tau^2 from null model (numeric).
#'   \item \code{tau2_cv} Cross-validated tau^2 estimate (numeric).
#'   \item \code{convergence_rate} Percentage of successful CV folds (numeric).
#'   \item \code{rmse} Root mean squared error of predictions (numeric).
#'   \item \code{mae} Mean absolute error of predictions (numeric).
#'   \item \code{correlation} Correlation between observed and predicted (numeric).
#'   \item \code{predictions} Vector of cross-validated predictions.
#'   \item \code{yi} Original effect sizes for reference.
#' }
#' @export
#'
#' @examples
#' \donttest{
#' k <- 30
#' yi <- rnorm(k, 0, sqrt(0.1))
#' vi <- runif(k, 0.01, 0.1)
#' mods <- cbind(1, rnorm(k))
#' result <- r2het_cv(yi, vi, mods, verbose = FALSE)
#' round(result$optimism * 100, 1)
#' }
r2het_cv <- function(yi, vi, mods = NULL, data = NULL, method = "REML",
                     cv_method = "loo", k_folds = 5, verbose = TRUE) {
  k <- length(yi)

  # Apparent
  apparent <- r2het(yi, vi, mods, data, method)

  if (is.null(mods)) {
    return(list(
      r2het_apparent = 0,
      r2het_corrected = 0,
      optimism = 0,
      tau2_null = apparent$tau2_null,
      tau2_cv = apparent$tau2_null,
      convergence_rate = 100,
      rmse = NA_real_,
      mae = NA_real_,
      correlation = NA_real_,
      predictions = rep(NA_real_, k),
      yi = yi  # include observed for plotting completeness
    ))
  }

  # Build moderator matrix
  if (inherits(mods, "formula")) {
    if (is.null(data)) stop("Data must be provided when using a formula for 'mods'.")
    mods_matrix <- model.matrix(mods, data)
  } else {
    mods_matrix <- as.matrix(mods)
  }
  p <- ncol(mods_matrix)

  if (k <= p + 1) {
    warning("Too few studies for stable cross-validation (k <= p+1)")
    return(list(
      r2het_apparent = apparent$r2het,
      r2het_corrected = 0,
      optimism = apparent$r2het,
      tau2_null = apparent$tau2_null,
      tau2_cv = NA_real_,
      convergence_rate = 0,
      rmse = NA_real_,
      mae = NA_real_,
      correlation = NA_real_,
      predictions = rep(NA_real_, k),
      yi = yi
    ))
  }

  # CV predictions
  yhat_cv <- rep(NA_real_, k)
  converged <- rep(FALSE, k)

  if (cv_method == "loo") {
    if (verbose) pb <- txtProgressBar(min = 0, max = k, style = 3)
    for (i in seq_len(k)) {
      tryCatch({
        fit_cv <- metafor::rma(
          yi = yi[-i], vi = vi[-i],
          mods = ~ mods_matrix[-i, , drop = FALSE] - 1,
          method = method
        )
        yhat_cv[i] <- sum(mods_matrix[i, ] * coef(fit_cv))
        converged[i] <- TRUE
      }, error = function(e) {
        yhat_cv[i] <- NA_real_
      })
      if (verbose) setTxtProgressBar(pb, i)
    }
    if (verbose) close(pb)
  } else if (cv_method == "kfold") {
    folds <- sample(rep(seq_len(k_folds), length.out = k))
    for (fold in seq_len(k_folds)) {
      test_idx <- which(folds == fold)
      train_idx <- which(folds != fold)
      tryCatch({
        fit_cv <- metafor::rma(
          yi = yi[train_idx], vi = vi[train_idx],
          mods = ~ mods_matrix[train_idx, , drop = FALSE] - 1,
          method = method
        )
        for (j in test_idx) {
          yhat_cv[j] <- sum(mods_matrix[j, ] * coef(fit_cv))
          converged[j] <- TRUE
        }
      }, error = function(e) {})
    }
  } else {
    stop("cv_method must be 'loo' or 'kfold'.")
  }

  valid <- converged & !is.na(yhat_cv)

  if (sum(valid) > k / 2) {
    tau2_cv <- mean(pmax(0, (yi[valid] - yhat_cv[valid])^2 - vi[valid]))
    r2het_cv_val <- if (apparent$tau2_null > 0) max(0, 1 - tau2_cv / apparent$tau2_null) else 0
    rmse <- sqrt(mean((yi[valid] - yhat_cv[valid])^2))
    mae <- mean(abs(yi[valid] - yhat_cv[valid]))
    correlation <- cor(yi[valid], yhat_cv[valid])
  } else {
    tau2_cv <- NA_real_
    r2het_cv_val <- NA_real_
    rmse <- NA_real_
    mae <- NA_real_
    correlation <- NA_real_
  }

  optimism <- if (!is.na(r2het_cv_val)) apparent$r2het - r2het_cv_val else NA_real_

  result <- list(
    r2het_apparent = apparent$r2het,
    r2het_corrected = r2het_cv_val,
    optimism = optimism,
    tau2_null = apparent$tau2_null,
    tau2_cv = tau2_cv,
    convergence_rate = sum(valid) / k * 100,
    rmse = rmse,
    mae = mae,
    correlation = correlation,
    predictions = if (sum(valid) > 0) yhat_cv else NULL,
    yi = yi
  )
  class(result) <- c("r2het_cv", "list")
  result
}

#' Bootstrap Confidence Intervals for R-squared
#'
#' @param yi Vector of effect sizes.
#' @param vi Vector of sampling variances.
#' @param mods Model matrix or formula for moderators.
#' @param data Optional data frame.
#' @param method Heterogeneity estimator.
#' @param B Number of bootstrap samples (default: 1000).
#' @param alpha Significance level for CIs (default: 0.05).
#' @param verbose Show progress bar.
#' @param parallel Logical. Use parallel processing? (default: FALSE).
#' @param n_cores Number of cores to use. If NULL, uses detectCores() - 1.
#'
#' @return A list with class "r2het_boot" containing:
#' \itemize{
#'   \item \code{ci_apparent} Bootstrap CI for apparent R^2_het (numeric vector of length 2).
#'   \item \code{ci_corrected} Bootstrap CI for corrected R^2_het (numeric vector of length 2).
#'   \item \code{mean_apparent} Bootstrap mean of apparent R^2_het (numeric).
#'   \item \code{mean_corrected} Bootstrap mean of corrected R^2_het (numeric).
#'   \item \code{se_apparent} Bootstrap SE of apparent R^2_het (numeric).
#'   \item \code{se_corrected} Bootstrap SE of corrected R^2_het (numeric).
#'   \item \code{prop_at_zero} Proportion of bootstrap samples with R^2_het < 0.01 (numeric).
#'   \item \code{convergence} Percentage of successful bootstrap samples (numeric).
#'   \item \code{bootstrap_values} List with vectors of bootstrap values for apparent and corrected.
#' }
#' @export
#'
#' @examples
#' # Note: Bootstrap with B=50 for quick demonstration.
#' # In practice, use B >= 500 for stable results.
#' k <- 30
#' yi <- rnorm(k, 0, sqrt(0.1))
#' vi <- runif(k, 0.01, 0.1)
#' mods <- cbind(1, rnorm(k))
#' boot_result <- r2het_boot(yi, vi, mods, B = 50, verbose = FALSE)
r2het_boot <- function(yi, vi, mods = NULL, data = NULL, method = "REML",
                       B = 1000, alpha = 0.05, verbose = TRUE,
                       parallel = FALSE, n_cores = NULL) {
  if (is.null(mods)) {
    return(list(
      ci_apparent = c(0, 0),
      ci_corrected = c(0, 0),
      mean_apparent = 0,
      mean_corrected = 0,
      se_apparent = 0,
      se_corrected = 0,
      prop_at_zero = 1,
      convergence = 100,
      bootstrap_values = list(apparent = numeric(0), corrected = numeric(0))
    ))
  }

  k <- length(yi)
  r2het_ap_vec <- numeric(B)
  r2het_cv_vec <- numeric(B)

  if (parallel) {
    if (!requireNamespace("parallel", quietly = TRUE) ||
        !requireNamespace("doParallel", quietly = TRUE) ||
        !requireNamespace("foreach", quietly = TRUE)) {
      warning("Parallel packages not installed. Falling back to sequential execution.")
      parallel <- FALSE
    }
  }

  if (parallel) {
    if (is.null(n_cores)) {
      n_cores <- max(1, parallel::detectCores() - 1)
    }

    cl <- parallel::makeCluster(n_cores)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    doParallel::registerDoParallel(cl)

    if (verbose) message(sprintf("Running bootstrap on %d cores...", n_cores))

    # Explicitly import %dopar% operator
    `%dopar%` <- foreach::`%dopar%`
    
    # We use 'b' as iterator
    # We export the functions explicitly to ensure they are available even if package not installed
    results <- foreach::foreach(b = seq_len(B), 
                                .packages = c("metafor", "stats"),
                                .export = c("r2het", "r2het_cv")) %dopar% {
      idx_boot <- sample(seq_len(k), k, replace = TRUE)
      yi_boot <- yi[idx_boot]
      vi_boot <- vi[idx_boot]
      
      ap_val <- NA_real_
      cv_val <- NA_real_

      if (inherits(mods, "formula")) {
        if (!is.null(data)) {
          data_boot <- data[idx_boot, , drop = FALSE]
          tryCatch({
            ap <- r2het(yi_boot, vi_boot, mods, data_boot, method)
            cv <- r2het_cv(yi_boot, vi_boot, mods, data_boot, method, verbose = FALSE)
            ap_val <- ap$r2het
            cv_val <- cv$r2het_corrected
          }, error = function(e) {})
        }
      } else {
        mods_boot <- mods[idx_boot, , drop = FALSE]
        tryCatch({
          ap <- r2het(yi_boot, vi_boot, mods_boot, data = NULL, method)
          cv <- r2het_cv(yi_boot, vi_boot, mods_boot, data = NULL, method, verbose = FALSE)
          ap_val <- ap$r2het
          cv_val <- cv$r2het_corrected
        }, error = function(e) {})
      }
      
      list(apparent = ap_val, corrected = cv_val)
    }
    
    # Unpack results
    r2het_ap_vec <- sapply(results, function(x) x$apparent)
    r2het_cv_vec <- sapply(results, function(x) x$corrected)

  } else {
    # Sequential execution
    if (verbose) pb <- txtProgressBar(min = 0, max = B, style = 3)

    for (b in seq_len(B)) {
      idx_boot <- sample(seq_len(k), k, replace = TRUE)
      yi_boot <- yi[idx_boot]
      vi_boot <- vi[idx_boot]

      if (inherits(mods, "formula")) {
        if (is.null(data)) {
          r2het_ap_vec[b] <- NA_real_
          r2het_cv_vec[b] <- NA_real_
        } else {
          data_boot <- data[idx_boot, , drop = FALSE]
          tryCatch({
            ap <- r2het(yi_boot, vi_boot, mods, data_boot, method)
            cv <- r2het_cv(yi_boot, vi_boot, mods, data_boot, method, verbose = FALSE)
            r2het_ap_vec[b] <- ap$r2het
            r2het_cv_vec[b] <- cv$r2het_corrected
          }, error = function(e) {
            r2het_ap_vec[b] <- NA_real_
            r2het_cv_vec[b] <- NA_real_
          })
        }
      } else {
        mods_boot <- mods[idx_boot, , drop = FALSE]
        tryCatch({
          ap <- r2het(yi_boot, vi_boot, mods_boot, data = NULL, method)
          cv <- r2het_cv(yi_boot, vi_boot, mods_boot, data = NULL, method, verbose = FALSE)
          r2het_ap_vec[b] <- ap$r2het
          r2het_cv_vec[b] <- cv$r2het_corrected
        }, error = function(e) {
          r2het_ap_vec[b] <- NA_real_
          r2het_cv_vec[b] <- NA_real_
        })
      }

      if (verbose) setTxtProgressBar(pb, b)
    }

    if (verbose) close(pb)
  }

  r2het_ap_vec <- r2het_ap_vec[!is.na(r2het_ap_vec)]
  r2het_cv_vec <- r2het_cv_vec[!is.na(r2het_cv_vec)]

  if (length(r2het_ap_vec) > B / 2) {
    ci_apparent <- quantile(r2het_ap_vec, c(alpha / 2, 1 - alpha / 2))
    mean_apparent <- mean(r2het_ap_vec)
    se_apparent <- sd(r2het_ap_vec)
  } else {
    ci_apparent <- c(NA_real_, NA_real_)
    mean_apparent <- NA_real_
    se_apparent <- NA_real_
  }

  if (length(r2het_cv_vec) > B / 2) {
    ci_corrected <- quantile(r2het_cv_vec, c(alpha / 2, 1 - alpha / 2))
    mean_corrected <- mean(r2het_cv_vec)
    se_corrected <- sd(r2het_cv_vec)
    prop_at_zero <- mean(r2het_cv_vec < 0.01)
  } else {
    ci_corrected <- c(NA_real_, NA_real_)
    mean_corrected <- NA_real_
    se_corrected <- NA_real_
    prop_at_zero <- NA_real_
  }

  list(
    ci_apparent = ci_apparent,
    ci_corrected = ci_corrected,
    mean_apparent = mean_apparent,
    mean_corrected = mean_corrected,
    se_apparent = se_apparent,
    se_corrected = se_corrected,
    prop_at_zero = prop_at_zero,
    convergence = (length(r2het_ap_vec) / B) * 100,
    bootstrap_values = list(apparent = r2het_ap_vec, corrected = r2het_cv_vec)
  )
}

#' Check for Overfitting in Meta-Regression
#'
#' @param yi Vector of effect sizes.
#' @param vi Vector of sampling variances.
#' @param mods Model matrix or formula for moderators.
#' @param data Optional data frame.
#' @param method Heterogeneity estimator.
#' @param B Number of bootstrap samples.
#' @param parallel Logical. Use parallel processing? (default: FALSE).
#' @param n_cores Number of cores to use. If NULL, uses detectCores() - 1.
#'
#' @return An object of class "metaoverfit" (a list) containing:
#' \itemize{
#'   \item \code{k} Number of studies (integer).
#'   \item \code{p} Number of parameters (integer).
#'   \item \code{k_per_p} Ratio of studies to parameters (numeric).
#'   \item \code{risk_category} Overfitting risk level: "None", "Low", "Moderate", "Severe", or "Extreme" (character).
#'   \item \code{expected_optimism} Expected range of optimism (character).
#'   \item \code{actual_optimism} Actual optimism percentage (numeric).
#'   \item \code{r2het_apparent} Apparent R^2_het percentage (numeric).
#'   \item \code{r2het_corrected} Corrected R^2_het percentage (numeric).
#'   \item \code{ci_corrected} 95% CI for corrected R^2_het (numeric vector).
#'   \item \code{recommendation} Text recommendation based on results (character).
#'   \item \code{convergence_rate} Percentage of successful CV folds (numeric).
#'   \item \code{bootstrap_values} List with vectors of bootstrap values for apparent and corrected.
#' }
#' @examples
#' \donttest{
#' # BCG vaccine data
#' data(dat.bcg, package = "metadat")
#' dat <- metafor::escalc(measure = "RR", ai = tpos, bi = tneg,
#'                        ci = cpos, di = cneg, data = dat.bcg)
#' result <- check_overfitting(dat$yi, dat$vi, mods = ~ ablat, data = dat, B = 50)
#' print(result)
#' }
#' @export
check_overfitting <- function(yi, vi, mods = NULL, data = NULL,
                              method = "REML", B = 500,
                              parallel = FALSE, n_cores = NULL) {
  k <- length(yi)

  if (is.null(mods)) {
    rpt <- list(
      k = k,
      p = 1,
      k_per_p = k,
      risk_category = "None",
      expected_optimism = "<10%",
      actual_optimism = 0,
      r2het_apparent = 0,
      r2het_corrected = 0,
      ci_corrected = c(NA_real_, NA_real_),
      recommendation = "No moderators - no overfitting risk",
      convergence_rate = 100,
      bootstrap_values = list(apparent = numeric(0), corrected = numeric(0))
    )
    class(rpt) <- c("metaoverfit", "list")
    return(rpt)
  }

  # Determine p
  if (inherits(mods, "matrix")) {
    p <- ncol(mods)
  } else if (inherits(mods, "formula")) {
    tmp <- metafor::rma(yi = yi, vi = vi, mods = mods, data = data, method = method)
    p <- length(coef(tmp))
  } else {
    stop("'mods' must be a matrix or a formula, or NULL.")
  }

  k_per_p <- k / p

  # Heuristic risk bands
  if (k < 20 || k_per_p < 5) {
    risk_category <- "Extreme"
    expected_optimism <- ">40%"
    recommendation <- "DO NOT conduct meta-regression - sample size too small"
  } else if (k_per_p < 10) {
    risk_category <- "Severe"
    expected_optimism <- "20-40%"
    recommendation <- "Results highly unreliable - consider exploratory only"
  } else if (k_per_p < 15) {
    risk_category <- "Moderate"
    expected_optimism <- "10-20%"
    recommendation <- "Interpret with caution - report optimism correction"
  } else {
    risk_category <- "Low"
    expected_optimism <- "<10%"
    recommendation <- "Acceptable sample size - still report optimism correction"
  }

  cv_result <- r2het_cv(yi, vi, mods, data, method, verbose = FALSE)
  boot_result <- r2het_boot(yi, vi, mods, data, method, B = B, verbose = FALSE,
                            parallel = parallel, n_cores = n_cores)

  rpt <- list(
    k = k,
    p = p,
    k_per_p = round(k_per_p, 1),
    risk_category = risk_category,
    expected_optimism = expected_optimism,
    actual_optimism = round(cv_result$optimism * 100, 1),
    r2het_apparent = round(cv_result$r2het_apparent * 100, 1),
    r2het_corrected = round(cv_result$r2het_corrected * 100, 1),
    ci_corrected = round(boot_result$ci_corrected * 100, 1),
    recommendation = recommendation,
    convergence_rate = cv_result$convergence_rate,
    bootstrap_values = boot_result$bootstrap_values
  )
  class(rpt) <- c("metaoverfit", "list")
  rpt
}

#' Sample Size Recommendation for Meta-Regression
#'
#' Calculates the minimum required number of studies for a meta-regression
#' analysis based on the number of parameters and target optimism level.
#'
#' @param p Number of parameters (including intercept).
#' @param target_optimism Maximum acceptable optimism (default: 0.10).
#' @param verbose Logical. If TRUE (default), prints recommendation to console.
#'
#' @return An object of class "sample_size_rec" (a list) containing:
#' \itemize{
#'   \item \code{p} Number of parameters (integer).
#'   \item \code{target_optimism} Target optimism level (numeric).
#'   \item \code{min_ratio} Minimum k/p ratio required (integer).
#'   \item \code{min_k} Minimum number of studies required (integer).
#' }
#' The object is returned invisibly. When printed, it displays a formatted
#' recommendation summary.
#' @examples
#' # Minimum studies for 3-predictor model
#' sample_size_recommendation(p = 3, target_optimism = 0.10)
#' @export
sample_size_recommendation <- function(p, target_optimism = 0.10, verbose = TRUE) {
  if (target_optimism <= 0.05) {
    min_ratio <- 20
  } else if (target_optimism <= 0.10) {
    min_ratio <- 15
  } else if (target_optimism <= 0.20) {
    min_ratio <- 10
  } else {
    min_ratio <- 5
  }

  min_k <- max(20, ceiling(p * min_ratio))

  result <- list(
    p = p,
    target_optimism = target_optimism,
    min_ratio = min_ratio,
    min_k = min_k
  )

  class(result) <- c("sample_size_rec", "list")

  if (verbose) {
    print(result)
  }

  invisible(result)
}

#' Print Method for sample_size_rec Objects
#'
#' @param x A sample_size_rec object.
#' @param ... Additional arguments (unused).
#'
#' @return No return value, called for side effects (printing to console).
#' @export
print.sample_size_rec <- function(x, ...) {
  message("Sample Size Recommendation")
  message(strrep("-", 40))
  message("Parameters (p): ", x$p)
  message("Target optimism: ", x$target_optimism * 100, "%")
  message("Minimum k/p ratio: ", x$min_ratio)
  message("MINIMUM k required: ", x$min_k)
  message("\nNote: This ensures optimism < ", x$target_optimism * 100, "%")
}

#' Print Method for metaoverfit Objects
#'
#' @param x A metaoverfit object.
#' @param ... Additional arguments (unused).
#'
#' @return No return value, called for side effects (printing to console).
#' @export
print.metaoverfit <- function(x, ...) {
  message("\n", strrep("=", 60))
  message("META-REGRESSION OVERFITTING ASSESSMENT")
  message(strrep("=", 60), "\n")

  message("Sample Size:")
  message("  k = ", x$k, " studies")
  message("  p = ", x$p, " parameters")
  message("  k/p ratio = ", x$k_per_p, "\n")

  message("Risk Assessment:")
  message("  Category: ", x$risk_category)
  message("  Expected optimism: ", x$expected_optimism)
  message("  Actual optimism: ", x$actual_optimism, "%")
  message("  CV convergence: ", x$convergence_rate, "%\n")

  message("R^2 for Heterogeneity:")
  message("  Apparent R^2_het: ", x$r2het_apparent, "%")
  message("  Corrected R^2_het: ", x$r2het_corrected, "%")
  message("  95% CI: [", x$ci_corrected[1], "%, ",
          x$ci_corrected[2], "%]\n")

  message("RECOMMENDATION:")
  message("  ", x$recommendation)
  message("\n", strrep("=", 60))
}

#' Plot Overfitting Diagnostics
#'
#' Creates diagnostic plots for assessing overfitting in meta-regression.
#'
#' @param result Output from \code{check_overfitting()} or \code{r2het_cv()}.
#' @param type "bar" for comparison of apparent vs corrected R^2_het,
#'   "scatter" for observed vs predicted (requires predictions), or
#'   "density" for bootstrap distributions (requires bootstrap_values).
#'
#' @return A ggplot2 object. If ggplot2 is not installed, the function
#'   will stop with an error message.
#' @examples
#' \donttest{
#' data(dat.bcg, package = "metadat")
#' dat <- metafor::escalc(measure = "RR", ai = tpos, bi = tneg,
#'                        ci = cpos, di = cneg, data = dat.bcg)
#' cv <- r2het_cv(dat$yi, dat$vi, mods = ~ ablat, data = dat)
#' plot_overfitting(cv, type = "bar")
#' }
#' @export
plot_overfitting <- function(result, type = "bar") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package required for plotting")
  }

  if (type == "bar") {
    # Detect scale: check_overfitting() stores values as percentages (>1),
    # while r2het_cv()/r2het_boot() store on 0-1 scale
    app_val <- result$r2het_apparent
    cor_val <- result$r2het_corrected
    if (!is.null(app_val) && !is.na(app_val) && app_val <= 1) {
      app_val <- app_val * 100
      cor_val <- cor_val * 100
    }
    df <- data.frame(
      Type  = c("Apparent", "Corrected"),
      R2het = c(app_val, cor_val)
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = Type, y = R2het, fill = Type)) +
      ggplot2::geom_bar(stat = "identity", width = 0.6) +
      ggplot2::scale_fill_manual(values = c("Apparent" = "#E74C3C",
                                            "Corrected" = "#27AE60")) +
      ggplot2::labs(
        x = "", y = "R^2_het (%)",
        title = "Overfitting in Meta-Regression",
        subtitle = paste(
          "Optimism:",
          round(app_val - cor_val, 1), "%"
        )
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "none")

  } else if (type == "scatter") {
    if (is.null(result$predictions) || all(is.na(result$predictions))) {
      stop("No predictions available in 'result' to draw a scatter plot.")
    }
    if (is.null(result$yi)) {
      stop("Observed values 'yi' not found in 'result' (call r2het_cv which returns 'yi').")
    }

    df <- data.frame(
      Observed  = result$yi,
      Predicted = result$predictions
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = Predicted, y = Observed)) +
      ggplot2::geom_point(size = 3, alpha = 0.7) +
      ggplot2::geom_abline(slope = 1, intercept = 0,
                           linetype = "dashed", color = "red") +
      ggplot2::labs(
        x = "Predicted Effect Size",
        y = "Observed Effect Size",
        title = "Cross-Validation Performance",
        subtitle = paste("Correlation:", round(result$correlation, 3))
      ) +
      ggplot2::theme_minimal()
      
  } else if (type == "density") {
    if (is.null(result$bootstrap_values) || 
        length(result$bootstrap_values$apparent) == 0) {
      stop("No bootstrap values available in 'result' to draw density plot.")
    }
    
    df_boot <- data.frame(
      Value = c(result$bootstrap_values$apparent, 
                result$bootstrap_values$corrected),
      Type = rep(c("Apparent", "Corrected"), 
                 each = length(result$bootstrap_values$apparent))
    )
    
    p <- ggplot2::ggplot(df_boot, ggplot2::aes(x = Value * 100, fill = Type)) +
      ggplot2::geom_density(alpha = 0.5) +
      ggplot2::scale_fill_manual(values = c("Apparent" = "#E74C3C",
                                            "Corrected" = "#27AE60")) +
      ggplot2::labs(
        x = "R^2_het (%)",
        y = "Density",
        title = "Bootstrap Distributions of R^2_het",
        subtitle = "Comparison of Apparent and Optimism-Corrected Estimates"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(legend.position = "top")
      
  } else {
    stop("type must be 'bar', 'scatter', or 'density'.")
  }

  p
}
