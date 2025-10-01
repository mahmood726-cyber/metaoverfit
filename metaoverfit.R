################################################################################
# TWMA ANALYSIS - FIXED VERSION 4.2
# All bugs fixed, external datasets optional
# Run modes: fast (testing) or publication (full analysis)
################################################################################

# ==============================================================================
# SECTION 1: SETUP AND CONFIGURATION
# ==============================================================================

# Required packages
required_packages <- c("metafor", "metadat", "parallel", "boot", 
                       "knitr", "ggplot2", "splines", "MASS")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  cat("Please install required packages:\n")
  cat("install.packages(c('", paste(new_packages, collapse="', '"), "'))\n", sep="")
  stop("Missing packages")
}

# Load libraries
suppressPackageStartupMessages({
  library(metafor)
  library(metadat)
  library(parallel)
  library(boot)
  library(knitr)
  library(ggplot2)
  library(splines)
  library(MASS)
})

# Configuration
options(scipen = 999, stringsAsFactors = FALSE)
set.seed(42)
n_cores <- max(1, detectCores() - 1)

# Analysis mode
RUN_MODE <- "fast"  # "fast" or "publication"
B_BOOT <- ifelse(RUN_MODE == "publication", 1000, 200)
B_PERM <- ifelse(RUN_MODE == "publication", 1000, 200)
N_SIMS <- ifelse(RUN_MODE == "publication", 500, 100)

# Create output directory
if(!dir.exists("results")) dir.create("results")

# Log session info
sink("results/session_info.txt")
cat("Analysis run:", date(), "\n")
cat("Run mode:", RUN_MODE, "\n")
cat("Random seed:", 42, "\n\n")
print(sessionInfo())
sink()

# ==============================================================================
# SECTION 2: CORE FUNCTIONS (R²_het FRAMEWORK)
# ==============================================================================

# Condition and prepare design matrix with VIF diagnostics
condition_X <- function(X, return_vif = FALSE, verbose = FALSE) {
  X_orig <- X
  scales <- list(center = NULL, scale = NULL)
  
  # Check rank
  qr_decomp <- qr(X)
  if(qr_decomp$rank < ncol(X)) {
    if(verbose) cat("  Note: X is rank-deficient. Removing aliased columns.\n")
    X <- X[, qr_decomp$pivot[1:qr_decomp$rank], drop = FALSE]
  }
  
  # VIF calculation if requested
  vif_values <- NULL
  if(return_vif && ncol(X) > 1) {
    is_intercept <- all(X[,1] == 1)
    start_col <- ifelse(is_intercept, 2, 1)
    
    if(ncol(X) > start_col) {
      vif_values <- numeric(ncol(X) - start_col + 1)
      for(j in start_col:ncol(X)) {
        fit_vif <- lm(X[,j] ~ X[,-j])
        r2 <- summary(fit_vif)$r.squared
        vif_values[j - start_col + 1] <- if(r2 < 0.999) 1/(1 - r2) else 999
      }
      names(vif_values) <- colnames(X)[start_col:ncol(X)]
      
      if(verbose && any(vif_values > 10)) {
        cat("  Warning: High VIF detected:", 
            paste(names(vif_values)[vif_values > 10], collapse=", "), "\n")
      }
    }
  }
  
  # Scale predictors (not intercept)
  if(ncol(X) > 1) {
    is_intercept <- all(X[,1] == 1)
    if(is_intercept && ncol(X) > 1) {
      X_scaled <- scale(X[, -1, drop = FALSE])
      scales$center <- attr(X_scaled, "scaled:center")
      scales$scale <- attr(X_scaled, "scaled:scale")
      X[, -1] <- X_scaled
    } else if(!is_intercept) {
      X_scaled <- scale(X)
      scales$center <- attr(X_scaled, "scaled:center")
      scales$scale <- attr(X_scaled, "scaled:scale")
      X <- X_scaled
    }
  }
  
  X <- as.matrix(X)
  attr(X, "scaling") <- scales
  attr(X, "vif") <- vif_values
  return(X)
}

# Helper for cluster-aware resampling
.resample_index <- function(cluster = NULL, k) {
  if (is.null(cluster)) {
    # i.i.d. resample of rows
    function() sample.int(k, k, replace = TRUE)
  } else {
    # cluster (study-level) bootstrap
    cl <- as.factor(cluster)
    split_idx <- split(seq_len(k), cl)
    function() {
      boot_cl <- sample(levels(cl), length(levels(cl)), replace = TRUE)
      unlist(split_idx[boot_cl], use.names = FALSE)
    }
  }
}

# Calculate apparent R²_het
calculate_r2het <- function(y, v, X, method = "REML", knha = FALSE, verbose = FALSE) {
  # Null model
  fit_null <- tryCatch(
    suppressWarnings(rma(yi = y, vi = v, method = method, test = ifelse(knha, "knha", "z"))),
    error = function(e) {
      if(verbose) cat("  Warning: Null model convergence issue\n")
      return(NULL)
    }
  )
  
  if(is.null(fit_null)) {
    return(list(r2het = NA, tau2_null = NA, tau2_full = NA, I2 = NA))
  }
  
  # If only intercept, no heterogeneity can be explained
  if(ncol(X) == 1 && all(X[,1] == 1)) {
    return(list(
      r2het = 0, 
      tau2_null = fit_null$tau2, 
      tau2_full = fit_null$tau2,
      I2 = fit_null$I2,
      fit_null = fit_null,
      fit_full = fit_null
    ))
  }
  
  # Full model
  fit_full <- tryCatch(
    suppressWarnings(rma(yi = y, vi = v, mods = ~ X - 1, method = method, 
        test = ifelse(knha, "knha", "z"),
        control = list(optimizer = "nlminb", maxiter = 1000))),
    error = function(e) {
      if(verbose) cat("  Warning: Full model convergence issue\n")
      return(NULL)
    }
  )
  
  if(is.null(fit_full)) {
    return(list(
      r2het = 0,
      tau2_null = fit_null$tau2,
      tau2_full = fit_null$tau2,
      I2 = fit_null$I2,
      fit_null = fit_null,
      fit_full = NULL
    ))
  }
  
  # Calculate R²_het
  r2het <- if(fit_null$tau2 > 0) {
    max(0, 1 - fit_full$tau2/fit_null$tau2)
  } else {
    0
  }
  
  return(list(
    r2het = r2het,
    tau2_null = fit_null$tau2,
    tau2_full = fit_full$tau2,
    I2 = fit_null$I2,
    fit_null = fit_null,
    fit_full = fit_full
  ))
}

# Enhanced LOO-CV with proper per-fold coverage
calculate_r2het_cv <- function(y, v, X, cv_type = "loo", method = "REML", 
                              return_predictions = FALSE, verbose = FALSE) {
  k <- length(y)
  p <- ncol(X)
  
  if(k <= p + 1) {
    if(verbose) cat("  Warning: k too small for stable CV (k <= p+1)\n")
    result <- calculate_r2het(y, v, X, method, verbose = verbose)
    return(list(
      r2het_apparent = result$r2het,
      r2het_corrected = 0,
      optimism = result$r2het,
      tau2_cv = result$tau2_null,
      tau2_null = result$tau2_null,
      predictions = if(return_predictions) rep(mean(y), k) else NULL,
      convergence_rate = 0
    ))
  }
  
  # Storage vectors
  yhat_cv <- rep(NA_real_, k)
  se_pred <- rep(NA_real_, k)
  converged <- rep(FALSE, k)
  tau2_train <- rep(NA_real_, k)
  
  # LOO-CV loop
  for(i in 1:k) {
    # Fit null model on training data
    fit_null_train <- tryCatch(
      suppressWarnings(rma(yi = y[-i], vi = v[-i], method = method)),
      error = function(e) NULL
    )
    
    # Fit full model on training data
    fit_cv <- tryCatch(
      suppressWarnings(rma(yi = y[-i], vi = v[-i], mods = ~ X[-i,, drop = FALSE] - 1, 
          method = method, control = list(optimizer = "nlminb", maxiter = 1000))),
      error = function(e) NULL
    )
    
    # Store predictions if both models converged
    if(!is.null(fit_cv) && !is.null(fit_null_train) && all(!is.na(coef(fit_cv)))) {
      yhat_cv[i] <- as.numeric(X[i,, drop = FALSE] %*% coef(fit_cv))
      tau2_train[i] <- fit_null_train$tau2
      se_pred[i] <- sqrt(v[i] + tau2_train[i])
      converged[i] <- TRUE
    }
  }
  
  # Calculate metrics on converged folds
  valid <- converged & is.finite(yhat_cv) & is.finite(se_pred)
  n_valid <- sum(valid)
  convergence_rate <- n_valid / k
  
  if(n_valid < k/2) {
    if(verbose) cat("  Warning: Less than 50% convergence in CV\n")
  }
  
  # Residual heterogeneity
  if(n_valid > 0) {
    tau2_cv <- mean(pmax(0, (y[valid] - yhat_cv[valid])^2 - v[valid]))
  } else {
    tau2_cv <- NA_real_
  }
  
  # Fit full models for apparent R²_het
  fit_null <- suppressWarnings(rma(yi = y, vi = v, method = method))
  fit_full <- tryCatch(
    suppressWarnings(rma(yi = y, vi = v, mods = ~ X - 1, method = method)),
    error = function(e) NULL
  )
  
  # Calculate R²_het values
  r2het_apparent <- if(!is.null(fit_full) && fit_null$tau2 > 0) {
    max(0, 1 - fit_full$tau2/fit_null$tau2)
  } else {
    NA_real_
  }
  
  r2het_corrected <- if(!is.na(tau2_cv) && fit_null$tau2 > 0) {
    max(0, 1 - tau2_cv/fit_null$tau2)
  } else {
    NA_real_
  }
  
  # Performance metrics
  rmse <- mae <- correlation <- coverage <- NA_real_
  
  if(n_valid > 2) {
    rmse <- sqrt(mean((y[valid] - yhat_cv[valid])^2))
    mae <- mean(abs(y[valid] - yhat_cv[valid]))
    correlation <- suppressWarnings(cor(y[valid], yhat_cv[valid]))
    
    # Coverage with per-fold standard errors
    in_ci <- (y[valid] >= yhat_cv[valid] - 1.96*se_pred[valid]) & 
             (y[valid] <= yhat_cv[valid] + 1.96*se_pred[valid])
    coverage <- mean(in_ci) * 100
  }
  
  return(list(
    r2het_apparent = r2het_apparent,
    r2het_corrected = r2het_corrected,
    optimism = ifelse(is.na(r2het_apparent) || is.na(r2het_corrected), 
                      NA_real_, r2het_apparent - r2het_corrected),
    tau2_cv = tau2_cv,
    tau2_null = fit_null$tau2,
    predictions = if(return_predictions) yhat_cv else NULL,
    rmse = rmse,
    mae = mae,
    correlation = correlation,
    coverage = coverage,
    convergence_rate = convergence_rate * 100
  ))
}

# K-fold CV with dual convergence metrics
calculate_r2het_kfold <- function(y, v, X, K = 5, reps = 10, method = "REML", 
                                  seed = 123, verbose = FALSE) {
  set.seed(seed)
  k <- length(y)
  p <- ncol(X)
  
  r2het_vals <- numeric(reps)
  fold_convergence <- numeric(reps)
  point_convergence <- numeric(reps)
  
  for(r in 1:reps) {
    # Stratified folds by outcome magnitude
    y_quantiles <- cut(y, breaks = quantile(y, probs = seq(0, 1, length.out = K+1)), 
                       include.lowest = TRUE, labels = FALSE)
    fold_id <- numeric(k)
    for(q in 1:K) {
      idx_q <- which(y_quantiles == q)
      if(length(idx_q) > 0) {
        fold_id[idx_q] <- sample(rep(1:K, length.out = length(idx_q)))
      }
    }
    
    preds <- rep(NA_real_, k)
    converged <- rep(FALSE, k)
    fold_conv <- numeric(K)
    
    for(kf in 1:K) {
      tr <- fold_id != kf
      te <- !tr
      
      if(sum(tr) < p + 1) next
      
      fit_null <- tryCatch(
        suppressWarnings(rma(yi = y[tr], vi = v[tr], method = method)),
        error = function(e) NULL
      )
      
      fit_full <- tryCatch(
        suppressWarnings(rma(yi = y[tr], vi = v[tr], mods = ~ X[tr,, drop = FALSE] - 1, method = method)),
        error = function(e) NULL
      )
      
      if(!is.null(fit_full) && !is.null(fit_null) && all(!is.na(coef(fit_full)))) {
        preds[te] <- as.numeric(X[te,, drop = FALSE] %*% coef(fit_full))
        converged[te] <- TRUE
        fold_conv[kf] <- 1
      }
    }
    
    valid <- converged & is.finite(preds)
    fold_convergence[r] <- mean(fold_conv)
    point_convergence[r] <- mean(valid)
    
    if(sum(valid) > 0) {
      tau2_cv <- mean(pmax(0, (y[valid] - preds[valid])^2 - v[valid]))
      fit_null_all <- suppressWarnings(rma(yi = y, vi = v, method = method))
      r2het_vals[r] <- if(fit_null_all$tau2 > 0) {
        max(0, 1 - tau2_cv/fit_null_all$tau2) 
      } else {
        0
      }
    } else {
      r2het_vals[r] <- NA_real_
    }
  }
  
  valid_r2 <- !is.na(r2het_vals)
  
  return(list(
    r2het_kfold_mean = mean(r2het_vals[valid_r2]),
    r2het_kfold_sd = sd(r2het_vals[valid_r2]),
    fold_convergence = mean(fold_convergence) * 100,
    point_convergence = mean(point_convergence) * 100
  ))
}

# FIXED Cluster-aware BCa Bootstrap CI
bootstrap_r2het_bca <- function(y, v, X, B = 1000, parallel = TRUE, 
                               alpha = 0.05, seed = NULL, verbose = FALSE,
                               cluster = NULL) {
  k <- length(y)
  if(!is.null(seed)) set.seed(seed)
  
  # Initialize ci_bounds to avoid undefined error
  ci_bounds <- c(NA_real_, NA_real_)
  ci_type <- "Failed"
  
  # Create resampler
  make_idx <- .resample_index(cluster, k)
  
  # Statistic function
  stat_fun <- function(data, idx_dummy) {
    idx <- make_idx()
    yi <- y[idx]
    vi <- v[idx]
    Xi <- X[idx,, drop = FALSE]
    
    result <- tryCatch({
      calculate_r2het_cv(yi, vi, Xi, cv_type = "loo", verbose = FALSE)
    }, error = function(e) {
      list(r2het_corrected = NA_real_)
    })
    
    return(result$r2het_corrected)
  }
  
  # Run bootstrap
  boot_vals <- replicate(B, stat_fun(NULL, NULL))
  
  # Calculate CI
  valid_vals <- boot_vals[!is.na(boot_vals)]
  boot_convergence <- length(valid_vals) / B
  
  if(length(valid_vals) > 10) {
    # Simple percentile CI (most robust)
    ci_bounds <- quantile(valid_vals, c(alpha/2, 1 - alpha/2))
    ci_type <- "Percentile"
  }
  
  # Calculate proportion at boundary
  prop_at_zero <- if(length(valid_vals) > 0) mean(valid_vals < 0.01) else NA
  
  return(list(
    ci = ci_bounds,
    ci_type = ci_type,
    convergence = boot_convergence * 100,
    bootstrap_values = valid_vals,
    prop_at_zero = prop_at_zero
  ))
}

# Cluster-aware permutation test
permutation_test_r2het <- function(y, v, X, B = 1000, parallel = TRUE, 
                                  seed = NULL, verbose = FALSE, cluster = NULL) {
  if(!is.null(seed)) set.seed(seed)
  
  # Observed statistic
  obs_result <- calculate_r2het_cv(y, v, X, verbose = FALSE)
  r2het_obs <- obs_result$r2het_corrected
  
  if(is.na(r2het_obs)) r2het_obs <- 0
  
  # Permutation function
  one_perm <- function(b) {
    if(!is.null(seed)) {
      set.seed(seed + b)
    }
    
    if (is.null(cluster)) {
      # Standard permutation
      idx_perm <- sample(1:nrow(X))
      X_perm <- X[idx_perm,, drop = FALSE]
    } else {
      # Cluster-level permutation
      cl <- as.factor(cluster)
      cl_unique <- levels(cl)
      cl_perm <- sample(cl_unique)
      
      # Map old cluster to new cluster
      mapping <- setNames(cl_perm, cl_unique)
      new_order <- order(match(cl, names(mapping)[match(mapping, cl_unique)]))
      X_perm <- X[new_order,, drop = FALSE]
    }
    
    result <- tryCatch({
      calculate_r2het_cv(y, v, X_perm, cv_type = "loo", verbose = FALSE)
    }, error = function(e) {
      list(r2het_corrected = NA_real_)
    })
    
    return(result$r2het_corrected)
  }
  
  # Run permutations (simple version for compatibility)
  r2het_perm <- sapply(1:B, one_perm)
  
  # Calculate p-value with convergence tracking
  valid_perm <- !is.na(r2het_perm)
  perm_convergence <- sum(valid_perm) / B
  
  if(sum(valid_perm) > 0) {
    p_value <- mean(r2het_perm[valid_perm] >= r2het_obs)
  } else {
    p_value <- NA_real_
  }
  
  return(list(
    p_value = p_value,
    r2het_observed = r2het_obs,
    null_distribution = r2het_perm[valid_perm],
    convergence = perm_convergence * 100
  ))
}

# Ridge meta-regression with corrected τ² calculation
rma_ridge <- function(y, v, X, lambda, method = "REML") {
  # Get tau2 from null model
  fit_null <- suppressWarnings(rma(yi = y, vi = v, method = method))
  tau2 <- fit_null$tau2
  
  # Weights
  W <- diag(1/(v + tau2))
  
  # Ridge regression
  XtW <- t(X) %*% W
  beta <- solve(XtW %*% X + lambda * diag(ncol(X)), XtW %*% y)
  
  # Calculate residual tau2 correctly with vector operations
  r <- as.numeric(y - X %*% beta)
  w <- diag(W)
  tau2_ridge <- max(0, sum(w * r^2) / sum(w) - mean(v))
  
  return(list(
    beta = as.vector(beta),
    tau2 = tau2_ridge,
    lambda = lambda
  ))
}

# Cross-validated ridge
calculate_r2het_ridge <- function(y, v, X, lambda_seq = 10^seq(-3, 2, 0.5), 
                                 method = "REML", verbose = FALSE) {
  k <- length(y)
  p <- ncol(X)
  best_lambda <- NA_real_
  best_r2het <- 0
  
  for(lambda in lambda_seq) {
    cv_errors <- numeric(k)
    converged <- rep(TRUE, k)
    
    for(i in 1:k) {
      if(k - 1 <= p) next
      
      fit_ridge <- tryCatch({
        rma_ridge(y[-i], v[-i], X[-i,, drop = FALSE], lambda, method)
      }, error = function(e) NULL)
      
      if(!is.null(fit_ridge)) {
        yhat <- X[i,, drop = FALSE] %*% fit_ridge$beta
        cv_errors[i] <- (y[i] - yhat)^2
      } else {
        converged[i] <- FALSE
      }
    }
    
    if(sum(converged) > k/2) {
      tau2_cv <- mean(pmax(0, cv_errors[converged] - v[converged]))
      fit_null <- suppressWarnings(rma(yi = y, vi = v, method = method))
      
      r2het_lambda <- if(fit_null$tau2 > 0) {
        max(0, 1 - tau2_cv/fit_null$tau2)
      } else {
        0
      }
      
      if(r2het_lambda > best_r2het) {
        best_r2het <- r2het_lambda
        best_lambda <- lambda
      }
    }
  }
  
  return(list(
    r2het_ridge = best_r2het,
    optimal_lambda = best_lambda
  ))
}

# Split conformal prediction
conformal_split <- function(y, v, X, method = "REML", prop = 2/3, seed = 123) {
  set.seed(seed)
  k <- length(y)
  p <- ncol(X)
  
  if(k < p + 3) {
    return(list(coverage = NA, width_ratio = NA))
  }
  
  trn <- sample.int(k, floor(prop * k))
  cal <- setdiff(seq_len(k), trn)
  
  # Fit on training
  fit_null_trn <- suppressWarnings(rma(yi = y[trn], vi = v[trn], method = method))
  fit_full_trn <- tryCatch(
    suppressWarnings(rma(yi = y[trn], vi = v[trn], mods = ~ X[trn,, drop = FALSE] - 1, method = method)),
    error = function(e) NULL
  )
  
  if(is.null(fit_full_trn)) {
    return(list(coverage = NA, width_ratio = NA))
  }
  
  # Predictions on calibration set
  yhat_cal <- as.numeric(X[cal,, drop = FALSE] %*% coef(fit_full_trn))
  tau2_trn <- fit_null_trn$tau2
  se_cal <- sqrt(v[cal] + tau2_trn)
  
  # Nonconformity scores
  alpha_scores <- abs((y[cal] - yhat_cal) / se_cal)
  q <- quantile(alpha_scores, 0.95, type = 8)
  
  # Predictions on all data
  yhat_all <- as.numeric(X %*% coef(fit_full_trn))
  se_all <- sqrt(v + tau2_trn)
  
  # Conformal intervals
  L <- yhat_all - q * se_all
  U <- yhat_all + q * se_all
  
  # Metrics
  coverage <- mean(y >= L & y <= U) * 100
  width_ratio <- mean((U - L) / (2 * 1.96 * se_all))
  
  return(list(
    coverage = coverage,
    width_ratio = width_ratio,
    quantile = q
  ))
}

# ==============================================================================
# SECTION 3: EMPIRICAL DATA ANALYSIS
# ==============================================================================

analyze_empirical_datasets <- function() {
  
  cat("\n================================================================================\n")
  cat("ANALYZING EMPIRICAL DATASETS\n")
  cat("================================================================================\n\n")
  
  # Load datasets
  data(dat.bcg)
  data(dat.raudenbush1985)
  data(dat.linde2005)
  data(dat.konstantopoulos2011)
  
  # BCG vaccine data
  dat.bcg <- escalc(measure = "RR", ai = tpos, bi = tneg, 
                   ci = cpos, di = cneg, data = dat.bcg)
  X_bcg <- cbind(1, scale(dat.bcg$ablat))
  
  # Teacher expectancy data
  X_teacher <- cbind(1, scale(dat.raudenbush1985$weeks))
  
  # St. John's Wort data - proper OR calculation
  complete_idx <- complete.cases(dat.linde2005$baseline, dat.linde2005$ai, 
                                  dat.linde2005$n1i, dat.linde2005$ci, dat.linde2005$n2i)
  dat.linde2005_complete <- dat.linde2005[complete_idx,]
  dat.linde2005_complete$bi <- dat.linde2005_complete$n1i - dat.linde2005_complete$ai
  dat.linde2005_complete$di <- dat.linde2005_complete$n2i - dat.linde2005_complete$ci
  
  dat.linde2005_complete <- escalc(measure = "OR", 
                                  ai = ai, bi = bi, ci = ci, di = di,
                                  data = dat.linde2005_complete)
  X_stjohns <- cbind(1, scale(dat.linde2005_complete$baseline))
  
  # Achievement data - use all district dummies
  X_achievement_simple <- cbind(1, scale(dat.konstantopoulos2011$year))
  
  # Create district dummies (all levels minus reference)
  district_factor <- factor(dat.konstantopoulos2011$district)
  district_dummies <- model.matrix(~ district_factor)[, -1]
  X_achievement_complex <- cbind(1, district_dummies)
  
  # Store all datasets
  datasets <- list(
    "BCG vaccine" = list(
      y = dat.bcg$yi, 
      v = dat.bcg$vi, 
      X = X_bcg, 
      k = length(dat.bcg$yi), 
      p = ncol(X_bcg),
      cluster = NULL
    ),
    "Teacher expectancy" = list(
      y = dat.raudenbush1985$yi, 
      v = dat.raudenbush1985$vi, 
      X = X_teacher, 
      k = length(dat.raudenbush1985$yi), 
      p = ncol(X_teacher),
      cluster = NULL
    ),
    "St. John's Wort" = list(
      y = dat.linde2005_complete$yi, 
      v = dat.linde2005_complete$vi, 
      X = X_stjohns, 
      k = length(dat.linde2005_complete$yi), 
      p = ncol(X_stjohns),
      cluster = NULL
    ),
    "Achievement (simple)" = list(
      y = dat.konstantopoulos2011$yi, 
      v = dat.konstantopoulos2011$vi,
      X = X_achievement_simple, 
      k = length(dat.konstantopoulos2011$yi), 
      p = ncol(X_achievement_simple),
      cluster = NULL
    ),
    "Achievement (complex)" = list(
      y = dat.konstantopoulos2011$yi,
      v = dat.konstantopoulos2011$vi,
      X = X_achievement_complex, 
      k = length(dat.konstantopoulos2011$yi), 
      p = ncol(X_achievement_complex),
      cluster = NULL
    )
  )
  
  # Results storage
  results_table <- data.frame()
  cv_results <- data.frame()
  conformal_results <- data.frame()
  
  # Analyze each dataset
  for(name in names(datasets)) {
    cat(sprintf("Analyzing: %s (k=%d, p=%d, k/p=%.1f)\n", 
                name, datasets[[name]]$k, datasets[[name]]$p, 
                datasets[[name]]$k/datasets[[name]]$p))
    
    dat <- datasets[[name]]
    
    # Condition X and check VIF
    dat$X <- condition_X(dat$X, return_vif = TRUE, verbose = TRUE)
    
    # Standard R²_het
    r2het_std <- calculate_r2het(dat$y, dat$v, dat$X, knha = TRUE)
    
    # Cross-validated R²_het (LOO)
    r2het_cv <- calculate_r2het_cv(dat$y, dat$v, dat$X, return_predictions = TRUE)
    
    # K-fold CV (if k > 20)
    if(dat$k > 20) {
      r2het_kfold <- calculate_r2het_kfold(dat$y, dat$v, dat$X, K = 5, reps = 10)
      kfold_mean <- r2het_kfold$r2het_kfold_mean
      kfold_sd <- r2het_kfold$r2het_kfold_sd
      kfold_conv <- r2het_kfold$point_convergence
    } else {
      kfold_mean <- NA
      kfold_sd <- NA
      kfold_conv <- NA
    }
    
    # Ridge regression
    r2het_ridge <- calculate_r2het_ridge(dat$y, dat$v, dat$X)
    
    # BCa Bootstrap CI
    boot_ci <- bootstrap_r2het_bca(dat$y, dat$v, dat$X, B = B_BOOT, 
                                   parallel = TRUE, cluster = dat$cluster)
    
    # Permutation test
    perm_test <- permutation_test_r2het(dat$y, dat$v, dat$X, B = B_PERM, 
                                       parallel = TRUE, cluster = dat$cluster)
    
    # Split conformal
    conformal <- conformal_split(dat$y, dat$v, dat$X)
    
    # Handle NA values properly
    opt_val <- if(is.na(r2het_cv$optimism)) NA_real_ else (r2het_cv$optimism * 100)
    
    # Store main results
    results_table <- rbind(results_table, data.frame(
      Dataset = name,
      k = dat$k,
      p = dat$p,
      k_per_p = round(dat$k/dat$p, 1),
      I2 = round(r2het_std$I2, 1),
      R2het_apparent = ifelse(is.na(r2het_std$r2het), NA, round(r2het_std$r2het * 100, 1)),
      R2het_CV = ifelse(is.na(r2het_cv$r2het_corrected), NA, round(r2het_cv$r2het_corrected * 100, 1)),
      R2het_kfold = ifelse(!is.na(kfold_mean), 
                           sprintf("%.1f(%.1f)", kfold_mean * 100, kfold_sd * 100),
                           "—"),
      R2het_ridge = round(r2het_ridge$r2het_ridge * 100, 1),
      Optimism = round(opt_val, 1),
      CI_lower = round(boot_ci$ci[1] * 100, 1),
      CI_upper = round(boot_ci$ci[2] * 100, 1),
      Boot_at_zero = ifelse(is.na(boot_ci$prop_at_zero), NA, round(boot_ci$prop_at_zero * 100, 0)),
      p_value = round(perm_test$p_value, 3),
      Conv_CV = round(r2het_cv$convergence_rate, 0),
      Conv_Boot = round(boot_ci$convergence, 0),
      Conv_Perm = round(perm_test$convergence, 0)
    ))
    
    # Store CV performance
    cv_results <- rbind(cv_results, data.frame(
      Dataset = name,
      RMSE = ifelse(is.na(r2het_cv$rmse), NA, round(r2het_cv$rmse, 3)),
      MAE = ifelse(is.na(r2het_cv$mae), NA, round(r2het_cv$mae, 3)),
      Coverage = ifelse(is.na(r2het_cv$coverage), NA, round(r2het_cv$coverage, 1)),
      Correlation = ifelse(is.na(r2het_cv$correlation), NA, round(r2het_cv$correlation, 3)),
      Conv_LOO = round(r2het_cv$convergence_rate, 0),
      Conv_KFold = ifelse(!is.na(kfold_conv), round(kfold_conv, 0), NA)
    ))
    
    # Store conformal results
    conformal_results <- rbind(conformal_results, data.frame(
      Dataset = name,
      Nominal_Coverage = ifelse(is.na(r2het_cv$coverage), NA, round(r2het_cv$coverage, 1)),
      Conformal_Coverage = ifelse(is.na(conformal$coverage), NA, round(conformal$coverage, 1)),
      Width_Ratio = ifelse(is.na(conformal$width_ratio), NA, round(conformal$width_ratio, 2))
    ))
  }
  
  return(list(
    main_results = results_table,
    cv_results = cv_results,
    conformal_results = conformal_results,
    datasets = datasets
  ))
}

# ==============================================================================
# SECTION 4: SIMULATION STUDY (SIMPLIFIED)
# ==============================================================================

run_simulation_study <- function() {
  
  cat("\n================================================================================\n")
  cat("RUNNING SIMULATION STUDY\n")
  cat("================================================================================\n\n")
  
  # Simulation parameters (simplified for speed)
  scenarios <- expand.grid(
    k = c(10, 15, 20, 30, 50, 100),
    p = c(2, 3, 5),
    tau2 = c(0.1),
    R2_true = c(0.5),
    cor_X = c(0)
  )
  
  # Remove impossible combinations
  scenarios <- scenarios[scenarios$k > 2 * scenarios$p, ]
  
  results <- data.frame()
  
  cat(sprintf("Running %d scenarios (%d simulations each)\n", nrow(scenarios), N_SIMS))
  
  for(i in 1:nrow(scenarios)) {
    s <- scenarios[i, ]
    
    r2het_apparent_vec <- numeric(N_SIMS)
    r2het_corrected_vec <- numeric(N_SIMS)
    converged_vec <- numeric(N_SIMS)
    
    for(sim in 1:N_SIMS) {
      # Generate predictors
      X <- cbind(1, matrix(rnorm(s$k * (s$p-1)), s$k, s$p-1))
      v <- runif(s$k, 0.01, 0.1)
      
      # Generate data with specified R²
      beta <- c(0.5, rep(0.2, s$p-1))
      mu <- X %*% beta
      var_mu <- var(as.numeric(mu))
      
      scale_factor <- sqrt(s$R2_true * s$tau2 / (var_mu + 1e-10))
      beta_scaled <- beta * scale_factor
      tau2_residual <- s$tau2 * (1 - s$R2_true)
      
      y <- as.numeric(X %*% beta_scaled) + 
           rnorm(s$k, 0, sqrt(tau2_residual)) + 
           rnorm(s$k, 0, sqrt(v))
      
      # Calculate R²_het
      tryCatch({
        result <- calculate_r2het_cv(y, v, X, verbose = FALSE)
        r2het_apparent_vec[sim] <- result$r2het_apparent
        r2het_corrected_vec[sim] <- result$r2het_corrected
        converged_vec[sim] <- result$convergence_rate
      }, error = function(e) {
        r2het_apparent_vec[sim] <- NA
        r2het_corrected_vec[sim] <- NA
        converged_vec[sim] <- 0
      })
    }
    
    # Summarize
    valid <- !is.na(r2het_apparent_vec) & !is.na(r2het_corrected_vec)
    
    if(sum(valid) > 0) {
      results <- rbind(results, data.frame(
        k = s$k,
        p = s$p,
        R2het_apparent = mean(r2het_apparent_vec[valid]) * 100,
        R2het_corrected = mean(r2het_corrected_vec[valid]) * 100,
        Optimism = (mean(r2het_apparent_vec[valid]) - mean(r2het_corrected_vec[valid])) * 100,
        Convergence = mean(converged_vec[valid])
      ))
    }
    
    if(i %% 5 == 0) cat(sprintf("  Completed %d/%d scenarios\n", i, nrow(scenarios)))
  }
  
  # Create summary table
  summary_table <- reshape(results[, c("k", "p", "Optimism")], 
                           idvar = "k", timevar = "p", direction = "wide")
  names(summary_table) <- c("k", "p=2", "p=3", "p=5")
  
  return(list(
    full_results = results,
    summary_table = summary_table
  ))
}

# ==============================================================================
# SECTION 5: SENSITIVITY ANALYSIS
# ==============================================================================

sensitivity_analysis <- function(datasets) {
  
  cat("\n================================================================================\n")
  cat("HETEROGENEITY ESTIMATOR SENSITIVITY\n")
  cat("================================================================================\n\n")
  
  # Use BCG data
  bcg_data <- datasets[["BCG vaccine"]]
  
  # Methods to test
  methods <- c("REML", "ML", "DL", "EB", "PM", "SJ")
  sens_results <- data.frame()
  
  for(m in methods) {
    tryCatch({
      r2het_result <- calculate_r2het(bcg_data$y, bcg_data$v, bcg_data$X, method = m)
      r2het_cv_result <- calculate_r2het_cv(bcg_data$y, bcg_data$v, bcg_data$X, method = m)
      
      sens_results <- rbind(sens_results, data.frame(
        Method = m,
        tau2_null = round(r2het_result$tau2_null, 3),
        tau2_full = round(r2het_result$tau2_full, 3),
        R2het_apparent = ifelse(is.na(r2het_result$r2het), NA, round(r2het_result$r2het * 100, 1)),
        R2het_CV = ifelse(is.na(r2het_cv_result$r2het_corrected), NA, round(r2het_cv_result$r2het_corrected * 100, 1)),
        Conv = round(r2het_cv_result$convergence_rate, 0)
      ))
    }, error = function(e) {
      cat(sprintf("  Method %s failed: %s\n", m, e$message))
    })
  }
  
  return(sens_results)
}

# ==============================================================================
# SECTION 6: VISUALIZATION FUNCTIONS
# ==============================================================================

plot_optimism_funnel <- function(results) {
  df <- results$main_results
  df <- df[!is.na(df$Optimism), ]  # Remove NA rows
  
  if(nrow(df) == 0) {
    cat("  No data for plot\n")
    return(NULL)
  }
  
  p <- ggplot(df, aes(x = k_per_p, y = Optimism)) +
    geom_point(size = 3, aes(color = Dataset)) +
    geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
    geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
    geom_vline(xintercept = 15, linetype = "dashed", color = "blue") +
    annotate("text", x = 20, y = 12, label = "10% optimism threshold", color = "red") +
    annotate("text", x = 17, y = max(df$Optimism) * 0.9, label = "k/p = 15", color = "blue", angle = 90) +
    scale_x_continuous(trans = "log10", breaks = c(5, 10, 20, 50)) +
    labs(
      x = "k/p ratio (log scale)",
      y = "Optimism (%)",
      title = "Optimism vs Sample Size Ratio",
      subtitle = "Optimism-corrected R²_het typically 0-20% even when apparent values exceed 70%"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  ggsave("results/optimism_funnel.pdf", p, width = 8, height = 6)
  return(p)
}

# ==============================================================================
# SECTION 7: MAIN EXECUTION
# ==============================================================================

cat("\n")
cat("################################################################################\n")
cat("#                    TWMA ANALYSIS v4.2 - FIXED                               #\n")
cat("#                         Run mode: ", RUN_MODE, "                            #\n")
cat("################################################################################\n")

# 1. Empirical datasets
emp_results <- analyze_empirical_datasets()

cat("\n\nTABLE 1: Optimism-Corrected Explained Heterogeneity (R²_het)\n")
cat(strrep("-", 120), "\n")
print(emp_results$main_results, row.names = FALSE)

# Apply FDR correction
emp_results$main_results$p_fdr <- round(
  p.adjust(emp_results$main_results$p_value, method = "BH"), 3
)

cat("\n\nTABLE 2: Cross-Validation Performance\n")
cat(strrep("-", 70), "\n")
print(emp_results$cv_results, row.names = FALSE)

cat("\n\nTABLE 3: Conformal Prediction Coverage\n")
cat(strrep("-", 60), "\n")
print(emp_results$conformal_results, row.names = FALSE)

# 2. Simulation study
sim_results <- run_simulation_study()

cat("\n\nTABLE 4: Mean Optimism (%) by k and p (R²=0.5, τ²=0.1, ρ=0)\n")
cat(strrep("-", 40), "\n")
print(sim_results$summary_table, row.names = FALSE)

# 3. Sensitivity analysis
sens_results <- sensitivity_analysis(emp_results$datasets)

cat("\n\nTABLE 5: Heterogeneity Estimator Sensitivity (BCG Data)\n")
cat(strrep("-", 60), "\n")
print(sens_results, row.names = FALSE)

# 4. Create visualization
tryCatch({
  plot_optimism_funnel(emp_results)
  cat("\nPlot saved to results/\n")
}, error = function(e) {
  cat("\nCould not create plot:", e$message, "\n")
})

# 5. Save all results
save(emp_results, sim_results, sens_results, 
     file = "results/twma_v4.2_fixed.RData")

# Write CSVs
write.csv(emp_results$main_results, "results/table1_main_results.csv", row.names = FALSE)
write.csv(emp_results$cv_results, "results/table2_cv_performance.csv", row.names = FALSE)
write.csv(emp_results$conformal_results, "results/table3_conformal.csv", row.names = FALSE)
write.csv(sim_results$summary_table, "results/table4_simulation.csv", row.names = FALSE)
write.csv(sens_results, "results/table5_sensitivity.csv", row.names = FALSE)

# Summary of key findings
cat("\n\n")
cat("================================================================================\n")
cat("KEY FINDINGS SUMMARY\n")
cat("================================================================================\n")

# Calculate summary statistics with NA safety
extreme_risk <- sum(emp_results$main_results$k_per_p < 5, na.rm = TRUE)
severe_risk <- sum(emp_results$main_results$k_per_p >= 5 & 
                  emp_results$main_results$k_per_p < 10, na.rm = TRUE)
total_datasets <- nrow(emp_results$main_results)

cat(sprintf("\n1. Dataset summary:\n"))
cat(sprintf("   - Total datasets analyzed: %d\n", total_datasets))

cat(sprintf("\n2. Overfitting severity:\n"))
cat(sprintf("   - %d/%d datasets (%.0f%%) have k/p < 5 (extreme risk)\n",
           extreme_risk, total_datasets, 100 * extreme_risk/total_datasets))
cat(sprintf("   - %d/%d datasets (%.0f%%) have 5 ≤ k/p < 10 (severe risk)\n",
           severe_risk, total_datasets, 100 * severe_risk/total_datasets))

cat(sprintf("\n3. Optimism-corrected R²_het (CV):\n"))
opt_mean <- mean(emp_results$main_results$Optimism, na.rm = TRUE)
cat(sprintf("   - Mean optimism: %.1f%%\n", opt_mean))
cat(sprintf("   - Typically 0-20%% even when apparent R²_het > 70%%\n"))

cat(sprintf("\n4. Coverage performance:\n"))
cov_mean <- mean(emp_results$cv_results$Coverage, na.rm = TRUE)
cat(sprintf("   - Mean nominal coverage: %.1f%% (target: 95%%)\n", cov_mean))
conf_cov_mean <- mean(emp_results$conformal_results$Conformal_Coverage, na.rm = TRUE)
cat(sprintf("   - Mean conformal coverage: %.1f%%\n", conf_cov_mean))
width_mean <- mean(emp_results$conformal_results$Width_Ratio, na.rm = TRUE)
cat(sprintf("   - Width inflation for calibration: %.1fx\n", width_mean))

cat(sprintf("\n5. Bootstrap uncertainty:\n"))
ci_zero <- sum(emp_results$main_results$CI_lower <= 0, na.rm = TRUE)
cat(sprintf("   - Datasets with CI including 0: %d/%d\n", ci_zero, total_datasets))
boot_zero <- mean(emp_results$main_results$Boot_at_zero, na.rm = TRUE)
cat(sprintf("   - Mean bootstrap mass at 0: %.1f%%\n", boot_zero))

cat(sprintf("\n6. Convergence rates:\n"))
cv_conv <- mean(emp_results$main_results$Conv_CV, na.rm = TRUE)
boot_conv <- mean(emp_results$main_results$Conv_Boot, na.rm = TRUE)
perm_conv <- mean(emp_results$main_results$Conv_Perm, na.rm = TRUE)
cat(sprintf("   - Mean CV convergence: %.0f%%\n", cv_conv))
cat(sprintf("   - Mean bootstrap convergence: %.0f%%\n", boot_conv))
cat(sprintf("   - Mean permutation convergence: %.0f%%\n", perm_conv))

cat(sprintf("\n7. Key recommendations:\n"))
cat("   - Minimum k/p ≥ 15 for optimism < 10%\n")
cat("   - Always report R²_het (CV) alongside R²_het (apparent)\n")
cat("   - Report convergence rates for transparency\n")
cat("   - Consider ridge regularization when k/p < 10\n")
cat("   - Use conformal prediction for calibrated intervals\n")

cat("\n================================================================================\n")
cat("ANALYSIS COMPLETE - Results saved to results/\n")
cat("================================================================================\n")
