# ============================================================================
# Advanced Tests for metaoverfit
# ============================================================================
# Tests for parallel processing, advanced plots, and more complex scenarios.

test_that("Parallel processing runs without error", {
  skip_on_cran()  # Skip on CRAN to avoid machine resource issues
  
  # Check if parallel packages are available
  if (!requireNamespace("parallel", quietly = TRUE) ||
      !requireNamespace("doParallel", quietly = TRUE) ||
      !requireNamespace("foreach", quietly = TRUE)) {
    skip("Parallel packages not installed")
  }
  
  set.seed(123)
  k <- 20
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))
  
  # Run small bootstrap in parallel
  res <- r2het_boot(yi, vi, mods, B = 20, parallel = TRUE, n_cores = 2, verbose = FALSE)
  
  expect_true(is.list(res))
  expect_length(res$ci_apparent, 2)
  
  if (res$convergence == 0) {
    # If convergence is 0, print some bootstrap values to debug
    print("Bootstrap values (apparent):")
    print(head(res$bootstrap_values$apparent))
  }
  
  expect_true(res$convergence > 0)
})

test_that("check_overfitting accepts parallel argument", {
  skip_on_cran()
  
  if (!requireNamespace("parallel", quietly = TRUE)) skip("Parallel packages not installed")

  set.seed(123)
  k <- 20
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))
  
  res <- check_overfitting(yi, vi, mods, B = 20, parallel = TRUE, n_cores = 2)
  
  expect_true(inherits(res, "metaoverfit"))
})

test_that("plot_overfitting supports density plot", {
  skip_if_not_installed("ggplot2")
  
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))
  
  res <- check_overfitting(yi, vi, mods, B = 50)
  
  # Density plot
  p <- plot_overfitting(res, type = "density")
  expect_true(inherits(p, "ggplot"))
  
  # Check failure on missing bootstrap values
  res_broken <- res
  res_broken$bootstrap_values <- NULL
  expect_error(plot_overfitting(res_broken, type = "density"), "No bootstrap values")
})
