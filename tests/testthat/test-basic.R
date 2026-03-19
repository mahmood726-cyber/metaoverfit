# Test suite for metaoverfit package

# ==============================
# Basic Structure Tests
# ==============================

test_that("r2het works on toy data", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)
  mods <- cbind(1, rnorm(10))

  res <- r2het(yi, vi, mods)
  expect_true(is.list(res))
  expect_true("r2het" %in% names(res))
})

# ==============================
# Statistical Validity Tests
# ==============================

test_that("r2het returns values in valid range", {
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))

  res <- r2het(yi, vi, mods)

  # R^2_het should be between 0 and 1
  expect_gte(res$r2het, 0)
  expect_lte(res$r2het, 1)

  # Adjusted R^2 should be <= apparent R^2
  expect_lte(res$r2het_adj, res$r2het)

  # tau2 values should be non-negative
  expect_gte(res$tau2_null, 0)
  expect_gte(res$tau2_full, 0)

  # k and p should be positive
  expect_gt(res$k, 0)
  expect_gt(res$p, 0)
})

test_that("r2het_cv produces corrected R^2 <= apparent R^2", {
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))

  res <- r2het_cv(yi, vi, mods, verbose = FALSE)

  # Corrected R^2 should be <= apparent R^2 (optimism correction)
  expect_lte(res$r2het_corrected, res$r2het_apparent)

  # Optimism should be non-negative
  expect_gte(res$optimism, 0)

  # Convergence rate should be between 0 and 100
  expect_gte(res$convergence_rate, 0)
  expect_lte(res$convergence_rate, 100)
})

test_that("bootstrap produces valid confidence intervals", {
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))

  res <- r2het_boot(yi, vi, mods, B = 100, verbose = FALSE)

  # CI should have 2 elements
  expect_length(res$ci_apparent, 2)
  expect_length(res$ci_corrected, 2)

  # CI lower bound <= upper bound
  expect_lte(res$ci_apparent[1], res$ci_apparent[2])
  expect_lte(res$ci_corrected[1], res$ci_corrected[2])

  # Values should be in [0, 1] range
  expect_gte(res$ci_apparent[1], 0)
  expect_lte(res$ci_apparent[2], 1)
})

# ==============================
# Edge Case Tests
# ==============================

test_that("r2het handles NULL moderators correctly", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)

  res <- r2het(yi, vi, mods = NULL)

  expect_equal(res$r2het, 0)
  expect_equal(res$r2het_adj, 0)
  expect_equal(res$tau2_null, res$tau2_full)
})

test_that("r2het_cv handles NULL moderators correctly", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)

  res <- r2het_cv(yi, vi, mods = NULL, verbose = FALSE)

  expect_equal(res$r2het_apparent, 0)
  expect_equal(res$r2het_corrected, 0)
  expect_equal(res$optimism, 0)
})

test_that("r2het_boot handles NULL moderators correctly", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)

  res <- r2het_boot(yi, vi, mods = NULL, B = 50, verbose = FALSE)

  expect_equal(res$ci_apparent, c(0, 0))
  expect_equal(res$ci_corrected, c(0, 0))
})

test_that("check_overfitting handles edge cases", {
  set.seed(123)
  k <- 15
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k), rnorm(k))  # p=3, k/p=5 < 10

  res <- check_overfitting(yi, vi, mods, B = 50)

  # Should detect severe risk
  expect_true(res$risk_category %in% c("Severe", "Extreme"))
})

test_that("k <= p+1 produces appropriate warning and NA results", {
  set.seed(123)
  k <- 5
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k), rnorm(k), rnorm(k))  # p=4, k=5, k <= p+1

  expect_warning(
    res <- r2het_cv(yi, vi, mods, verbose = FALSE),
    "Too few studies"
  )

  expect_true(is.na(res$tau2_cv) || res$tau2_cv == 0)
})

# ==============================
# Input Validation Tests
# ==============================

test_that("r2het validates input lengths", {
  yi <- rnorm(10, 0, 1)
  vi <- runif(5, 0.01, 0.1)  # Wrong length
  mods <- cbind(1, rnorm(10))

  expect_error(
    r2het(yi, vi, mods),
    "must have the same length"
  )
})

test_that("r2het_cv validates formula with data", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)

  expect_error(
    r2het_cv(yi, vi, mods = ~ ablat, data = NULL),
    "ablat"  # Error will be about missing object since data is NULL
  )
})

test_that("r2het_cv validates cv_method", {
  set.seed(123)
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)
  mods <- cbind(1, rnorm(10))

  expect_error(
    r2het_cv(yi, vi, mods, cv_method = "invalid"),
    "cv_method must be"
  )
})

# ==============================
# Formula Interface Tests
# ==============================

test_that("r2het works with formula interface", {
  set.seed(123)
  dat <- data.frame(
    yi = rnorm(20, 0, sqrt(0.1)),
    vi = runif(20, 0.01, 0.1),
    x1 = rnorm(20),
    x2 = rnorm(20)
  )

  res <- r2het(yi = dat$yi, vi = dat$vi,
               mods = ~ x1 + x2, data = dat)

  expect_true(is.list(res))
  expect_true("r2het" %in% names(res))
  expect_equal(res$p, 3)  # intercept + x1 + x2
})

test_that("r2het_cv works with formula interface", {
  set.seed(123)
  dat <- data.frame(
    yi = rnorm(20, 0, sqrt(0.1)),
    vi = runif(20, 0.01, 0.1),
    x1 = rnorm(20)
  )

  res <- r2het_cv(yi = dat$yi, vi = dat$vi,
                  mods = ~ x1, data = dat, verbose = FALSE)

  expect_true(is.list(res))
  expect_true("r2het_apparent" %in% names(res))
})

# ==============================
# Sample Size Recommendation Tests
# ==============================

test_that("sample_size_recommendation returns valid values", {
  res <- sample_size_recommendation(p = 2, target_optimism = 0.10, verbose = FALSE)

  expect_equal(res$p, 2)
  expect_equal(res$target_optimism, 0.10)
  expect_gt(res$min_k, 0)
  expect_gt(res$min_ratio, 0)

  # min_k should be at least max(20, p * ratio)
  expect_gte(res$min_k, 20)
})

test_that("sample_size_recommendation adjusts for different optimism targets", {
  res_low <- sample_size_recommendation(p = 2, target_optimism = 0.05, verbose = FALSE)
  res_high <- sample_size_recommendation(p = 2, target_optimism = 0.20, verbose = FALSE)

  # Lower optimism target should require larger sample size
  expect_gte(res_low$min_ratio, res_high$min_ratio)
})

# ==============================
# Print Method Tests
# ==============================

test_that("print.metaoverfit produces output", {
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))

  res <- check_overfitting(yi, vi, mods, B = 50)

  # Should produce output without error
  expect_message(print(res))
})

test_that("print.sample_size_rec produces output", {
  res <- sample_size_recommendation(p = 2, verbose = FALSE)

  expect_message(print(res))
})

# ==============================
# Class Tests
# ==============================

test_that("check_overfitting returns correct class", {
  set.seed(123)
  k <- 30
  yi <- rnorm(k, 0, sqrt(0.1))
  vi <- runif(k, 0.01, 0.1)
  mods <- cbind(1, rnorm(k))

  res <- check_overfitting(yi, vi, mods, B = 50)

  expect_true(inherits(res, "metaoverfit"))
  expect_true(inherits(res, "list"))
})

test_that("sample_size_recommendation returns correct class", {
  res <- sample_size_recommendation(p = 2, verbose = FALSE)

  expect_true(inherits(res, "sample_size_rec"))
  expect_true(inherits(res, "list"))
})
