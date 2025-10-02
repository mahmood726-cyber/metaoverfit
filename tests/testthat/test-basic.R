test_that("r2het works on toy data", {
  yi <- rnorm(10, 0, 1)
  vi <- runif(10, 0.01, 0.1)
  mods <- cbind(1, rnorm(10))

  res <- r2het(yi, vi, mods)
  expect_true(is.list(res))
  expect_true("r2het" %in% names(res))
})
