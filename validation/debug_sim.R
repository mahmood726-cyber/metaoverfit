# Debug script to find the actual error in simulation

.libPaths("C:/Users/user/AppData/Local/R/win-library/4.5")

# Load packages and metaoverfit from source
library(metafor)
setwd("C:/Users/user/OneDrive - NHS/Documents/metaoverfit")
devtools::load_all(".")

# Simple test case - same as scenario 1
set.seed(12345)
k <- 5
tau2 <- 0.05
beta_mod <- 0.3
p <- 2

# Generate data like the simulation
vi <- runif(k, 0.01, 0.1)
mods <- cbind(1, rnorm(k, 0, 1))
beta <- c(0, beta_mod)
ui <- rnorm(k, 0, sqrt(tau2))
yi <- mods %*% beta + ui + rnorm(k, 0, sqrt(vi))

cat("Data generated:\n")
cat(sprintf("  yi: %s\n", paste(round(as.vector(yi), 4), collapse = ", ")))
cat(sprintf("  vi: %s\n", paste(round(vi, 4), collapse = ", ")))
cat(sprintf("  mods dimensions: %d x %d\n", nrow(mods), ncol(mods)))

# Try to run check_overfitting
cat("\nCalling check_overfitting...\n")

tryCatch({
  result <- check_overfitting(
    yi = as.vector(yi),
    vi = vi,
    mods = mods,
    B = 50
  )
  cat("SUCCESS!\n")
  print(result)
}, error = function(e) {
  cat("ERROR:\n")
  cat(sprintf("  Message: %s\n", e$message))
  cat(sprintf("  Call: %s\n", deparse(e$call)))
})
