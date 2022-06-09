# gaussian_process_regression_practice.r
# Playing around with the basics of GP regression. 
#
# Andrew Roberts
# Working Directory: personal-study

library(mvtnorm)
library(plgp)

# Define kernel to be used in these examples: exponentiated quadratic distance with stationary variance 1. 
eps <- sqrt(.Machine$double.eps) # "Jitter" to ensure positive definite matrices
K <- function(X1, X2 = NA) {
  if(is.na(X2[1][1])) X2 <- X1
  exp(-distance(X1, X2))
}

# Sample from GP at finite subset of points (taking mean function to be 0 throughout this code).
n <- 100
X <- matrix(seq(-4, 4, length.out = n), ncol = 1)
Y <- rmvnorm(1, sigma = K(X))
plot(X, Y)

# Compare a few different draws from this MVN distribution. We'll also connect the dots with lines 
# to make this cleaner to look at. 
Y <- rmvnorm(3, sigma = K(X))
matplot(X, t(Y), type = 'l')
