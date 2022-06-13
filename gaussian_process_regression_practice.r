# gaussian_process_regression_practice.r
# Playing around with the basics of GP regression. 
#
# Andrew Roberts
# Working Directory: personal-study

library(mvtnorm)
library(plgp)

# Define kernel to be used in these examples: exponentiated quadratic distance with stationary variance 1. 
eps <- sqrt(.Machine$double.eps) 
K <- function(X1, X2 = NA) {
  if(is.na(X2[1][1])) X2 <- X1
  exp(-distance(X1, X2))
}

#
# Sampling from GP defined on the real line 
#

# Sample from GP at finite subset of points (taking mean function to be 0 throughout this code).
n <- 100
X <- matrix(seq(-4, 4, length.out = n), ncol = 1)
Y <- rmvnorm(1, sigma = K(X))
plot(X, Y)

# Compare a few different draws from this MVN distribution. We'll also connect the dots with lines 
# to make this cleaner to look at. 
Y <- rmvnorm(3, sigma = K(X))
matplot(X, t(Y), type = 'l')

#
# Playing around with the "kriging equations"; that is, the posterior/predictive distribution at some finite
# subset of locations conditional on observed values at some other set of locations.
#

# Observed locations and values
x_obs <- matrix(c(-4, -2, -1, 1, 2, 5), ncol = 1)
y_obs <- x_obs^2
plot(x_obs, y_obs, main = 'Observed data')

# Prediction locations
x_pred <- matrix(c(-3, 0, 3.5), ncol = 1)

# Predict one at a time, ignoring covariance between predictions. 
predict_mean_1d <- function(x) {
  x <- matrix(x, ncol = 1)
  eps <- sqrt(.Machine$double.eps) # "Jitter" to ensure positive definite matrices
  cross_cov <- K(x, x_obs)
  data_cov <- K(x_obs, x_obs) + diag(rep(eps, nrow(x_obs)))
  return(cross_cov %*% solve(data_cov) %*% y_obs)
}

predict_var_1d <- function(x) {
  x <- matrix(x, ncol = 1)
  prior_var <- K(x, x)
  cross_cov <- K(x, x_obs)
  data_cov <- K(x_obs, x_obs) + diag(rep(eps, nrow(x_obs)))
  return(prior_var - cross_cov %*% solve(data_cov) %*% t(cross_cov))
}

print("Predicted means: ")
print(sapply(x_pred, predict_mean_1d))

print("Predicted variances: ")
print(sapply(x_pred, predict_var_1d))

print("Predicted means pass through observed points (interpolation):")
print(sapply(x_obs, predict_mean_1d))

print("Predicted variances are (almost) zero at observed points; 
      not zero due to jitter and numerical error:")
print(sapply(x_obs, predict_var_1d))

#
# More substantial 1d example using kriging equations: 
#    - Predicting at larger number of points
#    - Generate samples from posterior
#    - Compare 1d prediction that ignores covariances with 1d prediction 
#      that includes covariances 
#


