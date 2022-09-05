# gaussian_process_regression_practice.r
# Playing around with the basics of GP regression. 
#
# Andrew Roberts
# Working Directory: personal-study

library(mvtnorm)
library(mlegp)
library(laGP)
library(plgp)

# Define kernel to be used in these examples: exponentiated quadratic distance with stationary variance 1. 
K <- function(X1, X2 = NA, l = 1.0) {
  if(is.na(X2[1][1])) X2 <- X1
  exp(-plgp::distance(X1, X2) / l)
}

# Polynomial Kernel (linear for p=1)
K.poly <- function(X1, X2 = NA, sig2 = 0, p = 1) {
  if(is.na(X2[1][1])) X2 <- X1
  (sig2 + tcrossprod(X1, X2))^p
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
predict_mean <- function(x_pred, x_obs, y_obs) {
  x_pred <- matrix(x_pred, ncol = 1)
  eps <- sqrt(.Machine$double.eps) # "Jitter" to ensure positive definite matrices
  cross_cov <- K(x_pred, x_obs)
  data_cov <- K(x_obs, x_obs) + diag(rep(eps, nrow(x_obs)))
  return(cross_cov %*% solve(data_cov) %*% y_obs)
}

predict_var_1d <- function(x_pred, x_obs, y_obs) {
  eps <- sqrt(.Machine$double.eps) 
  x_pred <- matrix(x_pred, ncol = 1)
  prior_var <- K(x_pred, x_pred)
  cross_cov <- K(x_pred, x_obs)
  data_cov <- K(x_obs, x_obs) + diag(rep(eps, nrow(x_obs)))
  return(prior_var - cross_cov %*% solve(data_cov) %*% t(cross_cov))
}

print("Predicted means: ")
print(sapply(x_pred, function(x) predict_mean(x, x_obs, y_obs)))

print("Predicted variances: ")
print(sapply(x_pred, function(x) predict_var_1d(x, x_obs, y_obs)))

print("Predicted means pass through observed points (interpolation):")
print(sapply(x_obs, function(x) predict_mean(x, x_obs, y_obs)))

print("Predicted variances are (almost) zero at observed points; 
      not zero due to jitter and numerical error:")
print(sapply(x_obs, function(x) predict_var_1d(x, x_obs, y_obs)))

# Predicting at many points - still "pointwise" predictions (ignoring covariances)
x_pred <- matrix(seq(-4, 5, length.out = 100), ncol = 1)
means_pred <- sapply(x_pred, function(x) predict_mean(x, x_obs, y_obs))
vars_pred <- sapply(x_pred, function(x) predict_var_1d(x, x_obs, y_obs))

# Drawing samples form "posterior". Plotting observed data (points), 
# three samples from posterior (gray lines) and 90% interval bars
# (red dashed lines).
y_posterior_samples <- rmvnorm(n = 3, means_pred, diag(vars_pred))
matplot(x_pred, t(y_posterior_samples), type = 'l', col='gray', lty=1)
matpoints(x_obs, y_obs, pch = 20, cex = 2)
interval_90_upper <- qnorm(.95, means_pred, sqrt(vars_pred))
interval_90_lower <- 2 * means_pred - interval_90_upper
lines(x_pred, interval_90_upper, lwd=2, lty=2, col=2)
lines(x_pred, interval_90_lower, lwd=2, lty=2, col=2)

# Now predict using multivariate formulas. This won't change the posterior means but
# it will account for correlation between the points and hence affect the posterior
# uncertainty bars. 
predict_cov <- function(x_pred, x_obs, y_obs) {
  eps <- sqrt(.Machine$double.eps) 
  x_pred <- matrix(x_pred, ncol = 1)
  prior_cov <- K(x_pred, x_pred)
  cross_cov <- K(x_pred, x_obs)
  data_cov <- K(x_obs, x_obs) + diag(rep(eps, nrow(x_obs)))
  return(prior_cov - cross_cov %*% solve(data_cov) %*% t(cross_cov))
}

# Plot posterior samples with uncertainty as before, but this time accounting for correlation. 
cov_pred <- predict_cov(x_pred, x_obs, y_obs)
y_posterior_samples_cov <- rmvnorm(n = 3, means_pred, cov_pred)
matplot(x_pred, t(y_posterior_samples_cov), type = 'l', col='gray', lty=1)
matpoints(x_obs, y_obs, pch = 20, cex = 2)
interval_90_upper <- qnorm(.95, means_pred, sqrt(diag(cov_pred)))
interval_90_lower <- 2 * means_pred - interval_90_upper
lines(x_pred, interval_90_upper, lwd=2, lty=2, col=2)
lines(x_pred, interval_90_lower, lwd=2, lty=2, col=2)

# Same example but using laGP package
gp <- newGP(x_obs, y_obs, d = 1, g = 0)
gp_pred <- predGP(gp, x_pred, nonug = TRUE)

y_posterior_samples_laGP <- rmvnorm(n = 3, gp_pred$mean, gp_pred$Sigma)
matplot(x_pred, t(y_posterior_samples_laGP), type = 'l', col='gray', lty=1)
matpoints(x_obs, y_obs, pch = 20, cex = 2)
interval_90_upper <- qnorm(.95, gp_pred$mean, sqrt(max(diag(gp_pred$Sigma), 0)))
interval_90_lower <- 2 * gp_pred$mean - interval_90_upper
lines(x_pred, interval_90_upper, lwd=2, lty=2, col=2)
lines(x_pred, interval_90_lower, lwd=2, lty=2, col=2)

# Same example but using mlegp package; couldn't get this to run when setting nugget to 
# 0 or sqrt(.Machine$double.eps) 
gp_mle <- mlegp(as.vector(x_obs), as.vector(y_obs))
gp_pred_mle <- predict.gp(gp_mle, x_pred, se.fit = TRUE)

y_posterior_samples_mle <- rmvnorm(n = 3, gp_pred_mle$fit, gp_pred_mle$se.fit)
matplot(x_pred, t(y_posterior_samples_laGP), type = 'l', col='gray', lty=1)
matpoints(x_obs, y_obs, pch = 20, cex = 2)
interval_90_upper <- qnorm(.95, gp_pred$mean, sqrt(max(diag(gp_pred$Sigma), 0)))
interval_90_lower <- 2 * gp_pred$mean - interval_90_upper
lines(x_pred, interval_90_upper, lwd=2, lty=2, col=2)
lines(x_pred, interval_90_lower, lwd=2, lty=2, col=2)


#
# Investigating mlegp package more closely
#

# First, another toy 1D example
n <- 8
X <- matrix(seq(0, 2*pi, length=n), ncol=1)
y <- sin(X)

gp_mle <- mlegp(X, y, nugget = 0, nugget.known = 1)
gp_mle$mu # Constant mean
gp_mle$Bhat # Constant mean equal to regression coefs in this case since constantMean == 1
gp_mle$beta # Lengthscale parameter
gp_mle$a # What is this?
gp_mle$sig2 # Overall unconditional GP variance (I think?)
gp_mle$nugget # Nugget was set to 0 in this case


X_test <- matrix(seq(-0.5, 2*pi + 0.5, length=100), ncol=1)
results <- predict.gp(gp_mle, X_test, se.fit = TRUE)

matplot(X_test, results$fit, type = 'l', col='blue', lty=1)
matpoints(X, y, pch = 20, cex = 2)
interval_90_upper <- qnorm(.95, results$fit, results$se.fit)
interval_90_lower <- 2 * results$fit - interval_90_upper

interval_90_upper <- results$fit + results$se.fit
interval_90_lower <- results$fit - results$se.fit

lines(X_test, interval_90_upper, lwd=2, lty=2, col=2)
lines(X_test, interval_90_lower, lwd=2, lty=2, col=2)


# Now, a 2D example
library(lhs) 
X <- randomLHS(40, 2) # Latin Hypercube design
X[,1] <- (X[,1] - 0.5)*6 + 1 # Map into interval [-2, 4]^2
X[,2] <- (X[,2] - 0.5)*6 + 1
y <- X[,1]*exp(-X[,1]^2 - X[,2]^2) # Function we want to learn
X_test <- seq(-2, 4, length=40)
X_test <- expand.grid(X_test, X_test)

gp <- mlegp(X, y, nugget = 0, nugget.known = 1)
gp$mu
gp$Bhat
gp$beta
gp$a

#
# Exploring kernels
#

X <- matrix(seq(0, 1, length.out = 1000), ncol = 1)

# Squared exponential kernel
y.sq.exp <- rmvnorm(3, sigma = K(X))
matplot(X, t(y.sq.exp), type = 'l')

# Linear kernel
y.lin <- rmvnorm(3, sigma = K.poly(X))
matplot(X, t(y.lin), type = 'l')

# Linear kernel with non-zero y-intercept
y.lin.intercept <- rmvnorm(3, sigma = K.poly(X, sig2 = 10))
matplot(X, t(y.lin.intercept), type = 'l')

# Squared exponential plus linear kernel
y <- rmvnorm(5, sigma = K(X) + K.poly(X))
matplot(X, t(y), type = 'l')

# Squared exponential with small lengthscale plus linear kernel with intercept
y <- rmvnorm(5, sigma = K(X, l = 0.1) + K.poly(X, sig2 = 10))
matplot(X, t(y), type = 'l')

# Quadratic
y <- rmvnorm(5, sigma = K.poly(X, p = 2))
matplot(X, t(y), type = 'l')

# Quadratic, on [-1, 1]
y <- rmvnorm(5, sigma = K.poly(rbind(-X, X), p = 2))
matplot(rbind(-X, X), t(y), type = 'l')

# Quadratic, shifting X so that zero variance point is at x = 0.5
y <- rmvnorm(5, sigma = K.poly(X - 0.5, p = 2))
matplot(X, t(y), type = 'l')

# Quadratic, shifting X so that zero variance point is at x = 0.5
# And adding variance constant
y <- rmvnorm(5, sigma = K.poly(X - 0.5, p = 2, sig2 = 2))
matplot(X, t(y), type = 'l')

# Squared Exponential plus Quadratic
y <- rmvnorm(5, sigma = K(X, l = 0.1) + K.poly(X - 0.5, p = 2))
matplot(X, t(y), type = 'l')








