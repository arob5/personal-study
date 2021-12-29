# sampling_methods.py
# Basic implementations of various random sampling methods. 
# Source: Monte Carlo Statistical Methods (Robert and Casella)
#
# Andrew Roberts

import numpy as np

def sample_exp(rate_param, N):
	"""
	Samples from exponential distribution, parameterized by rate (inverse mean) parameter. 
	Uses inverse transform sampling method. 

	Args:
		- rate_param: float, the rate parameter (often called lambda). 
		- N: int, number of samples. 

	Returns: 
		numpy array of length N. 
	"""

	uniform_samples = np.random.uniform(size = N)
	quantile_function = lambda u: -1 * np.log(1 - u) / rate_param
	exp_samples = np.array([quantile_function(u) for u in uniform_samples])

	return exp_samples
