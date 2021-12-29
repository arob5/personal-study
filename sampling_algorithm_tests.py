# sampling_methods.py
# Testing random sampling methods implemented in sampling_methods.py. 
#
# Andrew Roberts

from sampling_methods import sample_exp
import matplotlib.pyplot as plt
import numpy as np


# Exponential Samples (lambda = 1)
samples_exp = sample_exp(rate_param = 1, N = 10000)
plt.hist(samples_exp, bins = 100)
plt.show()



