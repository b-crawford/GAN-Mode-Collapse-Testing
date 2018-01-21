import numpy as np


# define actual distribution
real_mean_1 = 6
real_sd_1 = 1

real_mean_2 = 10
real_sd_2 = 100

sample_size = 3


which = np.random.choice((0,1),sample_size)
means = which*real_mean_1 + (1-which)*real_mean_2
sds = which*real_sd_1 + (1-which)*real_sd_2
samples = np.random.normal(means,sds,sample_size)

print means
print sds
print samples

def guassian_mixture(mu_1,sd_1,mu_2,sd_2,weights, sample_size):
    means = 10