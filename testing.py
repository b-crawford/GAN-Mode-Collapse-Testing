import numpy as np


# define actual distribution
real_mean_1 = 6
real_sd_1 = 1

real_mean_2 = 2
real_sd_2 = 0.5

sample_size = 3

real_dist_1 = np.random.normal(real_mean_1, real_sd_1, (sample_size, 1))
real_dist_2 = np.random.normal(real_mean_2, real_sd_2, (sample_size, 1))
real_dist = real_dist_1 + real_dist_2

print real_dist_1
print real_dist_2
print real_dist