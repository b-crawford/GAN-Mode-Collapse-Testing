



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

batch_size = 1000


real_mean_1 = 0
real_sd_1 = 1

real_mean_2 = 10
real_sd_2 = 1


for i in range(10):
    which = np.random.choice((0, 1), batch_size)  # bernoulli deciding which gaussian to sample from
    means = which * real_mean_1 + (1 - which) * real_mean_2  # chooses mean_1 if which = 1
    sds = which * real_sd_1 + (1 - which) * real_sd_2  # chooses sd_1 if which = 1
    real_dist = np.random.normal(means, sds, batch_size)  # generate samples
    real_dist = real_dist.reshape((batch_size, 1))
    sns.distplot(real_dist, hist=True, rug=False)
    plt.show()