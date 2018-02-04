
# coding: utf-8

# Import packages

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Structure: in each trial generate parameters, then for number_of_epochs generate a batch of size 'batch_size' each time from the input distribution and the real distribution
# and train the GAN on it. Randomly select which objective function for G and whether to do minibatch selection

# In[12]:

number_of_trails = 1000
number_of_epochs = 100000
batch_size = 2000
hidden_layer_size_d = 6
hidden_layer_size_g = 5


for it in range(1,number_of_trails+1):


    which_objective = np.random.choice((1, 2), 1)[0]
    minibatch_discrimination = np.random.choice((True, False), 1)[0]


    print 'Objective function: {}'.format(which_objective)
    print 'Minibatch discrimination: {}'.format(minibatch_discrimination)

    # Define actual distribution, Gaussian mixture model:

    # In[3]:


    real_mean_1 = 0
    real_sd_1 = 1

    real_mean_2 = 10
    real_sd_2 = 1


    # Discriminator and generator NNs

    # In[4]:


    def minibatch_l1(input):
        maxi = tf.reduce_max(input)
        mini = tf.reduce_min(input)
        return tf.subtract(maxi,mini)


    # Discriminator and generator NNs

    # In[5]:


    def discriminator(input, parameters, max_batch_dist, batch_mean):
        pre_1 = tf.add(tf.matmul(tf.to_float(input), parameters[0]), parameters[1])
        activ_1 = tf.tanh(pre_1)
        pre_2 = tf.add(tf.matmul(activ_1, parameters[2]), parameters[3])
        activ_2 = tf.tanh(pre_2)
        pre_3 = tf.add(tf.matmul(activ_2, parameters[4]), parameters[5])
        output1 = tf.sigmoid(pre_3)

        mini_1 = tf.add(tf.multiply(max_batch_dist,parameters[6]),parameters[7])

        mean_1 = tf.add(tf.multiply(batch_mean,parameters[8]),parameters[9])

        mixed1 = tf.add(tf.multiply(mini_1,parameters[10]),pre_3)
        mixed2 = tf.add(tf.multiply(mean_1,parameters[11]),mixed1)
        output = tf.sigmoid(mixed2)
        return output

    def simple_discriminator(input, parameters):
        pre_1 = tf.add(tf.matmul(tf.to_float(input), parameters[0]), parameters[1])
        activ_1 = tf.tanh(pre_1)
        pre_2 = tf.add(tf.matmul(activ_1, parameters[2]), parameters[3])
        activ_2 = tf.tanh(pre_2)
        pre_3 = tf.add(tf.matmul(activ_2, parameters[4]), parameters[5])
        output1 = tf.sigmoid(pre_3)
        return output1


    def generator(input, parameters):
        pre_1 = tf.add(tf.matmul(tf.to_float(input), parameters[0]), parameters[1])
        activ_1 = tf.tanh(pre_1)
        output = tf.add(tf.matmul(activ_1, parameters[2]), parameters[3])
        return output

    def batch_generator(input, parameters):
        pre_1 = tf.add(tf.matmul(tf.to_float(input), parameters[0]), parameters[1])
        activ_1 = tf.tanh(pre_1)
        output = tf.add(tf.matmul(activ_1, parameters[2]), parameters[3])
        return output






    # D parameters

    # In[6]:


    weight_d_1 = tf.Variable(tf.random_uniform([1, hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
    bias_d_1 = tf.Variable(tf.random_uniform([hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
    weight_d_2 = tf.Variable(tf.random_uniform([hidden_layer_size_d, hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
    bias_d_2 = tf.Variable(tf.random_uniform([hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
    weight_d_3 = tf.Variable(tf.random_uniform([hidden_layer_size_d, 1], minval=0, maxval=1, dtype=tf.float32))
    bias_d_3 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

    mini_weight_1 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))
    mini_bias_1 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

    mean_weight_1 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))
    mean_bias_1 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

    mixing_weight1 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))
    mixing_weight2 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

    d_parameters = [weight_d_1,bias_d_1, weight_d_2, bias_d_2,weight_d_3,
                    bias_d_3,mini_weight_1,mini_bias_1,mean_weight_1,mean_bias_1,mixing_weight1,mixing_weight2]

    simple_d_parameters = [weight_d_1,bias_d_1, weight_d_2, bias_d_2,weight_d_3, bias_d_3]



    # G parameters

    # In[7]:



    weight_g_1 = tf.Variable(tf.random_uniform([1, hidden_layer_size_g], minval=0, maxval=1, dtype=tf.float32))
    bias_g_1 = tf.Variable(tf.random_uniform([hidden_layer_size_g], minval=0, maxval=1, dtype=tf.float32))
    weight_g_2 = tf.Variable(tf.random_uniform([hidden_layer_size_g, 1], minval=0, maxval=1, dtype=tf.float32))
    bias_g_2 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

    g_parameters = [weight_g_1,bias_g_1, weight_g_2, bias_g_2]



    # Losses:

    # In[8]:


    real_dist_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
    generator_input_placeholder = tf.placeholder(tf.float32, shape=(None, 1))


    with tf.variable_scope("Discrim") as scope:
        real_batch_l1 = minibatch_l1(real_dist_placeholder)
        real_mean = tf.reduce_mean(real_dist_placeholder)
        discriminator_dict_1 = {False: simple_discriminator(real_dist_placeholder,simple_d_parameters),
                            True: discriminator(real_dist_placeholder, d_parameters, real_batch_l1,real_mean)}
        d_output_real = discriminator_dict_1[minibatch_discrimination]
        scope.reuse_variables()
        fake_batch_l1 = minibatch_l1(generator(generator_input_placeholder, g_parameters))
        fake_mean = tf.reduce_mean(generator(generator_input_placeholder, g_parameters))
        discriminator_dict_2 = {False: simple_discriminator(generator(generator_input_placeholder, g_parameters),simple_d_parameters),
                            True: discriminator(generator(generator_input_placeholder, g_parameters), d_parameters,fake_batch_l1,fake_mean)}
        d_output_fake = discriminator_dict_2[minibatch_discrimination]


    objectives = {1: tf.reduce_mean(tf.log(1-d_output_fake)) , 2: -tf.reduce_mean(tf.log(d_output_fake))}
    loss_d = tf.reduce_mean(-tf.log(d_output_real) - tf.log(1 - d_output_fake))
    loss_g = objectives[which_objective]




    # Train step:

    # In[ ]:


    learning_rate = tf.placeholder(tf.float32)

    train_g = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_g, var_list=g_parameters)
    train_d = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_d, var_list=d_parameters)

    minibatch_dict = {True: 'minibatch', False : 'no_minibatch'}

    data_directory = '/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/objective_{}_{}'.format(which_objective,minibatch_dict[minibatch_discrimination])
    os.chdir(data_directory)


# Run:

# In[ ]:



    # sample parameters
    learning_rate_vec = np.random.uniform(0.000001,0.1,1)

    res_matrix = np.zeros((len(learning_rate_vec), batch_size))
    learning_rate_out_vec = np.zeros((len(learning_rate_vec)))
    time_taken_out_vec = np.zeros((len(learning_rate_vec)))

    row =0
    for i, p in enumerate(learning_rate_vec):
        start_time = time.time()

        print 'Trial: {}/{}'.format(it,number_of_trails)
        print 'Step: {}/{}'.format(row+1, len(learning_rate_vec))
        print 'Learning Rate: {0}'.format(p)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # writer = tf.summary.FileWriter('./graphs', sess.graph)
            
            for step in tqdm(range(number_of_epochs)):
                generator_input = np.random.uniform(0, 1, (batch_size, 1))
                # sample data
                which = np.random.choice((0, 1), batch_size)  # bernoulli deciding which gaussian to sample from
                means = which * real_mean_1 + (1 - which) * real_mean_2  # chooses mean_1 if which = 1
                sds = which * real_sd_1 + (1 - which) * real_sd_2  # chooses sd_1 if which = 1
                real_dist = np.random.normal(means, sds, batch_size)  # generate samples
                real_dist = real_dist.reshape((batch_size, 1))

                sess.run(train_d, feed_dict={real_dist_placeholder: real_dist,
                                             generator_input_placeholder: generator_input,
                                             learning_rate : learning_rate_vec[i]})
                sess.run(train_g, feed_dict={real_dist_placeholder: real_dist,
                                             generator_input_placeholder: generator_input,
                                             learning_rate: learning_rate_vec[i]})
            
            generator_input = np.random.uniform(0, 1, (batch_size, 1))
            which = np.random.choice((0, 1), batch_size)  # bernoulli deciding which gaussian to sample from
            means = which * real_mean_1 + (1 - which) * real_mean_2  # chooses mean_1 if which = 1
            sds = which * real_sd_1 + (1 - which) * real_sd_2  # chooses sd_1 if which = 1
            real_dist = np.random.normal(means, sds, batch_size)  # generate samples
            real_dist = real_dist.reshape((batch_size, 1))

            generated = sess.run(generator(generator_input, g_parameters))
            
            res_matrix[row] = generated.reshape(batch_size)
            learning_rate_out_vec[row] = p
            time_taken_out_vec[row] = time.time()-start_time
            row = row + 1
            
            # sns.distplot(generated, hist=True, rug=False)
            # sns.distplot(real_dist, hist=False, rug=False)
            # plt.show()

    res_dataframe = pd.DataFrame(data=res_matrix.astype(float))
    learning_rate_dataframe = pd.DataFrame(data=learning_rate_out_vec.astype(float))
    time_dataframe = pd.DataFrame(data=time_taken_out_vec.astype(float))
	
    output_dataframe1 = pd.concat([learning_rate_dataframe.reset_index(drop=True), time_dataframe], axis=1)
    output_dataframe = pd.concat([output_dataframe1.reset_index(drop=True), res_dataframe], axis=1)

    with open("output.csv", 'a') as f:
        output_dataframe.to_csv(f, sep=',', header=False, float_format='%.9f', index=False)



