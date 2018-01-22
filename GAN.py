import tensorflow as tf
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Structure: loop through a big sample by taking minibatches, do this for a number of epochs . Do all of this for
# however many trials,randomising the learning rate parameters each time
number_of_trails = 200
number_of_epochs = 200
mini_batch_size = 10
sample_size = 2000
learn_steps = (number_of_epochs*sample_size) / mini_batch_size
hidden_layer_size_d = 6
hidden_layer_size_g = 5

# define actual distribution
real_mean_1 = 0
real_sd_1 = 1

real_mean_2 = 10
real_sd_2 = 1

def discriminator(input, parameters, minibatch_layer = False):
    pre_0 = tf.to_float(input)
    activ_1 = tf.add(tf.matmul(pre_0, parameters[0]), parameters[1])
    pre_1 = tf.tanh(activ_1)
    activ_2 = tf.add(tf.matmul(pre_1, parameters[2]), parameters[3])
    pre_2 = tf.tanh(activ_2)
    if minibatch_layer:
        activ_3 = tf.add(tf.matmul(pre_2, parameters[4]), parameters[5])
        activation = tf.reshape(activ_3, (-1, 5, 3))
        diffs = tf.expand_dims(activation, 3) - \
                tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        tf.concat([pre_2, minibatch_features], 1)
    else:
        activ_3 = tf.add(tf.matmul(pre_2, parameters[4]), parameters[5])
    output = tf.sigmoid(activ_3)
    return output

######
# my descriminator takes in one sample as input, for minibatch featres need to be able to take in more
######

def generator(input, parameters):
    pre_0 = tf.to_float(input)
    activ_1 = tf.add(tf.matmul(pre_0, parameters[0]), parameters[1])
    pre_1 = tf.tanh(activ_1)
    output = tf.add(tf.matmul(pre_1, parameters[2]), parameters[3])
    return output


# Create weights and biases variables
weight_d_1 = tf.Variable(tf.random_uniform([1, hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
bias_d_1 = tf.Variable(tf.random_uniform([hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
weight_d_2 = tf.Variable(tf.random_uniform([hidden_layer_size_d, hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
bias_d_2 = tf.Variable(tf.random_uniform([hidden_layer_size_d], minval=0, maxval=1, dtype=tf.float32))
weight_d_3 = tf.Variable(tf.random_uniform([hidden_layer_size_d, 1], minval=0, maxval=1, dtype=tf.float32))
bias_d_3 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))

d_parameters = [weight_d_1,bias_d_1, weight_d_2, bias_d_2,weight_d_3, bias_d_3]

weight_g_1 = tf.Variable(tf.random_uniform([1, hidden_layer_size_g], minval=0, maxval=1, dtype=tf.float32))
bias_g_1 = tf.Variable(tf.random_uniform([hidden_layer_size_g], minval=0, maxval=1, dtype=tf.float32))
weight_g_2 = tf.Variable(tf.random_uniform([hidden_layer_size_g, 1], minval=0, maxval=1, dtype=tf.float32))
bias_g_2 = tf.Variable(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32))


g_parameters = [weight_g_1,bias_g_1, weight_g_2, bias_g_2]


# losses
x = tf.placeholder(tf.float32, shape=(None, 1))
z = tf.placeholder(tf.float32, shape=(None, 1))
with tf.variable_scope("Discrim") as scope:
    mini_features = False
    D1 = discriminator(x, d_parameters, minibatch_layer= mini_features)
    scope.reuse_variables()
    D2 = discriminator(generator(z, g_parameters), d_parameters, minibatch_layer= mini_features)
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

learning_rate = tf.placeholder(tf.float32)

# Train step

train_g = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_g, var_list=g_parameters)
train_d = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_d, var_list=d_parameters)

data_directory = '/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/minibatch_features_{0}'.format(mini_features)
os.chdir(data_directory)

start_time = time.time()

for it in range(1,number_of_trails+1):
    # sample parameters
    learning_rate_vec = np.random.uniform(0.000001,0.1,10)

    res_matrix = np.zeros((len(learning_rate_vec), sample_size))
    learning_rate_out_vec = np.zeros((len(learning_rate_vec)))

    row =0
    for i, p in enumerate(learning_rate_vec):
        generator_input = np.random.uniform(0, 1, (sample_size, 1))
        # sample data
        which = np.random.choice((0, 1), sample_size) # bernoulli deciding which guassian to sample from
        means = which * real_mean_1 + (1 - which) * real_mean_2 # chooses mean_1 if which = 1
        sds = which * real_sd_1 + (1 - which) * real_sd_2 # chooses sd_1 if which = 1
        real_dist = np.random.normal(means, sds, sample_size) # generate samples
        real_dist = real_dist.reshape((sample_size, 1))

        print 'Trial: {}/{}'.format(it,number_of_trails)
        print 'Step: {}/{}'.format(row+1, len(learning_rate_vec))
        print 'Learning Rate: {0}'.format(p)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # writer = tf.summary.FileWriter('./graphs', sess.graph)
            for step in range(1, learn_steps):
                start_index = ((step - 1) * mini_batch_size) % sample_size
                end_index = (step * mini_batch_size) % sample_size

                x_minibatch = real_dist[start_index:end_index,:]
                z_minibatch = generator_input[start_index:end_index,:]

                sess.run(train_d, feed_dict={x: x_minibatch, z: z_minibatch, learning_rate: p})
                sess.run(train_g, feed_dict={x: x_minibatch, z: z_minibatch, learning_rate: p})

            generated = sess.run(generator(generator_input,g_parameters))
            res_matrix[row] = generated.reshape(sample_size)
            learning_rate_out_vec[row] = p
            row = row+1
            # writer.close()

            sns.distplot(generated, hist=True, rug=False)
            sns.distplot(real_dist, hist=False, rug=False)
            plt.show()

    res_dataframe = pd.DataFrame(data=res_matrix.astype(float))
    learning_rate_dataframe = pd.DataFrame(data=learning_rate_out_vec.astype(float))

    output_dataframe = pd.concat([learning_rate_dataframe.reset_index(drop=True), res_dataframe], axis=1)

    with open("output.csv", 'a') as f:
        output_dataframe.to_csv(f, sep=',', header=False, float_format='%.9f', index=False)


print 'Total time taken: {0} seconds'.format(time.time()- start_time)



os.system('afplay /System/Library/Sounds/Sosumi.aiff')
