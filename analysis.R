data = read.csv('/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/minibatch_features_True/output.csv',
                header = F)

plot(density(as.numeric(data[nrow(data),])))
