rm(list = ls(all=T))
library(ggplot2)
library(reshape2)
library(plyr)
source('/Users/Billy/PycharmProjects/GALR/mulitplot.R')

gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

cols = gg_color_hue(2)


minibatch_features_data = read.csv('/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/minibatch_features_True/output.csv',
                header = F)

no_minibatch_features_data = read.csv('/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/minibatch_features_False/output.csv',
                                   header = F)

entries_per_group = 1

min_rows = min(c(nrow(minibatch_features_data),nrow(no_minibatch_features_data)))

sample_size = min(entries_per_group,min_rows)


panel_1 = data.frame(x = melt(minibatch_features_data[sample(
                              1:nrow(minibatch_features_data),size = sample_size,replace = F),])[,2])
panel_2 = data.frame(x = melt(no_minibatch_features_data[sample(
                              1:nrow(no_minibatch_features_data),size = sample_size,replace = F),])[,2])


pull_max_density =function(panel){
  min_break = round_any(min(panel$x), diff(range(panel$x))/30, floor)
  max_break = round_any(max(panel$x), diff(range(panel$x))/30, ceiling)
  breaks = seq(min_break, max_break, diff(range(panel$x/30)))
  histo = hist(panel$x, breaks=breaks, plot=F)
  return(max(histo$density))
}

max_density = max(c(pull_max_density(panel_1),pull_max_density(panel_2)))

p1 = ggplot(data.frame(x=panel_1), aes(x)) + geom_histogram(fill = cols[2],aes(y = ..density..))+
  stat_function(fun = dnorm, args = list(mean = 4, sd = 0.5), lwd = 1, col = cols[1])+ xlab('')+ ylab('')+
  labs(title='Minibatch Features')+ theme(plot.title = element_text(size=8))+ylim(0,max_density)+xlim(2,6)
p2 = ggplot(data.frame(x=panel_2), aes(x)) + geom_histogram(fill = cols[2],aes(y = ..density..))+
  stat_function(fun = dnorm, args = list(mean = 4, sd = 0.5), lwd = 1, col = cols[1])+ xlab('')+ ylab('')+
  labs(title='No Minibatch Features')+ theme(plot.title = element_text(size=8))+ylim(0,max_density)+xlim(2,6)

multiplot(p1, p2, cols=2)

row_sd = function(data,row){
  return(sd(data[row,]))
}

median(sapply(1:nrow(minibatch_features_data),row_sd, data = minibatch_features_data))
median(sapply(1:nrow(no_minibatch_features_data),row_sd, data = no_minibatch_features_data))
