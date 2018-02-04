rm(list = ls(all=T))
suppressMessages(suppressWarnings(library(reshape2)))
library(plyr)
suppressMessages(suppressWarnings(library(ggplot2)))
library(scatterplot3d)
suppressMessages(suppressWarnings(library(data.table)))
suppressMessages(suppressWarnings(library(plotly)))
library(lattice)
source('/Users/Billy/PycharmProjects/GALR/mulitplot.R')

### Objective -------------------
which_objective = 1
minibatch = 1

which_minibatch = c('minibatch','no_minibatch')


### Colours ------------


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

cols = gg_color_hue(2)

### Data -------------------
folder = '/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/'
sub_folder = paste(c('objective_',which_objective,'_',which_minibatch[minibatch],'/'), collapse = '')
file = paste(c(folder,sub_folder,'output.csv'),collapse = '')

df = data.frame(as.matrix(fread(file, header = F)))
df = na.omit(df)

# Find mean density values for each entry for each distribution in the mixture -------

low_gauss_densities = cbind(df[,1],apply(df[,2:ncol(df)], c(1,2),dnorm, mean = 0,sd = 1))
high_gauss_densities = cbind(df[,1],apply(df[,2:ncol(df)], c(1,2),dnorm, mean = 10,sd = 1))

colnames(df)= colnames(low_gauss_densities)= colnames(high_gauss_densities)= c('learning_rate',2:ncol(df))

low_gauss_densities_means = rowMeans(low_gauss_densities)
high_gauss_densities_means = rowMeans(high_gauss_densities)

means_df = data.frame(low = low_gauss_densities_means,high = high_gauss_densities_means)

find_big_over_small = function(vec){
  return(max(vec)/min(vec))
}

big_over_small = apply(means_df,MARGIN = 1,find_big_over_small)
find_closest_to_1 = abs(big_over_small-1)

worst = as.numeric(df[which.max(big_over_small),2:ncol(df)])
best = as.numeric(df[which.min(find_closest_to_1),2:ncol(df)])

plot(density(worst))
plot(density(best))

df3 = data.frame(x=worst)

bin_number = 30

mixture_density = function(x){
  return(dnorm(x,10,1)/2 + dnorm(x,0,1)/2)
}

pull_max_density =function(vec){
  min_break = round_any(min(vec), diff(range(vec))/bin_number, floor)
  max_break = round_any(max(vec), diff(range(vec))/bin_number, ceiling)
  breaks = seq(min_break, max_break, diff(range(vec/bin_number)))
  histo = hist(vec, breaks=breaks, plot=F)
  return(max(histo$density))
}

max_density = max(c(pull_max_density(best),pull_max_density(best)))/1.5


p1 = ggplot(data.frame(x=worst), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='Worst')+ylab('Density')

p2 = ggplot(data.frame(x=best), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='Best')+ylab('Density')

multiplot(p1, p2, cols=2)

