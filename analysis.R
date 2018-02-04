rm(list = ls(all=T))
suppressMessages(suppressWarnings(library(reshape2)))
library(plyr)
suppressMessages(suppressWarnings(library(ggplot2)))
library(scatterplot3d)
suppressMessages(suppressWarnings(library(data.table)))
suppressMessages(suppressWarnings(library(plotly)))
library(lattice)
source('/Users/Billy/PycharmProjects/GALR/mulitplot.R')


### Colours ------------


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

cols = gg_color_hue(2)

### Data -------------------
dataframes = list()

item = 1
for(objective in 1:2){
  for(minibatch in c('minibatch','no_minibatch')){
    folder = '/Users/Billy/PycharmProjects/GAN-Mode-Collapse-Testing/data/'
    sub_folder = paste(c('objective_',objective,'_',minibatch,'/'), collapse = '')
    file = paste(c(folder,sub_folder,'output.csv'),collapse = '')
    df = data.frame(as.matrix(fread(file, header = F)))
    df = na.omit(df)
    dataframes[[item]] = df
    item =item+1
  }
}

# Find mean density values for each entry for each distribution in the mixture -------

return_big_over_small = function(df){
  low_gauss_densities = cbind(df[,1:2],apply(df[,3:ncol(df)], c(1,2),dnorm, mean = 0,sd = 1))
  high_gauss_densities = cbind(df[,1:2],apply(df[,3:ncol(df)], c(1,2),dnorm, mean = 10,sd = 1))
  
  colnames(df)= colnames(low_gauss_densities)= colnames(high_gauss_densities)= c('learning_rate','Time',3:ncol(df))
  
  low_gauss_densities_means = rowMeans(low_gauss_densities[,3:ncol(df)])
  high_gauss_densities_means = rowMeans(high_gauss_densities[,3:ncol(df)])
  
  means_df = data.frame(low = low_gauss_densities_means,high = high_gauss_densities_means)
  
  find_big_over_small = function(vec){
    return(max(vec)/min(vec))
  }
  
  big_over_small = as.numeric(apply(means_df,MARGIN = 1,find_big_over_small))
 
  return(big_over_small)
}

big_over_small_list = lapply(dataframes,return_big_over_small)

which_closest_to_1 = function(vec){
  vec_minus_1 = abs(vec-1)
  best = which.min(vec_minus_1)
  return(best)
}

best_data_rows = lapply(big_over_small_list,which_closest_to_1)

collected_rows_for_hist = list()
for(item in 1:4){
  collected_rows_for_hist[[item]] = as.numeric(dataframes[[item]][best_data_rows[[item]],3:ncol(dataframes[[item]])])
}


bin_number = 40

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

max_density = max(as.numeric(lapply(collected_rows_for_hist,pull_max_density)))

p1 = ggplot(data.frame(x=collected_rows_for_hist[[1]]), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='1 Minibatch')+ylab('Density')

p2 = ggplot(data.frame(x=collected_rows_for_hist[[2]]), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='1 No Minibatch')+ylab('Density')


p3 = ggplot(data.frame(x=collected_rows_for_hist[[3]]), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='2 Minibatch')+ylab('Density')

p4 = ggplot(data.frame(x=collected_rows_for_hist[[4]]), aes(x)) + 
  geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
  stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
  xlim(-5,15)+ylim(0,max_density)+labs(title='2 No Minibatch')+ylab('Density')

multiplot(p1, p2,p3,p4, cols=2)





