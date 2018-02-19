# Import -----

rm(list = ls(all=T))
suppressMessages(suppressWarnings(library(reshape2)))
library(plyr)
suppressMessages(suppressWarnings(library(ggplot2)))
library(scatterplot3d)
suppressMessages(suppressWarnings(library(data.table)))
suppressMessages(suppressWarnings(library(plotly)))
library(lattice)
source('/Users/Billy/PycharmProjects/GALR/mulitplot.R')
library(entropy)
library(FNN)
library(pbapply)


# Objective ----
# Objective 1 is min log(1-D(G(z)), Objective 2 is max log(D(G(z))

# Colours ------------


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

cols = gg_color_hue(2)

# Max density -------
bin_number = 30
pull_max_density =function(vec){
  min_break = round_any(min(vec), diff(range(vec))/bin_number, floor)
  max_break = round_any(max(vec), diff(range(vec))/bin_number, ceiling)
  breaks = seq(min_break, max_break, diff(range(vec/bin_number)))
  histo = hist(vec, breaks=breaks, plot=F)
  return(max(histo$density))
}


# Actual density ----

mixture_density = function(x){
  return(dnorm(x,10,1)/2 + dnorm(x,0,1)/2)
}

# Data -------------------
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
    rm(df)
  }
}

# K nearest neighbours ---------


length_vec = 10000
which_mean = sample(c(0,10),size = length_vec, replace =T)
real_dist = rnorm(n = length_vec,mean = which_mean,sd =1 )

kl_to_real = function(vec){
  vec_1 = as.numeric(vec)
  return(KL.divergence(real_dist,vec_1, k=5)[5])
}

kl_by_row = function(df){
  return(as.numeric(apply(df[,3:ncol(df)],1, kl_to_real)))
}


kl_rows = pblapply(X = dataframes,FUN = kl_by_row)

which_min_kl_div = lapply(kl_rows,FUN = function(vec){return(which.min(vec))})

collected_rows_for_hist = list()
for(item in 1:4){
  collected_rows_for_hist[[item]] = as.numeric(dataframes[[item]][which_min_kl_div[[item]],3:ncol(dataframes[[item]])])
}

which.median = function(x) {
  if (length(x) %% 2 != 0) {
    which(x == median(x))
  } else if (length(x) %% 2 == 0) {
    a = sort(x)[c(length(x)/2, length(x)/2+1)]
    which(x == a[1])
  }
}

which_med_kl_div = lapply(kl_rows,which.median)


hist(as.numeric(dataframes[[4]][which_med_kl_div[[4]],3:ncol(dataframes[[4]])]))

# Compare two median side by side general function ----

plot_median_side_by_side = function(index1,index2,title1,title2, y_max=0.5){
  collected_rows_for_hist_med = list()
  indice = c(index1,index2)
  for(item in indice){
    collected_rows_for_hist_med[[item]] = as.numeric(dataframes[[item]][which_med_kl_div[[item]],3:ncol(dataframes[[item]])])
  }
  
  p1 = ggplot(data.frame(x=collected_rows_for_hist_med[[index1]]), aes(x)) + 
    geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
    stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
    xlim(-5,15) + ylim(0,y_max)+
    labs(title=title1)+ylab('Density')
  
  p2 = ggplot(data.frame(x=collected_rows_for_hist_med[[index2]]), aes(x)) + 
    geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
    stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
    xlim(-5,15) + ylim(0,y_max)+
    labs(title=title2)+ylab('Density')
  
  suppressWarnings(multiplot(p1, p2, cols=2))
  
}

plot_median_side_by_side(1,2,'Minibatch Discrimination','No Minibatch Discrimination') # oringial objective
plot_median_side_by_side(3,4,'Minibatch Discrimination','No Minibatch Discrimination') # alternative
plot_median_side_by_side(2,4,'min log(1-D(G(z))','max log(D(G(z))') # non minibatch

# Compare two best side by side general function ----

plot_best_side_by_side = function(index1,index2,title1,title2, y_max=0.5){
  collected_rows_for_hist = list()
  indice = c(index1,index2)
  for(item in indice){
    collected_rows_for_hist[[item]] = as.numeric(dataframes[[item]][which_min_kl_div[[item]],3:ncol(dataframes[[item]])])
  }
  
  p1 = ggplot(data.frame(x=collected_rows_for_hist[[index1]]), aes(x)) + 
    geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
    stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
    xlim(-5,15) + ylim(0,y_max)+
    labs(title=title1)+ylab('Density')
  
  p2 = ggplot(data.frame(x=collected_rows_for_hist[[index2]]), aes(x)) + 
    geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
    stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
    xlim(-5,15) + ylim(0,y_max)+
    labs(title=title2)+ylab('Density')
  
  suppressWarnings(multiplot(p1, p2, cols=2))
  
}

plot_best_side_by_side(1,2,'Minibatch Discrimination','No Minibatch Discrimination') # oringial objective
plot_best_side_by_side(3,4,'Minibatch Discrimination','No Minibatch Discrimination') # alternative
plot_best_side_by_side(2,4,'min log(1-D(G(z))','max log(D(G(z))') # non minibatch


# Pick out 25% quantile and plot ------
which.25 = function(x) {
  return(which(x == sort(x)[round(length(x)/4)]))
}

which_25_kl_div = lapply(kl_rows,which.25)


plot_25 = function(data_index,title1 = '',y_max=0.5){
  bin_number = 40
  data = as.numeric(dataframes[[data_index]][which_25_kl_div[[data_index]],3:ncol(dataframes[[data_index]])])
  ggplot(data.frame(x=data), aes(x)) + 
    geom_histogram(fill = cols[2],aes(y = ..density..),bins = bin_number)+
    stat_function(fun = mixture_density, lwd = 1, col = cols[1])+
    xlim(-5,15) + ylim(0,y_max)+
    labs(title=title1)+ylab('Density')
}

plot_25(2) # oringinal, no minibatch

plot_25(4) # new, no minibatch




# Find average KL by group -----
means = as.numeric(lapply(kl_rows, mean))
means = data.frame(original_objective = means[1:2], changed_objective = means[3:4]) 
rownames(means) = c('Minibatch','No minibatch') 
means

