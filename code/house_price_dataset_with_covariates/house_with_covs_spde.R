library(Matrix)
library(fields)
library(readr)
library(INLA)
library(rSPDE)
library(inlabru)

options(mc.cores = 8)

#load data
house_train <- read_csv("data/house_train_with_covs.csv")
house_inter <- read_csv("data/house_interpolation_with_covs.csv")
house_extra <- read_csv("data/house_extrapolation_with_covs.csv")

coords_train <- as.matrix(house_train[, c("long","lat")])
coords_inter<-as.matrix(house_inter[,c("long","lat")])
coords_extra<-as.matrix(house_extra[,c("long","lat")])

#loading results from exact calculations
pred_exact_mean_train <- read_csv("exact_results/exact_pred_mean_train_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_mean_inter <- read_csv("exact_results/exact_pred_mean_inter_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_mean_extra <- read_csv("exact_results/exact_pred_mean_extra_house_with_covs.txt",col_names = FALSE)$X1

pred_exact_var_train <- read_csv("exact_results/exact_pred_var_train_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_var_inter <- read_csv("exact_results/exact_pred_var_inter_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_var_extra <- read_csv("exact_results/exact_pred_var_extra_house_with_covs.txt",col_names = FALSE)$X1

#define the mesh limits
pl.dom <- cbind(c(484000,488000, 500500,510000, 510000,539000,539000,520900,520900, 484000)/10000, 
                c(195000, 195000, 205000,215000,216500,216500,223450,226500,230000,230000)/10000)

compute_kl<-function(var1,var2,mean1,mean2){
  kl = log(sqrt(var2)/sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
  sum(kl)
}

#we define a function that will be called for every tuning parameter
run_spde_with_covs<-function(max.edge){
  set.seed(1)
  fitting_time <- system.time({ 
    #construct the mesh
    mesh <- inla.mesh.2d(loc.domain = pl.dom, max.e = max.edge)
    spde <- rspde.matern(mesh = mesh,nu=1.5)
    
    A <- inla.spde.make.A(mesh, loc = coords_train)
    stk <- inla.stack(
      data = list(resp = house_train$log_price),
      A = list(A,1,1,1,1,1,1,1,1,1,1,1,1,1),
      effects = list(i = 1:spde$n.spde,beta0=house_train$intercept,long=house_train$long,
                     lat=house_train$lat,age=house_train$age,TLA=house_train$TLA,
                     rooms=house_train$rooms,lotsize=house_train$lotsize,syear1994=house_train$syear1994,
                     syear1995=house_train$syear1995,syear1996=house_train$syear1996,syear1997=house_train$syear1997,
                     syear1998=house_train$syear1998,ages_sq=house_train$ages_sq),
      tag = 'train')
    
    #only fit this time
    res_eb<-inla(resp~0+ beta0+long+lat+age+TLA+rooms+lotsize+syear1994+syear1995+syear1996+
                   syear1997+syear1998+ages_sq+f(i,model=spde),
                 data=inla.stack.data(stk),
                 control.predictor=list(A=inla.stack.A(stk)),
                 control.inla = list(int.strategy = "eb"),num.threads=8 )
    })[3]
  
  
  pred_inter_time <- system.time({A_inter<-inla.spde.make.A(mesh,loc=coords_inter)
  stk.inter <- inla.stack(
    data = list(resp = NA),
    A = list(A_inter,1,1,1,1,1,1,1,1,1,1,1,1,1),
    effects = list(i = 1:spde$n.spde,beta0=house_inter$intercept,long=house_inter$long,
                   lat=house_inter$lat,age=house_inter$age,TLA=house_inter$TLA,
                   rooms=house_inter$rooms,lotsize=house_inter$lotsize,syear1994=house_inter$syear1994,
                   syear1995=house_inter$syear1995,syear1996=house_inter$syear1996,syear1997=house_inter$syear1997,
                   syear1998=house_inter$syear1998,ages_sq=house_inter$ages_sq),tag = 'inter')
  })[3]
  
  pred_extra_time <- system.time({A_extra<-inla.spde.make.A(mesh,loc=coords_extra)
  stk.extra <- inla.stack(
    data = list(resp = NA),
    A = list(A_extra,1,1,1,1,1,1,1,1,1,1,1,1,1),
    effects = list(i = 1:spde$n.spde,beta0=house_extra$intercept,long=house_extra$long,
                   lat=house_extra$lat,age=house_extra$age,TLA=house_extra$TLA,
                   rooms=house_extra$rooms,lotsize=house_extra$lotsize,syear1994=house_extra$syear1994,
                   syear1995=house_extra$syear1995,syear1996=house_extra$syear1996,syear1997=house_extra$syear1997,
                   syear1998=house_extra$syear1998,ages_sq=house_extra$ages_sq),tag = 'extra')
  })[3]
  
  #join all data stacks
  stk.full <- inla.stack(stk, stk.inter,stk.extra)
  
  #fit and prediction together, but using previous estimates to reduce time
  time_joint_pred<-system.time({ rpmu <- inla(resp~0+ beta0+long+lat+age+TLA+rooms+lotsize+syear1994+syear1995+syear1996+
                                                syear1997+syear1998+ages_sq+f(i,model=spde),
                                              data = inla.stack.data(stk.full),
                                              control.mode = list(theta = res_eb$mode$theta,
                                              restart = FALSE),
                                              control.predictor = list(A = inla.stack.A(
                                                stk.full), compute = TRUE),
                                              control.inla = list(int.strategy = "eb"),
                                              num.threads=8 ) })[3]
  
  pred_train_time<-time_joint_pred
  pred_inter_time<- pred_inter_time+time_joint_pred
  pred_extra_time<-pred_extra_time+time_joint_pred
  
  #extract train preds
  train_index<-inla.stack.index(stk.full, 'train')$data
  pred_train_mean<-rpmu$summary.fitted.values[train_index, 1]
  pred_train_var<-rpmu$summary.fitted.values[train_index, 2]^2+ 
    1/rpmu$summary.hyperpar["Precision for the Gaussian observations","mode"]
  
  #extract inter preds
  inter_index<-inla.stack.index(stk.full, 'inter')$data
  pred_inter_mean<-rpmu$summary.fitted.values[inter_index, 1]
  #add nugget to obtain variances for the observable process
  pred_inter_var<-rpmu$summary.fitted.values[inter_index, 2]^2+ 
    1/rpmu$summary.hyperpar["Precision for the Gaussian observations","mode"]
  
  #extract extra preds
  extra_index<-inla.stack.index(stk.full, 'extra')$data
  pred_extra_mean<-rpmu$summary.fitted.values[extra_index, 1]
  #add nugget to obtain variances for the observable process
  pred_extra_var<-rpmu$summary.fitted.values[extra_index, 2]^2+ 
    1/rpmu$summary.hyperpar["Precision for the Gaussian observations","mode"]
  
  #RMSE
  train_rmse<-sqrt(mean((house_train$log_price - pred_train_mean)^2))
  inter_rmse<-sqrt(mean((house_inter$log_price - pred_inter_mean)^2))
  extra_rmse<-sqrt(mean((house_extra$log_price - pred_extra_mean)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train_mean-house_train$log_price)^2)/pred_train_var +
                       0.5*log(2*pi*pred_train_var) )
  inter_score<-mean( (0.5*(pred_inter_mean-house_inter$log_price)^2)/pred_inter_var +
                       0.5*log(2*pi*pred_inter_var) )
  extra_score<-mean( (0.5*(pred_extra_mean-house_extra$log_price)^2)/pred_extra_var +
                       0.5*log(2*pi*pred_extra_var) )
  
  #crps
  crps <- function(predlist,trueobs) {
    z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
    scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                               2 * dnorm(z, 0, 1) - 1/sqrt(pi))
    return(scores)
  }
  
  train_crps<-mean(crps(list(mean=pred_train_mean,sd=sqrt(pred_train_var)),house_train$log_price))
  inter_crps<-mean(crps(list(mean=pred_inter_mean,sd=sqrt(pred_inter_var)),house_inter$log_price))
  extra_crps<-mean(crps(list(mean=pred_extra_mean,sd=sqrt(pred_extra_var)),house_extra$log_price))
  
  #rmse between predictive means
  train_rmse_mean<-sqrt(mean((pred_train_mean-pred_exact_mean_train)^2))
  inter_rmse_mean<-sqrt(mean((pred_inter_mean-pred_exact_mean_inter)^2))
  extra_rmse_mean<-sqrt(mean((pred_extra_mean-pred_exact_mean_extra)^2))
  
  #rmse between predictive variances
  train_rmse_var<-sqrt(mean((pred_train_var-pred_exact_var_train)^2))
  inter_rmse_var<-sqrt(mean((pred_inter_var-pred_exact_var_inter)^2))
  extra_rmse_var<-sqrt(mean((pred_extra_var-pred_exact_var_extra)^2))
  
  #kl divergence
  train_kl<-compute_kl(pred_exact_var_train,pred_train_var,pred_exact_mean_train,
                       pred_train_mean)
  inter_kl<-compute_kl(pred_exact_var_inter,pred_inter_var,pred_exact_mean_inter,
                       pred_inter_mean)
  extra_kl<-compute_kl(pred_exact_var_extra,pred_extra_var,pred_exact_mean_extra,
                       pred_extra_mean)
  
  # Create the filename
  filename <- paste0("results/house_with_covs/spde_house_with_covs_",max.edge)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("spde ", max.edge),
    
    paste0("time for fitting: ", fitting_time),
    paste0("univariate score train: ", train_score),
    paste0("univariate score interpolation: ", inter_score),
    paste0("univariate score extrapolation: ", extra_score),
    paste0("time for train univariate prediction: ", pred_train_time),
    paste0("time for interpolation univariate prediction: ", pred_inter_time),
    paste0("time for extrapolation univariate prediction: ", pred_extra_time),
    paste0("rmse train: ", train_rmse),
    paste0("rmse interpolation: ", inter_rmse),
    paste0("rmse extrapolation: ", extra_rmse),
    paste0("rmse mean train: ", train_rmse_mean),
    paste0("rmse mean interpolation: ", inter_rmse_mean),
    paste0("rmse mean extrapolation: ", extra_rmse_mean),
    paste0("rmse var train: ", train_rmse_var),
    paste0("rmse var interpolation: ", inter_rmse_var),
    paste0("rmse var extrapolation: ", extra_rmse_var),
    paste0("kl train: ", train_kl),
    paste0("kl interpolation: ", inter_kl),
    paste0("kl extrapolation: ", extra_kl),
    paste0("crps train: ", train_crps),
    paste0("crps interpolation: ", inter_crps),
    paste0("crps extrapolation: ", extra_crps),
    paste0("true negloglik: "),
    paste0("wrong negloglik: "),
    paste0("time for true negloglik evaluation: "),
    paste0("time for wrong negloglik evaluation: ")
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage (recall scaled coords)
run_spde_with_covs(3000/10000)