#The code is based on an implementation from Andrew Zammit-Mangion, to whom we give credits
#https://github.com/finnlindgren/heatoncomparison/blob/master/Code/FRK/FRK.R

library(FRK)
library(sp)
library(ggpubr)
library(gridExtra)
library(splancs)
library(gstat)
library(fields)
library(readxl)
library(readr)

opts_FRK$set("parallel",8L)
print(opts_FRK$get("parallel"))

#loading results from exact calculations to perform a comparison
pred_exact_mean_train <- read_csv("/data/pred_mean_train.txt",   col_names = FALSE)$X1
pred_exact_mean_inter <- read_csv("/data/pred_mean_inter.txt",   col_names = FALSE)$X1
pred_exact_mean_extra <- read_csv("/data/pred_mean_extra.txt",   col_names = FALSE)$X1

pred_exact_var_train <- read_csv("/data/pred_var_train.txt",   col_names = FALSE)$X1
pred_exact_var_inter <- read_csv("/data/pred_var_inter.txt",   col_names = FALSE)$X1
pred_exact_var_extra <- read_csv("/data/pred_var_extra.txt",   col_names = FALSE)$X1

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

compute_kl<-function(var1,var2,mean1,mean2){
  kl = log(sqrt(var2)/sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
  sum(kl)
}

#we define a function that will be called for every tuning parameter
run_frk<-function(nres){
  set.seed(nres)
  
  #load data
  house_train <- read_csv("data/house_train.csv")
  house_inter <- read_csv("data/house_interpolation.csv")
  house_extra <- read_csv("data/house_extrapolation.csv")
  
  #we center the data to match the 0 mean assumption
  train_mean<-mean(house_train$log_price)
  
  house_train$log_price<-house_train$log_price-train_mean
  house_inter$log_price<-house_inter$log_price-train_mean
  house_extra$log_price<-house_extra$log_price-train_mean
  
  data<-rbind(house_train, house_inter, house_extra)
  
  coordinates(house_train)  <- ~long+lat        # Convert to SpatialPointsDataFrame
  
  fitting_time <- system.time({ ## Make BAUs as SpatialPixels
    BAUs <- data                            # assign BAUs
    BAUs$log_price <- NULL                     # remove data from BAUs
    BAUs$fs <- 1                          # set fs variation to unity
    coordinates(BAUs)  <- ~long+lat       # convert to SpatialPointsDataFrame
    BAUs<-BAUs_from_points(BAUs)
    
    ## Make Data as SpatialPoints
    basis <- auto_basis(plane(),          # we are on the plane
                        data = house_train,       # data around which to make basis
                        regular = 0,      # irregular basis
                        nres = nres,         # 3 resolutions
                        scale_aperture = 1,
                        type = "Matern32")   # aperture scaling of basis functions 
    
    ## Estimate using ML
    S <- FRK(f = log_price ~ 1 ,                       # formula for SRE model
             data = house_train,                  # data
             basis = basis,               # Basis
             BAUs = BAUs,                 # BAUs
             tol = 1e-6,n_EM = 1000)                   # EM iterations
  })[3]
  
  #extract error variance
  est_nugget<-S@Ve[1,1]
  
  # Predict on training set
  pred_train_time<-system.time(pred_train <- predict(S, newdata = house_train))[3]
  
  # Predict on interpolation set
  coordinates(house_inter)<- ~long+lat
  pred_inter_time<-system.time(pred_inter <- predict(S, newdata = house_inter))[3]
  
  # Predict on extrapolation set
  coordinates(house_extra)<- ~long+lat
  pred_extra_time<-system.time(pred_extra <- predict(S, newdata = house_extra))[3]
  
  #add nugget to obtain variances for the observable process
  pred_train$var<-pred_train$var+est_nugget
  pred_inter$var<-pred_inter$var+est_nugget
  pred_extra$var<-pred_extra$var+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((house_train$log_price - pred_train$mu)^2))
  inter_rmse<-sqrt(mean((house_inter$log_price - pred_inter$mu)^2))
  extra_rmse<-sqrt(mean((house_extra$log_price - pred_extra$mu)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu-house_train$log_price)^2)/pred_train$var +
                       0.5*log(2*pi*pred_train$var) )
  inter_score<-mean( (0.5*(pred_inter$mu-house_inter$log_price)^2)/pred_inter$var +
                       0.5*log(2*pi*pred_inter$var) )
  extra_score<-mean( (0.5*(pred_extra$mu-house_extra$log_price)^2)/pred_extra$var +
                       0.5*log(2*pi*pred_extra$var) )
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train$mu,sd=sqrt(pred_train$var)),house_train$log_price))
  inter_crps<-mean(crps(list(mean=pred_inter$mu,sd=sqrt(pred_inter$var)),house_inter$log_price))
  extra_crps<-mean(crps(list(mean=pred_extra$mu,sd=sqrt(pred_extra$var)),house_extra$log_price))
  
  #rmse between predictive means
  train_rmse_mean<-sqrt(mean((pred_train$mu-pred_exact_mean_train)^2))
  inter_rmse_mean<-sqrt(mean((pred_inter$mu-pred_exact_mean_inter)^2))
  extra_rmse_mean<-sqrt(mean((pred_extra$mu-pred_exact_mean_extra)^2))
  
  #rmse between predictive variances
  train_rmse_var<-sqrt(mean((pred_train$var-pred_exact_var_train)^2))
  inter_rmse_var<-sqrt(mean((pred_inter$var-pred_exact_var_inter)^2))
  extra_rmse_var<-sqrt(mean((pred_extra$var-pred_exact_var_extra)^2))
  
  #kl divergence
  train_kl<-compute_kl(pred_exact_var_train,pred_train$var,pred_exact_mean_train,pred_train$mu)
  inter_kl<-compute_kl(pred_exact_var_inter,pred_inter$var,pred_exact_mean_inter,pred_inter$mu)
  extra_kl<-compute_kl(pred_exact_var_extra,pred_extra$var,pred_exact_mean_extra,pred_extra$mu)
  
  # Create the filename
  filename <- paste0("results/house/frk_house_",nres)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("FRK ", nres),
    
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
    paste0("time for wrong negloglik evaluation: "),
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_frk(1)