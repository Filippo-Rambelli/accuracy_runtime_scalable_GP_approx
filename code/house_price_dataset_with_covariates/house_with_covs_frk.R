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

compute_kl<-function(var1,var2,mean1,mean2){
  kl = log(sqrt(var2)/sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
  sum(kl)
}

#load data
house_train <- read_csv("data/house_train_with_covs.csv")
house_inter <- read_csv("data/house_interpolation_with_covs.csv")
house_extra <- read_csv("data/house_extrapolation_with_covs.csv")

house_train_no_covs<-house_train[, c("long", "lat", "log_price")]
coordinates(house_train_no_covs) <- ~long + lat

data<-rbind(house_train, house_inter, house_extra)
train_index<-1:nrow(house_train)
inter_index<-(nrow(house_train)+1):(nrow(house_train)+nrow(house_inter))
extra_index<-((nrow(house_train)+nrow(house_inter))+1):nrow(data)

coordinates(house_train)  <- ~long+lat        # Convert to SpatialPointsDataFrame

#loading results from exact calculations
pred_exact_mean_train <- read_csv("exact_results/exact_pred_mean_train_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_mean_inter <- read_csv("exact_results/exact_pred_mean_inter_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_mean_extra <- read_csv("exact_results/exact_pred_mean_extra_house_with_covs.txt",col_names = FALSE)$X1

pred_exact_var_train <- read_csv("exact_results/exact_pred_var_train_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_var_inter <- read_csv("exact_results/exact_pred_var_inter_house_with_covs.txt",col_names = FALSE)$X1
pred_exact_var_extra <- read_csv("exact_results/exact_pred_var_extra_house_with_covs.txt",col_names = FALSE)$X1

#we define a function that will be called for every tuning parameter
run_frk_with_covs<-function(nres){
  set.seed(nres)
  
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
    S <- FRK(f = log_price ~ 1 + long + lat  +  age + TLA + rooms +
             lotsize+ syear1994 + syear1995 + syear1996 + syear1997 + syear1998 + ages_sq, # formula for SRE model
             data = house_train_no_covs,                  # data
             basis = basis,               # Basis
             BAUs = BAUs,                 # BAUs
             tol = 1e-6,n_EM = 1000)                   # EM iterations
  })[3]
  
  #extract error variance
  est_nugget<-S@Ve[1,1]
  
  # Joint Prediction 
  pred_time<-system.time(predictions <- predict(S))[3]
  
  pred_mean_train <- predictions$mu[train_index]
  pred_mean_inter <- predictions$mu[inter_index]
  pred_mean_extra <- predictions$mu[extra_index]
  
  #add nugget to obtain variances for the observable process
  pred_var_train <-predictions$var[train_index]+est_nugget
  pred_var_inter <-predictions$var[inter_index]+est_nugget
  pred_var_extra <-predictions$var[extra_index]+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((house_train$log_price - pred_mean_train)^2))
  inter_rmse<-sqrt(mean((house_inter$log_price - pred_mean_inter)^2))
  extra_rmse<-sqrt(mean((house_extra$log_price - pred_mean_extra)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_mean_train-house_train$log_price)^2)/pred_var_train +
                       0.5*log(2*pi*pred_var_train) )
  inter_score<-mean( (0.5*(pred_mean_inter-house_inter$log_price)^2)/pred_var_inter +
                       0.5*log(2*pi*pred_var_inter) )
  extra_score<-mean( (0.5*(pred_mean_extra-house_extra$log_price)^2)/pred_var_extra +
                       0.5*log(2*pi*pred_var_extra) )
  
  #crps
  crps <- function(predlist,trueobs) {
    z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
    scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                               2 * dnorm(z, 0, 1) - 1/sqrt(pi))
    return(scores)
  }
  
  train_crps<-mean(crps(list(mean=pred_mean_train,sd=sqrt(pred_var_train)),house_train$log_price))
  inter_crps<-mean(crps(list(mean=pred_mean_inter,sd=sqrt(pred_var_inter)),house_inter$log_price))
  extra_crps<-mean(crps(list(mean=pred_mean_extra,sd=sqrt(pred_var_extra)),house_extra$log_price))
  
  #rmse between predictive means
  train_rmse_mean<-sqrt(mean((pred_mean_train-pred_exact_mean_train)^2))
  inter_rmse_mean<-sqrt(mean((pred_mean_inter-pred_exact_mean_inter)^2))
  extra_rmse_mean<-sqrt(mean((pred_mean_extra-pred_exact_mean_extra)^2))
  
  #rmse between predictive variances
  train_rmse_var<-sqrt(mean((pred_var_train-pred_exact_var_train)^2))
  inter_rmse_var<-sqrt(mean((pred_var_inter-pred_exact_var_inter)^2))
  extra_rmse_var<-sqrt(mean((pred_var_extra-pred_exact_var_extra)^2))
  
  #kl divergence
  train_kl<-compute_kl(pred_exact_var_train,pred_var_train,pred_exact_mean_train,pred_mean_train)
  inter_kl<-compute_kl(pred_exact_var_inter,pred_var_inter,pred_exact_mean_inter,pred_mean_inter)
  extra_kl<-compute_kl(pred_exact_var_extra,pred_var_extra,pred_exact_mean_extra,pred_mean_extra)
  
  # Create the filename
  filename <- paste0("results/house_with_covs/frk_house_with_covs_",nres)
  
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
    paste0("time for train univariate prediction: ", pred_time),
    paste0("time for interpolation univariate prediction: ", pred_time),
    paste0("time for extrapolation univariate prediction: ", pred_time),
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

#example usage
run_frk_with_covs(1)