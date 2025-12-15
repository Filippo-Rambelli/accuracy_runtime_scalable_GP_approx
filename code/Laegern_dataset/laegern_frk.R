#The code is based on an implementation from Andrew Zammit-Mangion, to whom we give credits
#https://github.com/finnlindgren/heatoncomparison/blob/master/Code/FRK/FRK.R

library(FRK)
library(sp)
library("ggpubr")
library(gridExtra)
library(splancs)
library(gstat)
library(fields)
library(readxl)
library(readr)

options(mc.cores = 8)
opts_FRK$set("parallel",8L)

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#we define a function that will be called for every tuning parameter
run_frk<-function(nres){
  set.seed(nres)
  
  #load data
  laegern_train <- read.csv("data/laegern_train.csv")
  laegern_inter <- read.csv("data/laegern_interpolation.csv")
  laegern_extra <- read.csv("data/laegern_extrapolation.csv")
  
  #we center the data to match the 0 mean assumption
  train_mean<-mean(laegern_train$CanopyHeight)
  
  laegern_train$CanopyHeight<-laegern_train$CanopyHeight-train_mean
  laegern_inter$CanopyHeight<-laegern_inter$CanopyHeight-train_mean
  laegern_extra$CanopyHeight<-laegern_extra$CanopyHeight-train_mean
  
  data<-rbind(laegern_train, laegern_inter, laegern_extra)
  
  coordinates(laegern_train)  <- ~x_coord+y_coord       # Convert to SpatialPointsDataFrame
  
  fitting_time <- system.time({ ## Make BAUs as SpatialPixels
    BAUs <- data                            # assign BAUs
    BAUs$CanopyHeight <- NULL                     # remove data from BAUs
    BAUs$fs <- 1                          # set fs variation to unity
    coordinates(BAUs)  <- ~x_coord+y_coord      # convert to SpatialPointsDataFrame
    gridded(BAUs) <- TRUE  
    
    ## Make Data as SpatialPoints
    basis <- auto_basis(plane(),          # we are on the plane
                        data = laegern_train,       # data around which to make basis
                        regular = 0,      # irregular basis
                        nres = nres,         # 3 resolutions
                        scale_aperture = 1,
                        type = "Matern32")   # aperture scaling of basis functions 
    
    ## Estimate using ML
    S <- FRK(f = CanopyHeight ~ 1 ,                       # formula for SRE model
             data = laegern_train,                  # data
             basis = basis,               # Basis
             BAUs = BAUs,                 # BAUs
             tol = 1e-6,n_EM = 1000)                   # EM iterations
  })[3]
  
  #extract error variance
  est_nugget<-S@Ve[1,1]
  
  # Predict on training set
  pred_train_time<-system.time(pred_train <- predict(S, newdata = laegern_train))[3]
  
  # Predict on interpolation set
  coordinates(laegern_inter)<- ~x_coord+y_coord
  pred_inter_time<-system.time(pred_inter <- predict(S, newdata = laegern_inter))[3]
  
  # Predict on extrapolation set
  coordinates(laegern_extra)<- ~x_coord+y_coord
  pred_extra_time<-system.time(pred_extra <- predict(S, newdata = laegern_extra))[3]
  
  #add nugget to obtain variances for the observable process
  pred_train$var<-pred_train$var+est_nugget
  pred_inter$var<-pred_inter$var+est_nugget
  pred_extra$var<-pred_extra$var+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((laegern_train$CanopyHeight - pred_train$mu)^2))
  inter_rmse<-sqrt(mean((laegern_inter$CanopyHeight - pred_inter$mu)^2))
  extra_rmse<-sqrt(mean((laegern_extra$CanopyHeight - pred_extra$mu)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu-laegern_train$CanopyHeight)^2)/pred_train$var +
                       0.5*log(2*pi*pred_train$var) )
  inter_score<-mean( (0.5*(pred_inter$mu-laegern_inter$CanopyHeight)^2)/pred_inter$var +
                       0.5*log(2*pi*pred_inter$var) )
  extra_score<-mean( (0.5*(pred_extra$mu-laegern_extra$CanopyHeight)^2)/pred_extra$var +
                       0.5*log(2*pi*pred_extra$var) )
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train$mu,sd=sqrt(pred_train$var)),
                        laegern_train$CanopyHeight))
  inter_crps<-mean(crps(list(mean=pred_inter$mu,sd=sqrt(pred_inter$var)),
                        laegern_inter$CanopyHeight))
  extra_crps<-mean(crps(list(mean=pred_extra$mu,sd=sqrt(pred_extra$var)),
                        laegern_extra$CanopyHeight))
  
  
  # Create the filename
  filename <- paste0("frk_laegern_nres",nres)
  
  # Open the file for writing
  file_path <- paste0("results/laegern/",filename, ".txt")
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
