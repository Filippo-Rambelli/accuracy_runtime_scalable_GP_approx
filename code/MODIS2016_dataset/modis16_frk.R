#The code is based on an implementation from Andrew Zammit-Mangion, to whom we give credits
#https://github.com/finnlindgren/heatoncomparison/blob/master/Code/FRK/FRK.R

library(FRK)
library(sp)
library(ggpubr)
library(gridExtra)
library(splancs)
library(gstat)
library(fields)
options(mc.cores = 8)
opts_FRK$set("parallel",8L)

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#load data
load("data/AllSatelliteTemps.RData")
load("data/SatelliteTemps.RData")

#we define a function that will be called for every tuning parameter
run_frk<-function(nres){
  set.seed(nres)
  
  data<-sat.temps
  
  ## Make Data as SpatialPoints
  train_data <- subset(data,!is.na(Temp))        # no missing data in data frame
  coordinates(train_data)  <- ~Lon+Lat        # Convert to SpatialPointsDataFrame
  
  fitting_time <- system.time({
    ## Make BAUs as SpatialPixels
    BAUs <- data                         # assign BAUs
    BAUs$Temp <- NULL                     # remove data from BAUs
    BAUs$fs <- 1                          # set fs variation to unity
    coordinates(BAUs)  <- ~Lon+Lat        # convert to SpatialPointsDataFrame
    gridded(BAUs) <- TRUE  
    
    basis <- auto_basis(plane(),          # we are on the plane
                        data = train_data,       # data around which to make basis
                        regular = 0,      # irregular basis
                        nres = nres,         # 3 resolutions
                        scale_aperture = 1,
                        type = "exp")   # aperture scaling of basis functions 
    ## Estimate using ML
    S <- FRK(f = Temp~1+Lon+Lat,                       # formula for SRE model
             data = train_data,                  # data
             basis = basis,               # Basis
             BAUs = BAUs,                 # BAUs
             tol = 1e-6,n_EM = 1000)                   # EM iterations
  })[3]
  
  #extract error variance
  est_nugget<-S@Ve[1,1]
  
  # Predict on training set
  pred_train_time<-system.time(pred_train <- predict(S, newdata = train_data))[3]
  
  #predict on test set
  test_data<-all.sat.temps[is.na(all.sat.temps$MaskTemp),]
  coordinates(test_data)<- ~Lon+Lat
  pred_test_time<-system.time(pred_test <- predict(S, newdata = test_data))[3]
  
  #add nugget to obtain variances for the observable process
  pred_train$var<-pred_train$var+est_nugget
  pred_test$var<-pred_test$var+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((train_data$Temp - pred_train$mu)^2))
  test_rmse<-sqrt(mean((test_data$TrueTemp - pred_test$mu)^2,na.rm = TRUE))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu-train_data$Temp)^2)/pred_train$var + 
                       0.5*log(2*pi*pred_train$var) )
  test_score<-mean( (0.5*(pred_test$mu-test_data$TrueTemp)^2)/pred_test$var + 
                      0.5*log(2*pi*pred_test$var) ,na.rm=T)
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train$mu,sd=sqrt(pred_train$var)),
                        train_data$Temp))
  test_crps<-mean(crps(list(mean=pred_test$mu,sd=sqrt(pred_test$var)),
                       test_data$TrueTemp),na.rm=TRUE)
  
  
  # Create the filename
  filename <- paste0("frk_modis16_nres",nres)
  
  # Open the file for writing
  file_path <- paste0("results/modis16/",filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("FRK gridded ", nres),
    
    paste0("time for fitting: ", fitting_time),
    paste0("univariate score train: ", train_score),
    paste0("univariate score test: ", test_score),
    paste0("time for train univariate prediction: ", pred_train_time),
    paste0("time for test univariate prediction: ", pred_test_time),
    paste0("rmse train: ", train_rmse),
    paste0("rmse test: ", test_rmse),
    paste0("crps train: ", train_crps),
    paste0("crps test: ", test_crps),
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
