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

wd = "/cluster/scratch/fabiopc"

opts_FRK$set("parallel",8L)
print(opts_FRK$get("parallel"))

range_denom_given_nu <- c(
  "0.5" = 3.00,
  "1.5" = 2.74,
  "2.5" = 2.65
)

file_map <- list(
  "0.5,0.5,1.5,1.5" = "data/combined_data_100k_r05.csv",
  "0.2,0.5,1.5,1.5" = "data/combined_data_100k_r02.csv",
  "0.05,0.5,1.5,1.5" = "data/combined_data_100k_r005.csv",
  "0.2,0.1,1.5,1.5" = "data/combined_data_100k_r02_n01.csv",
  "0.2,0.5,0.5,0.5" = "data/combined_data_100k_r02_s05.csv",
  "0.2,0.5,2.5,2.5" = "data/combined_data_100k_r02_s25.csv",
  "0.2,0.5,2.5,0.5" = "data/combined_data_100k_r02_s25.csv"  # Same file
)

run_frk<-function(range=0.2,nu_sim = 1.5, nu_estim = 1.5, nugget = 0.5,nres=1){
  set.seed(nres)
  
  filename_list <- c(range, nugget, nu_sim, nu_estim)
  key <- paste(filename_list, collapse = ",")
  filename <- file_map[[key]]
  full_data <- read.csv(file.path(wd, filename))
  nrep <- max(full_data$rep)
  range_denom <- range_denom_given_nu[as.character(nu_sim)]
  type = ifelse(nu_estim==1.5,"Matern32","exp") # to be changed!
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  fitting_times<-rep(0,nrep); nugget_list<-rep(0,nrep)
  
  for (i in 1:nrep){
    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    train_copy<-train[,c(1:3)]
    coordinates(train_copy)  <- ~x1+x2        # Convert to SpatialPointsDataFrame
    
    fitting_times[i] <- system.time({ ## Make BAUs as SpatialPixels
      BAUs <- data                            # assign BAUs
      BAUs$y <- NULL                     # remove data from BAUs
      BAUs$fs <- 1                          # set fs variation to unity
      coordinates(BAUs)  <- ~x1+x2      # convert to SpatialPointsDataFrame
      BAUs<-BAUs_from_points(BAUs)
      
      ## Make Data as SpatialPoints
      basis <- auto_basis(plane(),          # we are on the plane
                          data = train_copy,       # data around which to make basis
                          regular = 0,      # irregular basis
                          nres = nres,         # 3 resolutions
                          scale_aperture = 1,
                          type = type)   # aperture scaling of basis functions 
      
      ## Estimate using ML
      S <- FRK(f = y ~ 1 ,                       # formula for SRE model
               data = train_copy,                  # data
               basis = basis,               # Basis
               BAUs = BAUs,                 # BAUs
               tol = 1e-6,n_EM = 1000)                   # EM iterations
    })[3]
    
    nugget_list[i]<-S@Ve[1,1]
    
    if(i<=nrep/2){
      # Predict on training set
      coordinates(train)  <- ~x1+x2  
      pred_train_times[i]<-system.time(pred_train <- predict(S, newdata = train))[3]
      
      # Predict on interpolation set
      coordinates(interp)<- ~x1+x2  
      pred_inter_times[i]<-system.time(pred_inter <- predict(S, newdata = interp))[3]
      
      # Predict on extrapolation set
      coordinates(extrap)<- ~x1+x2  
      pred_extra_times[i]<-system.time(pred_extra <- predict(S, newdata = extrap))[3]
      
      #RMSE
      rmse_train_list[i]<-sqrt(mean((train$f - pred_train$mu)^2))
      rmse_inter_list[i]<-sqrt(mean((interp$f - pred_inter$mu)^2))
      rmse_extra_list[i]<-sqrt(mean((extrap$f - pred_extra$mu)^2))
      
      #log_score
      score_train_list[i]<-mean( (0.5*(pred_train$mu-train$f )^2)/pred_train$var + 
                                   0.5*log(2*pi*pred_train$var) )
      score_inter_list[i]<-mean( (0.5*(pred_inter$mu-interp$f)^2)/pred_inter$var + 
                                   0.5*log(2*pi*pred_inter$var) )
      score_extra_list[i]<-mean( (0.5*(pred_extra$mu-extrap$f)^2)/pred_extra$var + 
                                   0.5*log(2*pi*pred_extra$var) )
      
      rm(pred_train,pred_inter,pred_extra)
    }
    rm(data,train,train_copy,interp,extrap,BAUs,basis,S)
  }
  
  # Create the filename
  param_str = paste0("nres",nres)
  if (nu_estim != nu_sim) {
    filename_save <- sprintf("results/%s_100k_misspecified/frk_%s_100k_misspecified_%s",
                             range, range, param_str)
  } else if (nugget != 0.5) {
    filename_save <- sprintf("results/%s_100k_n%s/frk_%s_100k_n%s_%s",
                             range, nugget, range, nugget, param_str)
  } else if (nu_sim != 1.5) {
    filename_save <- sprintf("results/%s_100k_s%s/frk_%s_100k_s%s_%s",
                             range, nu_sim, range, nu_sim, param_str)
  } else {
    filename_save <- sprintf("results/%s_100k/frk_%s_100k_%s",
                             range, range, param_str)
  }
  
  # Open the file for writing
  file_path <- paste0(filename_save, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("FRK points ", nres),
    paste0("True range: ", range / range_denom),
    paste0("True nugget: ", nugget),
    paste0("True smoothness: ", nu_sim),
    paste0("Used smoothness: ", nu_estim),
    
    paste0("bias for GP range: "),
    paste0("MSE for GP range: "),
    paste0("bias for GP variance: "),
    paste0("MSE for GP variance: "),
    paste0("bias for error term variance: ",mean(nugget_list - nugget)),
    paste0("MSE for error term variance: ",mean((nugget_list - nugget)^2)),
    paste0("variance for bias of GP range: "),
    paste0("variance for bias GP of variance: "),
    paste0("variance for bias error of term variance: ",var(nugget_list)/length(nugget_list)),
    paste0("variance for MSE GP range: "),
    paste0("variance for MSE GP variance: "),
    paste0("variance for MSE error term variance: ",var((nugget_list - nugget)^2)/length(nugget_list)),
    paste0("mean time for parameter estimation: ", mean(fitting_times)),
    paste0("mean estimated negloglik true pars: "),
    paste0("mean estimated negloglik wrong pars: "),
    paste0("mean time for true loglik evaluation: "),
    paste0("mean time for wrong loglik evaluation: "),
    paste0("variance for negloglik true pars: "),
    paste0("variance for negloglik wrong pars: "),
    paste0("mean univariate score train: ", mean(score_train_list)),
    paste0("mean univariate score interpolation: ", mean(score_inter_list)),
    paste0("mean univariate score extrapolation: ", mean(score_extra_list)),
    paste0("variance univariate score train: ", var(score_train_list)/length(score_train_list)),
    paste0("variance univariate score interpolation: ", var(score_inter_list)/length(score_inter_list)),
    paste0("variance univariate score extrapolation: ", var(score_extra_list)/length(score_extra_list)),
    paste0("mean time for train univariate prediction: ", mean(pred_train_times)),
    paste0("mean time for interpolation univariate prediction: ", mean(pred_inter_times)),
    paste0("mean time for extrapolation univariate prediction: ", mean(pred_extra_times)),
    paste0("RMSE train: ", mean(rmse_train_list)),
    paste0("RMSE inter: ", mean(rmse_inter_list)),
    paste0("RMSE extra: ", mean(rmse_extra_list)),
    paste0("variance for RMSE train: ", var(rmse_train_list)/length(rmse_train_list)),
    paste0("variance for RMSE inter: ", var(rmse_inter_list)/length(rmse_inter_list)),
    paste0("variance for RMSE extra: ", var(rmse_extra_list)/length(rmse_extra_list))
    
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_frk(range=0.5,nu_sim = 1.5, nu_estim = 1.5, nugget = 0.5, nres=1)