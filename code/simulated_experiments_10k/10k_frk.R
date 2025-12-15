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

options(mc.cores = 8)
opts_FRK$set("parallel",8L)
print(opts_FRK$get("parallel"))

range_denom = 2.74

file_map <- list(
  "0.05" = "data/combined_data_r005.csv",
  "0.2"  = "data/combined_data_r02.csv",
  "0.5"  = "data/combined_data_r05.csv"
)

load_pred <- function(kind, stage, range) {
  fname <- sprintf("exact_results/exact_pred_%s_%s_%s.txt", kind, stage, range)
  read.csv(file.path(wd, fname))
}

compute_kl<-function(var1,var2,mean1,mean2){
  kl = log(sqrt(var2)/sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
  sum(kl)
}

#we define a function that will be called for every tuning parameter
run_frk<-function(range=0.2,nres=1){
  set.seed(nres)

  #load dataset
  range_key <- as.character(range)
  if (!range_key %in% names(file_map)) {
    stop("Invalid range value")
  }
  filename <- file_map[[range_key]]
  full_data <- read.csv(file.path(wd, filename))
  
  #load the predictive means and variances from exact calculations
  exact_pred_mean_train_whole <- load_pred("mean", "train", range)
  exact_pred_var_train_whole  <- load_pred("var",  "train", range)
  
  exact_pred_mean_inter_whole <- load_pred("mean", "inter", range)
  exact_pred_var_inter_whole  <- load_pred("var",  "inter", range)
  
  exact_pred_mean_extra_whole <- load_pred("mean", "extra", range)
  exact_pred_var_extra_whole  <- load_pred("var",  "extra", range)
  
  nugget=0.5
  nrep=max(full_data$rep)
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  rmse_mean_train_list<-rep(0,nrep/2); rmse_var_train_list<-rep(0,nrep/2)
  rmse_mean_inter_list<-rep(0,nrep/2); rmse_var_inter_list<-rep(0,nrep/2)
  rmse_mean_extra_list<-rep(0,nrep/2); rmse_var_extra_list<-rep(0,nrep/2)
  kl_train<-rep(0,nrep/2); kl_inter<-rep(0,nrep/2); kl_extra<-rep(0,nrep/2)
  
  fitting_times<-rep(0,nrep); nugget_list<-rep(0,nrep)
  
  for (i in 1:nrep){
    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    train_copy<-train[,c(1:3)]
    coordinates(train_copy)  <- ~x1+x2        # Convert to SpatialPointsDataFrame
    
    fitting_time <- system.time({ ## Make BAUs as SpatialPixels
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
                          type = "Matern32")   # aperture scaling of basis functions 
      
      ## Estimate using ML
      S <- FRK(f = y ~ 1 ,                       # formula for SRE model
               data = train_copy,                  # data
               basis = basis,               # Basis
               BAUs = BAUs,                 # BAUs
               tol = 1e-6,n_EM = 1000)                   # EM iterations
    })
    fitting_times[i]<-fitting_time[3]
    
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
      score_train_list[i]<-mean( (0.5*(pred_train$mu-train$f )^2)/pred_train$var + 0.5*log(2*pi*pred_train$var) )
      score_inter_list[i]<-mean( (0.5*(pred_inter$mu-interp$f)^2)/pred_inter$var + 0.5*log(2*pi*pred_inter$var) )
      score_extra_list[i]<-mean( (0.5*(pred_extra$mu-extrap$f)^2)/pred_extra$var + 0.5*log(2*pi*pred_extra$var) )
      
      #exact_calculations
      exact_pred_mean_train <-exact_pred_mean_train_whole[exact_pred_mean_train_whole$iteration==i,2]
      exact_pred_var_train<-exact_pred_var_train_whole[exact_pred_mean_train_whole$iteration==i,2]
      
      exact_pred_mean_inter<-exact_pred_mean_inter_whole[exact_pred_mean_train_whole$iteration==i,2]
      exact_pred_var_inter<-exact_pred_var_inter_whole[exact_pred_mean_train_whole$iteration==i,2]
      
      exact_pred_mean_extra<-exact_pred_mean_extra_whole[exact_pred_mean_train_whole$iteration==i,2] 
      exact_pred_var_extra<-exact_pred_var_extra_whole[exact_pred_mean_train_whole$iteration==i,2] 
      
      #kl
      kl_train[i]<-compute_kl(exact_pred_var_train,pred_train$var,exact_pred_mean_train,pred_train$mu)
      kl_inter[i]<-compute_kl(exact_pred_var_inter,pred_inter$var,exact_pred_mean_inter,pred_inter$mu)
      kl_extra[i]<-compute_kl(exact_pred_var_extra,pred_extra$var,exact_pred_mean_extra,pred_extra$mu)
      
      #rmse means
      rmse_mean_train_list[i]<-sqrt(mean((pred_train$mu-exact_pred_mean_train)^2))
      rmse_mean_inter_list[i]<-sqrt(mean((pred_inter$mu-exact_pred_mean_inter)^2))
      rmse_mean_extra_list[i]<-sqrt(mean((pred_extra$mu-exact_pred_mean_extra)^2))
      
      #rmse vars
      rmse_var_train_list[i]<-sqrt(mean((pred_train$var-exact_pred_var_train)^2))
      rmse_var_inter_list[i]<-sqrt(mean((pred_inter$var-exact_pred_var_inter)^2))
      rmse_var_extra_list[i]<-sqrt(mean((pred_extra$var-exact_pred_var_extra)^2))
      
      rm(pred_train,pred_inter,pred_extra,exact_pred_mean_train,exact_pred_var_train,exact_pred_mean_inter,exact_pred_var_inter,exact_pred_mean_extra,exact_pred_var_extra)
    }
    print(paste0("rep ",i," done"))
    rm(data,train,train_copy,interp,extrap,BAUs,basis,S)
  }
  
  # Create the filename
  filename <- paste0("results/",range,"/frk_",range,"_nres",nres)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("FRK points ", nres),
    paste0("True range: ", range / range_denom),
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
    paste0("mean exact negloglik true pars: "),
    paste0("true pars, mean diff negloglik: "),
    paste0("wrong pars, mean diff negloglik: "),
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
    paste0("mean rmse mean train: ",mean(rmse_mean_train_list)),
    paste0("mean rmse mean interpolation: ",mean(rmse_mean_inter_list)),
    paste0("mean rmse mean extrapolation: ",mean(rmse_mean_extra_list)),
    paste0("variance rmse mean train: ",var(rmse_mean_train_list)/length(rmse_mean_train_list)),
    paste0("variance rmse mean interpolation: ",var(rmse_mean_inter_list)/length(rmse_mean_inter_list)),
    paste0("variance rmse mean extrapolation: ",var(rmse_mean_extra_list)/length(rmse_mean_extra_list)),
    paste0("mean rmse var train: ",mean(rmse_var_train_list)),
    paste0("mean rmse var interpolation: ",mean(rmse_var_inter_list)),
    paste0("mean rmse var extrapolation: ",mean(rmse_var_extra_list)),
    paste0("variance rmse var train: ",var(rmse_var_train_list)/length(rmse_var_train_list)),
    paste0("variance rmse var interpolation: ",var(rmse_var_inter_list)/length(rmse_var_inter_list)),
    paste0("variance rmse var extrapolation: ",var(rmse_var_extra_list)/length(rmse_var_extra_list)),
    paste0("mean kl train: ",mean(kl_train)),
    paste0("mean kl interpolation: ",mean(kl_inter)),
    paste0("mean kl extrapolation: ",mean(kl_extra)),
    paste0("variance kl train: ",var(kl_train)/length(kl_train)),
    paste0("variance kl interpolation: ",var(kl_inter)/length(kl_inter)),
    paste0("variance kl extrapolation: ",var(kl_extra)/length(kl_extra)),
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
run_frk(range=0.2,nres=1)