library(GPvecchia)
library(Matrix)
library(fields)
library(readr)

options(mc.cores = 8)
wd = "/cluster/scratch/fabiopc"
filename="data/combined_data_100k_r005and02.csv"
range_denom = 2.74

#define custom functions for anisotropy
axis_aligned_distance <- function(locs1, locs2, hx, hy) {
  dx <- (locs1[,1] - locs2[,1]) / hx
  dy <- (locs1[,2] - locs2[,2]) / hy
  sqrt(dx^2 + dy^2)
}

ard_matern32_wrapper <- function(covparms) {
  function(locs1, locs2) {
    sigma2 <- covparms[1]
    hx     <- covparms[2]
    hy     <- covparms[3]
    h <- axis_aligned_distance(locs1, locs2, hx, hy)
    sqrt3h <- sqrt(3) * h
    return(sigma2 * (1 + sqrt3h) * exp(-sqrt3h))
  }
}

#load initial values for optimization from gpboost
init_values <- read.table(file.path(wd, "data/init_values_mra_ard_from_gpoboost.txt"), header = TRUE, sep = "\t")

#we define a function that will be called for every tuning parameter
run_mra_ard<-function(r=2){
  set.seed(r)
  full_data <- read.csv(file.path(wd, filename))
  nrep <- max(full_data$rep)
  nugget = 0.5
  true_range1 = 0.05/range_denom
  true_range2 = 0.2/range_denom
  marginal_var=1
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  fitting_times<-rep(0,nrep)
  variance_list<-rep(0,nrep); range1_list<-rep(0,nrep); range2_list<-rep(0,nrep); nugget_list<-rep(0,nrep)
  true_negloglik_list<-rep(0,nrep); wrong_negloglik_list<-rep(NA,nrep)
  true_negloglik_eval_times<-rep(0,nrep); wrong_negloglik_eval_times<-rep(NA,nrep)
  
  for (i in 1:nrep){

    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    mra.options.mra = list(r = c(r), J = 2)
    
    # Fit the model
    init_var<-init_values[init_values$rep==i,"GP_var"]
    init_nug<-init_values[init_values$rep==i,"Error_term"]
    init_hx<-init_values[init_values$rep==i,"GP_range_1"]
    init_hy<-init_values[init_values$rep==i,"GP_range_2"]
    theta.ini <- c(init_var,  init_hx,  init_hy, init_nug)
    #marginal var, range1, range2, nugget - same values as in gpboost initialization
    
    fitting_times[i] <- system.time({
      vecchia.est <- vecchia_estimate(
        data = train$y,
        covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2),
        locs = as.matrix(train[,1:2]), theta.ini = theta.ini,
        X = NULL,conditioning="mra", mra.options = mra.options.mra,reltol = 1e-6)
      
      vecchia.approx <- vecchia_specify(
        as.matrix(train[, c(1:2)]), 
        mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE)})[3]
    
    variance_list[i]<-vecchia.est$theta.hat[1]
    range1_list[i]<-vecchia.est$theta.hat[2]
    range2_list[i]<-vecchia.est$theta.hat[3]
    nugget_list[i]<-vecchia.est$theta.hat[4]
    
    #negloglik evaluation at true pars
    tryCatch({
      true_negloglik_eval_times[i] <- system.time({
        true_negloglik_list[[i]] <- -vecchia_likelihood(train$y, vecchia.approx, 
                                                        c(marginal_var, true_range1, true_range2), 
                                                        nugget,
                                                        covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2))
      })[3]
    }, error = function(e) {
      message("Error at iteration ", i, ": ", e$message)
      true_negloglik_eval_times[i] <- NA  
      true_negloglik_list[[i]] <- NA  
    })
    
    #negloglik evaluation at wrong pars
    tryCatch({
      wrong_negloglik_eval_times[i] <- system.time({
        wrong_negloglik_list[[i]] <- -vecchia_likelihood(train$y, vecchia.approx,  
                                                        c(2 *marginal_var, 2 *true_range1,  2 *true_range2), 
                                                        2 * nugget,
                                                        covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2))
      })[3]
    }, error = function(e) {
      message("Error at iteration ", i, ": ", e$message)
      wrong_negloglik_eval_times[i] <- NA  
      wrong_negloglik_list[[i]] <- NA  
    })

    print(i)
    if(i<=nrep/2){
      # Predict on the training set
      pred_train_times[i] <- system.time({
        pred_train <- vecchia_prediction(
          train$y, vecchia.approx, c(marginal_var, true_range1, true_range2), nugget,
          covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2)
        )
      })[3]
      # Predict on interpolation set
      pred_inter_times[i] <- system.time({
        vecchia.approx <- vecchia_specify(as.matrix(train[, c(1:2)]), 
                                          mra.options = mra.options.mra,
                                          conditioning = 'mra', verbose = TRUE,
                                          locs.pred = as.matrix(interp[, c(1:2)]))
        pred_inter <- vecchia_prediction(train$y, vecchia.approx, 
                                         c(marginal_var, true_range1, true_range2), nugget,
                                         covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2))})[3]
      
      # Predict on extrapolation set
      pred_extra_times[i] <- system.time({
        vecchia.approx <- vecchia_specify(as.matrix(train[, c(1:2)]), 
                                          mra.options = mra.options.mra,conditioning = 'mra',
                                          verbose = TRUE,locs.pred = as.matrix(extrap[, c(1:2)]))
        pred_extra <- vecchia_prediction(train$y, vecchia.approx, 
                                         c(marginal_var, true_range1, true_range2), nugget,
                                         covmodel = function(l1, l2) ard_matern32_wrapper(theta.ini[1:3])(l1, l2))})[3]
      
      #RMSE
      rmse_train_list[i]<-sqrt(mean((train$f - pred_train$mu.obs)^2))
      rmse_inter_list[i]<-sqrt(mean((interp$f - pred_inter$mu.pred)^2))
      rmse_extra_list[i]<-sqrt(mean((extrap$f - pred_extra$mu.pred)^2))
      
      #log_score
      score_train_list[i]<-mean( (0.5*(pred_train$mu.obs-train$f )^2)/pred_train$var.obs +
                                   0.5*log(2*pi*pred_train$var.obs) )
      score_inter_list[i]<-mean( (0.5*(pred_inter$mu.pred-interp$f)^2)/pred_inter$var.pred +
                                   0.5*log(2*pi*pred_inter$var.pred) )
      score_extra_list[i]<-mean( (0.5*(pred_extra$mu.pred-extrap$f)^2)/pred_extra$var.pred +
                                   0.5*log(2*pi*pred_extra$var.pred) )
      
      rm(pred_inter,pred_train,pred_extra)
    }
    rm(data,train,interp,extrap,vecchia.approx)
  }
  
  
  # Create the filename
  param_str = paste0("r",r)
  filename_save <- sprintf("results/ard_100k/mra_ard_100k_%s",
                           param_str)
  
  # Open the file for writing
  file_path <- paste0(filename_save, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("mra ", r),
    paste0("True range1: ", true_range1),
    paste0("True range2: ", true_range2),
    
    paste0("bias for GP range1: ", mean(range1_list - true_range1)),
    paste0("MSE for GP range1: ", mean((range1_list - true_range1)^2)),
    paste0("bias for GP range2: ", mean(range2_list - true_range2)),
    paste0("MSE for GP range2: ", mean((range2_list - true_range2)^2)),
    paste0("bias for GP variance: ", mean(variance_list - marginal_var)),
    paste0("MSE for GP variance: ", mean((variance_list - marginal_var)^2)),
    paste0("bias for error term variance: ", mean(nugget_list - nugget)),
    paste0("MSE for error term variance: ", mean((nugget_list - nugget)^2)),
    paste0("variance for bias of GP range1: ", var(range1_list)/length(range1_list)),
    paste0("variance for bias of GP range2: ", var(range2_list)/length(range2_list)),
    paste0("variance for bias GP of variance: ", var(variance_list)/length(variance_list)),
    paste0("variance for bias error of term variance: ", var(nugget_list)/length(nugget_list)),
    paste0("variance for MSE GP range1: ", var((range1_list - true_range1)^2)/length(range1_list)),
    paste0("variance for MSE GP range2: ", var((range2_list - true_range2)^2)/length(range2_list)),
    paste0("variance for MSE GP variance: ", var((variance_list - marginal_var)^2)/length(variance_list)),
    paste0("variance for MSE error term variance: ", var((nugget_list - nugget)^2)/length(nugget_list)),
    paste0("mean time for parameter estimation: ", mean(fitting_times)),
    paste0("mean estimated negloglik true pars: ", mean(true_negloglik_list, na.rm = TRUE)),
    paste0("mean estimated negloglik wrong pars: ", mean(wrong_negloglik_list, na.rm = TRUE)),
    paste0("mean time for true loglik evaluation: ", mean(true_negloglik_eval_times, na.rm = TRUE)),
    paste0("mean time for wrong loglik evaluation: ", mean(wrong_negloglik_eval_times, na.rm = TRUE)),
    paste0("variance for negloglik true pars: ", var(true_negloglik_list, na.rm = TRUE)/sum(!is.na(true_negloglik_list))),
    paste0("variance for negloglik wrong pars: ", var(wrong_negloglik_list, na.rm = TRUE)/sum(!is.na(wrong_negloglik_list))),
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
run_mra_ard(r=1)