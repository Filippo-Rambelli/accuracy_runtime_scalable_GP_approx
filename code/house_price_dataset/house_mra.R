library(GPvecchia)
library(Matrix)
library(fields)
library(readr)

options(mc.cores = 8)

#we manually define a function to optimize all the covariance parameters but the
#smoothness. This is achiever by modyfing the vecchia_estimate function in the
#package GPVecchia
vecchia_estimate_fixed_smoothness_1.5 <- function(
    data, locs, X, m = 20, covmodel = 'matern', theta.ini, output.level = 1,
                                    reltol = sqrt(.Machine$double.eps), ...) {
  
  ## default trend is constant over space (intercept)
  if (missing(X)) {
    beta.hat <- mean(data)
    z <- data - beta.hat
    trend <- 'constant'
  } else if (is.null(X)) {
    ## if X=NULL, do not estimate any trend
    beta.hat <- c()
    z <- data
    trend <- 'none'
  } else {
    ## otherwise, estimate and de-trend
    beta.hat <- Matrix::solve(crossprod(X), crossprod(X, data))
    z <- data - X %*% beta.hat
    trend <- 'userspecified'
  }
  
  ## specify vecchia approximation
  vecchia.approx <- vecchia_specify(locs, m, ...)
  
  ## initial covariance parameter values
  if (all(is.character(covmodel)) && covmodel == 'matern') {
    if (missing(theta.ini) || any(is.na(theta.ini))) {
      var.res <- stats::var(z)
      n <- length(z)
      dists.sample <- fields::rdist(locs[sample(1:n, min(n, 300)), ])
      theta.ini <- c(.9 * var.res, mean(dists.sample) / 4, .1 * var.res) # var, range, nugget
    }
  }
  
  ## specify vecchia loglikelihood
  negloglik.vecchia <- function(lgparms) {
    full_params <- exp(lgparms)
    full_params <- c(full_params[1:2], 1.5, full_params[3]) # Insert the fixed smoothness value
    l <- -vecchia_likelihood(z, vecchia.approx, full_params[-length(full_params)], full_params[length(full_params)], covmodel = covmodel)
    return(l)
  }
  
  ## find MLE of theta (given beta.hat)
  non1pars <- which(theta.ini != 1)
  parscale <- rep(1, length(theta.ini))
  parscale[non1pars] <- log(theta.ini[non1pars])
  
  opt.result <- stats::optim(par = log(theta.ini),
                             fn = negloglik.vecchia,
                             method = "Nelder-Mead",
                             control = list(
                               trace = 100, maxit = 300, parscale = parscale,
                               reltol = reltol
                             )) # trace=1 outputs iteration counts
  
  theta.hat <- exp(opt.result$par)
  theta.hat <- c(theta.hat[1:2], 1.5, theta.hat[3]) # Fix smoothness to 1.5
  names(theta.hat) <- c("variance", "range", "smoothness", "nugget")
  
  ## return estimated parameters
  if (output.level > 0) {
    cat('estimated trend coefficients:\n')
    print(beta.hat)
    cat('estimated covariance parameters:\n')
    print(theta.hat)
  }
  return(list(z = z, beta.hat = beta.hat, theta.hat = theta.hat,
              trend = trend, locs = locs, covmodel = covmodel))
}

#load data
house_train <- read_csv("data/house_train.csv")
house_inter <- read_csv("data/house_interpolation.csv")
house_extra <- read_csv("data/house_extrapolation.csv")

#we center the data to match the 0 mean assumption
train_mean<-mean(house_train$log_price)

house_train$log_price<-house_train$log_price-train_mean
house_inter$log_price<-house_inter$log_price-train_mean
house_extra$log_price<-house_extra$log_price-train_mean

#covariance parameters estimated from exact calculations
range=6.182155717795291139e+02
nugget=7.871528499294050407e-02
marginal_var=4.896777870079787598e-01

#loading results from exact calculations to perform a comparison
pred_exact_mean_train <- read_csv("exact_results/exact_pred_mean_train_house.txt",col_names = FALSE)$X1
pred_exact_mean_inter <- read_csv("exact_results/exact_pred_mean_inter_house.txt",col_names = FALSE)$X1
pred_exact_mean_extra <- read_csv("exact_results/exact_pred_mean_extra_house.txt",col_names = FALSE)$X1

pred_exact_var_train <- read_csv("exact_results/exact_pred_var_train_house.txt",col_names = FALSE)$X1
pred_exact_var_inter <- read_csv("exact_results/exact_pred_var_inter_house.txt",col_names = FALSE)$X1
pred_exact_var_extra <- read_csv("exact_results/exact_pred_var_extra_house.txt",col_names = FALSE)$X1

#we define a function that will be called for every tuning parameter
run_mra<-function(r){
  set.seed(r)
  
  #we keep J fixed and equal to 2, and we only vary r. M is then determined automatically
  mra.options.mra = list(r = c(r), J = 2)
  
  fitting_time <- system.time({
    vecchia.est<- vecchia_estimate_fixed_smoothness_1.5(
      data = house_train$log_price,
      locs = as.matrix(house_train[,1:2]),
      X = NULL,conditioning="mra", mra.options = mra.options.mra,reltol = 1e-6)
    
    vecchia.approx <- vecchia_specify(
      as.matrix(house_train[, c(1:2)]), 
      mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE)})[3]
  
  est_marginal_var=vecchia.est$theta.hat[1]
  est_range=vecchia.est$theta.hat[2]
  est_nugget=vecchia.est$theta.hat[4]
  
  
  #negloglik evaluation
  true_negloglik_eval_time <- system.time({
    true_negloglik <- -vecchia_likelihood(house_train$log_price, vecchia.approx, 
                                          c(marginal_var, range, 1.5), nugget)
  })[3]
  
  
  wrong_negloglik_eval_time <- system.time({
    wrong_negloglik<- -vecchia_likelihood(house_train$log_price, vecchia.approx, 
                                         c(2*marginal_var, 2*range, 1.5), 2 * nugget)
  })[3]
  
  
  
  # Predict on the training set
  pred_train_time <- system.time({
    pred_train <- vecchia_prediction(house_train$log_price, vecchia.approx, 
                                     c(est_marginal_var, est_range, 1.5), est_nugget
    )
  })[3]
  
  # Predict on interpolation set
  pred_inter_time <- system.time({
    vecchia.approx <- vecchia_specify(as.matrix(house_train[, c(1:2)]), 
                                      mra.options = mra.options.mra,conditioning = 'mra',
                                      verbose = TRUE,locs.pred = as.matrix(house_inter[, c(1:2)]))
    pred_inter <- vecchia_prediction(house_train$log_price, vecchia.approx,
                                     c(est_marginal_var, est_range, 1.5), est_nugget)})[3]
  
  # Predict on extrapolation set
  pred_extra_time <- system.time({
    vecchia.approx <- vecchia_specify(as.matrix(house_train[, c(1:2)]), 
                                      mra.options = mra.options.mra,
                                      conditioning = 'mra', verbose = TRUE,
                                      locs.pred = as.matrix(house_extra[, c(1:2)]))
    pred_extra <- vecchia_prediction(house_train$log_price, vecchia.approx, 
                                     c(est_marginal_var, est_range, 1.5), est_nugget)})[3]
  
  
  #add nugget to estimated variances
  pred_train$var.obs<-pred_train$var.obs+est_nugget
  pred_inter$var.pred<-pred_inter$var.pred+est_nugget
  pred_extra$var.pred<-pred_extra$var.pred+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((house_train$log_price - pred_train$mu.obs)^2))
  inter_rmse<-sqrt(mean((house_inter$log_price - pred_inter$mu.pred)^2))
  extra_rmse<-sqrt(mean((house_extra$log_price - pred_extra$mu.pred)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu.obs-house_train$log_price)^2)/pred_train$var.obs +
                       0.5*log(2*pi*pred_train$var.obs) )
  inter_score<-mean( (0.5*(pred_inter$mu.pred-house_inter$log_price)^2)/pred_inter$var.pred +
                       0.5*log(2*pi*pred_inter$var.pred) )
  extra_score<-mean( (0.5*(pred_extra$mu.pred-house_extra$log_price)^2)/pred_extra$var.pred +
                       0.5*log(2*pi*pred_extra$var.pred) )
  
  #crps
  crps <- function(predlist,trueobs) {
    z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
    scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                               2 * dnorm(z, 0, 1) - 1/sqrt(pi))
    return(scores)
  }
  
  train_crps<-mean(crps(list(mean=pred_train$mu.obs,sd=sqrt(pred_train$var.obs)),
                        house_train$log_price))
  inter_crps<-mean(crps(list(mean=pred_inter$mu.pred,sd=sqrt(pred_inter$var.pred)),
                        house_inter$log_price))
  extra_crps<-mean(crps(list(mean=pred_extra$mu.pred,sd=sqrt(pred_extra$var.pred)),
                        house_extra$log_price))
  
  #rmse between predictive means
  train_rmse_mean<-sqrt(mean((pred_train$mu.obs-pred_exact_mean_train)^2))
  inter_rmse_mean<-sqrt(mean((pred_inter$mu.pred-pred_exact_mean_inter)^2))
  extra_rmse_mean<-sqrt(mean((pred_extra$mu.pred-pred_exact_mean_extra)^2))
  
  #rmse between predictive variances
  train_rmse_var<-sqrt(mean((pred_train$var.obs-pred_exact_var_train)^2))
  inter_rmse_var<-sqrt(mean((pred_inter$var.pred-pred_exact_var_inter)^2))
  extra_rmse_var<-sqrt(mean((pred_extra$var.pred-pred_exact_var_extra)^2))
  
  #kl divergence
  compute_kl<-function(var1,var2,mean1,mean2){
    kl = log(sqrt(var2)/sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    sum(kl)
  }
  
  train_kl<-compute_kl(pred_exact_var_train,pred_train$var.obs,pred_exact_mean_train,pred_train$mu.obs)
  inter_kl<-compute_kl(pred_exact_var_inter,pred_inter$var.pred,pred_exact_mean_inter,pred_inter$mu.pred)
  extra_kl<-compute_kl(pred_exact_var_extra,pred_extra$var.pred,pred_exact_mean_extra,pred_extra$mu.pred)
  
  # Create the filename
  filename <- paste0("results/house/mra_house_",r)
  
  # Open the file for writing
  file_path <- paste0(filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("mra ", r),
    
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
    paste0("true negloglik: ", true_negloglik),
    paste0("wrong negloglik: ", wrong_negloglik),
    paste0("time for true negloglik evaluation: ", true_negloglik_eval_time),
    paste0("time for wrong negloglik evaluation: ", wrong_negloglik_eval_time),

  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_mra(1)
