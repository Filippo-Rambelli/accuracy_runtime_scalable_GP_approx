library(GPvecchia)
library(Matrix)
library(fields)
library(readr)

options(mc.cores = 8)

#we manually define a function to optimize all the covariance parameters but the
#smoothness. This is achiever by modyfing the vecchia_estimate function in the
#package GPVecchia
vecchia_estimate_fixed_smoothness_1.5 <- function(
    data, locs, X, m = 20, covmodel = 'matern', 
    theta.ini, output.level = 1,reltol = sqrt(.Machine$double.eps), ...) {
  
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
    l <- -vecchia_likelihood(z, vecchia.approx, full_params[-length(full_params)], 
                             full_params[length(full_params)], covmodel = covmodel)
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

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#we define a function that will be called for every tuning parameter
run_mra<-function(r){
  set.seed(r)
  laegern_train <- read.csv("data/laegern_train.csv")
  laegern_inter <- read.csv("data/laegern_interpolation.csv")
  laegern_extra <- read.csv("data/laegern_extrapolation.csv")
  
  #we center the data to match the 0 mean assumption
  train_mean<-mean(laegern_train$CanopyHeight)
  
  laegern_train$CanopyHeight<-laegern_train$CanopyHeight-train_mean
  laegern_inter$CanopyHeight<-laegern_inter$CanopyHeight-train_mean
  laegern_extra$CanopyHeight<-laegern_extra$CanopyHeight-train_mean
  
  #covariance parameters estimated by a Vecchia approximation with 240 neighbors
  nugget=8.146126889464754324e-07
  marginal_var=4.554596583854387398e-02
  range=1.596316574746951744e+01
  
  #we keep J fixed and equal to 2, and we only vary r. M is then determined automatically
  mra.options.mra = list(r = c(r), J = 2)
  
  fitting_time <- system.time({
    vecchia.est<- vecchia_estimate_fixed_smoothness_1.5(
      data = laegern_train$CanopyHeight,
      locs = as.matrix(laegern_train[,1:2]),
      X = NULL,conditioning="mra", mra.options = mra.options.mra,reltol = 1e-6)
    
    vecchia.approx <- vecchia_specify(
      as.matrix(laegern_train[, c(1:2)]), 
      mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE)})[3]
  
  est_marginal_var=vecchia.est$theta.hat[1]
  est_range=vecchia.est$theta.hat[2]
  est_nugget=vecchia.est$theta.hat[4]
  
  
  #negloglik evaluation
  true_negloglik_eval_time <- system.time({
    true_negloglik <- -vecchia_likelihood(laegern_train$CanopyHeight, vecchia.approx, 
                                          c(marginal_var, range, 1.5), nugget)
  })[3]
  
  
  wrong_negloglik_eval_time <- system.time({
    wrong_negloglik<- -vecchia_likelihood(laegern_train$CanopyHeight, vecchia.approx, 
                                         c(2*marginal_var, 2*range, 1.5), 2 * nugget)
  })[3]
  
  
  # Predict on the training set
  pred_train_time <- system.time({
    pred_train <- vecchia_prediction(
      laegern_train$CanopyHeight, vecchia.approx, c(est_marginal_var, est_range, 1.5), est_nugget
    )
  })[3]
  # Predict on interpolation set
  pred_inter_time <- system.time({
    vecchia.approx <- vecchia_specify(as.matrix(laegern_train[, c(1:2)]), 
                                      mra.options = mra.options.mra,conditioning = 'mra', 
                                      verbose = TRUE,locs.pred = as.matrix(laegern_inter[, c(1:2)]))
    pred_inter <- vecchia_prediction(laegern_train$CanopyHeight, vecchia.approx, 
                                     c(est_marginal_var, est_range, 1.5), est_nugget)})[3]
  
  # Predict on extrapolation set
  pred_extra_time <- system.time({
    vecchia.approx <- vecchia_specify(as.matrix(laegern_train[, c(1:2)]), 
                                      mra.options = mra.options.mra,conditioning = 'mra', 
                                      verbose = TRUE,locs.pred = as.matrix(laegern_extra[, c(1:2)]))
    pred_extra <- vecchia_prediction(laegern_train$CanopyHeight, vecchia.approx, 
                                     c(est_marginal_var, est_range, 1.5), est_nugget)})[3]
  
  
  #add nugget to estimated variances to obtain variances for the observable process
  pred_train$var.obs<-pred_train$var.obs+est_nugget
  pred_inter$var.pred<-pred_inter$var.pred+est_nugget
  pred_extra$var.pred<-pred_extra$var.pred+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((laegern_train$CanopyHeight - pred_train$mu.obs)^2))
  inter_rmse<-sqrt(mean((laegern_inter$CanopyHeight - pred_inter$mu.pred)^2))
  extra_rmse<-sqrt(mean((laegern_extra$CanopyHeight - pred_extra$mu.pred)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu.obs-laegern_train$CanopyHeight)^2)/pred_train$var.obs 
                     + 0.5*log(2*pi*pred_train$var.obs) )
  inter_score<-mean( (0.5*(pred_inter$mu.pred-laegern_inter$CanopyHeight)^2)/pred_inter$var.pred 
                     + 0.5*log(2*pi*pred_inter$var.pred) )
  extra_score<-mean( (0.5*(pred_extra$mu.pred-laegern_extra$CanopyHeight)^2)/pred_extra$var.pred 
                     + 0.5*log(2*pi*pred_extra$var.pred) )
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train$mu.obs,sd=sqrt(pred_train$var.obs)),
                        laegern_train$CanopyHeight))
  inter_crps<-mean(crps(list(mean=pred_inter$mu.pred,sd=sqrt(pred_inter$var.pred)),
                        laegern_inter$CanopyHeight))
  extra_crps<-mean(crps(list(mean=pred_extra$mu.pred,sd=sqrt(pred_extra$var.pred)),
                        laegern_extra$CanopyHeight))
  
  
  # Create the filename
  filename <- paste0("mra_laegern_r",r)
  
  # Open the file for writing
  file_path <- paste0("results/laegern/",filename, ".txt")
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
run_mra(2)
