library(GPvecchia)
library(Matrix)
library(fields)
library(readr)
options(mc.cores = 8)

#load data
load("data/AllSatelliteTemps.RData")
load("data/SatelliteTemps.RData")

#exponential covfun=matern covfun with smoothness=0.5
#we manually define a function to optimize all the covariance parameters but the
#smoothness. This is achiever by modyfing the vecchia_estimate function in the
#package GPVecchia
vecchia_estimate_fixed_smoothness_0.5 <- function(data, locs, X, m = 20, covmodel = 'matern',
                                                  theta.ini, output.level = 1,
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
    full_params <- c(full_params[1:2], 0.5, full_params[3]) # Insert the fixed smoothness value
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
  theta.hat <- c(theta.hat[1:2], 0.5, theta.hat[3]) # Fix smoothness to 1.5
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
  
  #covariance parameters estimated by a Vecchia approximation with 310 neighbours
  nugget=2.756430177414865382e-06
  marginal_var=6.096940872884346163e+00
  range=1.137103704281087202e-01
  
  #coefficients estimated by a Vecchia approximation with 310 neighbours  
  intercept=-2.437012419872512794e+02
  x1=1.773139234046750001e+00
  x2=-2.405353477654387007e+00
  
  train_data <- subset(sat.temps,!is.na(Temp)) 
  
  #we keep J fixed and equal to 2, and we only vary r. M is then determined automatically
  mra.options.mra = list(r = c(r), J = 2)
  
  fitting_time <- system.time({
    vecchia.est<- vecchia_estimate_fixed_smoothness_0.5(
      data = train_data$Temp,
      locs = as.matrix(train_data[,1:2]),
      X = as.matrix(cbind(1,train_data[,1:2])),conditioning="mra", 
      mra.options = mra.options.mra,reltol = 1e-6)
    
    vecchia.approx <- vecchia_specify(
      as.matrix(train_data[, c(1:2)]), 
      mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE)})[3]
  
  
  est_marginal_var=vecchia.est$theta.hat[1]
  est_range=vecchia.est$theta.hat[2]
  est_nugget=vecchia.est$theta.hat[4]
  
  
  #negloglik evaluation
  
  #detrend train
  trend<-as.matrix(cbind(1,train_data[,1:2]))%*%c(intercept,x1,x2)
  train_detrended<-train_data$Temp-trend
  
  true_negloglik_eval_time <- system.time({
    true_negloglik <- -vecchia_likelihood(train_detrended, vecchia.approx, 
                                          c(marginal_var, range, 0.5), nugget)
  })[3]
  
  
  wrong_negloglik_eval_time <- system.time({
    wrong_negloglik<- -vecchia_likelihood(train_detrended, vecchia.approx, 
                                         c(2*marginal_var, 2*range, 0.5), 2 * nugget)
  })[3]
  
  
  # Predict on the training set
  pred_train_time <- system.time({
    pred_train <- vecchia_prediction(
      train_data$Temp, vecchia.approx, c(est_marginal_var, est_range, 0.5), est_nugget
    )
  })[3]
  
  
  # Predict on the test set
  test_data<-all.sat.temps[is.na(all.sat.temps$MaskTemp),-3]
  test_data<-test_data[complete.cases(test_data),]
  
  pred_test_time <- system.time({
    pred_test <- vecchia_pred(
      vecchia.est, as.matrix(test_data[, c(1:2)]),X.pred=as.matrix(cbind(1,test_data[,1:2])),
      mra.options = mra.options.mra, conditioning = 'mra', verbose = TRUE
    )
  })[3]
  
  
  #add nugget to estimated variances to obtain variances for the observable process
  pred_train$var.obs<-pred_train$var.obs+est_nugget
  pred_test$var.pred<-pred_test$var.pred+est_nugget
  
  #RMSE
  train_rmse<-sqrt(mean((train_data$Temp - pred_train$mu.obs)^2))
  test_rmse<-sqrt(mean((test_data$TrueTemp - pred_test$mean.pred)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train$mu.obs-train_data$Temp)^2)/pred_train$var.obs 
                     + 0.5*log(2*pi*pred_train$var.obs) )
  test_score<-mean( (0.5*(pred_test$mean.pred-test_data$TrueTemp)^2)/pred_test$var.pred 
                    + 0.5*log(2*pi*pred_test$var.pred) )
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train$mu.obs,sd=sqrt(pred_train$var.obs)),train_data$Temp))
  test_crps<-mean(crps(list(mean=pred_test$mean.pred,sd=sqrt(pred_test$var.pred)),test_data$TrueTemp))
  
  
  # Create the filename
  filename <- paste0("mra_modi16_r",r)
  
  # Open the file for writing
  file_path <- paste0("results/modis16/",filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("mra ", r),
    
    paste0("time for fitting: ", fitting_time),
    paste0("univariate score train: ", train_score),
    paste0("univariate score test: ", test_score),
    paste0("time for train univariate prediction: ", pred_train_time),
    paste0("time for test univariate prediction: ", pred_test_time),
    paste0("rmse train: ", train_rmse),
    paste0("rmse test: ", test_rmse),
    paste0("crps train: ", train_crps),
    paste0("crps test: ", test_crps),
    paste0("true negloglik: ", true_negloglik),
    paste0("wrong negloglik: ", wrong_negloglik),
    paste0("time for true negloglik evaluation: ", true_negloglik_eval_time),
    paste0("time for wrong negloglik evaluation: ", wrong_negloglik_eval_time),
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_mra(3)