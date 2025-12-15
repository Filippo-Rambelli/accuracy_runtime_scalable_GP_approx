#The code is based on an implementation from Joe Guinnes, to whom we give credits
#https://github.com/joeguinness/npspec/blob/master/vignettes/challenge_surface_temp.R

library("npspec")
options(mc.cores = 8)

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#load data
load("data/AllSatelliteTemps.RData")

#convert data to matrix
tmpr <- matrix( all.sat.temps$MaskTemp, 500, 300 )

# get grid size
n1 <- nrow(tmpr)
n2 <- ncol(tmpr)
nvec_obs <- c(n1,n2)

# get pattern of missing values
y <- tmpr[1:nvec_obs[1],1:nvec_obs[2]]
observed <- !is.na(y)
nobs <- sum(observed)

# define locations and covariates
locs <- as.matrix( expand.grid( 1:nvec_obs[1], 1:nvec_obs[2] ) )
X <- array(NA, c(nvec_obs,3))
X[,,1] <- 1
X[,,2] <- array( locs[,1], nvec_obs)
X[,,3] <- array( locs[,2], nvec_obs)
# fit the model
set.seed(1)

embedding_factor=1.2

fitting_time <- 
  system.time(fit <- iterate_spec(y,observed, X = X, burn_iters = 20, 
                              par_spec_fun = spec_AR1, 
                              embed_fac = embedding_factor,
                              precond_method = "Vecchia", m = 10,
                              silent = TRUE, ncondsim = 50,
                              converge_tol = 1e-6))[3]
# predictions
pred_mat <- fit$condexp
pred_vec <- c(pred_mat)


# calculate the prediction variances based on the
# conditional simulations
cond_diff <- array(NA, dim(fit$condsim) )
pred_time<-system.time({
  for(j in 1:dim(fit$condsim)[3]) cond_diff[,,j] <- fit$condsim[,,j] - fit$condexp
  meansq <- function(x) 1/length(x)*sum(x^2)
  predvar_mat <- apply( cond_diff, c(1,2), meansq )
  predvar_vec <- c(predvar_mat)
})[3]

#rmse
train_index <- !is.na(all.sat.temps$MaskTemp)
test_index<-is.na(all.sat.temps$MaskTemp)

train_RMSE<-sqrt(mean((pred_vec[train_index]-all.sat.temps$TrueTemp[train_index])^2,na.rm=T))
test_RMSE<-sqrt(mean((pred_vec[test_index]-all.sat.temps$TrueTemp[test_index])^2,na.rm=T))

#log_score
train_score<-mean( (0.5*(pred_vec[train_index]-all.sat.temps$TrueTemp[train_index])^2)/predvar_vec[train_index] + 
                     0.5*log(2*pi*pred_vec[train_index]) )
test_score<-mean( (0.5*(pred_vec[test_index]-all.sat.temps$TrueTemp[test_index])^2)/predvar_vec[test_index] + 
                    0.5*log(2*pi*pred_vec[test_index]) ,na.rm=T)

#crps
train_crps<-mean(crps(list(mean=pred_vec[train_index],sd=sqrt(predvar_vec[train_index])),
                      all.sat.temps$TrueTemp[train_index]))
test_crps<-mean(crps(list(mean=pred_vec[test_index],sd=sqrt(predvar_vec[test_index])),
                     all.sat.temps$TrueTemp[test_index]),na.rm=TRUE)


# Create the filename
filename <- paste0("npspec_modis16_",embedding_factor)

# Open the file for writing
file_path <- paste0("results/modis16/",filename, ".txt")
file_conn <- file(file_path, "w")


# Write the data to the file
writeLines(c(
  paste0("Periodic embedding ", embedding_factor),
  
  paste0("time for fitting: ", fitting_time),
  paste0("univariate score train: ", train_score),
  paste0("univariate score test: ", test_score),
  paste0("time for train univariate prediction: ", pred_time),
  paste0("time for test univariate prediction: ", pred_time),
  paste0("rmse train: ", train_RMSE),
  paste0("rmse test: ", test_RMSE),
  paste0("crps train: ", train_crps),
  paste0("crps test: ", test_crps),
  paste0("true negloglik: "),
  paste0("wrong negloglik: "),
  paste0("time for true negloglik evaluation: "),
  paste0("time for wrong negloglik evaluation: "),
), file_conn)

# Close the file connection
close(file_conn)
