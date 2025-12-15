#The code is based on an implementation from Joe Guinnes, to whom we give credits
#https://github.com/joeguinness/npspec/blob/master/vignettes/challenge_surface_temp.R

library(npspec)
library(Matrix)

options(mc.cores = 8)

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#load data
train_data <- read.csv("data/MODIS_data_train.txt", row.names=1, sep="")
test_data<-read.csv("data/MODIS_data_test.txt", row.names=1, sep="")

full_data<-rbind(train_data,test_data)

#observations are on a grid, but were provided in 2 different sets. We will now
#reconstruct the grid
east_indices_train <- match(train_data$east, sort(unique(full_data$east)))
north_indices_train <- match(train_data$north, sort(unique(full_data$north)))

east_indices_test <- match(test_data$east, sort(unique(full_data$east)))
north_indices_test <- match(test_data$north, sort(unique(full_data$north)))

# Create a sparse matrix with dimensions based on unique values
nrows <- length(unique(full_data$east))
ncols <- length(unique(full_data$north))
train_matrix <- sparseMatrix(i = east_indices_train, j = north_indices_train, 
                             x = train_data$temp, dims = c(nrows, ncols))
test_matrix <- sparseMatrix(i = east_indices_test, j = north_indices_test, 
                            x = test_data$temp, dims = c(nrows, ncols))

# Replace 0s with NAs
train_matrix[train_matrix == 0] <- NA
test_matrix[test_matrix == 0] <- NA

train_matrix<-as.matrix(train_matrix)
test_matrix<-as.matrix(test_matrix)

tmpr <- matrix( train_matrix, nrows, ncols )

nvec_obs <- c(nrows, ncols)

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

fitting_time <- system.time(fit <- 
                              iterate_spec(y,observed, X = X, burn_iters = 20, 
                                           par_spec_fun = spec_AR1, 
                                           embed_fac = embedding_factor,
                                           precond_method = "Vecchia", m = 10,
                                           silent = TRUE, ncondsim = 50,converge_tol = 1e-6))[3]
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
train_index <- !is.na(train_matrix)
test_index<-!is.na(test_matrix)

train_RMSE<-sqrt(mean((pred_vec[train_index]-train_matrix[train_index])^2,na.rm=T))
test_RMSE<-sqrt(mean((pred_vec[test_index]-test_matrix[test_index])^2,na.rm=T))

#log_score
train_score<-mean( (0.5*(pred_vec[train_index]-train_matrix[train_index])^2)/predvar_vec[train_index] + 
                     0.5*log(2*pi*pred_vec[train_index]) )
test_score<-mean( (0.5*(pred_vec[test_index]-test_matrix[test_index])^2)/predvar_vec[test_index] + 
                    0.5*log(2*pi*pred_vec[test_index]) ,na.rm=T)

#crps
train_crps<-mean(crps(list(mean=pred_vec[train_index],sd=sqrt(predvar_vec[train_index])),
                      train_matrix[train_index]))
test_crps<-mean(crps(list(mean=pred_vec[test_index],sd=sqrt(predvar_vec[test_index])),
                     test_matrix[test_index]),na.rm=TRUE)


# Create the filename
filename <- paste0("npspec_modis23_",embedding_factor)

# Open the file for writing
file_path <- paste0("results/modis23/",filename, ".txt")
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