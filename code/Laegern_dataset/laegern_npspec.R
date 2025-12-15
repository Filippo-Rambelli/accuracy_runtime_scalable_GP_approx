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
laegern_train <- read.csv("data/laegern_train.csv")
laegern_inter <- read.csv("data/laegern_interpolation.csv")
laegern_extra <- read.csv("data/laegern_extrapolation.csv")

#we center the data to match the 0 mean assumption
train_mean<-mean(laegern_train$CanopyHeight)

laegern_train$CanopyHeight<-laegern_train$CanopyHeight-train_mean
laegern_inter$CanopyHeight<-laegern_inter$CanopyHeight-train_mean
laegern_extra$CanopyHeight<-laegern_extra$CanopyHeight-train_mean

full_data<-rbind(laegern_train, laegern_inter, laegern_extra)

x_indices_train <- match(laegern_train$x, sort(unique(full_data$x)))
y_indices_train <- match(laegern_train$y, sort(unique(full_data$y)))

x_indices_inter <- match(laegern_inter$x, sort(unique(full_data$x)))
y_indices_inter <- match(laegern_inter$y, sort(unique(full_data$y)))

x_indices_extra <- match(laegern_extra$x, sort(unique(full_data$x)))
y_indices_extra <- match(laegern_extra$y, sort(unique(full_data$y)))


# Create a sparse matrix with dimensions based on unique values
nrows <- length(unique(full_data$x))
ncols <- length(unique(full_data$y))
train_matrix <- sparseMatrix(i = x_indices_train, j = y_indices_train, x = laegern_train$CanopyHeight, dims = c(nrows, ncols))
inter_matrix <- sparseMatrix(i = x_indices_inter, j = y_indices_inter, x = laegern_inter$CanopyHeight, dims = c(nrows, ncols))
extra_matrix <- sparseMatrix(i = x_indices_extra, j = y_indices_extra, x = laegern_extra$CanopyHeight, dims = c(nrows, ncols))


# Replace 0s with NAs
train_matrix[train_matrix == 0] <- NA
inter_matrix[inter_matrix == 0] <- NA
extra_matrix[extra_matrix == 0] <- NA

train_matrix<-as.matrix(train_matrix)
inter_matrix<-as.matrix(inter_matrix)
extra_matrix<-as.matrix(extra_matrix)

canopy <- matrix( train_matrix, nrows, ncols )

nvec_obs <- c(nrows,ncols)

# get pattern of missing values
y <- canopy[1:nvec_obs[1],1:nvec_obs[2]]
observed <- !is.na(y)
nobs <- sum(observed)

# fit the model
set.seed(1)

embedding_factor=1.2

fitting_time <- system.time(fit <- iterate_spec(y,
                                                observed, X = NULL, burn_iters = 20, par_spec_fun = spec_AR1, embed_fac = embedding_factor,
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

#extract indexis
train_index <- !is.na(train_matrix)
inter_index<-!is.na(inter_matrix)
extra_index<-!is.na(extra_matrix)

#RMSE
train_rmse<-sqrt(mean((laegern_train$CanopyHeight - pred_vec[train_index])^2))
inter_rmse<-sqrt(mean((laegern_inter$CanopyHeight -pred_vec[inter_index])^2))
extra_rmse<-sqrt(mean((laegern_extra$CanopyHeight - pred_vec[extra_index])^2))

#log_score
train_score<-mean( (0.5*(pred_vec[train_index]-laegern_train$CanopyHeight)^2)/predvar_vec[train_index] + 0.5*log(2*pi*predvar_vec[train_index]) )
inter_score<-mean( (0.5*(pred_vec[inter_index]-laegern_inter$CanopyHeight)^2)/predvar_vec[inter_index] + 0.5*log(2*pi*predvar_vec[inter_index]) )
extra_score<-mean( (0.5*(pred_vec[extra_index]-laegern_extra$CanopyHeight)^2)/predvar_vec[extra_index] + 0.5*log(2*pi*predvar_vec[extra_index]) )

#crps
train_crps<-mean(crps(list(mean=pred_vec[train_index],sd=sqrt(predvar_vec[train_index])),laegern_train$CanopyHeight))
inter_crps<-mean(crps(list(mean=pred_vec[inter_index],sd=sqrt(predvar_vec[inter_index])),laegern_inter$CanopyHeight))
extra_crps<-mean(crps(list(mean=pred_vec[extra_index],sd=sqrt(predvar_vec[extra_index])),laegern_extra$CanopyHeight))


# Create the filename
filename <- paste0("npspec_laegern_",embedding_factor)

# Open the file for writing
file_path <- paste0("results/laegern/",filename, ".txt")
file_conn <- file(file_path, "w")

# Write the data to the file
writeLines(c(
  paste0("Periodic embedding ", embedding_factor),
  
  paste0("time for fitting: ", fitting_time),
  paste0("univariate score train: ", train_score),
  paste0("univariate score interpolation: ", inter_score),
  paste0("univariate score extrapolation: ", extra_score),
  paste0("time for train univariate prediction: ", pred_time),
  paste0("time for interpolation univariate prediction: ", pred_time),
  paste0("time for extrapolation univariate prediction: ", pred_time),
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
  
  paste0("sync time: ", round(fitting_time+pred_time , 4)),
  paste0("total time: ", round(fitting_time+pred_time , 4))
), file_conn)

# Close the file connection
close(file_conn)
