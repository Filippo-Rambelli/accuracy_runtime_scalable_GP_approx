library(GPvecchia)
library(Matrix)
library(fields)
library(readr)
library(INLA)
library(rSPDE)

options(mc.cores = 8)

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#we define a function that will be called for every tuning parameter
run_spde<-function(max.edge){
  set.seed(1)
  #load data
  laegern_train <- read.csv("data/laegern_train.csv")
  laegern_inter <- read.csv("data/laegern_interpolation.csv")
  laegern_extra <- read.csv("data/laegern_extrapolation.csv")
  
  #we center the data to match the 0 mean assumption
  train_mean<-mean(laegern_train$CanopyHeight)
  
  laegern_train$CanopyHeight<-laegern_train$CanopyHeight-train_mean
  laegern_inter$CanopyHeight<-laegern_inter$CanopyHeight-train_mean
  laegern_extra$CanopyHeight<-laegern_extra$CanopyHeight-train_mean
  
  coords_train <- as.matrix(laegern_train[, 1:2])
  coords_inter<-as.matrix(laegern_inter[,c(1:2)])
  coords_extra<-as.matrix(laegern_extra[,c(1:2)])
  
  #define the mesh limits
  pl.dom <- cbind(c(668500,670000, 675200, 675200,674000, 668500), 
                  c(258000, 258000,258300, 260400, 260400,260100))
  

  fitting_time <- system.time({ 
    #construct the mesh
    mesh <- inla.mesh.2d(loc.domain = pl.dom, max.e = max.edge)
    spde <- rspde.matern(mesh = mesh,nu=1.5)
    
    A <- inla.spde.make.A(mesh, loc = coords_train)
    stk <- inla.stack(
      data = list(resp = laegern_train$CanopyHeight),
      A = list(A),
      effects = list(i = 1:spde$n.spde),
      tag = 'train')
    
    #only fit this time
    res_eb<-inla(resp~0+ f(i,model=spde),
                 data=inla.stack.data(stk),
                 control.predictor=list(A=inla.stack.A(stk)),
                 control.inla = list(int.strategy = "eb"),num.threads=8)
    
    
  })[3]
  
  pred_inter_time <- system.time({A_inter<-inla.spde.make.A(mesh,loc=coords_inter)
  stk.inter <- inla.stack(
    data = list(resp = NA),
    A = list(A_inter),
    effects = list(i = 1:spde$n.spde),
    tag = 'inter')})[3]
  
  pred_extra_time <- system.time({A_extra<-inla.spde.make.A(mesh,loc=coords_extra)
  stk.extra <- inla.stack(
    data = list(resp = NA),
    A = list(A_extra),
    effects = list(i = 1:spde$n.spde),
    tag = 'extra')})[3]
  
  #join all data stacks
  stk.full <- inla.stack(stk, stk.inter,stk.extra)
  
  #fit and prediction together, but using previous estimates to reduce time
  time_joint_pred<-system.time({ rpmu <- inla(resp ~ 0+f(i, model = spde),
                                              data = inla.stack.data(stk.full),
                                              control.mode = list(theta = res_eb$mode$theta, 
                                                                  restart = FALSE),
                                              control.predictor = list(A = inla.stack.A(
                                                stk.full), compute = TRUE),
                                              control.inla = list(int.strategy = "eb"),
                                              num.threads=8) })[3]
  
  pred_train_time<-time_joint_pred
  pred_inter_time<- pred_inter_time+time_joint_pred
  pred_extra_time<-pred_extra_time+time_joint_pred
  
  #extract train preds
  train_index<-inla.stack.index(stk.full, 'train')$data
  pred_train_mean<-rpmu$summary.fitted.values[train_index, 1]
  #add nugget to obtain variances for the observable process
  pred_train_var<-rpmu$summary.fitted.values[train_index, 2]^2+ 
    1/rpmu$summary.hyperpar[1,"mode"]
  
  #extract inter preds
  inter_index<-inla.stack.index(stk.full, 'inter')$data
  pred_inter_mean<-rpmu$summary.fitted.values[inter_index, 1]
  #add nugget to obtain variances for the observable process
  pred_inter_var<-rpmu$summary.fitted.values[inter_index, 2]^2+ 
    1/rpmu$summary.hyperpar[1,"mode"]
  
  #extract extra preds
  extra_index<-inla.stack.index(stk.full, 'extra')$data
  pred_extra_mean<-rpmu$summary.fitted.values[extra_index, 1]
  #add nugget to obtain variances for the observable process
  pred_extra_var<-rpmu$summary.fitted.values[extra_index, 2]^2+ 
    1/rpmu$summary.hyperpar[1,"mode"]
  
  #RMSE
  train_rmse<-sqrt(mean((laegern_train$CanopyHeight - pred_train_mean)^2))
  inter_rmse<-sqrt(mean((laegern_inter$CanopyHeight - pred_inter_mean)^2))
  extra_rmse<-sqrt(mean((laegern_extra$CanopyHeight - pred_extra_mean)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train_mean-laegern_train$CanopyHeight)^2)/pred_train_var +
                       0.5*log(2*pi*pred_train_var) )
  inter_score<-mean( (0.5*(pred_inter_mean-laegern_inter$CanopyHeight)^2)/pred_inter_var +
                       0.5*log(2*pi*pred_inter_var) )
  extra_score<-mean( (0.5*(pred_extra_mean-laegern_extra$CanopyHeight)^2)/pred_extra_var +
                       0.5*log(2*pi*pred_extra_var) )
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train_mean,sd=sqrt(pred_train_var)),
                        laegern_train$CanopyHeight))
  inter_crps<-mean(crps(list(mean=pred_inter_mean,sd=sqrt(pred_inter_var)),
                        laegern_inter$CanopyHeight))
  extra_crps<-mean(crps(list(mean=pred_extra_mean,sd=sqrt(pred_extra_var)),
                        laegern_extra$CanopyHeight))
  
  # Create the filename
  filename <- paste0("spde_laegern_max_edge",max.edge)
  
  # Open the file for writing
  file_path <- paste0("results/laegern/",filename, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("spde ", max.edge),
    
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
    paste0("true negloglik: "),
    paste0("wrong negloglik: "),
    paste0("time for true negloglik evaluation: "),
    paste0("time for wrong negloglik evaluation: ")
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}


#example usage
run_spde(48)