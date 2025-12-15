library(Matrix)
library(fields)
library(readr)
library(INLA)
library(rSPDE)
options(mc.cores = 8)

#load data
load("data/AllSatelliteTemps.RData")

crps <- function(predlist,trueobs) {
  z <- as.numeric((trueobs - predlist$mean) / predlist$sd)
  scores <- predlist$sd * (z *(2 * pnorm(z, 0, 1) - 1) +
                             2 * dnorm(z, 0, 1) - 1/sqrt(pi))
  return(scores)
}

#we define a function that will be called for every tuning parameter
run_spde<-function(max.edge){
  set.seed(1)
  
  train_data <- all.sat.temps[!is.na(all.sat.temps$MaskTemp),-3]
  test_data<-all.sat.temps[is.na(all.sat.temps$MaskTemp),-3]
  
  test_data<-test_data[complete.cases(test_data),]
  
  coords_train <- as.matrix(train_data[, 1:2])
  coords_test<- as.matrix(test_data[, 1:2])
  
  #define the mesh limits
  pl.dom <- cbind(c(min(all.sat.temps$Lon)-0.1, max(all.sat.temps$Lon)+0.1, 
                    max(all.sat.temps$Lon)+0.1, min(all.sat.temps$Lon)-0.1), 
                  c(min(all.sat.temps$Lat)-0.1, min(all.sat.temps$Lat)-0.1, 
                    max(all.sat.temps$Lat)+0.1, max(all.sat.temps$Lat)+0.1))
  
  fitting_time <- system.time({
    #construct the mesh
    mesh <- inla.mesh.2d(loc.domain = pl.dom, max.e = max.edge)
    spde <- rspde.matern(mesh = mesh,nu=0.5)
    
    A <- inla.spde.make.A(mesh, loc = coords_train)
    stk <- inla.stack(
      data = list(resp = train_data$TrueTemp),
      A = list(A,1,1,1),
      effects = list(i = 1:spde$n.spde,beta0=rep(1, nrow(train_data))
                     ,Lon=train_data$Lon,Lat=train_data$Lat),
      tag = 'train')
    
    #only fit this time
    res_eb<-inla(resp~0+beta0+Lon+Lat+ f(i,model=spde),
                 data=inla.stack.data(stk),
                 control.predictor=list(A=inla.stack.A(stk)),
                 control.inla = list(int.strategy = "eb"),num.threads=8)
  })[3]
  
  pred_test_time <- system.time({A_test<-inla.spde.make.A(mesh,loc=coords_test)
  stk.test <- inla.stack(
    data = list(resp = NA),
    A = list(A_test,1,1,1),
    effects = list(i = 1:spde$n.spde,beta0=rep(1, nrow(test_data))
                   ,Lon=test_data$Lon,Lat=test_data$Lat),
    tag = 'test')})[3]
  
  #join all data stacks
  stk.full <- inla.stack(stk, stk.test)
  
  #fit and prediction together, but using previous estimates to reduce time
  time_joint_pred<-system.time({ rpmu <- inla(resp~0+beta0+Lon+Lat+ f(i,model=spde),
                                              data = inla.stack.data(stk.full),
                                              control.mode = list(theta = res_eb$mode$theta, 
                                                                  restart = FALSE),
                                              control.predictor = list(A = inla.stack.A(
                                                stk.full), compute = TRUE),
                                              control.inla = list(int.strategy = "eb"),
                                              num.threads=8) })[3]
  
  pred_train_time<-time_joint_pred
  pred_test_time<- pred_test_time+time_joint_pred
  
  #extract train preds
  train_index<-inla.stack.index(stk.full, 'train')$data
  pred_train_mean<-rpmu$summary.fitted.values[train_index, 1]
  #add nugget to obtain variances for the observable process
  pred_train_var<-rpmu$summary.fitted.values[train_index, 2]^2+ 
    1/rpmu$summary.hyperpar[1,"mode"] 
  
  #extract test preds
  test_index<-inla.stack.index(stk.full, 'test')$data
  pred_test_mean<-rpmu$summary.fitted.values[test_index, 1]
  #add nugget to obtain variances for the observable process
  pred_test_var<-rpmu$summary.fitted.values[test_index, 2]^2+ 
    1/rpmu$summary.hyperpar[1,"mode"]
  
  
  #RMSE
  train_rmse<-sqrt(mean((train_data$TrueTemp -pred_train_mean)^2))
  test_rmse<-sqrt(mean((test_data$TrueTemp - pred_test_mean)^2))
  
  #log_score
  train_score<-mean( (0.5*(pred_train_mean-train_data$TrueTemp )^2)/pred_train_var +
                       0.5*log(2*pi*pred_train_var) )
  test_score<-mean( (0.5*(pred_test_mean-test_data$TrueTemp)^2)/pred_test_var +
                      0.5*log(2*pi*pred_test_var) ,na.rm=T)
  
  #crps
  train_crps<-mean(crps(list(mean=pred_train_mean,sd=sqrt(pred_train_var)),train_data$TrueTemp))
  test_crps<-mean(crps(list(mean=pred_test_mean,sd=sqrt(pred_test_var)),test_data$TrueTemp))
  
  
  # Create the filename
  filename <- paste0("spde_modis16_max_edge",max.edge)
  
  # Open the file for writing
  file_path <- paste0("results/modis16/",filename, ".txt")
  file_conn <- file(file_path, "w")
  
  
  # Write the data to the file
  writeLines(c(
    paste0("spde ", max.edge),
    
    paste0("time for fitting: ", fitting_time),
    paste0("univariate score train: ", train_score),
    paste0("univariate score test: ", test_score),
    paste0("time for train univariate prediction: ", pred_train_time),
    paste0("time for test univariate prediction: ", pred_test_time),
    paste0("rmse train: ", train_rmse),
    paste0("rmse test: ", test_rmse),
    paste0("crps train: ", train_crps),
    paste0("crps test: ", test_crps),
    paste0("true negloglik: "),
    paste0("wrong negloglik: "),
    paste0("time for true negloglik evaluation: "),
    paste0("time for wrong negloglik evaluation: ")
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_spde(0.036)