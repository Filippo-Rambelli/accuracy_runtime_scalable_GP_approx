library(Matrix)
library(fields)
library(readr)
library(INLA)
library(rSPDE)

wd = "/cluster/scratch/fabiopc"

options(mc.cores = 8)

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
run_spde<-function(range=0.2,max.edge=0.1){
  set.seed(1)
  
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
  
  eff_range=range/range_denom
  nugget=0.5
  marginal_var=1
  nu = 1.5
  
  nrep=max(full_data$rep)
  
  rmse_train_list<-rep(0,nrep/2); rmse_inter_list<-rep(0,nrep/2); rmse_extra_list<-rep(0,nrep/2)
  score_train_list<-rep(0,nrep/2); score_inter_list<-rep(0,nrep/2); score_extra_list<-rep(0,nrep/2)
  pred_train_times<-rep(0,nrep/2);pred_inter_times<-rep(0,nrep/2);pred_extra_times<-rep(0,nrep/2)
  fitting_times<-rep(0,nrep); nugget_list<-rep(0,nrep); variance_list<-rep(0,nrep); range_list<-rep(0,nrep)
  rmse_mean_train_list<-rep(0,nrep/2); rmse_var_train_list<-rep(0,nrep/2)
  rmse_mean_inter_list<-rep(0,nrep/2); rmse_var_inter_list<-rep(0,nrep/2)
  rmse_mean_extra_list<-rep(0,nrep/2); rmse_var_extra_list<-rep(0,nrep/2)
  kl_train<-rep(0,nrep/2); kl_inter<-rep(0,nrep/2); kl_extra<-rep(0,nrep/2)
  
  for (i in 1:nrep){
    data<-full_data[full_data$rep==i,]
    train<-data[data$which=="train",]
    interp<-data[data$which=="interpolation",]
    extrap<-data[data$which=="extrapolation",]
    
    #define the mesh limits
    pl.dom <- cbind(c(0, 1, 1, 0), c(0, 0, 1, 1))
    
    coords_train <- as.matrix(train[, 1:2])
    coords_inter<-as.matrix(interp[,c(1:2)])
    coords_extra<-as.matrix(extrap[,c(1:2)])
    
    #only fit
    fitting_times[i] <- system.time({
      #construct the mesh
      mesh <- inla.mesh.2d(loc.domain = pl.dom, max.e = max.edge)
      spde <- rspde.matern(mesh = mesh,nu=nu)
      
      A <- inla.spde.make.A(mesh, loc = coords_train)
      stk <- inla.stack(
        data = list(resp = train$y),
        A = list(A),
        effects = list(i = 1:spde$n.spde),
        tag = 'train')
      
      #only fit this time
      res_eb<-inla(resp~0+ f(i,model=spde),
                   data=inla.stack.data(stk),
                   control.predictor=list(A=inla.stack.A(stk)),
                   control.inla = list(int.strategy = "eb"),num.threads=8)
      
    })[3]
    
    #store parameter estimates
    nugget_list[i]<-1/res_eb$summary.hyperpar["Precision for the Gaussian observations","mode"]
    
    result_fit<-rspde.result(res_eb,"i",spde, parameterization = "matern")
    variance_list[i]<-summary(result_fit)["std.dev","mode"]^2
    range_list[i]<-summary(result_fit)["range","mode"]/2 #division necessary to make the range comparable to the one used to simulate
    
    if(i<=nrep/2){
      #fit and predict together, but speedup such that it is mostly prediction time
      pred_inter_times[i] <- system.time({A_inter<-inla.spde.make.A(mesh,loc=coords_inter)
      stk.inter <- inla.stack(
        data = list(resp = NA),
        A = list(A_inter),
        effects = list(i = 1:spde$n.spde),
        tag = 'inter')})[3]
      
      pred_extra_times[i] <- system.time({A_extra<-inla.spde.make.A(mesh,loc=coords_extra)
      stk.extra <- inla.stack(
        data = list(resp = NA),
        A = list(A_extra),
        effects = list(i = 1:spde$n.spde),
        tag = 'extra')})
      
      #join all data stacks
      stk.full <- inla.stack(stk, stk.inter,stk.extra)
      
      #fit and prediction together, but using previous estimates to reduce time
      time_joint_pred<-system.time({ rpmu <- inla(resp ~ 0+f(i, model = spde),
                                                  data = inla.stack.data(stk.full),
                                                  control.mode = list(theta = res_eb$mode$theta, restart = FALSE),
                                                  control.predictor = list(A = inla.stack.A(
                                                    stk.full), compute = TRUE),
                                                  control.inla = list(int.strategy = "eb"),num.threads=8 ) })[3]
      pred_train_times[i]<-time_joint_pred
      pred_inter_times[i]<- pred_inter_times[i]+time_joint_pred
      pred_extra_times[i]<-pred_extra_times[i]+time_joint_pred
      
      #extract train preds
      train_index<-inla.stack.index(stk.full, 'train')$data
      pred_train_mean<-rpmu$summary.fitted.values[train_index, 1]
      pred_train_sd<-rpmu$summary.fitted.values[train_index, 2]
      
      #extract inter preds
      inter_index<-inla.stack.index(stk.full, 'inter')$data
      pred_inter_mean<-rpmu$summary.fitted.values[inter_index, 1]
      pred_inter_sd<-rpmu$summary.fitted.values[inter_index, 2]
      
      #extract extra preds
      extra_index<-inla.stack.index(stk.full, 'extra')$data
      pred_extra_mean<-rpmu$summary.fitted.values[extra_index, 1]
      pred_extra_sd<-rpmu$summary.fitted.values[extra_index, 2]
      
      #RMSE
      rmse_train_list[i]<-sqrt(mean((train$f - pred_train_mean)^2))
      rmse_inter_list[i]<-sqrt(mean((interp$f - pred_inter_mean)^2))
      rmse_extra_list[i]<-sqrt(mean((extrap$f - pred_extra_mean)^2))
      
      #log_score
      score_train_list[i]<-mean( (0.5*(pred_train_mean-train$f )^2)/pred_train_sd^2 +
                                   0.5*log(2*pi*pred_train_sd^2) )
      score_inter_list[i]<-mean( (0.5*(pred_inter_mean-interp$f)^2)/pred_inter_sd^2 +
                                   0.5*log(2*pi*pred_inter_sd^2) )
      score_extra_list[i]<-mean( (0.5*(pred_extra_mean-extrap$f)^2)/pred_extra_sd^2 +
                                   0.5*log(2*pi*pred_extra_sd^2) )
      
      #exact_calculations
      exact_pred_mean_train <-exact_pred_mean_train_whole[exact_pred_mean_train_whole$iteration==i,2]
      exact_pred_var_train<-exact_pred_var_train_whole[exact_pred_mean_train_whole$iteration==i,2]
      
      exact_pred_mean_inter<-exact_pred_mean_inter_whole[exact_pred_mean_train_whole$iteration==i,2]
      exact_pred_var_inter<-exact_pred_var_inter_whole[exact_pred_mean_train_whole$iteration==i,2]
      
      exact_pred_mean_extra<-exact_pred_mean_extra_whole[exact_pred_mean_train_whole$iteration==i,2] 
      exact_pred_var_extra<-exact_pred_var_extra_whole[exact_pred_mean_train_whole$iteration==i,2] 
      
      #kl
      kl_train[i]<-compute_kl(exact_pred_var_train,pred_train_sd^2,exact_pred_mean_train,pred_train_mean)
      kl_inter[i]<-compute_kl(exact_pred_var_inter,pred_inter_sd^2,exact_pred_mean_inter,pred_inter_mean)
      kl_extra[i]<-compute_kl(exact_pred_var_extra,pred_extra_sd^2,exact_pred_mean_extra,pred_extra_mean)
      
      #rmse means
      rmse_mean_train_list[i]<-sqrt(mean((pred_train_mean-exact_pred_mean_train)^2))
      rmse_mean_inter_list[i]<-sqrt(mean((pred_inter_mean-exact_pred_mean_inter)^2))
      rmse_mean_extra_list[i]<-sqrt(mean((pred_extra_mean-exact_pred_mean_extra)^2))
      
      #rmse vars
      rmse_var_train_list[i]<-sqrt(mean((pred_train_sd^2-exact_pred_var_train)^2))
      rmse_var_inter_list[i]<-sqrt(mean((pred_inter_sd^2-exact_pred_var_inter)^2))
      rmse_var_extra_list[i]<-sqrt(mean((pred_extra_sd^2-exact_pred_var_extra)^2))
      
      rm(A_inter,A_extra,stk.full,stk.inter,stk.extra,rpmu,pred_train_mean,pred_train_sd,
         pred_inter_mean,pred_inter_sd,pred_extra_mean,pred_extra_sd)
    }
    print(paste0("rep ",i," done"))
    rm(data,train,interp,extrap,mesh,spde,A,stk,res_eb,coords_train,coords_inter,coords_extra)
  }
  
  # Create the filename
  filename_save <- paste0("results/",range,"/spde_",range,"_max_edge",max.edge)
  
  # Open the file for writing
  file_path <- paste0(filename_save, ".txt")
  file_conn <- file(file_path, "w")
  
  # Write the data to the file
  writeLines(c(
    paste0("spde ", max.edge),
    paste0("True range: ",range/range_denom ),
    
    paste0("bias for GP range: ", mean(range_list - eff_range)),
    paste0("MSE for GP range: ", mean((range_list - eff_range)^2)),
    paste0("bias for GP variance: ", mean(variance_list - marginal_var)),
    paste0("MSE for GP variance: ", mean((variance_list - marginal_var)^2)),
    paste0("bias for error term variance: ", mean(nugget_list - nugget)),
    paste0("MSE for error term variance: ", mean((nugget_list - nugget)^2)),
    paste0("variance for bias of GP range: ", var(range_list)/length(range_list)),
    paste0("variance for bias GP of variance: ", var(variance_list)/length(variance_list)),
    paste0("variance for bias error of term variance: ", var(nugget_list)/length(nugget_list)),
    paste0("variance for MSE GP range: ", var((range_list - eff_range)^2)/length(range_list)),
    paste0("variance for MSE GP variance: ", var((variance_list - marginal_var)^2)/length(variance_list)),
    paste0("variance for MSE error term variance: ", var((nugget_list - nugget)^2)/length(nugget_list)),
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
    paste0("variance for RMSE extra: ", var(rmse_extra_list)/length(rmse_extra_list)),
    paste0("range_list: ", range_list)
  ), file_conn)
  
  # Close the file connection
  close(file_conn)
}

#example usage
run_spde(range=0.2,max.edge = 0.07)
