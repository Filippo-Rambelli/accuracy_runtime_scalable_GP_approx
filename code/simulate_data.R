#The code for isotropic GP simulation relies on a implementation from Fabio Sigrist, December 2023
#The code for anisotropic GP simulation relies on a implementation from Fabio Sigrist, November 2025

#Setup----
library(RandomFields)

#ignore displayed errors
setwd("C:/Users/filor/Desktop/tesi ETH/data")

#Define functions----
sim_gp_given_coords<-function(coords,
                              rho, #effective range
                              sigma2, #marginal variance
                              nu, #smoothness
                              sigma2_error, #variance of error (=nugget)
                              n #sample size of each partition
){
  if (rho == 0) {
    iid_no_GP <- TRUE
  } else {
    iid_no_GP <- FALSE
  }
  if (iid_no_GP) {
    eps <- rnorm(n,sd = sqrt(sigma2))
  } else {
    if (nu == 0.5) {
      RFmodel <- RMexp(var=sigma2, scale=rho)
    } else if (nu > 1e3) {
      RFmodel <- RMgauss(var=sigma2, scale=rho)
    } else {
      RFmodel <- RMmatern(var=sigma2, scale=rho, nu=nu)
    }
    sim <- RFsimulate(RFmodel, x=coords) # ignore warning
    eps <- sim$variable1
  }
  y <- eps + rnorm(n, sd=sqrt(sigma2_error))
  cbind(y,eps)
}

generate_dataset <- function(index,
                                rho,
                                n,
                                reps,
                                seed,
                                file_name,
                                sigma2_error = 0.5,
                                nu = 1.5,
                                range_denom =2.74 #nu-dependent normalization constant
                             ) {
  set.seed(seed)
  list_result <- vector("list", reps)
  
  for (j in 1:reps) {
    coords_train <- matrix(runif(2)/2, ncol=2)
    while (nrow(coords_train) < n) {
      coord_i <- runif(2)
      if (!(coord_i[1] >= 0.5 & coord_i[2] >= 0.5)) {
        coords_train <- rbind(coords_train, coord_i)
      }
    }
    
    coords_inter <- matrix(runif(2)/2, ncol=2)
    while (nrow(coords_inter) < n) {
      coord_i <- runif(2)
      if (!(coord_i[1] >= 0.5 & coord_i[2] >= 0.5)) {
        coords_inter <- rbind(coords_inter, coord_i)
      }
    }
    
    coords_extra <- matrix(1 - runif(n * 2) / 2, ncol=2)
    coords <- rbind(coords_train, coords_inter, coords_extra)
    
    yeps <- sim_gp_given_coords(coords, rho = rho / range_denom, nu = nu,
                                sigma2_error = sigma2_error, n = 3 * n)
    
    which <- c(rep("train", n), rep("interpolation", n), rep("extrapolation", n))
    full_data <- cbind(coords, yeps, which, j)
    colnames(full_data) <- c("x1", "x2", "y", "f", "which", "rep")
    
    list_result[[j]] <- full_data
  }
  
  combined_data <- do.call(rbind, list_result)
  write.csv(combined_data, file = file_name, row.names = FALSE)
  message(paste("Saved", file_name))
}

#Define ARD functions----
sim_ard_gp_given_coords<-function(coords,
                                  rho1, #effective range 1
                                  rho2, #effective range 2
                                  sigma2, #marginal variance
                                  nu, #smoothness
                                  sigma2_error, #variance of error (=nugget)
                                  n #sample size of each partition
){
  
  coords_simulate <- coords
  ## ARD covariance
  coords_simulate[,1] <- coords_simulate[,1] / rho1
  coords_simulate[,2] <- coords_simulate[,2] / rho2
  rho <- 1
  
  if (rho == 0) {
    iid_no_GP <- TRUE
  } else {
    iid_no_GP <- FALSE
  }
  
  if (iid_no_GP) {
    eps <- rnorm(n,sd = sqrt(sigma2))
  } else {
    if (nu == 0.5) {
      RFmodel <- RMexp(var=sigma2, scale=rho)
    } else if (nu > 1e3) {
      RFmodel <- RMgauss(var=sigma2, scale=rho)
    } else {
      RFmodel <- RMmatern(var=sigma2, scale=rho, nu=nu)
    }
    sim <- RFsimulate(RFmodel, x=coords_simulate) # ignore warning
    eps <- sim$variable1
  }
  y <- eps + rnorm(n, sd=sqrt(sigma2_error))
  cbind(y,eps)
}

generate_ard_dataset <- function(index,
                                 rho1,
                                 rho2,
                                 n,
                                 reps,
                                 seed,
                                 file_name,
                                 sigma2_error = 0.5,
                                 nu = 1.5,
                                 range_denom =2.74 #nu-dependent normalization constant
) {
  set.seed(seed)
  list_result <- vector("list", reps)
  
  for (j in 1:reps) {
    coords_train <- matrix(runif(2)/2, ncol=2)
    while (nrow(coords_train) < n) {
      coord_i <- runif(2)
      if (!(coord_i[1] >= 0.5 & coord_i[2] >= 0.5)) {
        coords_train <- rbind(coords_train, coord_i)
      }
    }
    
    coords_inter <- matrix(runif(2)/2, ncol=2)
    while (nrow(coords_inter) < n) {
      coord_i <- runif(2)
      if (!(coord_i[1] >= 0.5 & coord_i[2] >= 0.5)) {
        coords_inter <- rbind(coords_inter, coord_i)
      }
    }
    
    coords_extra <- matrix(1 - runif(n * 2) / 2, ncol=2)
    coords <- rbind(coords_train, coords_inter, coords_extra)
    
    yeps <- sim_ard_gp_given_coords(coords, rho1 = rho1 / range_denom, rho2 = rho2 / range_denom, 
                                    nu = nu, sigma2_error = sigma2_error, n = 3 * n)
    
    which <- c(rep("train", n), rep("interpolation", n), rep("extrapolation", n))
    full_data <- cbind(coords, yeps, which, j)
    colnames(full_data) <- c("x1", "x2", "y", "f", "which", "rep")
    
    list_result[[j]] <- full_data
  }
  
  combined_data <- do.call(rbind, list_result)
  write.csv(combined_data, file = file_name, row.names = FALSE)
  message(paste("Saved", file_name))
}

#Simulate datasets----
generate_dataset(index = 1, rho = 0.05, n = 10000, reps = 100, seed = 1,file_name = "combined_data_r005.csv")
generate_dataset(index = 2, rho = 0.2, n = 10000, reps = 100, seed = 2, file_name = "combined_data_r02.csv")
generate_dataset(index = 3, rho = 0.5, n = 10000, reps = 100, seed = 3, file_name = "combined_data_r05.csv")

generate_dataset(index = 4, rho = 0.05, n = 100000, reps = 20, seed = 4, file_name = "combined_data_100k_r005.csv")
generate_dataset(index = 5, rho = 0.2, n = 100000, reps = 20, seed = 5, file_name = "combined_data_100k_r02.csv")
generate_dataset(index = 6, rho = 0.5, n = 100000, reps = 20, seed = 6, file_name = "combined_data_100k_r05.csv")

generate_dataset(index = 7, rho = 0.2, n = 100000, reps = 20, seed = 7, file_name = "combined_data_100k_r02_n01.csv", sigma2_error = 0.1)
generate_dataset(index = 8, rho = 0.2, n = 100000, reps = 20, seed = 8, file_name = "combined_data_100k_r02_s05.csv", nu = 0.5,range_denom=3)
generate_dataset(index = 9, rho = 0.2, n = 100000, reps = 20, seed = 9, file_name = "combined_data_100k_r02_s25.csv", nu = 2.5,range_denom=2.65)

generate_ard_dataset(index=10, rho1 = 0.05, rho2 = 0.2, n = 100000, reps = 20, seed = 10, file_name = "combined_data_100k_r005and02.csv")

#Plot data (direct load)----
library(ggplot2)
dpi=70
full_data<-read.csv("combined_data_r02.csv")

plot_data<-as.data.frame(full_data)[full_data$rep==1,c(1,2,5)]
plot_data$which<-as.factor(plot_data$which)
levels(plot_data$which)<-c("Test_extra","Test_inter","Training")
plot_data$Set<-factor(plot_data$which, levels = c("Training","Test_inter","Test_extra"))

p1<-ggplot(plot_data, aes(x = x1, y = x2)) +
  geom_point(aes(color = Set), size = 0.4)  + 
  labs(x = "x1", y = "x2", color = "Set") + 
  theme(plot.title = element_text(size = 22, hjust = 0.5,face="bold"),
        axis.title=element_text(size=17),
        axis.text=element_text(size=14),
        legend.text = element_text(size = 18),  # Increase the size of legend text
        legend.title = element_text(size = 20, face = "bold")) + 
  scale_colour_manual(values = c( "#F8766D", "#00BA38","#619CFF"))+
  guides(colour = guide_legend(override.aes = list(size=2)))+xlab(bquote(X[1]))+
  ylab(bquote(X[2]))

ggsave("locations_example.png",p1, width = 9, height = 7,dpi=dpi)
