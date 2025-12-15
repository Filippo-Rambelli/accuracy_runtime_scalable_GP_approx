import os
import pandas as pd
import gpboost as gpb
import numpy as np
import math
from scipy.stats import norm
import time
import torch

wd = "/cluster/scratch/fabiopc"
def crps_gaussian(mu, sigma, x):
    """Compute the CRPS for a Gaussian predictive distribution."""
    standardized = (x - mu) / sigma
    return sigma * (standardized * (2 * norm.cdf(standardized) - 1) + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi))

def compute_kl(var1,var2,mean1,mean2):
    kl = torch.log(torch.sqrt(var2)/torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    return kl.sum()

#load datasets
train_set = pd.read_csv(os.path.join(wd,"data/house_train_with_covs.csv"))
interpolation_set = pd.read_csv(os.path.join(wd,"data/house_interpolation_with_covs.csv"))
extrapolation_set = pd.read_csv(os.path.join(wd,"data/house_extrapolation_with_covs.csv"))

#load results from exact calculations to perform a comparison
exact_pred_mean_train= np.loadtxt("exact_results/exact_pred_mean_train_house_with_covs.txt")
exact_pred_var_train = np.loadtxt("exact_results/exact_pred_var_train_house_with_covs.txt")
exact_pred_mean_inter = np.loadtxt("exact_results/exact_pred_mean_inter_house_with_covs.txt")
exact_pred_var_inter = np.loadtxt("exact_results/exact_pred_var_inter_house_with_covs.txt")
exact_pred_mean_extra = np.loadtxt("exact_results/exact_pred_mean_extra_house_with_covs.txt")
exact_pred_var_extra = np.loadtxt("exact_results/exact_pred_var_extra_house_with_covs.txt")

#covariance parameters estimated via exact calculations 
truth = np.array([0.05513030716341601, 0.20188955237350595 ,0.0603076839309954])
#regression coefficients estimated via exact calculations
coefs = np.array([12.614722511788656,-0.12436141170281728,-0.023595011562458013,-0.11568609738546891,0.5469768881001963,0.016115405756767467,0.1310218698487707,
                  0.03313185031725592,0.07571105963500542,0.10169076871530056,0.1395963246582103,0.18763742854493398,-0.36949136575186714])

#we center the data to match the 0 mean assumption
coords_train = train_set[['long', 'lat']].values 
y_train = train_set['log_price'].values
coords_interpolation = interpolation_set[['long', 'lat']].values
y_interpolation = interpolation_set['log_price'].values 
coords_extrapolation = extrapolation_set[['long', 'lat']].values 
y_extrapolation = extrapolation_set['log_price'].values 

#construct data matrices needed 
covariate_cols = [col for col in train_set.columns if col not in ['log_price']]
X_train = train_set[covariate_cols].values #already includes intercept
X_interpolation = interpolation_set[covariate_cols].values
X_extrapolation = extrapolation_set[covariate_cols].values

fixed_effects = X_train.dot(coefs)

#define a function for the Vecchia approximation that can be called for each tuning parameter
def gpboost_run_with_covs(gp_approx, **kwargs):
    """
    Generic GPBoost run function that works for all approximations.
    Parameters:
        gp_approx: str
            Type of approximation: "vecchia", "tapering", "fitc", or "full_scale_tapering"
        **kwargs:
            Additional approximation-specific parameters:
                - num_neighbors (for vecchia)
                - cov_fct_taper_range (for tapering/full_scale_tapering)
                - num_ind_points (for fitc/full_scale_tapering)
    """
    #define model
    model_args = dict(
        gp_coords=coords_train,
        cov_function="matern",
        cov_fct_shape=1.5,
        likelihood="gaussian",
        gp_approx=gp_approx,
    )
    model_args.update(kwargs)  # add specific args
    gp_model = gpb.GPModel(**model_args)
    
    #fitting
    start_time = time.time()
    gp_model.fit(y=y_train, X=X_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train,fixed_effects=fixed_effects)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    wrong_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train,fixed_effects=fixed_effects)
    end_time = time.time()
    wrong_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train, X_pred=X_train, predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    rmse_mean_train_comparison = math.sqrt(np.mean((pred_mean_train - exact_pred_mean_train)**2))
    rmse_var_train_comparison = math.sqrt(np.mean((pred_var_train - exact_pred_var_train)**2))
    kl_train = compute_kl(torch.from_numpy(exact_pred_var_train),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train),torch.from_numpy(pred_mean_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation, X_pred=X_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    rmse_mean_inter_comparison = math.sqrt(np.mean((pred_mean_inter - exact_pred_mean_inter)**2))
    rmse_var_inter_comparison = math.sqrt(np.mean((pred_var_inter - exact_pred_var_inter)**2))
    kl_inter = compute_kl(torch.from_numpy(exact_pred_var_inter),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter),torch.from_numpy(pred_mean_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation, X_pred=X_extrapolation ,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    rmse_mean_extra_comparison = math.sqrt(np.mean((pred_mean_extra - exact_pred_mean_extra)**2))
    rmse_var_extra_comparison = math.sqrt(np.mean((pred_var_extra - exact_pred_var_extra)**2))
    kl_extra = compute_kl(torch.from_numpy(exact_pred_var_extra),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra),torch.from_numpy(pred_mean_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    param_str = "_".join(f"{k}{v}" for k, v in kwargs.items())
    base_name = gp_approx.replace("_", "")
    filename_save = f"results/house_with_covs/{base_name}_house_with_covs_{param_str}".replace("__", "_")
    with open(filename_save + '.txt', 'w') as file:
        file.write(f"{gp_approx.capitalize()} run with {param_str}\n")

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score interpolation: ' + str(score_inter) + '\n')
        file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for interpolation univariate prediction: ' + str(pred_time_inter) + '\n')
        file.write('time for extrapolation univariate prediction: ' + str(pred_time_extra) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
        file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
        file.write('rmse mean train: ' + str(rmse_mean_train_comparison) + '\n')
        file.write('rmse mean interpolation: ' + str(rmse_mean_inter_comparison) + '\n')
        file.write('rmse mean extrapolation: ' + str(rmse_mean_extra_comparison) + '\n')
        file.write('rmse var train: ' + str(rmse_var_train_comparison) + '\n')
        file.write('rmse var interpolation: ' + str(rmse_var_inter_comparison) + '\n')
        file.write('rmse var extrapolation: ' + str(rmse_var_extra_comparison) + '\n')
        file.write('kl train: ' + str(kl_train.item()) + '\n')
        file.write('kl interpolation: ' + str(kl_inter.item()) + '\n')
        file.write('kl extrapolation: ' + str(kl_extra.item()) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('wrong negloglik: ' + str(wrong_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for wrong negloglik evaluation: ' + str(wrong_loglik_eval_time) + '\n')

    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, wrong_negloglik

#example usage
gpboost_run_with_covs(gp_approx="vecchia",num_neighbors=5)