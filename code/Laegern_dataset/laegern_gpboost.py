import pandas as pd
import gpboost as gpb
import numpy as np
import math
from scipy.stats import norm
import time
import torch

def crps_gaussian(mu, sigma, x):
    """Compute the CRPS for a Gaussian predictive distribution."""
    standardized = (x - mu) / sigma
    return sigma * (standardized * (2 * norm.cdf(standardized) - 1) + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi))

def compute_kl(var1,var2,mean1,mean2):
    kl = torch.log(torch.sqrt(var2)/torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    return kl.sum()

#load datasets
train_set = pd.read_csv("data/laegern_train.csv")
interpolation_set = pd.read_csv("data/laegern_interpolation.csv")
extrapolation_set = pd.read_csv("data/laegern_extrapolation.csv")

#we center the data to match the 0 mean assumption
coords_train = train_set[['x_coord', 'y_coord']].values 
y_train = train_set['CanopyHeight'].values - train_set['CanopyHeight'].mean()
coords_interpolation = interpolation_set[['x_coord', 'y_coord']].values
y_interpolation = interpolation_set['CanopyHeight'].values -train_set['CanopyHeight'].mean()
coords_extrapolation = extrapolation_set[['x_coord', 'y_coord']].values 
y_extrapolation = extrapolation_set['CanopyHeight'].values - train_set['CanopyHeight'].mean()


#covariance parameters estimated by a Vecchia approximation with 240 neighbors
truth = np.array([8.146126889464754324e-07,4.554596583854387398e-02,1.596316574746951744e+01])

#define a function that can be called for each approximation
def gpboost_run(gp_approx, **kwargs):
    """
    Generic GPBoost run function that works for all approximations.
        gp_approx: str
            Type of approximation: "vecchia", "tapering", "fitc", or "full_scale_tapering"
        **kwargs:
            Additional approximation-specific parameters:
                - num_neighbors (for vecchia)
                - cov_fct_taper_range (for tapering/full_scale_tapering)
                - num_ind_points (for fitc/full_scale_tapering)
    """
    # Build model (different kwargs depending on approximation)
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
    gp_model.fit(y=y_train)
    end_time = time.time()
    fitting_time = end_time - start_time

    #likelihood evaluation
    start_time= time.time()
    true_negloglik = gp_model.neg_log_likelihood(cov_pars=truth,y=y_train)
    end_time = time.time()
    true_loglik_eval_time = end_time - start_time

    start_time= time.time()
    wrong_negloglik = gp_model.neg_log_likelihood(cov_pars=truth*2,y=y_train)
    end_time = time.time()
    wrong_loglik_eval_time = end_time - start_time
    
    #train
    start_time= time.time() 
    pred_train = gp_model.predict(gp_coords_pred=coords_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #interpolation
    start_time= time.time()
    pred_inter = gp_model.predict(gp_coords_pred= coords_interpolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_inter = end_time - start_time
    rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
    pred_mean_inter = pred_inter['mu']
    pred_var_inter = pred_inter['var']
    score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
    crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

    #extrapolation
    start_time= time.time()
    pred_extra = gp_model.predict(gp_coords_pred= coords_extrapolation,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_extra = end_time - start_time
    rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
    pred_mean_extra = pred_extra['mu']
    pred_var_extra = pred_extra['var']
    score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
    crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

    #saving results
    param_str = "_".join(f"{k}{v}" for k, v in kwargs.items())
    base_name = gp_approx.replace("_", "")
    filename = f"results/laegern/{base_name}_laegern_{param_str}".replace("__", "_")
    with open(filename + '.txt', 'w') as file:
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
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps interpolation: ' + str(crps_inter) + '\n')
        file.write('crps extrapolation: ' + str(crps_extra) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('wrong negloglik: ' + str(wrong_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for wrong negloglik evaluation: ' + str(wrong_loglik_eval_time) + '\n')  

    del gp_model, pred_train, pred_inter, pred_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, true_negloglik, wrong_negloglik

#example usage
gpboost_run("vecchia", num_neighbors=5)