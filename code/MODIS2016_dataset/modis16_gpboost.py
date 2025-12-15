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

#load data
train_data = pd.read_csv("data/HEATON_train.csv")
test_data = pd.read_csv("data/HEATON_test.csv")

y_train = train_data['TrueTemp']
coords_train = train_data[['Lat', 'Lon']]

y_test = test_data['TrueTemp']
coords_test = test_data[['Lat', 'Lon']]

#covariance parameters estimated by a Vecchia approximation with 310 neighbours
truth = np.array([2.756430177414865382e-06,6.096940872884346163e+00,1.137103704281087202e-01])
#coefficients estimated by a Vecchia approximation with 310 neighbours  
coefs = np.array([-2.437012419872512794e+02,1.773139234046750001e+00,-2.405353477654387007e+00])

#construct data matrices needed 
X_train = np.column_stack((np.ones(len(coords_train)), coords_train))
X_test = np.column_stack((np.ones(len(coords_test)), coords_test))

fixed_effects = X_train.dot(coefs)

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
        cov_function="exponential",
        likelihood="gaussian",
        gp_approx=gp_approx,
    )
    model_args.update(kwargs)  # add specific args
    gp_model = gpb.GPModel(**model_args)

    #fitting
    start_time = time.time()
    gp_model.fit(y=y_train,X=X_train)
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
    pred_train = gp_model.predict(gp_coords_pred= coords_train,X_pred=X_train,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_train = end_time - start_time
    rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
    pred_mean_train= pred_train['mu']
    pred_var_train = pred_train['var']
    score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
    crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

    #test
    start_time= time.time()
    pred_test = gp_model.predict(gp_coords_pred= coords_test,X_pred=X_test,predict_response=True,predict_var=True)
    end_time = time.time()
    pred_time_test = end_time - start_time
    rmse_test = math.sqrt(np.mean((pred_test['mu'] - y_test)**2))
    pred_mean_test = pred_test['mu']
    pred_var_test = pred_test['var']
    score_test = np.mean((0.5*(pred_mean_test - y_test)**2)/pred_var_test + 0.5*np.log(2*np.pi*pred_var_test))
    crps_test = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_test, np.sqrt(pred_var_test), y_test)])

    #saving results
    param_str = "_".join(f"{k}{v}" for k, v in kwargs.items())
    base_name = gp_approx.replace("_", "")
    filename = f"results/modis16/{base_name}_modis16_{param_str}".replace("__", "_")
    with open(filename + '.txt', 'w') as file:
        file.write(f"{gp_approx.capitalize()} run with {param_str}\n")

        file.write('time for fitting: ' + str(fitting_time) + '\n')
        file.write('univariate score train: ' + str(score_train) + '\n')
        file.write('univariate score test: ' + str(score_test) + '\n')
        file.write('time for train univariate prediction: ' + str(pred_time_train) + '\n')
        file.write('time for test univariate prediction: ' + str(pred_time_test) + '\n')
        file.write('rmse train: ' + str(rmse_train) + '\n')
        file.write('rmse test: ' + str(rmse_test) + '\n')
        file.write('crps train: ' + str(crps_train) + '\n')
        file.write('crps test: ' + str(crps_test) + '\n')
        file.write('true negloglik: ' + str(true_negloglik) + '\n')
        file.write('wrong negloglik: ' + str(wrong_negloglik) + '\n')
        file.write('time for true negloglik evaluation: ' + str(true_loglik_eval_time) + '\n')
        file.write('time for wrong negloglik evaluation: ' + str(wrong_loglik_eval_time) + '\n')
    
    del gp_model, pred_train, pred_test, pred_mean_train, pred_var_train, pred_mean_test, pred_var_test, true_negloglik, wrong_negloglik

#example usage
gpboost_run("vecchia", num_neighbors=5)