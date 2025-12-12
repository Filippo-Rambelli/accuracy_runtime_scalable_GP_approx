import os
import pandas as pd
import gpboost as gpb
import numpy as np
import math
from scipy.stats import norm

wd = "/cluster/scratch/fabiopc/"
def crps_gaussian(mu, sigma, x):
    """Compute the CRPS for a Gaussian predictive distribution."""
    standardized = (x - mu) / sigma
    return sigma * (standardized * (2 * norm.cdf(standardized) - 1) + 2 * norm.pdf(standardized) - 1 / np.sqrt(np.pi))

#load datasets
train_set = pd.read_csv(os.path.join(wd,"data/house_train_with_covs.csv"))
interpolation_set = pd.read_csv(os.path.join(wd,"data/house_interpolation_with_covs.csv"))
extrapolation_set = pd.read_csv(os.path.join(wd,"data/house_extrapolation_with_covs.csv"))

#we use coovariates -> no need to center data
coords_train = train_set[['long', 'lat']].values 
y_train = train_set['log_price'].values 
coords_interpolation = interpolation_set[['long', 'lat']].values
y_interpolation = interpolation_set['log_price'].values 
coords_extrapolation = extrapolation_set[['long', 'lat']].values 
y_extrapolation = extrapolation_set['log_price'].values 

#construct data matrices needed 
covariate_cols = [col for col in train_set.columns if col not in ['log_price']]
X_train = train_set[covariate_cols].values
X_interpolation = interpolation_set[covariate_cols].values
X_extrapolation = extrapolation_set[covariate_cols].values

#fitting
gp_model = gpb.GPModel(gp_coords=coords_train,  cov_function="matern",cov_fct_shape=1.5,
            likelihood="gaussian", gp_approx="none")
gp_model.fit(y=y_train, X=X_train)

#save exact parameters
gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']
cov_pars = np.array([[error_term_value], [gp_var_value], [gp_range_value]]).flatten()

param_dict = {
    'GP_range': gp_range_value,
    'GP_var': gp_var_value,
    'Error_term': error_term_value
}

with open('exact_results/pars_exact_house_with_covs.txt', 'w') as f:
    for key, value in param_dict.items():
        f.write(f"{key}: {value}\n")

#save exact coefficients
coefs = np.ravel(gp_model.get_coef())
coefs_dict = dict(zip(covariate_cols, coefs))

with open('exact_results/coefs_exact_house_with_covs.txt', 'w') as f:
    for key, value in coefs_dict.items():
        f.write(f"{key}: {value}\n")

#likelihood evaluation
fixed_effects = X_train.dot(coefs)
true_negloglik = gp_model.neg_log_likelihood(cov_pars=cov_pars,y=y_train,fixed_effects=fixed_effects)
wrong_negloglik = gp_model.neg_log_likelihood(cov_pars=cov_pars*2,y=y_train,fixed_effects=fixed_effects)

#train
pred_train = gp_model.predict(gp_coords_pred=coords_train, X_pred=X_train, predict_response=True, predict_var=True)
rmse_train = math.sqrt(np.mean((pred_train['mu'] - y_train)**2))
pred_mean_train= pred_train['mu']
pred_var_train = pred_train['var']
score_train = np.mean((0.5*(pred_mean_train - y_train)**2)/pred_var_train + 0.5*np.log(2*np.pi*pred_var_train))
crps_train = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_train, np.sqrt(pred_var_train), y_train)])

#interpolation
pred_inter = gp_model.predict(gp_coords_pred=coords_interpolation, X_pred=X_interpolation, predict_response=True, predict_var=True)
rmse_inter = math.sqrt(np.mean((pred_inter['mu'] - y_interpolation)**2))
pred_mean_inter = pred_inter['mu']
pred_var_inter = pred_inter['var']
score_inter = np.mean((0.5*(pred_mean_inter - y_interpolation)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
crps_inter = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_inter, np.sqrt(pred_var_inter), y_interpolation)])

#extrapolation
pred_extra = gp_model.predict(gp_coords_pred=coords_extrapolation, X_pred=X_extrapolation , predict_response=True, predict_var=True)
rmse_extra = math.sqrt(np.mean((pred_extra['mu'] - y_extrapolation)**2))
pred_mean_extra = pred_extra['mu']
pred_var_extra = pred_extra['var']
score_extra = np.mean((0.5*(pred_mean_extra - y_extrapolation)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
crps_extra = np.mean([crps_gaussian(mu, sigma, x) for mu, sigma, x in zip(pred_mean_extra, np.sqrt(pred_var_extra), y_extrapolation)])

#Predictions
pred_files = {
    "exact_results/exact_pred_mean_train_house_with_covs.txt": pred_mean_train,
    "exact_results/exact_pred_var_train_house_with_covs.txt": pred_var_train,
    "exact_results/exact_pred_mean_inter_house_with_covs.txt": pred_mean_inter,
    "exact_results/exact_pred_var_inter_house_with_covs.txt": pred_var_inter,
    "exact_results/exact_pred_mean_extra_house_with_covs.txt": pred_mean_extra,
    "exact_results/exact_pred_var_extra_house_with_covs.txt": pred_var_extra
}

# Save each to a separate file, one value per line
for fname, data in pred_files.items():
    with open(fname, 'w') as file:
        for val in data:
            file.write(f"{val}\n")

#Metrics
filename = "exact_results/exact_calculations_house_with_covs"
with open(filename + '.txt', 'w') as file:
    file.write('univariate score train: ' + str(score_train) + '\n')
    file.write('univariate score interpolation: ' + str(score_inter) + '\n')
    file.write('univariate score extrapolation: ' + str(score_extra) + '\n')
    file.write('rmse train: ' + str(rmse_train) + '\n')
    file.write('rmse interpolation: ' + str(rmse_inter) + '\n')
    file.write('rmse extrapolation: ' + str(rmse_extra) + '\n')
    file.write('crps train: ' + str(crps_train) + '\n')
    file.write('crps interpolation: ' + str(crps_inter) + '\n')
    file.write('crps extrapolation: ' + str(crps_extra) + '\n')
    file.write('true negloglik: ' + str(true_negloglik) + '\n')
    file.write('wrong negloglik: ' + str(wrong_negloglik) + '\n')
