import os
import pandas as pd
import gpboost as gpb
import numpy as np
from scipy.stats import multivariate_normal
import random
from scipy.spatial import distance
import time
import torch
from sklearn.gaussian_process.kernels import Matern
from pathlib import Path

range_denom = 2.74

wd = "/cluster/scratch/fabiopc"

file_map = {
    0.5: "data/combined_data_r05.csv",
    0.2: "data/combined_data_r02.csv",
    0.05: "data/combined_data_r005.csv"
}

def load_pred(kind, stage,range_par):
    fname = f"exact_results/exact_pred_{kind}_{stage}_{range_par}.txt"
    return pd.read_csv(
        os.path.join(wd, fname),
        dtype={"iteration": int, "value": float}
)

def load_lik(kind,range_par):
    fname = f"exact_results/{kind}_exact_negloglik_values_{range_par}.txt"
    return pd.read_csv(
        os.path.join(wd, fname),
        header=None,      
        names=["value"],     
        dtype=float,       
    )["value"].to_numpy()   

def compute_kl(var1,var2,mean1,mean2):
    kl = torch.log(torch.sqrt(var2)/torch.sqrt(var1)) + (var1 + (mean1 - mean2)**2)/(2*var2) - 0.5
    return kl.sum()

#define a function that can be called for each approximation
def gpboost_run(range_par, gp_approx, **kwargs):
    """
    Generic GPBoost run function that works for all approximations.
    Parameters:
        range_par: float
            The practical range parameter (e.g. 0.5, 0.2, 0.05)
        gp_approx: str
            Type of approximation: "vecchia", "tapering", "fitc", or "full_scale_tapering"
        **kwargs:
            Additional approximation-specific parameters:
                - num_neighbors (for vecchia)
                - cov_fct_taper_range (for tapering/full_scale_tapering)
                - num_ind_points (for fitc/full_scale_tapering)
    """
    filename = file_map.get(range_par)
    if filename is None:
        print("wrong range given")
        return
    
    df = pd.read_csv(os.path.join(wd, filename))

    df['rep'] = df['rep'].astype(int)
    nrep = max(df['rep'])

    #load exact prediction and log-likelihood files dynamically
    exact_pred_mean_train_whole = load_pred("mean", "train", range_par)
    exact_pred_var_train_whole  = load_pred("var",  "train", range_par)

    exact_pred_mean_inter_whole = load_pred("mean", "inter", range_par)
    exact_pred_var_inter_whole  = load_pred("var",  "inter", range_par)

    exact_pred_mean_extra_whole = load_pred("mean", "extra", range_par)
    exact_pred_var_extra_whole  = load_pred("var",  "extra", range_par)

    true_exact_negloglik_values = load_lik("true", range_par)
    wrong_exact_negloglik_values = load_lik("wrong", range_par)

    #global parameters
    true_range = range_par/range_denom
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #Parameter estimation 
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list(); param_estimation_time = list()

    #Likelihood evaluation & comparison
    true_negloglik_eval_time= list(); wrong_negloglik_eval_time= list()
    true_estimated_negloglik_values = list(); wrong_estimated_negloglik_values = list()

    #Prediction accuracy 
    scores_train= list(); scores_inter = list(); scores_extra = list()
    rmse_train_list = list()
    rmse_inter_list = list()
    rmse_extra_list = list() 
    train_pred_accuracy_time = list(); inter_pred_accuracy_time = list(); extra_pred_accuracy_time = list()

    #comparison to exact calculations
    rmse_mean_train = list() ; rmse_var_train = list(); kl_train = list()
    rmse_mean_inter = list() ; rmse_var_inter = list(); kl_inter = list()
    rmse_mean_extra = list() ; rmse_var_extra = list(); kl_extra = list()

    for i in range(1, nrep + 1):
        df._clear_item_cache()
        data_rep = df.loc[df['rep'].values == i].copy(deep=True)
        data_rep.reset_index(drop=True, inplace=True)

        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ####GPBOOST  
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
        
        #Parameter estimation
        start_time = time.time()
        gp_model.fit(y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        param_estimation_time.append(execution_time)

        gp_range_value = gp_model.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Likelihood evaluation

        #true
        start_time = time.time()
        true_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        true_negloglik_eval_time.append(execution_time)
        true_estimated_negloglik_values.append(true_negloglik_eval)
        
        #wrong
        start_time = time.time()
        wrong_negloglik_eval=gp_model.neg_log_likelihood(cov_pars=truth*2, y=y_train)
        end_time = time.time()
        execution_time = end_time - start_time
        wrong_negloglik_eval_time.append(execution_time)
        wrong_estimated_negloglik_values.append(wrong_negloglik_eval)

        #Prediction accuracy
        if i<=nrep/2:
            ####TRAIN
            #univariate gpboost
            start_time = time.time()
            pred_resp_train = gp_model.predict(gp_coords_pred=coords_train, cov_pars=truth,
                            predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time = end_time - start_time
            train_pred_accuracy_time.append(elapsed_time)

            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']
            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())

            #exact univariate gpboost (loaded)
            exact_pred_mean_train = exact_pred_mean_train_whole[exact_pred_mean_train_whole['iteration'] == i]['value']
            exact_pred_var_train = exact_pred_var_train_whole[exact_pred_var_train_whole['iteration'] == i]['value']
            
            rmse_mean_train.append(np.sqrt(np.mean((exact_pred_mean_train-pred_mean_train) ** 2)).item())
            rmse_var_train.append(np.sqrt(np.mean((exact_pred_var_train-pred_var_train) ** 2)).item())

            kl_train.append(compute_kl(torch.from_numpy(exact_pred_var_train.to_numpy()),torch.from_numpy(pred_var_train),torch.from_numpy(exact_pred_mean_train.to_numpy()),torch.from_numpy(pred_mean_train)))  
            
            ####INTERPOLATION 
            #univariate gpboost
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            start_time = time.time()
            pred_resp_inter= gp_model.predict(gp_coords_pred=coords_inter, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time= end_time - start_time
            inter_pred_accuracy_time.append(elapsed_time)

            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']
            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #exact univariate gpboost (loaded)
            exact_pred_mean_inter = exact_pred_mean_inter_whole[exact_pred_mean_inter_whole['iteration'] == i]['value']
            exact_pred_var_inter = exact_pred_var_inter_whole[exact_pred_var_inter_whole['iteration'] == i]['value']
            
            rmse_mean_inter.append(np.sqrt(np.mean((exact_pred_mean_inter-pred_mean_inter) ** 2)).item())
            rmse_var_inter.append(np.sqrt(np.mean((exact_pred_var_inter-pred_var_inter) ** 2)).item())

            kl_inter.append(compute_kl(torch.from_numpy(exact_pred_var_inter.to_numpy()),torch.from_numpy(pred_var_inter),torch.from_numpy(exact_pred_mean_inter.to_numpy()),torch.from_numpy(pred_mean_inter)))

            #####EXTRAPOLATION 
            #univariate gpboost 
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            gp_model.set_prediction_data(vecchia_pred_type="order_obs_first_cond_obs_only")
            start_time = time.time()
            pred_resp_extra= gp_model.predict(gp_coords_pred=coords_extra, cov_pars=truth,
                                    predict_var=True, predict_response=False)
            end_time = time.time()
            elapsed_time += end_time - start_time
            extra_pred_accuracy_time.append(elapsed_time)

            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']
            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)

            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            #exact univariate gpboost (loaded)
            exact_pred_mean_extra = exact_pred_mean_extra_whole[exact_pred_mean_extra_whole['iteration'] == i]['value']
            exact_pred_var_extra = exact_pred_var_extra_whole[exact_pred_var_extra_whole['iteration'] == i]['value']
            
            rmse_mean_extra.append(np.sqrt(np.mean((exact_pred_mean_extra-pred_mean_extra) ** 2)).item())
            rmse_var_extra.append(np.sqrt(np.mean((exact_pred_var_extra-pred_var_extra) ** 2)).item())

            kl_extra.append(compute_kl(torch.from_numpy(exact_pred_var_extra.to_numpy()),torch.from_numpy(pred_var_extra),torch.from_numpy(exact_pred_mean_extra.to_numpy()),torch.from_numpy(pred_mean_extra)))
            
            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra, exact_pred_mean_train, 
            exact_pred_var_train, exact_pred_mean_inter, exact_pred_var_inter, exact_pred_mean_extra, exact_pred_var_extra

        del train_df, data_rep, coords_train, f_train,  y_train, gp_model 
        torch.cuda.empty_cache()
        print(f"rep {i} done")
   
    #computing results
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)

    true_negloglik_diff_values = [
    est - exact for est, exact in zip(true_estimated_negloglik_values, true_exact_negloglik_values)
    ]
    wrong_negloglik_diff_values = [
        est - exact for est, exact in zip(wrong_estimated_negloglik_values, wrong_exact_negloglik_values)
    ]

    mean_time_param_estimation = np.mean(param_estimation_time)
    mean_estimated_negloglik_true_pars = np.mean(true_estimated_negloglik_values)
    mean_exact_negloglik_true_pars = np.mean(true_exact_negloglik_values)
    mean_diff_negloglik_true_pars = np.mean(true_negloglik_diff_values)
    mean_diff_negloglik_wrong_pars = np.mean(wrong_negloglik_diff_values)
    mean_time_eval_negloglik_true_pars = np.mean(true_negloglik_eval_time)
    mean_time_eval_negloglik_wrong_pars = np.mean(wrong_negloglik_eval_time)
    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_time_univ_pred_train = np.mean(train_pred_accuracy_time)
    mean_time_univ_pred_inter = np.mean(inter_pred_accuracy_time)
    mean_time_univ_pred_extra = np.mean(extra_pred_accuracy_time)

    mean_rmse_mean_train = np.mean(rmse_mean_train)
    mean_rmse_mean_inter = np.mean(rmse_mean_inter)
    mean_rmse_mean_extra = np.mean(rmse_mean_extra)
    mean_rmse_var_train = np.mean(rmse_var_train)
    mean_rmse_var_inter = np.mean(rmse_var_inter)
    mean_rmse_var_extra = np.mean(rmse_var_extra)
    mean_kl_train = np.mean(kl_train)
    mean_kl_inter = np.mean(kl_inter)
    mean_kl_extra = np.mean(kl_extra)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    #saving results
    param_str = "_".join(f"{k}{v}" for k, v in kwargs.items())
    base_name = gp_approx.replace("_", "")
    filename_save = f"results/{range_par}/{base_name}_{range_par}_{param_str}".replace("__", "_")
    with open(filename_save + '.txt', 'w') as file:

        file.write(f"{gp_approx.capitalize()} run with {param_str}\n")
        file.write('True range: ' + str(true_range) + '\n')

        file.write('bias for GP range: ' + str(bias_gp_range) + '\n')
        file.write('MSE for GP range: ' + str(mse_gp_range) + '\n')
        file.write('bias for GP variance: ' + str(bias_gp_var) + '\n')
        file.write('MSE for GP variance: ' + str(mse_gp_var) + '\n')
        file.write('bias for error term variance: ' + str(bias_error_term) + '\n')
        file.write('MSE for error term variance: ' + str(mse_error_term) + '\n')
        file.write('variance for bias of GP range: ' + str(np.var(gp_range_hat)/len(gp_range_hat)) + '\n')
        file.write('variance for bias GP of variance: ' + str(np.var(gp_var_hat)/len(gp_var_hat)) + '\n')
        file.write('variance for bias error of term variance: ' + str(np.var(error_term_hat)/len(error_term_hat)) + '\n')
        file.write('variance for MSE GP range: ' + str(np.var((np.array(gp_range_hat)-true_range)**2)/len(gp_range_hat)) + '\n')
        file.write('variance for MSE GP variance: ' + str(np.var((np.array(gp_var_hat)-true_gp_var)**2)/len(gp_var_hat)) + '\n')
        file.write('variance for MSE error term variance: ' + str(np.var((np.array(error_term_hat)-true_error_term)**2)/len(error_term_hat)) + '\n')

        file.write('mean time for parameter estimation: ' + str(mean_time_param_estimation) + '\n')
        file.write('mean estimated negloglik true pars: '  + str(mean_estimated_negloglik_true_pars) + '\n')
        file.write('mean exact negloglik true pars: ' + str(mean_exact_negloglik_true_pars) + '\n')
        file.write('true pars, mean diff negloglik: ' + str(mean_diff_negloglik_true_pars) + '\n')
        file.write('wrong pars, mean diff negloglik: ' + str(mean_diff_negloglik_wrong_pars) + '\n')
        file.write('mean time for true loglik evaluation: ' + str(mean_time_eval_negloglik_true_pars) + '\n')
        file.write('mean time for wrong loglik evaluation: ' + str(mean_time_eval_negloglik_wrong_pars) + '\n')
        file.write('variance for negloglik true pars: ' + str(np.var(true_estimated_negloglik_values)/len(true_estimated_negloglik_values)) + '\n')
        file.write('variance for negloglik wrong pars: ' + str(np.var(wrong_estimated_negloglik_values)/len(wrong_estimated_negloglik_values)) + '\n')

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)/len(scores_train)) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)/len(scores_inter)) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)/len(scores_extra)) + '\n')
        file.write('mean time for train univariate prediction: ' + str(mean_time_univ_pred_train) + '\n')
        file.write('mean time for interpolation univariate prediction: ' + str(mean_time_univ_pred_inter) + '\n')
        file.write('mean time for extrapolation univariate prediction: ' + str(mean_time_univ_pred_extra) + '\n')

        file.write('mean rmse mean train: ' + str(mean_rmse_mean_train) + '\n')
        file.write('mean rmse mean interpolation: ' + str(mean_rmse_mean_inter) + '\n')
        file.write('mean rmse mean extrapolation: ' + str(mean_rmse_mean_extra) + '\n')
        file.write('variance rmse mean train: ' + str(np.var(rmse_mean_train)/len(rmse_mean_train)) + '\n')
        file.write('variance rmse mean interpolation: ' + str(np.var(rmse_mean_inter)/len(rmse_mean_inter)) + '\n')
        file.write('variance rmse mean extrapolation: ' + str(np.var(rmse_mean_extra)/len(rmse_mean_extra)) + '\n')
        file.write('mean rmse var train: ' + str(mean_rmse_var_train) + '\n')
        file.write('mean rmse var interpolation: ' + str(mean_rmse_var_inter) + '\n')
        file.write('mean rmse var extrapolation: ' + str(mean_rmse_var_extra) + '\n')
        file.write('variance rmse var train: ' + str(np.var(rmse_var_train)/len(rmse_var_train)) + '\n')
        file.write('variance rmse var interpolation: ' + str(np.var(rmse_var_inter)/len(rmse_var_inter)) + '\n')
        file.write('variance rmse var extrapolation: ' + str(np.var(rmse_var_extra)/len(rmse_var_extra)) + '\n')
        file.write('mean kl train: ' + str(mean_kl_train) + '\n')
        file.write('mean kl interpolation: ' + str(mean_kl_inter) + '\n')
        file.write('mean kl extrapolation: ' + str(mean_kl_extra) + '\n')
        file.write('variance kl train: ' + str(np.var(kl_train)/len(kl_train)) + '\n')
        file.write('variance kl interpolation: ' + str(np.var(kl_inter)/len(kl_inter)) + '\n')
        file.write('variance kl extrapolation: ' + str(np.var(kl_extra)/len(kl_extra)) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')

        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)/len(rmse_train_list)) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)/len(rmse_inter_list)) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)/len(rmse_extra_list)) + '\n')

#example usage
gpboost_run(0.5, "vecchia", num_neighbors=5)