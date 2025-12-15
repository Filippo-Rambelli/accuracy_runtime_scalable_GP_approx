import pandas as pd
import gpboost as gpb
import numpy as np
import torch
import os

range_denom = 2.74

#wd = "C:/Users/filor/Desktop/tesi ETH/data"
wd = "/cluster/scratch/fabiopc"
file_map = {
    0.5: "data/combined_data_r05.csv",
    0.2: "data/combined_data_r02.csv",
    0.05: "data/combined_data_r005.csv"
}

def exact_run(range_par):
    filename = file_map.get(range_par)
    if filename is None:
        print("wrong range given")
        return
    
    df = pd.read_csv(os.path.join(wd, filename))

    df['rep'] = df['rep'].astype(int)
    nrep = max(df['rep'])

    #global parameters
    true_range = range_par / range_denom
    true_gp_var = 1
    true_error_term = 0.5
    truth = np.array([[true_error_term], [true_gp_var], [true_range]]).flatten()

    #log-likelihood storage
    true_negloglik_values = []
    wrong_negloglik_values = []

    #prediction storage
    pred_mean_train_all, pred_var_train_all = [], []
    pred_mean_inter_all, pred_var_inter_all = [], []
    pred_mean_extra_all, pred_var_extra_all = [], []

    #results storage
    gp_range_hat= list(); gp_var_hat = list(); error_term_hat = list()
    scores_train= list(); scores_inter = list(); scores_extra = list()
    rmse_train_list = list(); rmse_inter_list = list(); rmse_extra_list = list() 

    for i in range(1, nrep + 1):
        df._clear_item_cache()
        data_rep = df.loc[df['rep'].values == i].copy(deep=True)
        data_rep.reset_index(drop=True, inplace=True)

        train_df = data_rep[data_rep['which'] == 'train']
        coords_train = train_df[['x1', 'x2']].values 
        y_train = train_df['y'].values
        f_train = train_df['f'].values

        ##Exact GPBoost model
        gp_model_exact = gpb.GPModel(
            gp_coords=coords_train, cov_function="matern", cov_fct_shape=1.5,
            likelihood="gaussian"
        )

        #Parameter estimation
        gp_model_exact.fit(y=y_train)

        gp_range_value = gp_model_exact.get_cov_pars().loc['Param.', 'GP_range']
        gp_var_value = gp_model_exact.get_cov_pars().loc['Param.', 'GP_var']
        error_term_value = gp_model_exact.get_cov_pars().loc['Param.', 'Error_term']

        gp_range_hat.append(gp_range_value)
        gp_var_hat.append(gp_var_value)
        error_term_hat.append(error_term_value)

        #Exact likelihood evaluation
        exact_true_negloglik = gp_model_exact.neg_log_likelihood(cov_pars=truth, y=y_train)
        exact_wrong_negloglik = gp_model_exact.neg_log_likelihood(cov_pars=truth * 2, y=y_train)

        true_negloglik_values.append(exact_true_negloglik)
        wrong_negloglik_values.append(exact_wrong_negloglik)

        #Prediction accuracy (only first half of reps)
        if i <= nrep / 2:
            #TRAIN predictions
            pred_resp_train = gp_model_exact.predict(
                gp_coords_pred=coords_train, cov_pars=truth,
                predict_var=True, predict_response=False
            )
            pred_mean_train = pred_resp_train['mu']
            pred_var_train = pred_resp_train['var']

            score_train = np.mean(((pred_mean_train - f_train)**2)/(2*pred_var_train) + 0.5*np.log(2*np.pi*pred_var_train))
            scores_train.append(score_train)

            rmse_train_list.append(np.sqrt(np.mean((f_train - pred_mean_train) ** 2)).item())

            #store with iteration label
            for val in pred_mean_train:
                pred_mean_train_all.append((i, val))
            for val in pred_var_train:
                pred_var_train_all.append((i, val))

            #INTERPOLATION predictions
            inter_df = data_rep[data_rep['which'] == 'interpolation']
            coords_inter = inter_df[['x1', 'x2']].values
            f_inter = inter_df['f'].values

            pred_resp_inter = gp_model_exact.predict(
                gp_coords_pred=coords_inter, cov_pars=truth,
                predict_var=True, predict_response=False
            )
            pred_mean_inter = pred_resp_inter['mu']
            pred_var_inter = pred_resp_inter['var']

            score_inter = np.mean((0.5*(pred_mean_inter - f_inter)**2)/pred_var_inter + 0.5*np.log(2*np.pi*pred_var_inter))
            scores_inter.append(score_inter)

            rmse_inter_list.append(np.sqrt(np.mean((f_inter - pred_mean_inter) ** 2)).item())

            #store with iteration label
            for val in pred_mean_inter:
                pred_mean_inter_all.append((i, val))
            for val in pred_var_inter:
                pred_var_inter_all.append((i, val))

            #EXTRAPOLATION predictions
            extra_df = data_rep[data_rep['which'] == 'extrapolation']
            coords_extra = extra_df[['x1', 'x2']].values
            f_extra = extra_df['f'].values

            pred_resp_extra = gp_model_exact.predict(
                gp_coords_pred=coords_extra, cov_pars=truth,
                predict_var=True, predict_response=False
            )
            pred_mean_extra = pred_resp_extra['mu']
            pred_var_extra = pred_resp_extra['var']

            score_extra = np.mean((0.5*(pred_mean_extra - f_extra)**2)/pred_var_extra + 0.5*np.log(2*np.pi*pred_var_extra))
            scores_extra.append(score_extra)

            rmse_extra_list.append(np.sqrt(np.mean((f_extra - pred_mean_extra) ** 2)).item())

            #store with iteration label
            for val in pred_mean_extra:
                pred_mean_extra_all.append((i, val))
            for val in pred_var_extra:
                pred_var_extra_all.append((i, val))

            del inter_df, extra_df, coords_inter, coords_extra, f_inter, f_extra, pred_resp_train, pred_resp_inter, pred_resp_extra, 
            pred_mean_train, pred_var_train, pred_mean_inter, pred_var_inter, pred_mean_extra, pred_var_extra

        del train_df, data_rep, coords_train, f_train, y_train, gp_model_exact
        torch.cuda.empty_cache()
        print(f"rep {i} done")

    # === SAVE RESULTS ===
    #Likelihoods
    filename = f"exact_results/true_exact_negloglik_values_{range_par}.txt"
    with open(filename, 'w') as file:
        for value in true_negloglik_values:
            file.write(f"{value:.5f}\n")

    filename = f"exact_results/wrong_exact_negloglik_values_{range_par}.txt"
    with open(filename, 'w') as file:
        for value in wrong_negloglik_values:
            file.write(f"{value:.5f}\n")

    #Predictions
    pred_files = {
        f"exact_results/exact_pred_mean_train_{range_par}.txt": pred_mean_train_all,
        f"exact_results/exact_pred_var_train_{range_par}.txt": pred_var_train_all,
        f"exact_results/exact_pred_mean_inter_{range_par}.txt": pred_mean_inter_all,
        f"exact_results/exact_pred_var_inter_{range_par}.txt": pred_var_inter_all,
        f"exact_results/exact_pred_mean_extra_{range_par}.txt": pred_mean_extra_all,
        f"exact_results/exact_pred_var_extra_{range_par}.txt": pred_var_extra_all
    }

    for fname, data in pred_files.items():
        with open(fname, 'w') as file:
            file.write("iteration,value\n")
            for (it, val) in data:
                file.write(f"{it},{val}\n")

    print(f"Saved all likelihoods and predictions for range {range_par}")

    #Metrics
    mse_gp_range = np.mean((np.array(gp_range_hat) - true_range) ** 2)
    bias_gp_range = np.mean(np.array(gp_range_hat) - true_range)
    mse_gp_var = np.mean((np.array(gp_var_hat) - true_gp_var) ** 2)
    bias_gp_var = np.mean(np.array(gp_var_hat) - true_gp_var)
    mse_error_term = np.mean((np.array(error_term_hat) - true_error_term) ** 2)
    bias_error_term = np.mean(np.array(error_term_hat) - true_error_term)

    mean_univ_score_train = np.mean(scores_train)
    mean_univ_score_inter = np.mean(scores_inter)
    mean_univ_score_extra = np.mean(scores_extra)

    mean_rmse_train = np.mean(rmse_train_list)
    mean_rmse_inter = np.mean(rmse_inter_list)
    mean_rmse_extra = np.mean(rmse_extra_list)

    filename = "exact_calculations_" +str(range_par)
    with open(filename + '.txt', 'w') as file:

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

        file.write('mean univariate score train: ' + str(mean_univ_score_train) + '\n')
        file.write('mean univariate score interpolation: ' + str(mean_univ_score_inter) + '\n')
        file.write('mean univariate score extrapolation: ' + str(mean_univ_score_extra) + '\n')
        file.write('variance univariate score train: ' + str(np.var(scores_train)/len(scores_train)) + '\n')
        file.write('variance univariate score interpolation: ' + str(np.var(scores_inter)/len(scores_inter)) + '\n')
        file.write('variance univariate score extrapolation: ' + str(np.var(scores_extra)/len(scores_extra)) + '\n')

        file.write('RMSE train: ' + str(mean_rmse_train) + '\n')
        file.write('RMSE inter: ' + str(mean_rmse_inter) + '\n')
        file.write('RMSE extra: ' + str(mean_rmse_extra) + '\n')
        file.write('variance for RMSE train: ' + str(np.var(rmse_train_list)/len(rmse_train_list)) + '\n')
        file.write('variance for RMSE inter: ' + str(np.var(rmse_inter_list)/len(rmse_inter_list)) + '\n')
        file.write('variance for RMSE extra: ' + str(np.var(rmse_extra_list)/len(rmse_extra_list)) + '\n')

#Run example
exact_run(0.5)