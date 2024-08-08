# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:27:07 2023

@author: emma_
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import scanpy as sc
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint as sp_randint, uniform
import tensorflow as tf 
import os

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

print(tf.config.list_physical_devices())

def run_lightgbm_random_search(data, ages, param_dist, n_iter_search, n_splits=5, output_csv='LightGBM_ParaTunes.csv'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    best_score = float('inf')
    best_params = {}
    best_metrics = {}

    param_sampler = ParameterSampler(param_dist, n_iter=n_iter_search, random_state=42)
    for params in param_sampler:
        cv_metrics = {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'mape': []}

        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = ages[train_index], ages[test_index]

            lgbm_model = lgb.LGBMRegressor(**params, random_state=42)
            lgbm_model.fit(X_train, y_train)
            predictions = lgbm_model.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

            cv_metrics['mse'].append(mse)
            cv_metrics['rmse'].append(rmse)
            cv_metrics['mae'].append(mae)
            cv_metrics['r2'].append(r2)
            cv_metrics['mape'].append(mape)

        avg_metrics = {f'avg_{metric}': np.mean(values) for metric, values in cv_metrics.items()}

        if avg_metrics['avg_mse'] < best_score:
            best_score = avg_metrics['avg_mse']
            best_params = params
            best_metrics = avg_metrics

        results.append({
            **params,
            **avg_metrics
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    return results_df, best_params, best_metrics

# Define the parameter distribution for LightGBM
param_dist = {
    'num_leaves': sp_randint(20, 150),
    'max_depth': sp_randint(-1, 20),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': sp_randint(100, 1000),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

# Number of iterations for Randomized Search
n_iter_search = 20

# Load .h5ad file
adata = sc.read_h5ad('/home/emma/data/specific_combinations_all_genes_raw.h5ad')

sc.pp.filter_genes_dispersion(adata, subset=False, min_disp=.5, max_disp=None, min_mean=.0125, max_mean=10, n_bins=20, n_top_genes=None, log=True, copy=True)
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pp.scale(adata, max_value=10, zero_center=False)

# Extract gene expression data and ages
data = adata.X.toarray()
adata.obs['age'] = adata.obs['age'].apply(lambda x: int(x.replace('m', '')))
ages = adata.obs['age'].to_numpy()

# Define the path for the output file
output_path = '/home/emma/result/LightGBM_ParaTunes.csv'

# Run randomized search with cross-validation and save results to CSV
results_df, best_params, best_metrics = run_lightgbm_random_search(data, ages, param_dist, n_iter_search, output_csv=output_path)

print("Results saved to", output_path)
print("Best Params:", best_params)
print("Best Metrics:", best_metrics)
