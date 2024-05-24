# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:13:58 2024

@author: emma_
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:57:49 2024

@author: emma_
"""

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor #not lightGBMRegressor
import gc

def lightGBM_regression(data, ages, params, n_splits=5, output='metrics.csv'): #forgot colon and comma between each argument
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'mse':[], 'rmse':[], 'mae':[], 'mape':[], 'r2':[]}
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = ages[train_index], ages[test_index]

        print("Current params:", params)  # Debugging line
        lgb_model = LGBMRegressor(**params, random_state=42)
        lgb_model.fit(X_train, y_train)
        predictions = lgb_model.predict(X_test)

        mse = mean_squared_error(predictions, y_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(predictions, y_test)
        mape = np.mean(np.abs((y_test - predictions)/y_test)) * 100
        r2 = r2_score(y_test, predictions)

        metrics['mse'].append(mse)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['mape'].append(mape)
        metrics['r2'].append(r2)

    avg_metrics = {f'avg_{metric}': np.mean(values) for metric, values in metrics.items()}

    if output:
        pd.DataFrame([avg_metrics]).to_csv(output, index=False)

    return avg_metrics


adata = sc.read_h5ad('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/specific_combinations_all_genes_raw_02.h5ad')

sc.pp.filter_genes_dispersion(adata, subset=False, min_disp=0.5, max_disp=None,
                              min_mean=0.025, max_mean=10, n_bins=20, n_top_genes=None, copy=False, log=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pp.scale(adata, max_value=10, zero_center=False)

adata.obs['age'] = adata.obs['age'].apply(lambda x:int(x.replace('m','')))
selected_age_groups = [1, 3, 18, 21, 24, 30]
mask = adata.obs['age'].isin(selected_age_groups) # Create a boolean mask
data = adata.X.toarray()
data_selected = data[mask]
ages = adata.obs['age'].to_numpy()
ages_selected = ages[mask]

# Index into the data array using the boolean mask


params = {
    'colsample_bytree': 0.6999304858576277,
    'learning_rate': 0.023999698964084628,
    'max_depth': 14,
    'n_estimators': 882,
    'num_leaves': 70,
    'subsample': 0.6912309956335814}

output = "C:/Users/emma_/OneDrive/Desktop/Aging/Figure_4_specific combinations/result/lightgbm_regressor_metrics_3to30m_raw_02.csv"
avg_metrics = lightGBM_regression(data_selected , ages_selected, params, n_splits=5, output='metrics.csv')

print("Results saved to", output)
print("Best Metrics:", avg_metrics)