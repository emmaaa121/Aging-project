import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor 
import gc

def lightGBM_regression(data, ages, params, n_splits=5, output='metrics.csv'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'mse':[], 'rmse':[], 'mae':[], 'mape':[], 'r2':[]}
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = ages[train_index], ages[test_index]
        
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


adata = sc.read_h5ad('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/specific_combinations_all_genes_raw.h5ad')

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.filter_genes_dispersion(adata, subset=False, min_disp=0.5, max_disp=None,
                              min_mean=0.025, max_mean=10, n_bins=20, n_top_genes=None, copy=False, log=True)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10, zero_center=False)

data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
adata.obs['age'] = adata.obs['age'].apply(lambda x: int(x.replace('m','')))
ages = adata.obs['age'].to_numpy()
selected_ages = [3, 18, 21, 24, 30] #integers, not strings
mask = adata.obs['age'].isin(selected_ages)
adata_selected = adata[mask]
data_selected = data[mask]
ages_selected = ages[mask]

params = {
    'colsample_bytree': 0.6999304858576277,
    'learning_rate': 0.023999698964084628,
    'max_depth': 14,
    'n_estimators': 882,
    'num_leaves': 70,
    'subsample': 0.6912309956335814}

output = "C:/Users/emma_/OneDrive/Desktop/Aging/Figure_4_specific combinations/result/lightgbm_regressor_metrics_3to30m.csv"
avg_metrics = lightGBM_regression(data_selected , ages_selected, params, n_splits=5, output='metrics.csv')
print("Results saved to", output)
print("Best Metrics:", avg_metrics)

# Train a new model on the entire dataset for SHAP analysis
lgb_model_full = LGBMRegressor(**params, random_state=42)
lgb_model_full.fit(data_selected, ages_selected)
explainer = shap.TreeExplainer(lgb_model_full)
shap_values = explainer.shap_values(data_selected)

fig1, ax1 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, data_selected, feature_names=adata_selected.var_names, plot_type='bar')
fig1.savefig('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/global_feature_importance.png', dpi=300)  

fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, data_selected, feature_names=adata_selected.var_names, plot_type='dot')
fig2.savefig('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/local_explanation_summary.png', dpi=300)  

