import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import tensorflow as tf 
import shap
import os
from scipy.sparse import csr_matrix


# Load .h5ad file
adata = sc.read_h5ad('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/specific_combinations_all_genes_raw.h5ad')

sc.pp.filter_genes_dispersion(adata, subset=False, min_disp=.5, max_disp=None, min_mean=.0125, max_mean=10, n_bins=20, n_top_genes=None, log=True, copy=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pp.scale(adata, max_value=10, zero_center=False)


adata.obs['age'] = adata.obs['age'].apply(lambda x:int(x.replace('m','')))
selected_age_groups = [3, 18, 21, 24, 30]
adata_selected = adata[adata.obs['age'].isin(selected_age_groups)]
data_selected = adata_selected.X.toarray()
ages_selected = adata_selected.obs['age'].to_numpy()
genes_selected= adata_selected.var_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_selected, ages_selected, test_size=0.2, random_state=42)

# Define LightGBM model with specific parameters
lgb_params = {
    'colsample_bytree': 0.6999304858576277,
    'learning_rate': 0.023999698964084628,
    'max_depth': 14,
    'n_estimators': 882,
    'num_leaves': 70,
    'subsample': 0.6912309956335814
    }
model = lgb.LGBMRegressor(**lgb_params)

# Train the LightGBM model
model.fit(X_train, y_train)

# Create a SHAP explainer and calculate SHAP values for all samples
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_selected) #has to feed densed array

# Save the global feature importance plot
fig1, ax1 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, data_selected, feature_names=genes_selected, plot_type='bar')
plt.tight_layout()
fig1.savefig('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/global_feature_importance.png', dpi=300)  # Save the figure
plt.close(fig1)  # Close the figure to free memory

fig2, ax2 = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, data_selected, feature_names=genes_selected, plot_type='dot')
plt.tight_layout()
fig2.savefig('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/local_explanation_summary.png', dpi=300)  # Save the figure
plt.close(fig2)  # Close the figure to free memory
