import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

adata = sc.read_h5ad('C:/Users/emma_/OneDrive/Desktop/Aging/Figure_3_differential_exp/specific_combinations_all_genes_raw.h5ad')  # make sure to provide the correct path to your file
sc.pp.log1p(adata)
sc.pp.scale(adata)

genes = [
'Rpl13a', 'Sparc', 'Cfl1', 'Tmsb10', 'Rps29', 'C130026I21Rik',
'Plp1', 'Malat1', 'Ifitm2', 'Dcn', 'Bpifa1', 'Rps28',
'Gsn', 'Cytl1', 'Comt', 'Aes', 'Mir703',
'Gpx3', 'H2-D1', 'Snrpc'
]

ages_of_interest = ['18m', '21m', '24m', '30m']
adata_filtered = adata[adata.obs['age'].isin(ages_of_interest)]

# Sort the AnnData object by age
adata_filtered.obs['age'] = pd.Categorical(adata_filtered.obs['age'], categories=ages_of_interest, ordered=True)
adata_filtered = adata_filtered[adata_filtered.obs.sort_values('age').index]

# Perform Mann-Whitney U test for each gene between 18m and 24m
p_values_18_24 = {}
for gene in genes:
    expression_data_18 = adata_filtered[adata_filtered.obs['age'] == '18m', gene].X.toarray().flatten()
    expression_data_24 = adata_filtered[adata_filtered.obs['age'] == '24m', gene].X.toarray().flatten()
    
    stat, p_val = mannwhitneyu(expression_data_18, expression_data_24, alternative='two-sided')
    p_values_18_24[gene] = p_val

# Sort genes by p-value between 18m and 24m
sorted_genes = sorted(p_values_18_24, key=p_values_18_24.get)

heatmap_data = pd.DataFrame(index=ages_of_interest, columns=sorted_genes)
for gene in sorted_genes:
    for age in ages_of_interest:
        gene_expression = adata_filtered[adata_filtered.obs['age'] == age, gene].X.toarray().flatten()
        heatmap_data.at[age, gene] = np.mean(gene_expression)
heatmap_data_float64 = heatmap_data.astype(np.float64)

# Replace the long gene name with 'Rik'
heatmap_data_float64.columns = heatmap_data_float64.columns.str.replace('C130026I21Rik', 'Rik')

plt.figure(figsize=(10, 8))
cmap = sns.light_palette("navy", as_cmap=True)
ax = sns.heatmap(heatmap_data_float64, cmap=cmap, yticklabels=True, xticklabels=True, cbar=True)
#plt.title('Log-transformed and Scaled Heatmap of Gene Expression by Age')
#ax.set_xlabel('Age', fontsize=10)
#ax.set_ylabel('Genes', fontsize=10)

ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()
plt.show()
