# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:10:35 2023

@author: emma_
"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import genomap.genoTraj as gpt
import genomap.genoVis as gpv  
import gc  # Import garbage collector  


import sys
print(sys.path)

import colorDict


# Load an h5ad file
adata = sc.read_h5ad('C:/Users/emma_/OneDrive/Desktop/Aging/tabula-muris-senis-bbknn-processed-official-annotations.h5ad')

# sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
# #sc.pp.highly_variable_genes(adata, n_top_genes = 2000)
# sc.pp.filter_genes_dispersion(adata, subset=False, min_disp=.5, max_disp=None, 
# min_mean=.0125, max_mean=10, n_bins=20, n_top_genes=None, log=True, copy=True)
# sc.pp.log1p(adata)
# sc.pp.scale(adata, max_value=10, zero_center=False)
# # Compute neighbors
# sc.tl.pca(adata, n_comps=121)
# Map ages to age groups
map_age_to_group = {
            "1m": "1m, 3m",
            "3m": "1m, 3m",
            "18m": "18m, 21m",
            "21m": "18m, 21m",
            "24m": "24m, 30m",
            "30m": "24m, 30m"
        }
adata.obs['age_group'] = adata.obs['age'].map(map_age_to_group)

# Color dictionaries for plotting
agegroup_color_dict = colorDict.agegroup_color_dict()
#cluster_color_dict = colorDict.cluster_color_dict()

def plot_visualization_generic(embedding, categories, color_map, xlabel, ylabel, plot_title, file_name):
    plt.figure(figsize=(12, 10))
    plt.rcParams.update({'font.size': 43})
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    unique_categories = np.unique(categories)
    if plot_title.endswith('by age_group'):
        desired_order = ["1m, 3m", "18m, 21m", "24m, 30m"]
        unique_categories = [cat for cat in desired_order if cat in unique_categories]
        
    for cat in unique_categories:
        indices = np.where(categories == cat)[0]
        cluster_points = embedding[indices, :]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.6,
                    c=color_map.get(cat, '#000000'), label=cat, marker='o', s=100)
        
        # Check if plot title ends with 'X-genoVis' to add cluster numbers
        if plot_title.endswith('by cluster') and xlabel != 'genoTraj1':
            centroid = cluster_points.mean(axis=0)
            #plt.text(centroid[0], centroid[1], str(cat), fontsize=30, ha='center', va='center', fontweight='bold')
            plt.text(centroid[0], centroid[1], str(cat), fontsize=40, ha='center', va='center', fontweight='bold', color='white', # Choose a color that contrasts well with your points
                     bbox=dict(facecolor='black', alpha=0.5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    if plot_title.endswith('by age_group'):
        leg = plt.legend(title='categories', title_fontsize=30, fontsize=30)
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor('none')
        
    plt.tight_layout()
    # Save the plot in the specified directory with the provided file name
    save_path = r'C:/Users/emma_/OneDrive/Desktop/Aging/Figure_1_cluster_traj//' + file_name
    plt.savefig(save_path, dpi=600)
    plt.show()


# Process each cell type for visualization
cell_types = ['naive T cell','T cell', 'B cell','basophil','CD8-positive, alpha-beta T cell','mature NK T cell','CD4-positive, alpha-beta T cell','regulatory T cell','immature NKT cell',
              'immature T cell','double negative T cell','mature alpha-beta T cell','DN3 thymocyte','DN4 thymocyte','precursor B cell','late pro-B cell','immature B cell',
              'naive B cell','pancreatic B cell','early pro-B cell','plasma cell','macrophage','alveolar macrophage','lung macrophage','kupffer cell','leukogyte'
              'lymphocyte', 'mast cell', 'astrocyte', 'NK cell', 'monocyte','myeloid cell','myeloid leukocyte','neutrophil', 'classical monocyte', 'dentritic cell',
              'intermediate monocyte','granulocyte','thymocyte','professional antigen presenting cell', 'microglial cell','granulocyte monocyte progenitor cell',
              'granulocytopoietic cell','hematopoietic stem cell']


tissues = adata.obs['tissue'].unique()



for cell_type in cell_types:
  for tissue in tissues:
    try:
      print(f'Processing {cell_type}s in {tissue}...')
      adata_filtered = adata[(adata.obs['cell_ontology_class'] == cell_type) & (adata.obs['tissue'] == tissue)]

      # Perform genoVis on the PCA results
      pca_results = adata_filtered.obsm['X_pca']
      resVis = gpv.genoVis(pca_results, n_clusters=5, colNum=32, rowNum=32)
      adata_filtered.obs['cluster'] = resVis[1]  # Assign to adata_filtered
      adata_filtered.obsm['X-genovis'] = resVis[0]
      
      data = adata_filtered.X.toarray()
      trajectory_coordinates = gpt.genoTraj(data) # Using GenoTraj
      adata_filtered.obsm['X_trajectory'] = trajectory_coordinates
      
            # Save the result
      file_path = f'C:/Users/emma_/OneDrive/Desktop/Aging/{cell_type}_in_{tissue}_genoVis3cluster_bbknn.h5ad'
      sc.write(file_path, adata_filtered)
      
      # Creating a color map for 'X-genovis' using tab20 colormap
      unique_clusters = adata_filtered.obs['cluster'].unique().tolist()
      cluster_color_dict = {cluster: color for cluster, color in zip(unique_clusters, plt.cm.jet(np.linspace(0, 1, len(unique_clusters))))}


      colormap = {
                'age_group': agegroup_color_dict,
                'cluster': cluster_color_dict  # Updated to use the new color map
                }


      attributes = ['age_group', 'cluster']
      for attr in attributes:
          plot_visualization_generic(adata_filtered.obsm['X-genovis'], adata_filtered.obs[attr], colormap[attr],
                                     'genoVis1', 'genoVis2', f'by {attr}', f'{cell_type}_in_{tissue}_by_{attr}_genoVis_bbknn.png')
      

      
      attributes = ['age_group', 'cluster']
      for attr in attributes:
          plot_visualization_generic(adata_filtered.obsm['X_trajectory'], adata_filtered.obs[attr], colormap[attr],
                 '#genoTraj1', 'genoTraj2', f'{cell_type}_{tissue} by {attr}', f'{cell_type}_in_{tissue}_by_{attr}_genoTraj_bbknn.png')

      # Cleanup
      del adata_filtered
      gc.collect()  # Explicitly run the garbage collector

    except Exception as e:
        print(f"Error processing {cell_type} in {tissue}: {e}")