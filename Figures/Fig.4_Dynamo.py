import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd  # Import pandas for data frame operations
import matplotlib.pyplot as plt
import genomap.genoTraj as gp
import tensorflow as tf 
import sys
import os
import dynamo as dyn
from dynamo.preprocessing import Preprocessor
dyn.dynamo_logger.main_silence()
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

adata = sc.read_h5ad('/home/emma/data/10X_P7_2_3_marrow_bbknn_annotated_final.h5ad')
genes = ['Lars2', 'Mgp', 'Rpl13a',  'Ubb', 'Ets1', 'Cfl1', 'Tmsb10', 'S100a6', 'Rps29', 'Rps27', 'Gm6981', 'Map3k1', 'Wtap', 'Plp1', 'Pdcd4']
preprocessor = Preprocessor(gene_append_list=genes)
preprocessor.preprocess_adata(adata, recipe="monocle")
print(adata) 

dyn.tl.dynamics(adata, model="stochastic")
dyn.tl.reduceDimension(adata, n_pca_components=30)
dyn.tl.louvain(adata, resolution=0.2)


dyn.tl.cell_velocities(adata, method="pearson", other_kernels_dict={"transform": "sqrt"})
dyn.tl.cell_velocities(adata, basis="pca")
dyn.vf.VectorField(adata, basis='pca')
print(adata)

#dyn.pl.streamline_plot(adata, color=["clusters"], basis="umap", show_legend="on data", 
											#show_arrowed_spines=True, save_show_or_return='save', save_kwargs={"dpi":300})
gene = 'Pdcd4'
dyn.pd.perturbation(adata, gene, [-100], emb_basis="umap")
#dyn.pd.KO(adata, gene)
print(adata)


scv.pl.velocity_embedding_stream(adata, basis='umap_perturbation', color='cell_type', save='/home/emma/result/perturbation/Pdcd4_perturbation.png', dpi=300)

# Gene specific velocity plots
scv.pl.velocity(adata, ['Pdcd4'], perc = [5, 60], save='/home/emma/result/perturbation/Rpl13a_velocity_expression.png', dpi=300)
scv.pl.scatter(adata, basis= 'umap' , color= 'Pdcd4',  perc = [40, 95],  save='/home/emma/result/perturbation/Pdcd4_expression.png', dpi=300)

# Velocity confidence
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
scv.tl.velocity_confidence(adata)
keys = [Table 2: Aging Profile Highlighting Pronounced Genotype Changes]
scv.pl.scatter(adata, c=keys, perc=[5, 95], save='/home/emma/result/perturbation/a_scvelo_velocity_length_confidence.png', dpi=300)

# Velocity pseudotime
scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata, color='velocity_pseudotime', save='/home/emma/result/perturbation/a_scvelo_velocity_pseudotime.png', dpi=300)