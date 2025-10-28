
import os
import spatialdata as sd
import numpy as np
import geopandas as gpd
from sklearn.neighbors import kneighbors_graph
import torch
import infercnvpy as cnv
import scanpy as sc
import scipy.sparse
import pandas as pd

# 1. Load the SpatialData object
data_dir = "datasets"
sdata_path = os.path.join(data_dir, "Xenium5K_human_prostate.zarr")
sdata = sd.read_zarr(sdata_path)

# 2. Explore the SpatialData object
sdata.tables['table'].layers['raw'] = sdata.tables['table'].X.copy()
sc.pp.normalize_total(sdata.tables['table'])
sc.pp.log1p(sdata.tables['table'])

# Downsample the data
sc.pp.subsample(sdata.tables['table'], fraction=0.1, random_state=0)

# Get the subsampled cell coordinates
subsampled_cell_ids = sdata.tables['table'].obs_names

# Extract cell centroids from cell boundaries
centroids = sdata.shapes["cell_boundaries"].geometry.centroid

# Make a dataframe with cell_id and x/y coordinates
cell_coords = gpd.GeoDataFrame({
    "x": centroids.x,
    "y": centroids.y
})
cell_coords = cell_coords.loc[subsampled_cell_ids]

expression_matrix = sdata.tables['table'].X.toarray()
print(f"Expression matrix shape: {expression_matrix.shape}")

# 2.2. Spatial Graph
n_neighbors = 6
spatial_graph_sparse_scipy = kneighbors_graph(cell_coords, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
print("Built k-NN graph in scipy sparse format.")

coo = spatial_graph_sparse_scipy.tocoo()
indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
values = torch.from_numpy(coo.data).float()
shape = torch.Size(coo.shape)
spatial_graph_sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)

# 2.3. Gene Bins
cnv.io.genomic_position_from_biomart(sdata.tables['table'], adata_gene_id="gene_ids" ,species="hsapiens")

def create_gene_bins(gene_info_df, genes_per_bin=100):
    gene_info = gene_info_df.copy()
    gene_info = gene_info.dropna(subset=['chromosome', 'start'])
    gene_info['chromosome'] = gene_info['chromosome'].astype(str)
    chrom_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    gene_info['chromosome'] = pd.Categorical(gene_info['chromosome'], categories=chrom_order, ordered=True)
    sorted_genes = gene_info.sort_values(['chromosome', 'start'])
    gene_bins = {}
    current_chrom = None
    bin_counter = 0
    for i in range(0, len(sorted_genes), genes_per_bin):
        bin_df = sorted_genes.iloc[i:i+genes_per_bin]
        chrom = bin_df['chromosome'].iloc[0]
        if chrom != current_chrom:
            bin_counter = 0
            current_chrom = chrom
        bin_name = f"{chrom}_bin{bin_counter}"
        gene_indices = bin_df.index.map(lambda x: gene_info_df.index.get_loc(x)).tolist()
        gene_bins[bin_name] = gene_indices
        bin_counter += 1
    return gene_bins

gene_info_df = sdata.tables['table'].var
genes_per_bin = 20
gene_bins = create_gene_bins(gene_info_df, genes_per_bin=genes_per_bin)
print(f"Created {len(gene_bins)} bins with approximately {genes_per_bin} genes per bin.")

# Save the Processed Data
if not isinstance(expression_matrix, torch.Tensor):
    expression_matrix = torch.from_numpy(expression_matrix).float()

processed_data = {
    'expression_matrix': expression_matrix,
    'spatial_graph': spatial_graph_sparse_tensor,
    'gene_bins': gene_bins
}

output_dir = "datasets/processed"
output_path = os.path.join(output_dir, "processed_data.pt")
torch.save(processed_data, output_path)
print(f"Processed data saved to: {output_path}")
