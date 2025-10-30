import os
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import argparse
from clearcnv.data import create_gene_bins

def main():
    parser = argparse.ArgumentParser(description="Prepare data for ClearCNV model.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .h5ad file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the processed data .pt file.')
    parser.add_argument('--fraction', type=float, default=0.1, help='Fraction of cells to subsample.')
    parser.add_argument('--n_neighbors', type=int, default=6, help='Number of neighbors for the spatial graph.')
    parser.add_argument('--genes_per_bin', type=int, default=20, help='Number of genes per bin.')
    args = parser.parse_args()

    # 1. Load the AnnData object
    adata = sc.read_h5ad(args.input_path)
    
    # 2. Preprocessing

    # Downsample the data
    if args.fraction < 1.0:
        sc.pp.subsample(adata, fraction=args.fraction, random_state=0)

    # Get cell coordinates
    cell_coords = adata.obsm['spatial']
    
    expression_matrix = adata.X.toarray()
    print(f"Expression matrix shape: {expression_matrix.shape}")

    # 3. Spatial Graph
    spatial_graph_sparse_scipy = kneighbors_graph(cell_coords, n_neighbors=args.n_neighbors, mode='connectivity', include_self=False)
    print("Built k-NN graph in scipy sparse format.")

    coo = spatial_graph_sparse_scipy.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(coo.data).float()
    shape = torch.Size(coo.shape)
    spatial_graph_sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)

    # 4. Gene Bins
    gene_info_df = adata.var
    gene_bins = create_gene_bins(gene_info_df, genes_per_bin=args.genes_per_bin)
    print(f"Created {len(gene_bins)} bins with approximately {args.genes_per_bin} genes per bin.")

    # 5. Cell Status
    cell_status = adata.obs['status']

    # 6. Save the Processed Data
    if not isinstance(expression_matrix, torch.Tensor):
        expression_matrix = torch.from_numpy(expression_matrix).float()

    processed_data = {
        'expression_matrix': expression_matrix,
        'spatial_graph': spatial_graph_sparse_tensor,
        'gene_bins': gene_bins,
        'cell_status': cell_status.to_numpy()
    }

    torch.save(processed_data, args.output_path)
    print(f"Processed data saved to: {args.output_path}")

if __name__ == '__main__':
    main()