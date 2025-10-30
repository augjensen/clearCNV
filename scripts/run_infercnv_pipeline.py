import os
import sys
sys.path.append('src')
import infercnvpy as cnv
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the inferCNV pipeline on raw and denoised data.')
    parser.add_argument('--raw_h5ad_path', type=str, required=True, help='Path to the raw AnnData file.')
    parser.add_argument('--denoised_h5ad_path', type=str, required=True, help='Path to the denoised AnnData file.')
    parser.add_argument('--raw_infercnv_h5ad_path', type=str, required=True, help='Path to save the raw infercnv AnnData file.')
    parser.add_argument('--denoised_infercnv_h5ad_path', type=str, required=True, help='Path to save the denoised infercnv AnnData file.')
    parser.add_argument('--raw_heatmap_path', type=str, required=True, help='Path to save the raw heatmap.')
    parser.add_argument('--denoised_heatmap_path', type=str, required=True, help='Path to save the denoised heatmap.')
    args = parser.parse_args()

    # Load the raw and denoised data
    adata = ad.read_h5ad(args.raw_h5ad_path)
    adata_denoised = ad.read_h5ad(args.denoised_h5ad_path)

    ## Here I just want to make sure that the adata and adata_denoised have the 'status' .obs and tumor and normal annotations to be able to proceed with the cnv inference. The annotations should not happen here, but been done before. 

    def check_data(adata, name):
        print(f"Checking {name} data...")
        required_var_cols = ['chromosome', 'start', 'end']
        required_obs_cols = ['status']
        
        for col in required_var_cols:
            if col not in adata.var.columns:
                raise ValueError(f"Column '{col}' not found in {name}.var")
                
        for col in required_obs_cols:
            if col not in adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in {name}.obs")
                
        if 'tumor' not in adata.obs['status'].unique() or 'normal' not in adata.obs['status'].unique():
            raise ValueError(f"'status' column in {name}.obs must contain both 'tumor' and 'normal' values.")
            
        print(f"{name} data checks passed.")

    check_data(adata, "raw adata")
    check_data(adata_denoised, "denoised adata")


    # Run inferCNV on Raw Data
    cnv.tl.infercnv(adata, window_size = 50, reference_key='status', reference_cat='normal')
    print("After infercnv on raw data:")
    print(f"adata.uns keys: {adata.uns.keys()}")
    print(f"adata.obsm keys: {adata.obsm.keys()}")
    cnv_matrix_raw = adata.obsm['X_cnv'].toarray()
    print(f"Shape of raw X_cnv: {cnv_matrix_raw.shape}")
    print(f"Min of raw X_cnv: {np.min(cnv_matrix_raw)}")
    print(f"Max of raw X_cnv: {np.max(cnv_matrix_raw)}")
    print(f"Mean of raw X_cnv: {np.mean(cnv_matrix_raw)}")

    # Run inferCNV on Denoised Data
    cnv.tl.infercnv(adata_denoised,  window_size = 50, reference_key='status', reference_cat='normal')
    print("After infercnv on denoised data:")
    print(f"adata_denoised.uns keys: {adata_denoised.uns.keys()}")
    print(f"adata_denoised.obsm keys: {adata_denoised.obsm.keys()}")
    cnv_matrix_denoised = adata_denoised.obsm['X_cnv'].toarray()
    print(f"Shape of denoised X_cnv: {cnv_matrix_denoised.shape}")
    print(f"Min of denoised X_cnv: {np.min(cnv_matrix_denoised)}")
    print(f"Max of denoised X_cnv: {np.max(cnv_matrix_denoised)}")
    print(f"Mean of denoised X_cnv: {np.mean(cnv_matrix_denoised)}")

    # PCA → neighbors → Leiden
    for ad_obj in [adata, adata_denoised]:
        cnv.tl.pca(ad_obj, n_comps=30)
        cnv.pp.neighbors(ad_obj, n_neighbors=15, n_pcs=30)
        cnv.tl.leiden(ad_obj, resolution=0.5)

    # Plot separately
    print('Adata (Raw)')
    cnv.pl.chromosome_heatmap(adata, groupby='status')
    plt.savefig(args.raw_heatmap_path)
    print('Adata_denoised')
    cnv.pl.chromosome_heatmap(adata_denoised, groupby='status')
    plt.savefig(args.denoised_heatmap_path)

    # Save the AnnData objects with infercnv results
    adata.write_h5ad(args.raw_infercnv_h5ad_path)
    adata_denoised.write_h5ad(args.denoised_infercnv_h5ad_path)
    print("Saved raw and denoised AnnData objects with infercnv results.")

if __name__ == '__main__':
    main()
