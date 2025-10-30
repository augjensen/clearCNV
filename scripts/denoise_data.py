# This script denoises the data using the trained Masked Autoencoder model.

import os
import torch
import anndata as ad
from clearcnv.model import MaskedAutoencoder

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Denoise data using a trained clearCNV model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input AnnData file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the denoised AnnData file.')
    args = parser.parse_args()

    # Load the AnnData object with the raw data
    adata = ad.read_h5ad(args.data_path)

    # Load the trained model
    n_genes = adata.shape[1]
    model = MaskedAutoencoder(n_genes=n_genes)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Denoise the data
    with torch.no_grad():
        # The model returns a tuple (mu, theta, pi), we only need mu for the denoised expression
        denoised_outputs, _ = model(torch.tensor(adata.X.toarray()))
        mu, _, _ = denoised_outputs

    # Create a new AnnData object for the denoised data
    adata_denoised = adata.copy()
    adata_denoised.X = mu.numpy()

    # Save the denoised data
    adata_denoised.write_h5ad(args.output_path)

    print(f"Denoised data saved to {args.output_path}")
