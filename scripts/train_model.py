# This script will train the Masked Autoencoder model.

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from torch.utils.data import DataLoader, TensorDataset
from clearcnv.model import MaskedAutoencoder
from clearcnv.train import train_model
from clearcnv.loss import TotalLoss
from clearcnv.data import load_processed_data

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load the processed data ---
    data_path = "datasets/processed/processed_data.pt"
    expression_matrix, spatial_graph, gene_bins = load_processed_data(data_path)

    # --- Create the DataLoader ---
    indices = torch.arange(expression_matrix.shape[0])
    dataset = TensorDataset(expression_matrix, indices)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model, optimizer, and loss function
    n_genes = expression_matrix.shape[1]
    model = MaskedAutoencoder(n_genes=n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TotalLoss(spatial_weight=0)

    # Train the model
    output_dir = "clearcnv_outputs/models"
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=100,
        gene_bins=gene_bins,
        spatial_graph=spatial_graph,
        save_path=output_dir,
        print_every_n_batches=1000
    )
