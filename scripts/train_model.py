# This script will train the Masked Autoencoder model.

import torch
from torch.utils.data import DataLoader, TensorDataset
from clearcnv.model import MaskedAutoencoder
from clearcnv.train import train_model
from clearcnv.loss import CustomLoss

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, optimizer, and loss function
    n_genes = 1000
    model = MaskedAutoencoder(n_genes=n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CustomLoss()

    # Create dummy data, dataloader, gene_bins, and spatial_graph
    # Replace with your actual data
    dummy_data = torch.randn(100, n_genes)
    dummy_dataset = TensorDataset(dummy_data)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=32)
    dummy_gene_bins = {}
    dummy_spatial_graph = torch.randn(100, 100)

    # Train the model
    train_model(
        model=model,
        dataloader=dummy_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=100,
        gene_bins=dummy_gene_bins,
        spatial_graph=dummy_spatial_graph,
    )
