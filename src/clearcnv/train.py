# This file will contain the training loop for the model.

import torch

def train_model(model, dataloader, optimizer, loss_fn, device, num_epochs, gene_bins, spatial_graph):
    """
    Trains the Masked Autoencoder model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_fn: The loss function.
        device (torch.device): The device to train on.
        num_epochs (int): The number of epochs to train for.
        gene_bins (dict): A dictionary mapping gene indices to genomic bins.
        spatial_graph (torch.Tensor): The spatial graph for the spatial penalty.
    """
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            reconstructed, mask = model(batch)

            # Compute loss
            loss = loss_fn(reconstructed, batch, mask, gene_bins, spatial_graph)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
