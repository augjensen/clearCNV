# This file will contain the training loop for the model.

import torch

import os

def train_model(model, dataloader, optimizer, loss_fn, device, num_epochs, gene_bins, spatial_graph, save_path=None, print_every_n_batches=100, mask_rate=0.15):
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
        print_every_n_batches (int): The frequency of printing the loss.
    """
    model.to(device)
    spatial_graph = spatial_graph.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            expression_batch, indices_batch = batch
            expression_batch = expression_batch.to(device)
            indices_batch = indices_batch.to(device)
            
            # Forward pass
            (mu, theta, pi), mask = model(expression_batch, mask_rate=mask_rate)

            # Compute loss
            loss = loss_fn(mu, theta, pi, expression_batch, mask, gene_bins, spatial_graph, indices_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if (i + 1) % print_every_n_batches == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(dataloader)}")

        if save_path:
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                os.makedirs(save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pt"))
                print(f"Model saved to {os.path.join(save_path, f'model_epoch_{epoch+1}.pt')}")
