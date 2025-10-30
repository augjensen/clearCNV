
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from torch.utils.data import DataLoader, TensorDataset
from clearcnv.model import MaskedAutoencoder
from clearcnv.train import train_model
from clearcnv.loss import TotalLoss
from clearcnv.data import load_processed_data
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

def main():
    parser = argparse.ArgumentParser(description='Train the Masked Autoencoder model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed data .pt file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained models.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--spatial_weight', type=float, default=0.0, help='Weight for the spatial loss component.')
    parser.add_argument('--genomic_weight', type=float, default=0.0, help='Weight for the genomic loss component.')
    parser.add_argument('--mask_rate', type=float, default=0.15, help='Masking rate for the Masked Autoencoder.')
    parser.add_argument('--print_every_n_batches', type=int, default=1000, help='How often to print training progress.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps for the learning rate scheduler.')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load the processed data ---
    expression_matrix, spatial_graph, gene_bins = load_processed_data(args.data_path)

    # --- Create the DataLoader ---
    indices = torch.arange(expression_matrix.shape[0])
    dataset = TensorDataset(expression_matrix, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    n_genes = expression_matrix.shape[1]
    model = MaskedAutoencoder(n_genes=n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = TotalLoss(spatial_weight=args.spatial_weight, genomic_weight=args.genomic_weight)
    
    # Scheduler
    total_steps = args.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps, eta_min=1e-6)

    # Train the model
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=args.num_epochs,
        gene_bins=gene_bins,
        spatial_graph=spatial_graph,
        scheduler=scheduler,
        warmup_steps=args.warmup_steps,
        save_path=args.output_dir,
        print_every_n_batches=args.print_every_n_batches,
        mask_rate=args.mask_rate
    )

if __name__ == '__main__':
    main()
