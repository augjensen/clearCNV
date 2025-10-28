
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for denoising spatial transcriptomics data.
    """
    def __init__(self, n_genes, encoder_dims=[256, 64], bottleneck_dim=32, decoder_dims=[64, 256], conv_kernel_size=3):
        """
        Initializes the MaskedAutoencoder model.

        Args:
            n_genes (int): The number of genes in the input data.
            encoder_dims (list): The dimensions of the encoder layers.
            bottleneck_dim (int): The dimension of the bottleneck layer.
            decoder_dims (list): The dimensions of the decoder layers.
            conv_kernel_size (int): The kernel size for the 1D convolutional layer.
        """
        super(MaskedAutoencoder, self).__init__()
        self.n_genes = n_genes

        # Encoder
        encoder_layers = []
        input_dim = n_genes
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(input_dim, dim))
            encoder_layers.append(nn.ReLU())
            input_dim = dim
        encoder_layers.append(nn.Linear(input_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        input_dim = bottleneck_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(input_dim, dim))
            decoder_layers.append(nn.ReLU())
            input_dim = dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer for mu, theta and pi
        self.decoder_mu = nn.Linear(input_dim, n_genes)
        self.decoder_theta = nn.Linear(input_dim, n_genes)
        self.decoder_pi = nn.Linear(input_dim, n_genes)

        # 1D Convolutional layer for genomic adjacency
        self.conv1d_mu = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.conv1d_theta = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        self.conv1d_pi = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)


    def forward(self, x, mask_rate=0.15):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, n_genes).
            mask_rate (float): The rate of masking.

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): The mean of the negative binomial distribution.
                - theta (torch.Tensor): The dispersion of the negative binomial distribution.
                - pi (torch.Tensor): The zero-inflation logits.
            torch.Tensor: The mask tensor of shape (batch_size, n_genes).
        """
        # Create a mask
        mask = torch.rand(x.shape) < mask_rate
        masked_x = x.clone()
        masked_x[mask] = 0

        # Encode the masked input
        encoded = self.encoder(masked_x)
        
        # Decode
        decoded_hidden = self.decoder(encoded)
        mu = torch.exp(self.decoder_mu(decoded_hidden))
        theta = F.softplus(self.decoder_theta(decoded_hidden))
        pi = self.decoder_pi(decoded_hidden)

        # Apply 1D convolution
        mu_reshaped = mu.unsqueeze(1)
        mu_conv = self.conv1d_mu(mu_reshaped)
        mu = mu_conv.squeeze(1)

        theta_reshaped = theta.unsqueeze(1)
        theta_conv = self.conv1d_theta(theta_reshaped)
        theta = theta_conv.squeeze(1)

        pi_reshaped = pi.unsqueeze(1)
        pi_conv = self.conv1d_pi(pi_reshaped)
        pi = pi_conv.squeeze(1)

        return (mu, theta, pi), mask
