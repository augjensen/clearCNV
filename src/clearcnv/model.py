
import torch
import torch.nn as nn

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
        decoder_layers.append(nn.Linear(input_dim, n_genes))
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        
        # 1D Convolutional layer for genomic adjacency
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)


    def forward(self, x, mask_rate=0.15):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, n_genes).
            mask_rate (float): The rate of masking.

        Returns:
            torch.Tensor: The reconstructed tensor of shape (batch_size, n_genes).
            torch.Tensor: The mask tensor of shape (batch_size, n_genes).
        """
        # Create a mask
        mask = torch.rand(x.shape) < mask_rate
        masked_x = x.clone()
        masked_x[mask] = 0

        # Encode the masked input
        encoded = self.encoder(masked_x)
        
        # Decode
        decoded_mlp = self.decoder_mlp(encoded)

        # Apply 1D convolution
        # The input to Conv1d should be (batch_size, in_channels, sequence_length)
        decoded_mlp_reshaped = decoded_mlp.unsqueeze(1)
        decoded_conv = self.conv1d(decoded_mlp_reshaped)
        decoded = decoded_conv.squeeze(1)

        return decoded, mask
