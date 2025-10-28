
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ZINBLoss, self).__init__()
        self.eps = eps

    def forward(self, mu, theta, pi, x):
        # Ensure parameters are positive
        mu = torch.clamp(mu, min=self.eps)
        theta = torch.clamp(theta, min=self.eps)

        # Zero-inflation component
        log_p_zero = F.logsigmoid(pi)
        log_one_minus_p_zero = F.logsigmoid(-pi)

        # Negative Binomial component
        log_theta_mu_eps = torch.log(theta + mu + self.eps)
        log_nb_pmf_for_0 = theta * (torch.log(theta + self.eps) - log_theta_mu_eps)

        # Combine for zero counts
        case_zero_log_likelihood = torch.logaddexp(log_p_zero, log_one_minus_p_zero + log_nb_pmf_for_0)

        # For non-zero counts
        nb_ll = (
            torch.lgamma(theta + x)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
            + x * (torch.log(mu + self.eps) - log_theta_mu_eps)
            + theta * (torch.log(theta + self.eps) - log_theta_mu_eps)
        )
        case_non_zero_log_likelihood = log_one_minus_p_zero + nb_ll

        nll = -torch.where(x == 0, case_zero_log_likelihood, case_non_zero_log_likelihood)
        return nll.mean()

class TotalLoss(nn.Module):
    """
    Total loss function for the Masked Autoencoder.
    It includes a reconstruction loss, a spatial smoothness penalty, and a genomic smoothness penalty.
    """
    def __init__(self, spatial_weight=0.1, genomic_weight=0.1):
        """
        Initializes the TotalLoss.

        Args:
            spatial_weight (float): The weight for the spatial smoothness penalty.
            genomic_weight (float): The weight for the genomic smoothness penalty.
        """
        super(TotalLoss, self).__init__()
        self.spatial_weight = spatial_weight
        self.genomic_weight = genomic_weight
        self.reconstruction_loss = ZINBLoss()

    def forward(self, mu, theta, pi, original, mask, gene_bins, spatial_graph, indices_batch):
        """
        Computes the total loss.

        Args:
            mu (torch.Tensor): The mean of the NB distribution.
            theta (torch.Tensor): The dispersion of the NB distribution.
            pi (torch.Tensor): The zero-inflation probability.
            original (torch.Tensor): The original input data.
            mask (torch.Tensor): The mask used for the input.
            gene_bins (dict): A dictionary mapping gene indices to genomic bins.
            spatial_graph (torch.Tensor): The spatial graph.
            indices_batch (torch.Tensor): The indices of the cells in the current batch.

        Returns:
            torch.Tensor: The total loss.
        """
        # 1. Reconstruction Loss (only on masked elements)
        recon_loss = self.reconstruction_loss(mu[mask], theta[mask], pi[mask], original[mask])

        # 2. Spatial Smoothness Penalty
        if self.spatial_weight > 0:
            spatial_loss = self.spatial_smoothness_penalty(mu, spatial_graph, indices_batch)
        else:
            spatial_loss = 0

        # 3. Genomic Smoothness Penalty
        genomic_loss = self.genomic_smoothness_penalty(mu, gene_bins)
        
        # Total Loss
        total_loss = recon_loss + self.spatial_weight * spatial_loss + self.genomic_weight * genomic_loss
        return total_loss

    def spatial_smoothness_penalty(self, reconstructed, spatial_graph, indices_batch):
        """
        Computes the spatial smoothness penalty using sparse matrix operations.
        This encourages neighboring cells to have similar expression profiles.
        We use a graph Laplacian regularization for this.
        L = D - A, where D is the degree matrix and A is the adjacency matrix.
        The penalty is trace(X^T * L * X), where X is the reconstructed data.
        """
        # Ensure reconstructed is float
        reconstructed = reconstructed.float()
        
        # Move spatial_graph to the same device as reconstructed
        spatial_graph = spatial_graph.to(reconstructed.device)

        # Convert spatial_graph to dense for batch indexing if it's sparse
        if spatial_graph.is_sparse:
            spatial_graph_dense = spatial_graph.to_dense()
        else:
            spatial_graph_dense = spatial_graph

        # Select the part of the spatial graph relevant to the current batch
        batch_adj = spatial_graph_dense[indices_batch, :][:, indices_batch]

        # Degree matrix D for the batch
        deg = torch.sum(batch_adj, dim=1)

        # Laplacian L = D - A (A is the batch_adj)
        
        # D is a diagonal matrix, so D@reconstructed is equivalent to element-wise multiplication
        d_times_x = deg.unsqueeze(1) * reconstructed
        
        # A is a dense matrix, so we use matrix multiplication
        a_times_x = torch.matmul(batch_adj, reconstructed)
        
        laplacian_times_x = d_times_x - a_times_x

        # The penalty is trace(X^T * L * X) = sum(X * (L*X))
        penalty = torch.sum(reconstructed * laplacian_times_x)

        return penalty / (reconstructed.shape[0] * reconstructed.shape[0]) # Normalize by batch size squared

    def genomic_smoothness_penalty(self, reconstructed, gene_bins):
        """
        Computes the genomic smoothness penalty.
        This encourages genes within the same genomic bin to have similar expression values.
        """
        penalty = 0
        for bin_name, gene_indices in gene_bins.items():
            if len(gene_indices) > 1:
                bin_expressions = reconstructed[:, gene_indices]
                # We want to minimize the variance within each bin
                penalty += torch.var(bin_expressions, axis=1).mean()
        return penalty / len(gene_bins) # Normalize by number of bins
