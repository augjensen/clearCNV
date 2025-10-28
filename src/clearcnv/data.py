
import torch

def load_processed_data(data_path):
    """
    Loads the processed data from the given path.

    Args:
        data_path (str): The path to the processed data file.

    Returns:
        tuple: A tuple containing the expression matrix, spatial graph, and gene bins.
    """
    processed_data = torch.load(data_path)
    expression_matrix = processed_data['expression_matrix']
    spatial_graph = processed_data['spatial_graph']
    gene_bins = processed_data['gene_bins']
    return expression_matrix, spatial_graph, gene_bins
