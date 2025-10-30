import torch
import pandas as pd

def load_processed_data(data_path):
    """
    Loads the processed data from the given path.

    Args:
        data_path (str): The path to the processed data file.

    Returns:
        tuple: A tuple containing the expression matrix, spatial graph, and gene bins.
    """
    processed_data = torch.load(data_path, weights_only=False)
    expression_matrix = processed_data['expression_matrix']
    spatial_graph = processed_data['spatial_graph']
    gene_bins = processed_data['gene_bins']
    return expression_matrix, spatial_graph, gene_bins

def create_gene_bins(gene_info_df, genes_per_bin=100):
    gene_info = gene_info_df.copy()
    gene_info = gene_info.dropna(subset=['chromosome', 'start'])
    gene_info['chromosome'] = gene_info['chromosome'].astype(str)
    # Ensure chromosome format is 'chrX'
    gene_info['chromosome'] = gene_info['chromosome'].apply(lambda x: f'chr{x}' if not x.startswith('chr') else x)
    chrom_order = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
    gene_info['chromosome'] = pd.Categorical(gene_info['chromosome'], categories=chrom_order, ordered=True)
    sorted_genes = gene_info.sort_values(['chromosome', 'start'])
    
    gene_bins = {}
    current_chrom = None
    bin_counter = 0
    
    # Get the original index positions from the unsorted dataframe
    original_indices = {gene_id: i for i, gene_id in enumerate(gene_info_df.index)}

    for i in range(0, len(sorted_genes), genes_per_bin):
        bin_df = sorted_genes.iloc[i:i+genes_per_bin]
        chrom = bin_df['chromosome'].iloc[0]
        
        if chrom != current_chrom:
            bin_counter = 0
            current_chrom = chrom
            
        bin_name = f"{chrom}_bin{bin_counter}"
        
        # Map sorted gene names back to their original indices
        gene_indices = [original_indices[gene_id] for gene_id in bin_df.index]
        gene_bins[bin_name] = gene_indices
        bin_counter += 1
        
    return gene_bins