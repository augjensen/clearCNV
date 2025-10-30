import os


"""
FastMCP quickstart example.
"""

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool


# Helper function to run shell commands
def _run_shell_command(command, description, directory):
    # This will be replaced by the actual tool call in Gemini
    # For now, it's a placeholder
    print(f"Executing: {command} in {directory} ({description})")
    # In a real MCP server, this would execute the command and return output
    # For this exercise, we assume the scripts handle their own output/errors
    pass

# Tool for loading and preparing data
@mcp.tool()
def load_data_tool(input_h5ad_path: str, output_pt_path: str, fraction: float = 1.0, n_neighbors: int = 6, genes_per_bin: int = 20) -> str:
    """
    Loads the raw .h5ad dataset, processes it, and saves the output as a .pt file.
    """
    command = f"python scripts/load_data.py --input_path {input_h5ad_path} --output_path {output_pt_path} --fraction {fraction} --n_neighbors {n_neighbors} --genes_per_bin {genes_per_bin}"
    _run_shell_command(command, "Preparing data for ClearCNV model.", "clearCNV")
    return output_pt_path

# Tool for training the model
@mcp.tool()
def train_model_tool(data_path: str, output_dir: str, num_epochs: int = 100, lr: float = 1e-4, batch_size: int = 8, warmup_steps: int = 1000, spatial_weight: float = 0.0, genomic_weight: float = 0.0, mask_rate: float = 0.15, print_every_n_batches: int = 1000) -> str:
    """
    Trains the Masked Autoencoder model and saves the trained model checkpoints.
    """
    command = f"python scripts/train_model.py --data_path {data_path} --output_dir {output_dir} --num_epochs {num_epochs} --lr {lr} --batch_size {batch_size} --warmup_steps {warmup_steps} --spatial_weight {spatial_weight} --genomic_weight {genomic_weight} --mask_rate {mask_rate} --print_every_n_batches {print_every_n_batches}"
    _run_shell_command(command, "Training the Masked Autoencoder model.", "clearCNV")
    return output_dir

# Tool for denoising data
@mcp.tool()
def denoise_data_tool(model_path: str, raw_h5ad_path: str, denoised_h5ad_path: str) -> str:
    """
    Denoises the raw data using a trained Masked Autoencoder model.
    """
    command = f"python scripts/denoise_data.py --model_path {model_path} --data_path {raw_h5ad_path} --output_path {denoised_h5ad_path}"
    _run_shell_command(command, "Denoising data using a trained clearCNV model.", "clearCNV")
    return denoised_h5ad_path

# Tool for running the inferCNV pipeline
@mcp.tool()
def run_infercnv_pipeline_tool(raw_h5ad_path: str, denoised_h5ad_path: str, raw_infercnv_h5ad_path: str, denoised_infercnv_h5ad_path: str, raw_heatmap_path: str, denoised_heatmap_path: str) -> str:
    """
    Runs the inferCNV analysis on both raw and denoised data, generates heatmaps, and saves AnnData objects.
    """
    command = f"python scripts/run_infercnv_pipeline.py --raw_h5ad_path {raw_h5ad_path} --denoised_h5ad_path {denoised_h5ad_path} --raw_infercnv_h5ad_path {raw_infercnv_h5ad_path} --denoised_infercnv_h5ad_path {denoised_infercnv_h5ad_path} --raw_heatmap_path {raw_heatmap_path} --denoised_heatmap_path {denoised_heatmap_path}"
    _run_shell_command(command, "Running the inferCNV pipeline.", "clearCNV")
    return "Heatmaps generated and AnnData objects saved."

# Tool for evaluating CNV metrics
@mcp.tool()
def evaluate_cnv_tool(raw_adata_path: str, denoised_adata_path: str, cell_type_key: str, tumor_cell_type: str, control_cell_type: str) -> str:
    """
    Evaluates CNV analysis results and generates a scorecard.
    """
    command = f"python scripts/evaluate_cnv.py --raw_adata_path {raw_adata_path} --denoised_adata_path {denoised_adata_path} --cell_type_key {cell_type_key} --tumor_cell_type {tumor_cell_type} --control_cell_type {control_cell_type}"
    _run_shell_command(command, "Evaluating CNV analysis results.", "clearCNV")
    return "CNV evaluation scorecard generated."





if __name__ == "__main__":
   # Run as an MCP stdio server (no prints to stdout!)
   mcp.run(transport="stdio")