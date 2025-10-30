# clearCNV: Masked Autoencoder Denoising for Improved in situ CNV Inference

This repository contains the code and documentation for the DDLS Final Project 2025, focusing on using a Masked Autoencoder (mAE) with **Zero-Inflated Negative Binomial (ZINB) loss** to denoise spatial transcriptomics data (Xenium) for more robust Copy Number Variation (CNV) inference. **This project was developed with the assistance of a Gemini AI agent, and the full chat history can be found in the `gemini_chat_history/` folder.**

## Project Overview

The project aims to develop an unsupervised mAE model that takes raw cell-by-gene counts, spatial coordinates, and gene genomic positions as input. The model learns a low-dimensional representation and reconstructs denoised gene expression using a **Zero-Inflated Negative Binomial (ZINB) distribution**, effectively enhancing contiguous chromosomal patterns (CNV-like) while avoiding spurious signals. The ultimate goal is to improve the reliability and robustness of downstream CNV inference using tools like `infercnvpy`.

## Workflow

The overall workflow involves the following steps:

1.  **Data Preparation**: Load the raw `.h5ad` dataset, including normalized cell x gene counts, spatial coordinates, and gene genomic positions. The data is then processed for model input (subsampling, spatial graph, gene bins).
2.  **Model Training**: Train a Masked Autoencoder (mAE) with **ZINB loss** to learn a denoising representation of the spatial transcriptomics data.
3.  **Data Denoising**: Apply the trained mAE to the raw data to generate a denoised expression matrix.
4.  **CNV Inference**: Run the `infercnvpy` pipeline on both the raw and denoised data to infer CNVs.
5.  **Evaluation**: Quantitatively evaluate the improvement in CNV signal quality using defined metrics (Global CNV Contrast, Control Noise, Coherence, False-Positive Guard) and visualize results with heatmaps.

## Setup

### Prerequisites

*   Python 3.8+
*   `pip` for package installation

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your_username/clearCNV.git
    cd clearCNV
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Running the Project (Prostate Dataset Example)

### Using with Gemini CLI

To make the MCP toolset visible to the Gemini CLI, follow these steps:

1.  **Copy Settings File:** Copy the provided `settings.json` file to your Gemini configuration directory:
    ```bash
    mkdir -p /root/.gemini
    cp settings.json /root/.gemini/settings.json
    gemini 
    ```
2.  **Start Gemini and Load Tools:** Once you are logged into the Gemini chat window, simply type `start` to load the MCP toolset and begin interacting with the `clearCNV` workflow.

______


### This is an overview of the workflow that should be executed:

### 1. Data Preparation

The `load_data.py` script is responsible for loading the raw `.h5ad` file, extracting necessary information (counts, spatial coordinates, gene positions), and preparing it for model training.

**Input:** `datasets/Xenium5K_human_prostate_subset.h5ad`
**Output:** `datasets/processed/processed_data.pt`

To prepare the data:

```bash
python3 scripts/load_data.py --input_path datasets/Xenium5K_human_prostate_subset.h5ad --output_path datasets/processed/processed_data.pt --fraction 0.1 --n_neighbors 6 --genes_per_bin 20
```

### 2. Model Training

The `train_model.py` script trains the Masked Autoencoder.

**Input:** `datasets/processed/processed_data.pt`
**Output:** Trained model checkpoints in `clearcnv_outputs/models/`

To train the model:

```bash
python3 scripts/train_model.py --data_path datasets/processed/processed_data.pt --output_dir clearcnv_outputs/models --lr 1e-4 --num_epochs 100 --batch_size 8 --spatial_weight 0.0 --genomic_weight 0.0 --mask_rate 0.15 --print_every_n_batches 1000 --warmup_steps 1000
```
*Note: Adjust parameters as needed for your specific training run.*

### 3. Data Denoising

The `denoise_data.py` script uses the trained model to denoise the raw `.h5ad` data.

**Input:**
*   Trained model: `clearcnv_outputs/models/model_epoch_100.pt` (or the path to your best model checkpoint)
*   Raw `.h5ad` data: `datasets/Xenium5K_human_prostate_subset.h5ad`
**Output:** Denoised `.h5ad` data: `datasets/Xenium5K_human_prostate_subset_denoised.h5ad`

To denoise the data:

```bash
python3 scripts/denoise_data.py --model_path clearcnv_outputs/models/model_epoch_100.pt --data_path datasets/Xenium5K_human_prostate_subset.h5ad --output_path datasets/Xenium5K_human_prostate_subset_denoised.h5ad
```
*Note: Replace `model_epoch_100.pt` with the actual filename of your best trained model.*

### 4. CNV Inference

The `run_infercnv_pipeline.py` script runs `infercnvpy` on both the raw and denoised data.

**Input:**
*   Raw `.h5ad` data: `datasets/Xenium5K_human_prostate_subset.h5ad`
*   Denoised `.h5ad` data: `datasets/Xenium5K_human_prostate_subset_denoised.h5ad`
**Output:**
*   Raw inferCNV AnnData: `datasets/Xenium5K_human_prostate_subset_raw_infercnv.h5ad`
*   Denoised inferCNV AnnData: `datasets/Xenium5K_human_prostate_subset_denoised_infercnv.h5ad`
*   Heatmaps: `clearcnv_outputs/raw_heatmap.png`, `clearcnv_outputs/denoised_heatmap.png`

To run the inferCNV pipeline:

```bash
python3 scripts/run_infercnv_pipeline.py --raw_h5ad_path datasets/Xenium5K_human_prostate_subset.h5ad --denoised_h5ad_path datasets/Xenium5K_human_prostate_subset_denoised.h5ad --raw_infercnv_h5ad_path datasets/Xenium5K_human_prostate_subset_raw_infercnv.h5ad --denoised_infercnv_h5ad_path datasets/Xenium5K_human_prostate_subset_denoised_infercnv.h5ad --raw_heatmap_path clearcnv_outputs/raw_heatmap.png --denoised_heatmap_path clearcnv_outputs/denoised_heatmap.png
```

### 5. Evaluation

The `evaluate_cnv.py` script calculates the quantitative metrics and generates a scorecard.

**Input:**
*   Raw inferCNV AnnData: `datasets/Xenium5K_human_prostate_subset_raw_infercnv.h5ad`
*   Denoised inferCNV AnnData: `datasets/Xenium5K_human_prostate_subset_denoised_infercnv.h5ad`
*   Cell type key: `status` (assuming this column exists in your `.obs` dataframe)
*   Tumor cell type: `tumor` (example)
*   Control cell type: `normal` (example)
**Output:** Scorecard (printed to console).

To evaluate the CNV results:

```bash
python3 scripts/evaluate_cnv.py --raw_adata_path datasets/Xenium5K_human_prostate_subset_raw_infercnv.h5ad --denoised_adata_path datasets/Xenium5K_human_prostate_subset_denoised_infercnv.h5ad --cell_type_key status --tumor_cell_type tumor --control_cell_type normal
```
*Note: Adjust `cell_type_key`, `tumor_cell_type`, and `control_cell_type` to match your dataset's annotations.*

