# This script will evaluate the trained model.

import torch
from clearcnv.model import MaskedAutoencoder
from clearcnv.evaluate import evaluate_model

if __name__ == "__main__":
    # Load the trained model
    model = MaskedAutoencoder(n_genes=1000) # Example n_genes
    # model.load_state_dict(torch.load("path/to/your/model.pth"))
    model.eval()

    # Load your raw and denoised data
    # raw_matrix = ...
    # denoised_matrix = ...

    # Define tumor and control cells
    # tumor_cells = ...
    # control_cells = ...

    # Evaluate the model
    # evaluation_metrics = evaluate_model(
    #     raw_matrix=raw_matrix,
    #     denoised_matrix=denoised_matrix,
    #     tumor_cells=tumor_cells,
    #     control_cells=control_cells,
    # )

    # print("Evaluation Metrics:")
    # print(evaluation_metrics)
