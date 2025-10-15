
# Masked Autoencoder Denoising for Better in situ CNV Inference

This project aims to improve Copy Number Variation (CNV) inference from spatial transcriptomics data by using a masked autoencoder (mAE) to denoise the data.

## Directory Structure

```
.
├── data/
├── notebooks/
├── scripts/
│   ├── run_baseline_cnv.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/
│   └── clearcnv/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
└── README.md
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model, run the following command:

```bash
python scripts/train_model.py
```

## Saving Chat History

To save your Gemini chat history for documentation purposes, follow these steps:

1.  **Copy the chat history:**

    From the Gemini web interface, type 
    ```bash
    /chat save 'NAME_OF_CHAT'
    ```

2.  **Create a directory for the chat history:**

    ```bash
    mkdir -p gemini_chat_history
    ```

3.  **Save the chat history to the gemini_chat_history folder:**

    Type in a terminal:
    ```bash
    cp /root/.gemini/tmp/*/checkpoint-'NAME_OF_CHAT'.json ./gemini_chat_history/
    ```
    
    For example, if you named your chat "01_startup", you would run:
    ```bash
    cp /root/.gemini/tmp/*/checkpoint-01_startup.json ./gemini_chat_history/
    ```

    cp ./gemini_chat_history/checkpoint-01_startup.json /root/.gemini/tmp/*/ 
    