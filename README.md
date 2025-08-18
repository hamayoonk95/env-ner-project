# Environmental Named Entity Recognition (NER) Pipeline

This project implements a complete Named Entity Recognition (NER) pipeline tailored to environmental science texts. It combines rule-based annotation using curated vocabularies with statistical and deep learning models to identify five key entity types:

-   ENV_PROCESS (e.g. erosion, rainfall)
-   HABITAT (e.g. marsh, coastal grassland)
-   MEASUREMENT (e.g. kg, °C)
-   POLLUTANT (e.g. microplastics, nitrogen dioxide)
-   TAXONOMY (e.g. Eurasian Ibis, Panthera leo)

Three models are trained and evaluated:

-   CRF baseline model
-   SpaCy CNN-based NER
-   SpaCy Transformer-based NER

The final CNN model was selected for deployment due to its balance of accuracy, generalisation, and robustness.

## Setup Instructions

1. Clone the Repository

    `git clone https://github.com/hamayoonk95/env-ner-project.git`

    `cd env-ner-project`

2. Create Virtual Environment

    `python -m venv .venv`

    `source .venv/bin/activate # On Windows: .venv\Scripts\activate`

3. Install Dependencies

    `pip install -r requirements.txt`

4. Launch Jupyter Notebook

    `jupyter notebook`

## Folder Structure

The repository contains sample data to allow quick testing and notebook execution:

```
data/
├── raw_data/           # Original texts from PubMed, news, and UKCEH
├── processed/          # Cleaned versions of raw texts
├── segmented_text/     # Sentence-segmented versions
├── json/              # Weakly annotated data in SpaCy-compatible .jsonl format
└── spaCy/              # .spacy binary files for training
```

The models folder contains:

```
models/
├── crf/
│   └── final_crf_model.joblib
└── spaCy/
    ├── cnn_best/          # Best CNN config from validation
    ├── cnn_final/         # Final retrained CNN
    ├── transformer_best/   # Best transformer config from validation
```

## Using the Full Dataset

To rerun all notebooks with the full corpus:

1. Download the full dataset ZIP from the provided location:

    [Download full_data.zip](https://goldsmithscollege-my.sharepoint.com/:u:/g/personal/hkhan010_campus_goldsmiths_ac_uk/EbDkzHNVrexNlIzsn6UqwMABjeyRBewKBTaJn0_VywaIsg?e=4eg4Q3)

2. Unzip it inside the root project directory:

    `unzip full_data.zip`

This will replace the existing sample data folder

## Using All Trained Models

To access all the trained models:

1. Download all_models.zip from the link below:

    [Download all_models.zip](https://goldsmithscollege-my.sharepoint.com/:u:/g/personal/hkhan010_campus_goldsmiths_ac_uk/ESu0pIZondZJi_cMZZnKBdMBCdsbSK6qkfp2DZXtKSu7Jg?e=e7rVfQ)

2. Unzip it into the project root:

    `unzip all_models.zip`

The following models are already included in the repository:

-   models/spaCy/cnn_best/ (best validation CNN)
-   models/spaCy/cnn_final/ (final retrained CNN)
-   models/spaCy/transformer_best/ (best transformer model)
-   models/crf/final_crf_model.joblib (selected CRF)

Other models are optional and stored externally to reduce file size.

## GPU Support (Optional)

By default, the project runs on CPU (`--gpu-id -1` in SpaCy training commands).  

### 1. CPU (Default)
No changes needed.  
Dependencies are installed from `requirements.txt`, which uses the CPU-only version of PyTorch.  

### 2. NVIDIA GPU
If you have an NVIDIA GPU and want to enable acceleration:  

1. Install GPU requirements from the separate file:  

    ```bash
    pip install -r requirements-gpu.txt
    ```

   (This file points to the official CUDA-enabled PyTorch wheels.  
   Adjust the CUDA version if needed — see [PyTorch install guide](https://pytorch.org/get-started/locally/).)  

2. Update SpaCy training commands in `05_spacy_models.ipynb` and `06_evaluation.ipynb`:  
   - change `--gpu-id -1` → `--gpu-id 0`  

3. Verify GPU is detected:  

    ```python
    import torch
    print(torch.cuda.is_available())  # should return True
    ```

### 3. AMD or Apple Silicon GPUs
PyTorch provides alternative backends (ROCm for AMD on Linux, MPS for Apple Silicon).  
These are **not included in this repository**.  
If you are on AMD/Apple hardware, follow the [official PyTorch install docs](https://pytorch.org/get-started/locally/) to set up the correct build.
