# EARD

This is the official PyTorch implementation for our paper (ID: 505).

-----

## Environment Setup

To get started, follow these simple steps to set up your environment.

1.  **Create a Conda environment**:

    ```bash
    conda create -n EARD python=3.11
    ```

2.  **Activate the environment**:

    ```bash
    conda activate EARD
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

-----

## Hardware Recommendations

For optimal training performance, we recommend using a single **NVIDIA RTX 4090 GPU**.

-----

## Project Structure

The project is organized into modules for each backbone model.

```
EARD/
├── CDAE/
│   ├── logs/                   # Training logs
│   ├── models/                 # Saved model checkpoints
│   ├── config.conf             # Configuration file for CDAE
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── evaluate.py             # Evaluation script
│   ├── main_CDAE.py            # CDAE training with EARD
│   ├── main_CDAE_vanilla.py    # CDAE baseline training
│   └── model.py                # CDAE model definition
│
├── NCF/
│   ├── logs/                   # Training logs
│   ├── models/                 # Saved model checkpoints
│   ├── config.conf             # Configuration file for GMF/NeuMF
│   ├── data_utils.py           # Data preprocessing utilities
│   ├── evaluate.py             # Evaluation script
│   ├── main.py                 # GMF/NeuMF training with EARD
│   ├── main_vanilla.py         # GMF/NeuMF baseline training
│   └── model.py                # GMF and NeuMF model definitions
│
├── data/                       # Preprocessed datasets
│   ├── amazon_book/
│   ├── movielens/
│   └── yelp/
│
├── Hessian_valid.py            # Hyperparameter concavity analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusion rules
```

> **Note:** The GMF and NeuMF models share the implementation in the `./NCF` directory, while CDAE has a separate implementation in `./CDAE` due to differences in data processing.

-----

## Supported Models and Datasets

### Models (Argument name)

  * GMF (`GMF`)
  * NeuMF (`NeuMF-end`)
  * CDAE

### Datasets (Argument name)

  * ML-1M (`movielens`)
  * Yelp (`yelp`)
  * Amazon-Book (`amazon_book`)

> **Important**: Due to file size limitations for supplementary materials on OpenReview, we are only able to provide the ML-1M dataset in this repository.

### Key Hyperparameters

  * `factor_lower`: Corresponds to the lower bound of the entity weight ($\alpha$).
  * `factor_upper`: Corresponds to the upper bound of the entity weight ($\beta$).

-----

## Getting Started

### GMF & NeuMF

1.  Navigate to the NCF directory:

    ```bash
    cd ./NCF
    ```

2.  **To train a model with EARD**, specify the model, dataset, and hyperparameters.

    **Example**: Train **NeuMF** on **ML-1M** using $\alpha = 0.5$ and $\beta = 1.5$.

    ```bash
    python main.py --model NeuMF-end --dataset movielens --factor_lower 0.5 --factor_upper 1.5
    ```

    > You can also set these hyperparameters in `config.conf`. Note that command-line arguments override the values in `config.conf`.

3.  **To train a vanilla baseline** (without EARD), use `main_vanilla.py`.

    **Example**: Train the **NeuMF** vanilla baseline on **ML-1M**.

    ```bash
    python main_vanilla.py --model NeuMF-end --dataset movielens
    ```

    > The `factor_lower` and `factor_upper` arguments are not used in the vanilla training script.

-----

### CDAE

1.  Navigate to the CDAE directory:

    ```bash
    cd ./CDAE
    ```

2.  **To train the CDAE model**, run `main_CDAE.py`.

    **Example**: Train **CDAE** on **ML-1M**.

    ```bash
    python main_CDAE.py --dataset movielens
    ```

    > Hyperparameters $\alpha$ and $\beta$ can be configured using `--factor_lower` and `--factor_upper` as shown in the GMF/NeuMF examples.

-----

## Hyperparameters for RQ1 Experiments

The following table shows the $\alpha$ and $\beta$ values used for our main experiments (RQ1) across each model and dataset.

| Model | ML-1M       | Yelp        | Amazon-Book |
| :---- | :---------- | :---------- | :---------- |
| GMF   | \[1.0, 2.0] | \[0.9, 1.0] | \[0.14, 0.4] |
| NeuMF | \[0.5, 1.5] | \[0.05, 0.1] | \[0.05, 0.1] |
| CDAE  | \[0.5, 1.5] | \[0.1, 0.5] | \[0.1, 0.5] |