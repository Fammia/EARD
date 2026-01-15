# From Entity Reliability to Clean Feedback: An Entity-Aware Denoising Framework Beyond Interaction-Level Signals

[![Conference](https://img.shields.io/badge/WWW-2026-brightgreen)](https://www2026.thewebconf.org/) [![Framework](https://img.shields.io/badge/PyTorch-v2.1.2-orange)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **"From Entity Reliability to Clean Feedback: An Entity-Aware Denoising Framework Beyond Interaction-Level Signals"**, which has been accepted by **The Web Conference (WWW) 2026**.

## 📖 Abstract

Implicit feedback in recommender systems is inherently noisy, containing false-positive interactions that degrade model performance. Existing denoising methods primarily focus on identifying noisy interactions based on individual loss values (interaction-level signals), often overlooking the intrinsic reliability of the entities (users and items) involved.

**EARD (Entity-Aware Denoising)** is a novel framework that shifts the focus from interaction-level signals to **entity reliability**. By analyzing the loss distributions of users and items, EARD effectively distinguishes between hard-but-clean samples and noisy samples. The framework dynamically adjusts the importance of training samples through a multi-faceted weighting mechanism, leading to more robust and accurate recommendations.

**Key Contributions:**
*   **Entity-Centric Perspective**: We propose to evaluate noise through the lens of user and item reliability, moving beyond simple interaction-level loss filtering.
*   **Adaptive Reweighting**: We introduce a dynamic reweighting strategy based on the Empirical Cumulative Distribution Function (ECDF) of losses to adaptively down-weight unreliable signals.
*   **Model-Agnostic Design**: EARD is a general framework that can be seamlessly integrated with various collaborative filtering backbones (e.g., GMF, NeuMF, CDAE).

## 🛠️ Environment Setup

Please follow the steps below to set up the environment for reproducing our results.

### 1. Create Conda Environment
```bash
conda create -n EARD python=3.11
conda activate EARD
```

### 2. Install Dependencies
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations
For optimal training performance, we recommend using a single **NVIDIA RTX 4090D GPU** (or equivalent).

## 📂 Project Structure

The project is organized by backbone models to ensure modularity and ease of use.

```
EARD/
├── CDAE/                   # Implementation for CDAE backbone
│   ├── logs/               # Training logs
│   ├── models/             # Saved model checkpoints
│   ├── config.conf         # Configuration file
│   ├── data_utils.py       # Data loading and processing
│   ├── evaluate.py         # Evaluation metrics (Recall, NDCG, etc.)
│   ├── main_CDAE.py        # EARD training script for CDAE
│   ├── main_CDAE_vanilla.py# Baseline training script
│   └── model.py            # CDAE model architecture
│
├── NCF/                    # Implementation for GMF and NeuMF backbones
│   ├── logs/               # Training logs
│   ├── models/             # Saved model checkpoints
│   ├── config.conf         # Configuration file
│   ├── data_utils.py       # Data loading and processing
│   ├── evaluate.py         # Evaluation metrics
│   ├── main.py             # EARD training script for GMF/NeuMF
│   ├── main_vanilla.py     # Baseline training script
│   └── model.py            # GMF/NeuMF model architectures
│
├── data/                   # Dataset directory
│   ├── amazon_book/
│   ├── movielens/          # ML-1M dataset included
│   └── yelp/
│
├── Hessian_valid.py        # Script for hyperparameter concavity analysis
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Supported Models and Datasets

### Models
*   **GMF**: Generalized Matrix Factorization
*   **NeuMF**: Neural Matrix Factorization
*   **CDAE**: Collaborative Denoising Auto-Encoder

### Datasets  (Included in /data/)
*   **ML-1M** (`movielens`): Movie ratings dataset.
*   **Yelp** (`yelp`): Business reviews dataset.
*   **Amazon-Book** (`amazon_book`): Book purchase dataset.

## ⚡ Getting Started

### 1. Training GMF & NeuMF
Navigate to the `NCF` directory:
```bash
cd NCF
```

**Train with EARD:**
To train a model (e.g., NeuMF) with the EARD framework, specify the model name, dataset, and entity weight bounds ($\alpha$ and $\beta$).

```bash
# Example: Train NeuMF on ML-1M with alpha=0.5, beta=1.5
python main.py --model NeuMF-end --dataset movielens --factor_lower 0.5 --factor_upper 1.5
```

**Train Vanilla Baseline:**
To train the original model without EARD denoising:

```bash
python main_vanilla.py --model NeuMF-end --dataset movielens
```

### 2. Training CDAE
Navigate to the `CDAE` directory:
```bash
cd CDAE
```

**Train with EARD:**
```bash
# Example: Train CDAE on ML-1M
python main_CDAE.py --dataset movielens --factor_lower 0.5 --factor_upper 1.5
```

**Key Hyperparameters:**
*   `--factor_lower`: Lower bound of the entity weight ($\alpha$).
*   `--factor_upper`: Upper bound of the entity weight ($\beta$).
*   `--dataset`: Dataset name (`movielens`, `yelp`, `amazon_book`).
*   `--model`: Model name (only for NCF directory: `GMF`, `NeuMF-end`).

## 📊 Reproducibility (RQ1 Experiments)

To reproduce the main results reported in the paper (RQ1), please use the following hyperparameter settings for $\alpha$ (`factor_lower`) and $\beta$ (`factor_upper`):

| Model | ML-1M ($\alpha, \beta$) | Yelp ($\alpha, \beta$) | Amazon-Book ($\alpha, \beta$) |
| :---- | :---------------------- | :--------------------- | :---------------------------- |
| **GMF**   | [1.0, 2.0]              | [0.9, 1.0]             | [0.14, 0.4]                   |
| **NeuMF** | [0.5, 1.5]              | [0.05, 0.1]            | [0.05, 0.1]                   |
| **CDAE**  | [0.5, 1.5]              | [0.1, 0.5]             | [0.1, 0.5]                    |

## 📝 Citation

If you find this code or our paper useful, please consider citing:

```bibtex
@article{liu2025entity,
  title={From Entity Reliability to Clean Feedback: An Entity-Aware Denoising Framework Beyond Interaction-Level Signals},
  author={Liu, Ze and Wang, Xianquan and Liu, Shuochen and Ma, Jie and Xu, Huibo and Han, Yupeng and Zhang, Kai and Zhou, Jun},
  journal={arXiv preprint arXiv:2508.10851},
  year={2025}
}
```