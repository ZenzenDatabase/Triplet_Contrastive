# Contrastive vs. Triplet Loss: Variance & Optimization Analysis

This repository provides code and experiments for comparing contrastive and triplet loss functions in deep metric learning. We analyze how each loss shapes embedding geometry—focusing on intra-class dispersion and inter-class separation—and examine optimization behavior ("greediness") via diagnostics such as loss decay, active ratio, and gradient norms.

## Features
- **Synthetic Data Generation:** Controlled 128‑D toy datasets with tunable cluster tightness, label overlap, and outliers.
- **Model Architectures:**
  - Simple MLP for toy experiments
  - ConvNetEmbedder (MNIST/CIFAR)
  - Vision Transformer (ViT‑B/32) for fine‑grained retrieval (CIFAR‑10, CARS196, CUB‑200‑2011) via HuggingFace¹
- **Loss Functions:** PyTorch implementations of contrastive and triplet losses, with margin sampling strategies.
- **Training & Evaluation:** Scripts for training on synthetic and real data, and evaluation metrics including Recall@K and embedding visualization (t‑SNE).
- **Diagnostics:** Automatically track and plot loss curves, active ratio, and gradient norms to quantify optimization patterns.

## Repository Structure
```bash
├── data/                      # Optional: placeholder for real datasets (CIFAR-10, MNIST, CARS196, CUB-200)
├── experiments/               # Training and evaluation scripts
│   ├── train_toy.py           # Toy data generation and MLP training
│   ├── train_classification.py# ConvNet training on MNIST/CIFAR-10
│   └── train_retrieval.py     # ViT‑based retrieval on CARS196, CUB‑200
├── models/                    # Model definitions
│   ├── mlp.py
│   ├── convnet.py
│   ├── vit_embedder.py
│   └── losses.py
├── utils/                     # Data loaders, diagnostics, visualization
│   ├── synthetic_data.py
│   ├── metrics.py             # Active ratio, gradient norm, loss decay
│   └── visualize.py           # t‑SNE plotting
├── notebooks/                 # Example Jupyter notebooks
│   └── analysis.ipynb         # Embedding geometry & greedy optimization plots
├── README.md                  # This file
└── requirements.txt           # Dependencies

## Installation

1. Clone the repository (to be updated):
   ```bash
   git clone https://github.com/yourusername/deep-metric-compare.git
   cd deep-metric-compare
2. Install dependencies:
   pip install -r requirements.txt

## Usage

1. Toy Data Experiments

Generate synthetic clusters and train a simple MLP:
   ```bash
  python experiments/train_toy.py \
  --n_classes 10 \
  --samples_per_class 200 \
  --mode_spread 1.4 \
  --overlap_prob 0.1 \
  --outlier_frac 0.05 \
  --epochs 100 \
  --batch_size 64

2. Classification (MNIST / CIFAR‑10)
  ```bash
  python experiments/train_classification.py \
  --dataset cifar10 \
  --loss triplet \
  --epochs 50 \
  --batch_size 64
   
4. Retrieval (CARS196 / CUB‑200‑2011)
   ```bash
  python experiments/train_retrieval.py \
  --dataset cub \
  --loss contrastive \
  --epochs 30 \
  --batch_size 64 \
  --freeze_vit_backbone True

## Diagnostics & Visualization

After training, run the notebook to visualize embedding geometry and optimization metrics:
```bash
jupyter notebook notebooks/analysis.ipynb

Citation and Acknowledgments

Please cite our work if you use this code:

Comparing Contrastive and Triplet Loss: Variance Analysis and Optimization Behavior

¹ Vision Transformer (ViT) implementation via HuggingFace Transformers: https://huggingface.co/docs/transformers/v4.13.0/en/model_doc/vit
