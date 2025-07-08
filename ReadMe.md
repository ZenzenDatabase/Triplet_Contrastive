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

## Installation

1. Clone the repository (to be updated):
   ```bash
   git clone https://github.com/... (to be updated)
   ```
2. Install dependencies:
   pip install -r requirements.txt

3. Python==3.12

## Usage

1. Toy Data Experiments

Generate synthetic clusters and train a simple MLP:
   ```bash
  python toydata.py
  ```

2. Classification (MNIST / CIFAR‑10)
  ```bash
  python Classification.py
   ```

3. Retrieval (CIFAR‑10 / CARS196 / CUB‑200‑2011)
```bash
  python Retrieval.py
  ```
