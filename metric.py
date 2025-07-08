import numpy as np
import torch.nn.functional as F
import torch

def compute_intra_class_variance(embeddings, labels):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    unique_labels = labels.unique()
    variances = {}
    for c in unique_labels:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        class_embeds = embeddings[idx]
        centroid = class_embeds.mean(dim=0)
        var = ((class_embeds - centroid).pow(2).sum(dim=1)).mean().item()
        variances[c.item()] = var
    return variances

def compute_inter_class_variance(embeddings, labels):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    unique_labels = labels.unique()
    centroids = []

    for c in unique_labels:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        class_embeds = embeddings[idx]
        centroid = class_embeds.mean(dim=0)
        centroids.append(centroid)

    centroids = torch.stack(centroids)  # shape: [num_classes, embedding_dim]

    # Compute pairwise distances between all centroids
    dists = torch.cdist(centroids, centroids, p=2)
    num_classes = len(unique_labels)

    # We only care about upper-triangular (excluding diagonal)
    triu_indices = torch.triu_indices(num_classes, num_classes, offset=1)
    upper_dists = dists[triu_indices[0], triu_indices[1]]

    inter_mean = upper_dists.mean().item()
    inter_std = upper_dists.std().item()
    return inter_mean, inter_std
