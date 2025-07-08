import torch
import torch.nn as nn
import torch.nn.functional as F

# Contrastive loss
def contrastive_loss(x1, x2, label, margin=1.0):
    dist = F.pairwise_distance(x1, x2)
    return (label * dist.pow(2) + (1 - label) * F.relu(margin - dist).pow(2)).mean()

# Triplet loss
def triplet_loss(anchor, pos, neg, margin=1.0):
    d_pos = F.pairwise_distance(anchor, pos)
    d_neg = F.pairwise_distance(anchor, neg)
    return F.relu(d_pos - d_neg + margin).mean()

