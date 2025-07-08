import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import random
from plt_figure import *

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def synthesis_toydata(n_classes=10, samples_per_class=200, dim=128, 
                                 mode_spread=1.4, overlap_prob=0.1, outlier_frac=0.05):
    #1|0.2
    #1.5|0.1
    #*1.4｜0.2
    #1|0.01
    data = []
    labels = []
    
    # Create base centers randomly
    centers = torch.randn(n_classes, dim) * 5

    for c in range(n_classes):
        center = centers[c]

        # Create random positive definite covariance matrix with correlations
        A = torch.randn(dim, dim)
        # covariance matrix, symmetric positive definite
        cov = A @ A.T  
        
        # Cholesky decomposition for sampling correlated noise
        L = torch.linalg.cholesky(cov + torch.eye(dim) * 1e-3)
        
        n_in_class = samples_per_class
        for _ in range(n_in_class):
            noise = L @ torch.randn(dim)
            point = center + noise * mode_spread
            
            # Assign point to another class with some probability (overlap)
            if torch.rand(1).item() < overlap_prob:
                other_class = torch.randint(0, n_classes, (1,)).item()
                labels.append(other_class)
            else:
                labels.append(c)
            
            data.append(point)
        
    # Add uniform random outliers (background noise)
    n_outliers = int(len(data) * outlier_frac)
    for _ in range(n_outliers):
        outlier_point = torch.randn(dim) * 15
        data.append(outlier_point)
        labels.append(-1)  # Outlier class
    
    return torch.stack(data), torch.tensor(labels)


# data, labels = synthesis_toydata()
# data = F.normalize(data,dim=1)

# tensors = {
#     'data': data,
#     'label': labels
# }

# torch.save(tensors, 'synthesis_toydata.pt')

# loaded_tensors = torch.load('synthesis_toydata.pt')
# data, labels = loaded_tensors['data'], loaded_tensors['label'] 