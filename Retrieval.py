# Required Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm # For progress bars
import os
from PIL import Image # For CUB-200 image loading

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Transforms ---
# Common transform for MNIST/CIFAR-10
mnist_cifar_transform = transforms.Compose([
    transforms.ToTensor()
])

# Specific transform for CUB-200 (RGB images, varying sizes)
# We'll resize to 128x128 for consistency with the CNN input
cub_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to a fixed size
    transforms.ToTensor(),
    # Optional: Normalize for pre-trained models, but not strictly necessary for simple CNN
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset Loading ---
# CIFAR-10 (existing)
print("Loading CIFAR-10 datasets...")
cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=mnist_cifar_transform, download=True)
cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=mnist_cifar_transform, download=True)
print("CIFAR-10 datasets loaded.")

# --- Custom CUB-200-2011 Dataset Class ---
class CUB200Dataset(Dataset):
    """
    Custom Dataset for CUB-200-2011.
    Assumes the dataset is extracted to a structure like:
    ./data/CUB_200_2011/
    ├── images/
    ├── images.txt
    ├── image_class_labels.txt
    └── train_test_split.txt
    """
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = train

        # Paths to annotation files
        images_file = os.path.join(root_dir, 'images.txt')
        labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        splits_file = os.path.join(root_dir, 'train_test_split.txt')

        # Read image paths and IDs
        self.image_paths = {}
        with open(images_file, 'r') as f:
            for line in f:
                img_id, path = line.strip().split()
                self.image_paths[int(img_id)] = os.path.join(root_dir, 'images', path)

        # Read image labels
        self.image_labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                # CUB labels are 1-indexed, convert to 0-indexed
                self.image_labels[int(img_id)] = int(label) - 1

        # Read train/test split and filter data
        self.data = [] # List of (image_path, label) tuples
        with open(splits_file, 'r') as f:
            for line in f:
                img_id, is_training_img = line.strip().split()
                img_id = int(img_id)
                is_training_img = int(is_training_img)

                if (self.is_train and is_training_img == 1) or \
                   (not self.is_train and is_training_img == 0):
                    self.data.append((self.image_paths[img_id], self.image_labels[img_id]))

        print(f"CUB-200-2011 {'Training' if train else 'Test'} set loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB') # CUB images are RGB

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Custom Dataset Classes for Pair and Triplet Generation ---
# These classes are generic and will work with any dataset that returns (image, label)
class PairDataset(Dataset):
    """
    Dataset to generate pairs (anchor, positive/negative) on-the-fly.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Extract all targets once to create label_to_indices map
        all_targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        self.labels_set = list(set(all_targets))
        self.label_to_indices = {label: np.where(np.array(all_targets) == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]

        # Determine if it's a positive or negative pair
        should_be_same = random.randint(0, 1) # 0 for positive, 1 for negative

        if should_be_same:
            # Positive pair
            positive_index = index
            # Loop until a different positive sample is found
            while True:
                candidate_index = random.choice(self.label_to_indices[anchor_label])
                if candidate_index != index:
                    positive_index = candidate_index
                    break
            positive_img, _ = self.dataset[positive_index]
            pair_label = 0 # 0 for similar
        else:
            # Negative pair
            negative_label = random.choice([l for l in self.labels_set if l != anchor_label])
            negative_index = random.choice(self.label_to_indices[negative_label])
            negative_img, _ = self.dataset[negative_index]
            pair_label = 1 # 1 for dissimilar

        return anchor_img, positive_img if should_be_same else negative_img, torch.tensor(pair_label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

class TripletDataset(Dataset):
    """
    Dataset to generate triplets (anchor, positive, negative) on-the-fly.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Extract all targets once to create label_to_indices map
        all_targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        self.labels_set = list(set(all_targets))
        self.label_to_indices = {label: np.where(np.array(all_targets) == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        anchor_img, anchor_label = self.dataset[index]

        # Get positive sample (same class as anchor)
        positive_index = index
        while True: # Ensure positive is not the anchor itself
            candidate_index = random.choice(self.label_to_indices[anchor_label])
            if candidate_index != index:
                positive_index = candidate_index
                break
        positive_img, _ = self.dataset[positive_index]

        # Get negative sample (different class from anchor)
        negative_label = random.choice([l for l in self.labels_set if l != anchor_label])
        negative_index = random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self.dataset[negative_index]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset)

# --- Feature Extractor Model ---
class ConvNetEmbedder(nn.Module):
    """
    A convolutional neural network for learning embeddings.
    Dynamically calculates the flattened size based on input dimensions.
    """
    def __init__(self, input_channels: int, input_height: int, input_width: int, embedding_dim: int = 64):
        """
        Initializes the ConvNetEmbedder.

        Args:
            input_channels (int): Number of input channels (e.g., 1 for MNIST, 3 for CIFAR-10/CUB).
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
            embedding_dim (int): Desired dimension of the output embeddings.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1), # Same padding
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), # Same padding
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), # Added a layer for CUB complexity
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Compute correct flattened size using actual height & width
        # Pass a dummy tensor through the feature extractor to get the output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            out = self.features(dummy)
            self.flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256), # Increased size
            nn.ReLU(),
            nn.Linear(256, embedding_dim), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The L2-normalized embedding.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return F.normalize(x, dim=-1) 
    
class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    Based on Hadsell, Chopra, LeCun (2006) 'Dimensionality Reduction by Learning an Invariant Mapping'
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Calculates the contrastive loss.

        Args:
            output1 (torch.Tensor): Embedding of the first input.
            output2 (torch.Tensor): Embedding of the second input.
            label (torch.Tensor): 0 if similar, 1 if dissimilar.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Loss = (1-Y) * D^2 + Y * max(0, margin - D)^2
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2.0)

# --- Training Loops ---
def train_contrastive(model: nn.Module, train_dataset: Dataset, epochs: int = 50, batch_size: int = 64):
    """
    Trains the model using Contrastive Loss.

    Args:
        model (nn.Module): The embedding model to train.
        train_dataset (Dataset): The dataset to create pairs from.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        nn.Module: The trained model.
    """
    contrastive_dataset = PairDataset(train_dataset)
    # Using num_workers=0 for custom datasets with random operations to avoid multiprocessing issues
    train_loader = DataLoader(contrastive_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = ContrastiveLoss()
    model.train()

    print(f"\n--- Training with Contrastive Loss ---")
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, label in tqdm(train_loader, desc=f"Contrastive Epoch {epoch+1}/{epochs}"):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            out1, out2 = model(x1), model(x2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Contrastive Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}')
    return model

def train_triplet(model: nn.Module, train_dataset: Dataset, epochs: int = 50, batch_size: int = 64):
    """
    Trains the model using Triplet Loss.

    Args:
        model (nn.Module): The embedding model to train.
        train_dataset (Dataset): The dataset to create triplets from.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        nn.Module: The trained model.
    """
    triplet_dataset = TripletDataset(train_dataset)
    # Using num_workers=0 for custom datasets with random operations to avoid multiprocessing issues
    train_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    print(f"\n--- Training with Triplet Loss ---")
    for epoch in range(epochs):
        total_loss = 0
        for a, p, n in tqdm(train_loader, desc=f"Triplet Epoch {epoch+1}/{epochs}"):
            a, p, n = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()
            out_a, out_p, out_n = model(a), model(p), model(n)
            loss = triplet_loss_fn(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Triplet Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}')
    return model

# --- Evaluation Functions ---
def embed_data(model: nn.Module, loader: DataLoader):
    """Helper function to extract embeddings and labels."""
    model.eval() # Set model to evaluation mode
    embs, labs = [], []
    with torch.no_grad(): # Disable gradient calculations for inference
        for x, y in tqdm(loader, desc="Extracting embeddings"):
            x = x.to(device)
            emb = model(x)
            embs.append(emb.cpu())
            labs.append(y)
    return torch.cat(embs), torch.cat(labs)

def evaluate_knn(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
    """
    Evaluates the model's embeddings using k-NN classification.

    Args:
        model (nn.Module): The trained embedding model.
        train_loader (DataLoader): DataLoader for the training set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    print("\n--- Evaluating k-NN classification accuracy ---")
    X_train, y_train = embed_data(model, train_loader)
    X_test, y_test = embed_data(model, test_loader)

    print("Fitting k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train.numpy(), y_train.numpy())

    print("Predicting with k-NN and calculating accuracy...")
    acc = knn.score(X_test.numpy(), y_test.numpy())
    print(f'k-NN classification accuracy: {acc:.4f}')

def evaluate_retrieval(model: nn.Module, test_loader: DataLoader):
    """
    Evaluates the model's embeddings for image retrieval using Recall@K.

    Args:
        model (nn.Module): The trained embedding model.
        test_loader (DataLoader): DataLoader for the test set (used as both query and gallery).
    """
    print("\n--- Evaluating Image Retrieval (Recall@K) ---")

    # Extract embeddings for the entire test set (which acts as query and gallery)
    query_embeddings, query_labels = embed_data(model, test_loader)
    gallery_embeddings, gallery_labels = query_embeddings, query_labels 
    
    num_queries = query_embeddings.size(0)
    recalls = {1: 0, 5: 0, 10: 0} # Initialize recall counts for K values

    # Iterate through each query
    for i in tqdm(range(num_queries), desc="Calculating Recall@K"):
        query_emb = query_embeddings[i].unsqueeze(0) 
        query_label = query_labels[i].item()

        similarities = torch.matmul(query_emb, gallery_embeddings.T).squeeze(0)

        similarities[i] = -float('inf')

        # Get the indices of the top K most similar items
        # torch.topk returns values and indices, we need indices
        top_k_indices = torch.topk(similarities, k=max(recalls.keys()), largest=True).indices

        # Check if any of the top K retrieved items belong to the same class as the query
        retrieved_labels = gallery_labels[top_k_indices].numpy()

        for k_val in recalls.keys():
            if query_label in retrieved_labels[:k_val]:
                recalls[k_val] += 1

    # Calculate recall percentages
    for k_val in recalls.keys():
        recalls[k_val] = recalls[k_val] / num_queries
        print(f"Recall@{k_val}: {recalls[k_val]:.4f}")


def run_experiments(dataset_name: str = 'mnist', loss_type: str = 'contrastive', epochs: int = 50, batch_size: int = 64):
    """
    Runs a complete experiment for a given dataset and loss type.

    Args:
        dataset_name (str): 'mnist', 'cifar', or 'cub'.
        loss_type (str): 'contrastive' or 'triplet'.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training and evaluation DataLoaders.
    """
    print(f"\n===== Running Experiment: Dataset={dataset_name.upper()}, Loss Type={loss_type.capitalize()} =====")

    train_set = None
    test_set = None
    input_channels = 0
    input_height = 0
    input_width = 0

    if dataset_name == 'mnist':
        train_set = mnist_train_dataset
        test_set = mnist_test_dataset
        dummy_x, _ = train_set[0]
        channels, height, width = dummy_x.size(0), dummy_x.size(1), dummy_x.size(2)
    elif dataset_name == 'cifar':
        train_set = cifar10_train_dataset
        test_set = cifar10_test_dataset
        dummy_x, _ = train_set[0]
        channels, height, width = dummy_x.size(0), dummy_x.size(1), dummy_x.size(2)
    elif dataset_name == 'cub':
        cub_root_dir = '/home/zeng/Documents/code/datasets/cub2011/CUB_200_2011/CUB_200_2011' # Ensure this path is correct
        if not os.path.exists(os.path.join(cub_root_dir, 'images')):
            print(f"Error: CUB-200-2011 dataset not found at '{cub_root_dir}'.")
            print("Please download and extract 'CUB_200_2011.tgz' into the './data/' directory.")
            print("Skipping CUB-200-2011 experiment.")
            return

        train_set = CUB200Dataset(cub_root_dir, train=True, transform=cub_transform)
        test_set = CUB200Dataset(cub_root_dir, train=False, transform=cub_transform)
        channels = 3 
        height = 128
        width = 128
    else:
        raise ValueError("dataset_name must be 'mnist', 'cifar', or 'cub'")

    # Initialize model
    model = ConvNetEmbedder(input_channels=channels, input_height=height, input_width=width).to(device)

    # Train the model based on loss type
    if loss_type == 'contrastive':
        model = train_contrastive(model, train_set, epochs=epochs, batch_size=batch_size)
    elif loss_type == 'triplet':
        model = train_triplet(model, train_set, epochs=epochs, batch_size=batch_size)
    else:
        raise ValueError("loss_type must be 'contrastive' or 'triplet'")

    # Prepare DataLoaders for evaluation
    train_loader_eval = DataLoader(train_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader_eval = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    # Evaluate using k-NN (optional for CUB, but included for consistency)
    evaluate_knn(model, train_loader_eval, test_loader_eval)

    # Evaluate using Image Retrieval (Recall@K)
    evaluate_retrieval(model, test_loader_eval)

    print(f"===== Experiment Finished: Dataset={dataset_name.upper()}, Loss Type={loss_type.capitalize()} =====")



if __name__ == '__main__':

    common_epochs_mnist_cifar = 30
    common_epochs_cub = 30         
    common_batch_size = 64

    print("Starting experiments...")

    # MNIST Experiments (classification with k-NN and retrieval)
#     run_experiments('mnist', 'contrastive', epochs=common_epochs_mnist_cifar, batch_size=common_batch_size)
#     run_experiments('mnist', 'triplet', epochs=common_epochs_mnist_cifar, batch_size=common_batch_size)

    # CIFAR-10 Experiments (classification with k-NN and retrieval)
    run_experiments('cifar', 'contrastive', epochs=common_epochs_mnist_cifar, batch_size=common_batch_size)
    run_experiments('cifar', 'triplet', epochs=common_epochs_mnist_cifar, batch_size=common_batch_size)

    # CUB-200-2011 Experiments (Focus on retrieval)
#     run_experiments('cub', 'contrastive', epochs=common_epochs_cub, batch_size=common_batch_size)
#     run_experiments('cub', 'triplet', epochs=common_epochs_cub, batch_size=common_batch_size)

    print("All experiments completed.")