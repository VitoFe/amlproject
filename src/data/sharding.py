"""
Data Sharding Strategies for Federated Learning

This module implements IID and non-IID data sharding to simulate
statistical heterogeneity in federated learning scenarios.
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset


class ShardingStrategy(Enum):
    """Enumeration of data sharding strategies."""
    IID = "iid"
    NON_IID = "non_iid"


def create_client_splits(
    dataset: Dataset,
    num_clients: int,
    strategy: ShardingStrategy = ShardingStrategy.IID,
    nc: int = 10,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Split dataset into disjoint subsets for federated learning clients.
    
    This function creates K disjoint partitions of the training data,
    simulating the federated setting where each client has its own private data.
    
    Args:
        dataset: Full training dataset with .targets attribute
        num_clients: Number of clients (K)
        strategy: Sharding strategy (IID or NON_IID)
        nc: Number of classes per client for NON_IID sharding
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping client_id -> array of sample indices
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Handle Subset case
        labels = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        raise ValueError("Dataset must have 'targets' attribute")
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    if strategy == ShardingStrategy.IID:
        return _iid_sharding(indices, num_clients)
    else:
        return _non_iid_sharding(indices, labels, num_clients, nc)


def _iid_sharding(
    indices: np.ndarray,
    num_clients: int
) -> Dict[int, np.ndarray]:
    """
    IID sharding: uniformly random distribution of samples.
    
    Each client receives approximately equal number of samples,
    uniformly distributed over class labels.
    
    Args:
        indices: Array of sample indices
        num_clients: Number of clients
    
    Returns:
        Dictionary mapping client_id -> sample indices
    """
    np.random.shuffle(indices)
    client_splits = np.array_split(indices, num_clients)
    
    return {i: split for i, split in enumerate(client_splits)}


def _non_iid_sharding(
    indices: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    nc: int
) -> Dict[int, np.ndarray]:
    """
    Non-IID sharding: each client only has samples from Nc classes.
    
    This simulates statistical heterogeneity where clients have
    non-overlapping or partially overlapping label distributions.
    
    Args:
        indices: Array of sample indices
        labels: Array of corresponding labels
        num_clients: Number of clients
        nc: Number of classes per client
    
    Returns:
        Dictionary mapping client_id -> sample indices
    """
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = {c: indices[labels == c] for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
    
    # Assign classes to clients
    # Each client gets nc classes, with some classes potentially shared
    client_splits = {i: [] for i in range(num_clients)}
    
    # Calculate how many clients should receive each class
    # to ensure balanced distribution
    classes_per_client = nc
    total_class_assignments = num_clients * classes_per_client
    clients_per_class = total_class_assignments // num_classes
    
    # Create class assignments for each client
    class_assignment = []
    for c in range(num_classes):
        class_assignment.extend([c] * clients_per_class)
    
    # Handle remainder
    remaining = total_class_assignments - len(class_assignment)
    if remaining > 0:
        class_assignment.extend(np.random.choice(num_classes, remaining, replace=False).tolist())
    
    np.random.shuffle(class_assignment)
    
    # Assign classes to clients
    client_classes = {i: [] for i in range(num_clients)}
    for i, c in enumerate(class_assignment):
        client_id = i % num_clients
        if c not in client_classes[client_id]:
            client_classes[client_id].append(c)
    
    # Ensure each client has exactly nc classes
    for client_id in range(num_clients):
        while len(client_classes[client_id]) < nc:
            available_classes = [c for c in range(num_classes) if c not in client_classes[client_id]]
            if available_classes:
                client_classes[client_id].append(np.random.choice(available_classes))
            else:
                break
        client_classes[client_id] = client_classes[client_id][:nc]
    
    # Distribute samples to clients
    # Track position within each class
    class_positions = {c: 0 for c in range(num_classes)}
    
    # Count how many clients need each class
    class_client_count = {c: 0 for c in range(num_classes)}
    for client_id in range(num_clients):
        for c in client_classes[client_id]:
            class_client_count[c] += 1
    
    # Calculate samples per client per class
    samples_per_client_class = {}
    for c in range(num_classes):
        if class_client_count[c] > 0:
            samples_per_client_class[c] = len(class_indices[c]) // class_client_count[c]
    
    # Assign samples
    for client_id in range(num_clients):
        client_indices = []
        for c in client_classes[client_id]:
            start = class_positions[c]
            end = start + samples_per_client_class.get(c, 0)
            client_indices.extend(class_indices[c][start:end].tolist())
            class_positions[c] = end
        
        client_splits[client_id] = np.array(client_indices)
    
    return client_splits


def get_sharding_stats(
    client_splits: Dict[int, np.ndarray],
    labels: np.ndarray
) -> Dict[str, any]:
    """
    Compute statistics about the data sharding.
    
    Args:
        client_splits: Dictionary mapping client_id -> sample indices
        labels: Full array of labels
    
    Returns:
        Dictionary with sharding statistics
    """
    num_clients = len(client_splits)
    samples_per_client = [len(split) for split in client_splits.values()]
    
    # Classes per client
    classes_per_client = []
    for split in client_splits.values():
        if len(split) > 0:
            client_labels = labels[split] if isinstance(split, np.ndarray) else labels[list(split)]
            classes_per_client.append(len(np.unique(client_labels)))
    
    return {
        'num_clients': num_clients,
        'total_samples': sum(samples_per_client),
        'samples_per_client_mean': np.mean(samples_per_client),
        'samples_per_client_std': np.std(samples_per_client),
        'samples_per_client_min': np.min(samples_per_client),
        'samples_per_client_max': np.max(samples_per_client),
        'classes_per_client_mean': np.mean(classes_per_client),
        'classes_per_client_std': np.std(classes_per_client),
    }
