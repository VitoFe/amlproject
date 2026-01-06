"""
CIFAR-100 Dataset Loading and Preprocessing

This module handles loading CIFAR-100 from torchvision with proper
train/validation/test splitting and data augmentation.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np


def get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    """
    Get data transforms for CIFAR-100.
    
    DINO ViT expects 224x224 images, so we resize CIFAR-100 (32x32) accordingly.
    
    Args:
        train: Whether to apply training augmentation
        img_size: Target image size (default 224 for ViT)
    
    Returns:
        Composed transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def get_cifar100_datasets(
    data_dir: str = "./data",
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Subset, Subset, datasets.CIFAR100]:
    """
    Download and prepare CIFAR-100 datasets with train/val/test split.
    
    The original CIFAR-100 has no validation split, so we create one
    from the training set. This follows best practices for hyperparameter tuning.
    
    Args:
        data_dir: Directory to store/load data
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Download datasets
    train_full = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )
    
    val_dataset_base = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=False)
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )
    
    # Create train/val split
    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), 
        [n_train, n_val],
        generator=generator
    )
    
    # Create subsets
    train_dataset = Subset(train_full, train_indices.indices)
    val_dataset = Subset(val_dataset_base, val_indices.indices)
    
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: Subset,
    val_dataset: Subset,
    test_dataset: datasets.CIFAR100,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_dataset: Training subset
        val_dataset: Validation subset  
        test_dataset: Test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_client_dataloader(
    dataset: datasets.CIFAR100,
    indices: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for a specific client's data subset.
    
    This is used in federated learning where each client has its own
    disjoint subset of the training data.
    
    Args:
        dataset: Full CIFAR-100 dataset
        indices: Indices of samples belonging to this client
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader for client's data
    """
    client_subset = Subset(dataset, indices)
    
    return DataLoader(
        client_subset,
        batch_size=min(batch_size, len(indices)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
