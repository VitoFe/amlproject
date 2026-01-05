"""
Fisher Information and Gradient Mask Calibration

This module implements parameter sensitivity computation using
the Fisher Information Matrix (diagonal approximation) and 
gradient mask calibration strategies.

Reference: Iurada et al., "Efficient Model Editing with Task-Localized 
Sparse Fine-tuning". ICLR 2025.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from enum import Enum
from tqdm import tqdm


class MaskStrategy(Enum):
    """Strategies for selecting which parameters to mask/unmask."""
    LEAST_SENSITIVE = "least_sensitive"  # Default: update low-sensitivity params
    MOST_SENSITIVE = "most_sensitive"    # Update high-sensitivity params
    LOWEST_MAGNITUDE = "lowest_magnitude"  # Update low-magnitude params
    HIGHEST_MAGNITUDE = "highest_magnitude"  # Update high-magnitude params
    RANDOM = "random"  # Random selection


def compute_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
    num_samples: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher Information Matrix for model parameters.
    
    The Fisher Information measures parameter sensitivity - how much
    the loss changes with respect to parameter perturbations.
    
    F_ii = E[(∂L/∂θ_i)²]
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for computing Fisher
        criterion: Loss function
        device: Device for computation
        num_samples: Maximum samples to use (None = all)
        show_progress: Whether to show progress bar
    
    Returns:
        Dict mapping parameter name to Fisher diagonal tensor
    """
    model.eval()
    
    # Initialize Fisher accumulators
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() 
              if p.requires_grad}
    
    n_samples = 0
    max_samples = num_samples or float('inf')
    
    iterator = tqdm(dataloader, desc="Computing Fisher") if show_progress else dataloader
    
    for batch_idx, (inputs, targets) in enumerate(iterator):
        if n_samples >= max_samples:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Forward pass
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Accumulate squared gradients
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[name] += (p.grad ** 2) * batch_size
        
        n_samples += batch_size
    
    # Normalize by number of samples
    for name in fisher:
        fisher[name] /= n_samples
    
    return fisher


def compute_weight_magnitude(
    model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute absolute magnitude of model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict mapping parameter name to magnitude tensor
    """
    return {name: torch.abs(p.data.clone()) 
            for name, p in model.named_parameters() 
            if p.requires_grad}


def calibrate_gradient_mask(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    sparsity_ratio: float = 0.9,
    strategy: MaskStrategy = MaskStrategy.LEAST_SENSITIVE,
    num_calibration_rounds: int = 1,
    device: str = 'cuda',
    num_fisher_samples: Optional[int] = None,
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Calibrate gradient masks using specified strategy.
    
    This determines which parameters should be updated (mask=1) 
    and which should be frozen (mask=0) during fine-tuning.
    
    For least-sensitive masking, we identify parameters with low
    Fisher information (low sensitivity) and allow updates only to those.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for calibration
        criterion: Loss function
        sparsity_ratio: Fraction of parameters to mask (freeze)
        strategy: Mask calibration strategy
        num_calibration_rounds: Number of rounds for Fisher averaging
        device: Device for computation
        num_fisher_samples: Samples per calibration round
        show_progress: Whether to show progress bar
    
    Returns:
        Dict mapping parameter name to binary mask tensor
    """
    if strategy == MaskStrategy.RANDOM:
        return _calibrate_random_mask(model, sparsity_ratio)
    
    if strategy in [MaskStrategy.LOWEST_MAGNITUDE, MaskStrategy.HIGHEST_MAGNITUDE]:
        scores = compute_weight_magnitude(model)
    else:
        # Compute Fisher over multiple rounds and average
        scores = None
        for round_idx in range(num_calibration_rounds):
            if show_progress:
                print(f"Calibration round {round_idx + 1}/{num_calibration_rounds}")
            
            fisher = compute_fisher_information(
                model, dataloader, criterion, device,
                num_samples=num_fisher_samples,
                show_progress=show_progress
            )
            
            if scores is None:
                scores = fisher
            else:
                for name in scores:
                    scores[name] += fisher[name]
        
        # Average over rounds
        for name in scores:
            scores[name] /= num_calibration_rounds
    
    # Create masks based on strategy
    return _create_masks_from_scores(model, scores, sparsity_ratio, strategy)


def _calibrate_random_mask(
    model: nn.Module,
    sparsity_ratio: float
) -> Dict[str, torch.Tensor]:
    """
    Create random gradient masks.
    
    Args:
        model: PyTorch model
        sparsity_ratio: Fraction of parameters to mask
    
    Returns:
        Random binary masks
    """
    masks = {}
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            mask = torch.ones_like(p)
            n_mask = int(p.numel() * sparsity_ratio)
            flat_mask = mask.view(-1)
            indices = torch.randperm(p.numel())[:n_mask]
            flat_mask[indices] = 0
            masks[name] = mask
    
    return masks


def _create_masks_from_scores(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    sparsity_ratio: float,
    strategy: MaskStrategy
) -> Dict[str, torch.Tensor]:
    """
    Create binary masks based on parameter scores.
    
    Args:
        model: PyTorch model
        scores: Dict of parameter scores (Fisher or magnitude)
        sparsity_ratio: Fraction of parameters to mask
        strategy: Mask strategy
    
    Returns:
        Binary masks (1 = update, 0 = freeze)
    """
    # Flatten all scores to find global threshold
    all_scores = torch.cat([s.flatten() for s in scores.values()])
    
    # Determine threshold based on strategy
    if strategy in [MaskStrategy.LEAST_SENSITIVE, MaskStrategy.LOWEST_MAGNITUDE]:
        # Keep lowest-scoring parameters (unmask them)
        # Mask the highest-scoring ones
        threshold = torch.quantile(all_scores, 1 - sparsity_ratio)
        keep_below = True
    else:  # MOST_SENSITIVE or HIGHEST_MAGNITUDE
        # Keep highest-scoring parameters (unmask them)
        # Mask the lowest-scoring ones
        threshold = torch.quantile(all_scores, sparsity_ratio)
        keep_below = False
    
    masks = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            if keep_below:
                masks[name] = (scores[name] <= threshold).float()
            else:
                masks[name] = (scores[name] >= threshold).float()
    
    return masks


def get_mask_sparsity(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute sparsity statistics for gradient masks.
    
    Args:
        masks: Dict of gradient masks
    
    Returns:
        Dict with sparsity statistics
    """
    total_params = 0
    masked_params = 0
    per_layer = {}
    
    for name, mask in masks.items():
        layer_total = mask.numel()
        layer_masked = (mask == 0).sum().item()
        total_params += layer_total
        masked_params += layer_masked
        per_layer[name] = layer_masked / layer_total
    
    return {
        'global_sparsity': masked_params / total_params if total_params > 0 else 0.0,
        'total_params': total_params,
        'masked_params': masked_params,
        'trainable_params': total_params - masked_params,
        'per_layer_sparsity': per_layer
    }


def merge_masks(
    masks_list: list,
    mode: str = 'union'
) -> Dict[str, torch.Tensor]:
    """
    Merge multiple gradient masks.
    
    Args:
        masks_list: List of mask dicts
        mode: 'union' (OR) or 'intersection' (AND)
    
    Returns:
        Merged masks
    """
    if not masks_list:
        return {}
    
    result = {name: mask.clone() for name, mask in masks_list[0].items()}
    
    for masks in masks_list[1:]:
        for name in result:
            if name in masks:
                if mode == 'union':
                    result[name] = torch.maximum(result[name], masks[name])
                else:  # intersection
                    result[name] = torch.minimum(result[name], masks[name])
    
    return result
