import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from enum import Enum
from tqdm import tqdm
import numpy as np


class MaskStrategy(Enum):
    """Strategies for selecting which parameters to mask/unmask."""
    LEAST_SENSITIVE = "least_sensitive"
    MOST_SENSITIVE = "most_sensitive"
    LOWEST_MAGNITUDE = "lowest_magnitude"
    HIGHEST_MAGNITUDE = "highest_magnitude"
    RANDOM = "random"


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
    """
    model.eval()
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

    for name in fisher:
        fisher[name] /= n_samples
    
    return fisher


def compute_weight_magnitude(
    model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute absolute magnitude of model parameters.
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
    This determines which parameters should be updated (mask=1) 
    and which should be frozen (mask=0) during fine-tuning.
    """
    if strategy == MaskStrategy.RANDOM:
        return _calibrate_random_mask(model, sparsity_ratio)
    
    if strategy in [MaskStrategy.LOWEST_MAGNITUDE, MaskStrategy.HIGHEST_MAGNITUDE]:
        scores = compute_weight_magnitude(model)
    else:
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
        
        for name in scores:
            scores[name] /= num_calibration_rounds
    
    return _create_masks_from_scores(model, scores, sparsity_ratio, strategy)


def _calibrate_random_mask(
    model: nn.Module,
    sparsity_ratio: float
) -> Dict[str, torch.Tensor]:
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
    
    Returns:
        Binary masks (1 = update, 0 = freeze)
    """
    
    # flatten all scores to find global threshold

    all_scores_list = [s.view(-1).detach().cpu() for s in scores.values()]
    all_scores_np = torch.cat(all_scores_list).numpy()
    
    if strategy in [MaskStrategy.LEAST_SENSITIVE, MaskStrategy.LOWEST_MAGNITUDE]:
        # keep lowest-scoring parameters (unmask them)
        # mask the highest-scoring ones
        threshold = np.quantile(all_scores_np, 1 - sparsity_ratio)
        keep_below = True
    elif strategy in [MaskStrategy.MOST_SENSITIVE, MaskStrategy.HIGHEST_MAGNITUDE]:
        # MOST_SENSITIVE or HIGHEST_MAGNITUDE
        # keep highest-scoring parameters (unmask them)
        # mask the lowest-scoring onesarsity_ratio)
        threshold = np.quantile(all_scores_np, sparsity_ratio)
        keep_below = False
    elif strategy == MaskStrategy.RANDOM:
         return _calibrate_random_mask(model, sparsity_ratio)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy}")

    first_device = next(iter(scores.values())).device
    threshold_tensor = torch.tensor(threshold, device=first_device)

    masks = {}
    for name, p in model.named_parameters():
        if p.requires_grad and name in scores:
            score = scores[name]
            if keep_below:
                # <= threshold
                masks[name] = (score <= threshold_tensor).float()
            else:
                # >= threshold
                masks[name] = (score >= threshold_tensor).float()
    
    return masks


def get_mask_sparsity(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
