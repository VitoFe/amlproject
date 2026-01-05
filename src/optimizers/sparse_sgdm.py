"""
Sparse SGDM Optimizer

Implementation of SGD with Momentum that supports gradient masking
for sparse fine-tuning based on parameter sensitivity.

Reference: Iurada et al., "Efficient Model Editing with Task-Localized 
Sparse Fine-tuning". ICLR 2025.
"""

import torch
from torch.optim import Optimizer
from typing import Dict, Optional, Iterable


class SparseSGDM(Optimizer):
    """
    Sparse SGD with Momentum optimizer.
    
    Extends standard SGDM to support gradient masking, where updates
    are zeroed out for parameters whose corresponding mask entry is 0.
    
    This enables task-localized sparse fine-tuning by only updating
    a subset of parameters (e.g., least-sensitive weights).
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        gradient_masks: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize SparseSGDM optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            dampening: Dampening for momentum
            nesterov: Whether to use Nesterov momentum
            gradient_masks: Dict mapping param name to binary mask tensor
                           (1 = update, 0 = freeze)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")
        
        super().__init__(params, defaults)
        
        self.gradient_masks = gradient_masks or {}
        self._param_to_name = {}
    
    def set_gradient_masks(self, masks: Dict[str, torch.Tensor]) -> None:
        """
        Set gradient masks for sparse updates.
        
        Args:
            masks: Dict mapping parameter names to binary mask tensors
        """
        self.gradient_masks = masks
    
    def set_param_names(self, param_names: Dict[int, str]) -> None:
        """
        Set mapping from parameter id to name.
        
        Args:
            param_names: Dict mapping id(param) to parameter name
        """
        self._param_to_name = param_names
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with gradient masking.
        
        This implements the SGDM update rule with optional gradient masking:
        
        1. Apply weight decay: grad = grad + weight_decay * param
        2. Update momentum buffer: m = momentum * m + (1 - dampening) * grad
        3. Apply Nesterov (optional): update = grad + momentum * m
        4. Apply gradient mask: update = update * mask
        5. Update parameters: param = param - lr * update
        
        Args:
            closure: Optional closure for loss computation
        
        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Get parameter name for mask lookup
                param_id = id(p)
                param_name = self._param_to_name.get(param_id)
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Apply gradient mask (sparse update)
                if param_name is not None and param_name in self.gradient_masks:
                    mask = self.gradient_masks[param_name]
                    if mask.shape == d_p.shape:
                        d_p = d_p * mask
                
                # Update parameter
                p.add_(d_p, alpha=-lr)
        
        return loss
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """
        Compute sparsity statistics of gradient masks.
        
        Returns:
            Dict with sparsity statistics
        """
        if not self.gradient_masks:
            return {'total_sparsity': 0.0}
        
        total_params = 0
        masked_params = 0
        
        for name, mask in self.gradient_masks.items():
            total_params += mask.numel()
            masked_params += (mask == 0).sum().item()
        
        return {
            'total_sparsity': masked_params / total_params if total_params > 0 else 0.0,
            'total_params': total_params,
            'masked_params': masked_params,
            'trainable_params': total_params - masked_params
        }


def create_sparse_optimizer(
    model: torch.nn.Module,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    gradient_masks: Optional[Dict[str, torch.Tensor]] = None
) -> SparseSGDM:
    """
    Factory function to create SparseSGDM optimizer for a model.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay
        gradient_masks: Optional gradient masks
    
    Returns:
        Configured SparseSGDM optimizer
    """
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = SparseSGDM(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_masks=gradient_masks
    )
    
    # Set parameter name mapping
    param_names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    optimizer.set_param_names(param_names)
    
    return optimizer
