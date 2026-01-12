import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm


class AverageMeter:
    def __init__(self, name: str = 'metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1,)
) -> Dict[str, float]:
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model logits [B, C]
        targets: Ground truth labels [B]
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        Dict with accuracy for each k
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    result = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result[f'top{k}_accuracy'] = (correct_k / batch_size).item()
    
    return result


def compute_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> float:
    """
    Compute average loss over a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        criterion: Loss function
        device: Device
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)
    
    return total_loss / n_samples if n_samples > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
    show_progress: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device
        show_progress: Whether to show progress bar
    
    Returns:
        Dict with loss and accuracy metrics
    """
    model.eval()
    
    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('accuracy')
    
    iterator = tqdm(dataloader, desc="Evaluating") if show_progress else dataloader
    
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).float().mean()
        
        batch_size = inputs.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy.item(), batch_size)
    
    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg,
        'top1_accuracy': acc_meter.avg
    }


def compute_per_class_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = 'cuda'
) -> Dict[int, float]:
    """
    Compute per-class accuracy.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        num_classes: Number of classes
        device: Device
    
    Returns:
        Dict mapping class_id to accuracy
    """
    model.eval()
    
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for c in range(num_classes):
                mask = targets == c
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += ((predicted == targets) & mask).sum().item()
    
    per_class_acc = {}
    for c in range(num_classes):
        if total_per_class[c] > 0:
            per_class_acc[c] = correct_per_class[c].item() / total_per_class[c].item()
        else:
            per_class_acc[c] = 0.0
    
    return per_class_acc
