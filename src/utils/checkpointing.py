import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    name: str = "checkpoint"
) -> str:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = checkpoint_path / f"{name}_epoch{epoch}.pt"
    torch.save(checkpoint, filepath)
    
    latest_path = checkpoint_path / f"{name}_latest.pt"
    torch.save(checkpoint, latest_path)
    
    config_path = checkpoint_path / f"{name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return str(filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
        'timestamp': checkpoint.get('timestamp')
    }


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        name: str = "checkpoint",
        max_to_keep: int = 5,
        keep_best: int = 3,
        metric_name: str = "val_accuracy",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            name: Checkpoint name prefix
            max_to_keep: Max recent checkpoints to keep
            keep_best: Number of best checkpoints to keep
            metric_name: Metric to track for best checkpoints
            mode: 'max' or 'min' for metric optimization
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metrics: List[tuple] = []  # (metric_value, path)
        self.recent_checkpoints: List[str] = []
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> str:
        filepath = save_checkpoint(
            str(self.checkpoint_dir),
            model, optimizer, scheduler,
            epoch, metrics, config, self.name
        )
        
        self.recent_checkpoints.append(filepath)
        
        metric_value = metrics.get(self.metric_name, 0)
        self.best_metrics.append((metric_value, filepath))
        
        self._cleanup()
        
        return filepath
    
    def _cleanup(self):
        reverse = self.mode == 'max'
        self.best_metrics.sort(key=lambda x: x[0], reverse=reverse)
        
        best_paths = {path for _, path in self.best_metrics[:self.keep_best]}
        recent_paths = set(self.recent_checkpoints[-self.max_to_keep:])
        
        all_paths = set(self.recent_checkpoints)
        to_keep = best_paths | recent_paths
        to_delete = all_paths - to_keep
        
        for path in to_delete:
            if os.path.exists(path):
                os.remove(path)
        
        self.best_metrics = [(m, p) for m, p in self.best_metrics if p in to_keep]
        self.recent_checkpoints = [p for p in self.recent_checkpoints if p in to_keep]
    
    def get_best_checkpoint(self) -> Optional[str]:
        if not self.best_metrics:
            return None
        return self.best_metrics[0][1]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        latest_path = self.checkpoint_dir / f"{self.name}_latest.pt"
        if latest_path.exists():
            return str(latest_path)
        return None
