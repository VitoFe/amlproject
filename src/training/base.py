"""
Base Trainer Class

Provides the foundation for all training paradigms with common
functionality for training loops, evaluation, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from typing import Dict, Optional, Tuple, List, Any
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..utils.metrics import AverageMeter, evaluate
from ..utils.checkpointing import CheckpointManager, load_checkpoint
from ..utils.logging import get_logger


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.
    
    Provides common functionality for:
    - Model initialization
    - Optimizer and scheduler setup
    - Checkpointing
    - Metrics tracking
    - Evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        experiment_name: str = 'experiment'
    ):
        """
        Initialize base trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device for computation
            experiment_name: Name for logging and checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Initialize criterion with optional label smoothing for regularization
        label_smoothing = config.get('label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Initialize logger
        self.logger = get_logger(experiment_name)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            name=experiment_name,
            metric_name='val_accuracy',
            mode='max'
        )
        
        # Metrics history
        self.metrics_history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'learning_rate': []
        }
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
    
    def _create_optimizer(
        self,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4
    ) -> torch.optim.Optimizer:
        """
        Create optimizer for trainable parameters.
        
        Args:
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay
        
        Returns:
            Optimizer
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        total_epochs: int = 50,
        warmup_epochs: int = 5
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type of scheduler ('cosine', 'step', 'none')
            total_epochs: Total training epochs
            warmup_epochs: Warmup epochs (not used in basic schedulers)
        
        Returns:
            LR scheduler or None
        """
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
        elif scheduler_type == 'step':
            return StepLR(optimizer, step_size=total_epochs // 3, gamma=0.1)
        else:
            return None
    
    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        show_progress: bool = True
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('accuracy')
        
        iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") if show_progress else self.train_loader
        
        for inputs, targets in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == targets).float().mean()
            
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy.item(), batch_size)
            
            if show_progress:
                iterator.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.4f}'
                })
        
        return loss_meter.avg, acc_meter.avg
    
    def _evaluate(
        self,
        dataloader: DataLoader,
        show_progress: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            show_progress: Whether to show progress bar
        
        Returns:
            Dict with loss and accuracy
        """
        return evaluate(self.model, dataloader, self.criterion, self.device, show_progress)
    
    def evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set."""
        return self._evaluate(self.test_loader, show_progress=True)
    
    def evaluate_val(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        return self._evaluate(self.val_loader)
    
    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float]
    ) -> str:
        """
        Save training checkpoint.
        
        Args:
            optimizer: Current optimizer
            scheduler: Current scheduler
            metrics: Current metrics
        
        Returns:
            Path to saved checkpoint
        """
        return self.checkpoint_manager.save(
            self.model, optimizer, scheduler,
            self.current_epoch, metrics, self.config
        )
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
        
        Returns:
            Checkpoint metadata
        """
        metadata = load_checkpoint(
            checkpoint_path, self.model, optimizer, scheduler, self.device
        )
        self.current_epoch = metadata['epoch']
        return metadata
    
    def try_resume(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> bool:
        """
        Try to resume from latest checkpoint.
        
        Args:
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
        
        Returns:
            True if resumed, False otherwise
        """
        latest = self.checkpoint_manager.get_latest_checkpoint()
        if latest:
            self.logger.info(f"Resuming from checkpoint: {latest}")
            self.load_checkpoint(latest, optimizer, scheduler)
            return True
        return False
    
    @abstractmethod
    def train(self, **kwargs) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Final metrics
        """
        pass
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get training metrics history."""
        return self.metrics_history
