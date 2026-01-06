"""
Early Stopping Utility

Provides early stopping functionality to prevent overfitting
by monitoring validation metrics during training.
"""

from typing import Optional


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    
    Attributes:
        patience: Number of epochs with no improvement after which training stops
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
        best_score: Best metric value seen
        counter: Number of epochs since last improvement
        early_stop: Whether to stop training
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change in metric to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value (e.g., validation accuracy or loss)
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping! Best was {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement over best."""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:  # mode == 'min'
            return score < self.best_score - self.min_delta
    
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
