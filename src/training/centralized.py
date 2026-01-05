"""
Centralized Training

Standard centralized training baseline for CIFAR-100 with DINO ViT.
This serves as the performance benchmark for federated approaches.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any

from .base import BaseTrainer
from ..utils.logging import log_metrics


class CentralizedTrainer(BaseTrainer):
    """
    Trainer for centralized (non-federated) training.
    
    This is the standard training procedure where all data is
    available on a single machine. Used as a benchmark.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        experiment_name: str = 'centralized'
    ):
        """
        Initialize centralized trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration with keys:
                - epochs: Number of training epochs
                - learning_rate: Initial learning rate
                - momentum: SGD momentum
                - weight_decay: L2 regularization
                - scheduler: LR scheduler type
                - warmup_epochs: Warmup epochs
            device: Device for computation
            experiment_name: Name for logging
        """
        super().__init__(
            model, train_loader, val_loader, test_loader,
            config, device, experiment_name
        )
        
        # Extract config
        self.epochs = config.get('epochs', 50)
        self.lr = config.get('learning_rate', 0.001)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.scheduler_type = config.get('scheduler', 'cosine')
        self.warmup_epochs = config.get('warmup_epochs', 5)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer(
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = self._create_scheduler(
            self.optimizer,
            scheduler_type=self.scheduler_type,
            total_epochs=self.epochs,
            warmup_epochs=self.warmup_epochs
        )
    
    def train(
        self,
        resume: bool = True,
        save_every: int = 10,
        use_wandb: bool = False
    ) -> Dict[str, float]:
        """
        Train the model using standard centralized training.
        
        Args:
            resume: Whether to try resuming from checkpoint
            save_every: Save checkpoint every N epochs
            use_wandb: Whether to log to W&B
        
        Returns:
            Final test metrics
        """
        # Try to resume
        if resume:
            self.try_resume(self.optimizer, self.scheduler)
        
        self.logger.info(f"Starting centralized training for {self.epochs} epochs")
        self.logger.info(f"Config: LR={self.lr}, Momentum={self.momentum}, "
                        f"Weight Decay={self.weight_decay}, Scheduler={self.scheduler_type}")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(self.optimizer)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_val()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
            
            # Store metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['train_accuracy'].append(train_acc)
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])
            self.metrics_history['learning_rate'].append(current_lr)
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': current_lr
            }
            log_metrics(self.logger, metrics, epoch, use_wandb=use_wandb)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_metrics['accuracy'] > self.best_val_accuracy:
                self.save_checkpoint(self.optimizer, self.scheduler, {
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                })
            
            # Track best
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.logger.info(f"New best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Final evaluation
        self.logger.info("Training complete. Evaluating on test set...")
        
        # Load best model
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.load_checkpoint(best_checkpoint)
        
        test_metrics = self.evaluate_test()
        self.metrics_history['test_loss'].append(test_metrics['loss'])
        self.metrics_history['test_accuracy'].append(test_metrics['accuracy'])
        
        self.logger.info(f"Final Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return test_metrics
    
    def hyperparameter_search(
        self,
        lr_range: List[float] = [1e-4, 5e-4, 1e-3, 5e-3],
        epochs_per_trial: int = 10
    ) -> Dict[str, Any]:
        """
        Simple grid search for learning rate.
        
        Args:
            lr_range: List of learning rates to try
            epochs_per_trial: Number of epochs per trial
        
        Returns:
            Dict with best hyperparameters and results
        """
        self.logger.info(f"Starting hyperparameter search over LRs: {lr_range}")
        
        # Store original state
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        results = {}
        
        for lr in lr_range:
            # Reset model
            self.model.load_state_dict(original_state)
            
            # Create new optimizer
            optimizer = self._create_optimizer(lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
            scheduler = self._create_scheduler(optimizer, self.scheduler_type, epochs_per_trial)
            
            # Train for limited epochs
            best_val_acc = 0.0
            for epoch in range(epochs_per_trial):
                self._train_epoch(optimizer, show_progress=False)
                val_metrics = self.evaluate_val()
                best_val_acc = max(best_val_acc, val_metrics['accuracy'])
                if scheduler:
                    scheduler.step()
            
            results[lr] = best_val_acc
            self.logger.info(f"LR={lr}: Best Val Acc = {best_val_acc:.4f}")
        
        # Find best
        best_lr = max(results, key=results.get)
        self.logger.info(f"Best LR: {best_lr} with accuracy {results[best_lr]:.4f}")
        
        # Reset model
        self.model.load_state_dict(original_state)
        
        return {
            'best_lr': best_lr,
            'best_accuracy': results[best_lr],
            'all_results': results
        }
