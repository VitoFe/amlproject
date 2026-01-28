import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional

from .base import BaseTrainer
from ..utils.logging import log_metrics
from ..utils.early_stopping import EarlyStopping


class CentralizedTrainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        experiment_name: str = "centralized",
    ):
        """
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
                - early_stopping_patience: Epochs to wait before stopping
            device: Device for computation
            experiment_name: Name for logging
        """
        super().__init__(
            model,
            train_loader,
            val_loader,
            test_loader,
            config,
            device,
            experiment_name,
        )

        self.epochs = config.get("epochs", 50)
        self.lr = config.get("learning_rate", 0.001)
        self.momentum = config.get("momentum", 0.9)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.scheduler_type = config.get("scheduler", "cosine")
        self.warmup_epochs = config.get("warmup_epochs", 5)
        self.early_stopping_patience = config.get("early_stopping_patience", 10)

        self.optimizer = self._create_optimizer(
            lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

        self.scheduler = self._create_scheduler(
            self.optimizer,
            scheduler_type=self.scheduler_type,
            total_epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        self.early_stopping: Optional[EarlyStopping] = None
        if self.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.early_stopping_patience,
                min_delta=0.001,
                mode="max",
                verbose=True,
            )

    def train(
        self,
        resume: bool = True,
        save_every: int = 10,
        use_wandb: bool = False,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Args:
            resume: Whether to try resuming from checkpoint
            save_every: Save checkpoint every N epochs
            use_wandb: Whether to log to W&B
            early_stopping_patience: Override patience
        """
        if early_stopping_patience is not None:
            if early_stopping_patience > 0:
                self.early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=0.001,
                    mode="max",
                    verbose=True,
                )
            else:
                self.early_stopping = None

        if resume:
            self.try_resume(self.optimizer, self.scheduler)

        self.logger.info(f"Starting centralized training for {self.epochs} epochs")
        self.logger.info(
            f"Config: LR={self.lr}, Momentum={self.momentum}, "
            f"Weight Decay={self.weight_decay}, Scheduler={self.scheduler_type}"
        )
        if self.early_stopping:
            self.logger.info(
                f"Early stopping enabled with patience={self.early_stopping.patience}"
            )

        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            train_loss, train_acc = self._train_epoch(self.optimizer)
            val_metrics = self.evaluate_val()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["train_accuracy"].append(train_acc)
            self.metrics_history["val_loss"].append(val_metrics["loss"])
            self.metrics_history["val_accuracy"].append(val_metrics["accuracy"])
            self.metrics_history["learning_rate"].append(current_lr)

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "lr": current_lr,
            }
            log_metrics(self.logger, metrics, epoch, use_wandb=use_wandb)

            if (epoch + 1) % save_every == 0 or val_metrics[
                "accuracy"
            ] > self.best_val_accuracy:
                self.save_checkpoint(
                    self.optimizer,
                    self.scheduler,
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    },
                )

            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.logger.info(
                    f"New best validation accuracy: {self.best_val_accuracy:.4f}"
                )

            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["accuracy"], epoch):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}!")
                    self.logger.info(
                        f"Best validation accuracy was {self.early_stopping.best_score:.4f} at epoch {self.early_stopping.best_epoch}"
                    )
                    break

        self.logger.info("Training complete. Evaluating on test set...")

        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.load_checkpoint(best_checkpoint)

        test_metrics = self.evaluate_test()
        self.metrics_history["test_loss"].append(test_metrics["loss"])
        self.metrics_history["test_accuracy"].append(test_metrics["accuracy"])

        self.logger.info(f"Final Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")

        return test_metrics

    def hyperparameter_search(
        self,
        lr_range: List[float] = [1e-4, 5e-4, 1e-3, 5e-3],
        epochs_per_trial: int = 10,
    ) -> Dict[str, Any]:
        """
        Simple grid search for learning rate
        """
        self.logger.info(f"Started hyperparameter search over LRs: {lr_range}")

        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        results = {}

        for lr in lr_range:
            self.model.load_state_dict(original_state)

            optimizer = self._create_optimizer(
                lr=lr, momentum=self.momentum, weight_decay=self.weight_decay
            )
            scheduler = self._create_scheduler(
                optimizer, self.scheduler_type, epochs_per_trial
            )

            best_val_acc = 0.0
            for epoch in range(epochs_per_trial):
                self._train_epoch(optimizer, show_progress=False)
                val_metrics = self.evaluate_val()
                best_val_acc = max(best_val_acc, val_metrics["accuracy"])
                if scheduler:
                    scheduler.step()

            results[lr] = best_val_acc
            self.logger.info(f"LR={lr}: Best Val Acc = {best_val_acc:.4f}")

        best_lr = max(results, key=results.get)
        self.logger.info(f"Best LR: {best_lr} with accuracy {results[best_lr]:.4f}")

        self.model.load_state_dict(original_state)

        return {
            "best_lr": best_lr,
            "best_accuracy": results[best_lr],
            "all_results": results,
        }
