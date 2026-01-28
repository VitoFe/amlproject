"""
Federated Sparse Fine-tuning Trainer

Combines Federated Learning with Task Arithmetic techniques:
sparse fine-tuning using gradient masks based on parameter sensitivity.

Reference: Iurada et al., "Efficient Model Editing with Task-Localized
Sparse Fine-tuning". ICLR 2025.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, Optional, List, Any
from copy import deepcopy

from .federated import FederatedTrainer, _get_trainable_state, _set_trainable_state
from ..optimizers.sparse_sgdm import create_sparse_optimizer
from ..optimizers.fisher import calibrate_gradient_mask, MaskStrategy, get_mask_sparsity


class FederatedSparseTrainer(FederatedTrainer):
    """
    Federated Learning with Sparse Fine-tuning.

    Extends FedAvg with gradient masking based on parameter sensitivity,
    allowing clients to update only a subset of parameters chosen to
    minimize interference during aggregation.

    - Gradient mask calibration using Fisher Information
    - SparseSGDM optimizer for masked updates
    - Multiple mask strategies (least/most sensitive, magnitude, random)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda",
        experiment_name: str = "federated_sparse",
    ):
        super().__init__(
            model,
            train_dataset,
            val_loader,
            test_loader,
            config,
            device,
            experiment_name,
        )

        sparse_config = config.get("sparse", {})
        self.sparsity_ratio = sparse_config.get("sparsity_ratio", 0.9)
        self.calibration_rounds = sparse_config.get("calibration_rounds", 5)
        self.fisher_samples = sparse_config.get("fisher_samples", 512)

        strategy_str = sparse_config.get("mask_strategy", "least_sensitive")
        self.mask_strategy = MaskStrategy(strategy_str)

        self.gradient_masks: Optional[Dict[str, torch.Tensor]] = None

        self.logger.info(f"Sparse Fine-tuning Setup:")
        self.logger.info(f"  - Sparsity ratio: {self.sparsity_ratio}")
        self.logger.info(f"  - Calibration rounds: {self.calibration_rounds}")
        self.logger.info(f"  - Mask strategy: {self.mask_strategy.value}")
        self.logger.info(f"  - Fisher samples: {self.fisher_samples}")

    def calibrate_masks(
        self, calibration_loader: Optional[DataLoader] = None
    ) -> Dict[str, torch.Tensor]:
        self.logger.info(
            f"Calibrating gradient masks with strategy: {self.mask_strategy.value}"
        )

        if calibration_loader is None:
            calibration_loader = self.val_loader

        self.gradient_masks = calibrate_gradient_mask(
            self.model,
            calibration_loader,
            self.criterion,
            sparsity_ratio=self.sparsity_ratio,
            strategy=self.mask_strategy,
            num_calibration_rounds=self.calibration_rounds,
            device=self.device,
            num_fisher_samples=self.fisher_samples,
            show_progress=True,
        )

        sparsity_stats = get_mask_sparsity(self.gradient_masks)
        self.logger.info(f"Mask calibration complete:")
        self.logger.info(
            f"  - Global sparsity: {sparsity_stats['global_sparsity']:.4f}"
        )
        self.logger.info(
            f"  - Trainable params: {sparsity_stats['trainable_params']:,}"
        )
        self.logger.info(f"  - Masked params: {sparsity_stats['masked_params']:,}")

        return self.gradient_masks

    def _client_update(
        self, client_id: int, global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self.gradient_masks is None:
            raise RuntimeError(
                "Gradient masks not calibrated.. call calibrate_masks() first!"
            )

        # Fast: load only trainable params
        _set_trainable_state(self.model, global_state)
        self.model.train()

        optimizer = create_sparse_optimizer(
            self.model,
            lr=self.client_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            gradient_masks=self.gradient_masks,
        )

        client_loader = self._get_client_dataloader(client_id)

        step = 0
        while step < self.local_steps:
            for inputs, targets in client_loader:
                if step >= self.local_steps:
                    break

                inputs, targets = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )

                optimizer.zero_grad(set_to_none=True)

                # Use AMP if enabled (inherited from parent)
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()
                    # SparseSGDM handles masking in step(), AMP scaler works normally
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                step += 1

        # Fast: return only trainable params
        return _get_trainable_state(self.model)

    def train(
        self,
        resume: bool = True,
        save_every: int = 50,
        use_wandb: bool = False,
        calibrate_masks: bool = True,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, float]:
        if calibrate_masks and self.gradient_masks is None:
            self.calibrate_masks()
        return super().train(
            resume=resume,
            save_every=save_every,
            use_wandb=use_wandb,
            early_stopping_patience=early_stopping_patience,
        )

    def compare_mask_strategies(
        self, strategies: List[str] = None, num_rounds: int = 100
    ) -> Dict[str, Dict[str, float]]:
        if strategies is None:
            strategies = [
                "least_sensitive",
                "most_sensitive",
                "lowest_magnitude",
                "highest_magnitude",
                "random",
            ]

        self.logger.info(f"Comparing mask strategies: {strategies}")

        # Need full state for resetting model between strategy comparisons
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        original_num_rounds = self.num_rounds
        self.num_rounds = num_rounds

        results = {}

        for strategy in strategies:
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Testing strategy: {strategy}")
            self.logger.info(f"{'=' * 50}")

            self.model.load_state_dict(original_state)
            self.best_val_accuracy = 0.0
            self.current_epoch = 0

            self.mask_strategy = MaskStrategy(strategy)
            self.gradient_masks = None  # Force recalibration

            self.calibrate_masks()

            for key in self.metrics_history:
                self.metrics_history[key] = []

            test_metrics = super().train(resume=False, save_every=num_rounds + 1)

            results[strategy] = {
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "best_val_accuracy": self.best_val_accuracy,
            }

            self.logger.info(
                f"Strategy {strategy}: Test Acc = {test_metrics['accuracy']:.4f}"
            )

        self.num_rounds = original_num_rounds
        self.model.load_state_dict(original_state)

        return results


def run_all_mask_strategies_experiment(
    model_fn,
    train_dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    base_config: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    strategies = [
        "least_sensitive",
        "most_sensitive",
        "lowest_magnitude",
        "highest_magnitude",
        "random",
    ]

    results = {}

    for strategy in strategies:
        model = model_fn()

        config = deepcopy(base_config)
        config["sparse"]["mask_strategy"] = strategy

        trainer = FederatedSparseTrainer(
            model=model,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device,
            experiment_name=f"federated_sparse_{strategy}",
        )

        test_metrics = trainer.train(resume=False)

        results[strategy] = {
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "best_val_accuracy": trainer.best_val_accuracy,
        }

    return results
