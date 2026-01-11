"""
Federated Learning Trainer (FedAvg)

Implementation of Federated Averaging as described in:
"Communication-Efficient Learning of Deep Networks from Decentralized Data"
McMahan et al., AISTATS 2017

This simulates federated learning with sequential client training
on a single GPU, which is functionally equivalent to parallel training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from .base import BaseTrainer
from ..data.dataset import get_client_dataloader, get_transforms
from ..data.sharding import create_client_splits, ShardingStrategy, get_sharding_stats
from ..utils.logging import log_metrics
from ..utils.early_stopping import EarlyStopping
from ..models.dino_vit import get_model_state_dict, set_model_state_dict


class FederatedTrainer(BaseTrainer):
    """
    Federated Learning trainer implementing FedAvg.
    
    Key parameters:
    - K: Total number of clients
    - C: Fraction of clients participating per round
    - J: Number of local training steps per client
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
        experiment_name: str = 'federated'
    ):
        """
        Initialize federated trainer.
        
        Args:
            model: PyTorch model
            train_dataset: Full training dataset (will be sharded)
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration with keys:
                - num_clients (K): Total number of clients
                - participation_rate (C): Fraction per round
                - local_steps (J): Local training steps
                - num_rounds: Communication rounds
                - learning_rate: Client learning rate
                - sharding.strategy: 'iid' or 'non_iid'
                - sharding.nc: Classes per client for non-iid
            device: Device for computation
            experiment_name: Name for logging
        """
        # Create a temporary train_loader for base class
        # (actual training uses client subsets)
        temp_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        
        super().__init__(
            model, temp_loader, val_loader, test_loader,
            config, device, experiment_name
        )
        
        self.train_dataset = train_dataset
        
        # Federated learning parameters
        self.num_clients = config.get('num_clients', 100)  # K
        self.participation_rate = config.get('participation_rate', 0.1)  # C
        self.local_steps = config.get('local_steps', 4)  # J
        self.num_rounds = config.get('num_rounds', 500)
        self.client_lr = config.get('learning_rate', 0.001)  # Lower LR for pretrained models
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.batch_size = config.get('batch_size', 64)
        self.early_stopping_patience = config.get('early_stopping_patience', 100)  # More patience for FL
        
        # Sharding configuration
        sharding_config = config.get('sharding', {})
        strategy_str = sharding_config.get('strategy', 'iid')
        self.sharding_strategy = ShardingStrategy(strategy_str)
        self.nc = sharding_config.get('nc', 10)
        
        # Create client data splits
        self._create_client_splits()
        
        # Number of clients per round
        self.clients_per_round = max(1, int(self.num_clients * self.participation_rate))
        
        # Setup early stopping
        self.early_stopping: Optional[EarlyStopping] = None
        if self.early_stopping_patience > 0:
            self.early_stopping = EarlyStopping(
                patience=self.early_stopping_patience,
                min_delta=0.001,
                mode='max',
                verbose=True
            )
        
        self.logger.info(f"Federated Learning Setup:")
        self.logger.info(f"  - Total clients (K): {self.num_clients}")
        self.logger.info(f"  - Participation rate (C): {self.participation_rate}")
        self.logger.info(f"  - Clients per round: {self.clients_per_round}")
        self.logger.info(f"  - Local steps (J): {self.local_steps}")
        self.logger.info(f"  - Sharding: {self.sharding_strategy.value}")
        if self.sharding_strategy == ShardingStrategy.NON_IID:
            self.logger.info(f"  - Classes per client (Nc): {self.nc}")
        if self.early_stopping:
            self.logger.info(f"  - Early stopping patience: {self.early_stopping_patience} rounds")
    
    def _create_client_splits(self):
        """Create disjoint data splits for clients."""
        self.client_splits = create_client_splits(
            self.train_dataset,
            num_clients=self.num_clients,
            strategy=self.sharding_strategy,
            nc=self.nc,
            seed=self.config.get('seed', 42)
        )
        
        # Get labels for stats
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        elif hasattr(self.train_dataset, 'dataset'):
            labels = np.array(self.train_dataset.dataset.targets)[self.train_dataset.indices]
        else:
            labels = np.zeros(len(self.train_dataset))
        
        # Log sharding stats
        stats = get_sharding_stats(self.client_splits, labels)
        self.logger.info(f"Sharding statistics:")
        self.logger.info(f"  - Samples per client: {stats['samples_per_client_mean']:.1f} ± {stats['samples_per_client_std']:.1f}")
        self.logger.info(f"  - Classes per client: {stats['classes_per_client_mean']:.1f} ± {stats['classes_per_client_std']:.1f}")
    
    def _get_client_dataloader(self, client_id: int) -> DataLoader:
        """Get dataloader for a specific client."""
        indices = self.client_splits[client_id]
        
        # Get the base dataset
        if hasattr(self.train_dataset, 'dataset'):
            base_dataset = self.train_dataset.dataset
            # Map indices through the subset
            actual_indices = [self.train_dataset.indices[i] for i in indices]
        else:
            base_dataset = self.train_dataset
            actual_indices = indices
        
        return get_client_dataloader(
            base_dataset,
            actual_indices,
            batch_size=self.batch_size,
            num_workers=0,  # Avoid multiprocessing issues in sequential simulation
            pin_memory=True
        )
    
    def _client_update(
        self,
        client_id: int,
        global_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local training for a single client.
        
        Args:
            client_id: Client identifier
            global_state: Global model state dict
        
        Returns:
            Updated local model state dict
        """
        # Load global model
        self.model.load_state_dict(global_state)
        self.model.train()
        
        # Create client optimizer
        optimizer = torch.optim.SGD(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.client_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Get client data
        client_loader = self._get_client_dataloader(client_id)
        
        # Perform local steps
        step = 0
        while step < self.local_steps:
            for inputs, targets in client_loader:
                if step >= self.local_steps:
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                step += 1
        
        return get_model_state_dict(self.model)
    
    def _aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates using FedAvg.
        
        Standard FedAvg: weighted average based on dataset sizes.
        
        Args:
            client_states: List of client state dicts
            client_weights: Optional weights for averaging
        
        Returns:
            Aggregated global state dict
        """
        if client_weights is None:
            # Equal weighting
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            # Normalize weights
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]
        
        # Initialize with zeros
        aggregated = {}
        for key in client_states[0]:
            aggregated[key] = torch.zeros_like(client_states[0][key])
        
        # Weighted average
        for state, weight in zip(client_states, client_weights):
            for key in aggregated:
                aggregated[key] += weight * state[key]
        
        return aggregated
    
    def train(
        self,
        resume: bool = True,
        save_every: int = 50,
        use_wandb: bool = False,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train using Federated Averaging.
        
        Args:
            resume: Whether to try resuming
            save_every: Save checkpoint every N rounds
            use_wandb: Whether to log to W&B
            early_stopping_patience: Override patience (None uses config, 0 disables)
        
        Returns:
            Final test metrics
        """
        # Override early stopping if specified
        if early_stopping_patience is not None:
            if early_stopping_patience > 0:
                self.early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=0.001,
                    mode='max',
                    verbose=True
                )
            else:
                self.early_stopping = None
        if resume:
            self.try_resume()
        
        self.logger.info(f"Starting federated training for {self.num_rounds} rounds")
        
        start_round = self.current_epoch
        global_state = get_model_state_dict(self.model)
        
        for round_idx in range(start_round, self.num_rounds):
            self.current_epoch = round_idx
            
            # Select clients for this round
            selected_clients = np.random.choice(
                self.num_clients,
                size=self.clients_per_round,
                replace=False
            )
            
            # Perform client updates
            client_states = []
            client_weights = []
            
            for client_id in tqdm(selected_clients, desc=f"Round {round_idx}", leave=False):
                local_state = self._client_update(client_id, global_state)
                client_states.append(local_state)
                client_weights.append(len(self.client_splits[client_id]))
            
            # Aggregate updates
            global_state = self._aggregate(client_states, client_weights)
            
            # Update global model
            self.model.load_state_dict(global_state)
            
            # Evaluate
            val_metrics = self.evaluate_val()
            
            # Store metrics
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log
            if (round_idx + 1) % 10 == 0 or round_idx == 0:
                log_metrics(
                    self.logger,
                    {'val_loss': val_metrics['loss'], 'val_acc': val_metrics['accuracy']},
                    round_idx,
                    prefix='',
                    use_wandb=use_wandb
                )
            
            # Save checkpoint
            if (round_idx + 1) % save_every == 0 or val_metrics['accuracy'] > self.best_val_accuracy:
                self.save_checkpoint(
                    torch.optim.SGD(self.model.parameters(), lr=self.client_lr),  # Dummy optimizer for checkpoint
                    None,
                    {'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy']}
                )
            
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.logger.info(f"Round {round_idx}: New best val accuracy: {self.best_val_accuracy:.4f}")
            
            # Check early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['accuracy'], round_idx):
                    self.logger.info(f"Early stopping triggered at round {round_idx}!")
                    self.logger.info(f"Best validation accuracy was {self.early_stopping.best_score:.4f} at round {self.early_stopping.best_epoch}")
                    break
        
        # Final evaluation
        self.logger.info("Training complete. Evaluating on test set...")
        
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.load_checkpoint(best_checkpoint)
        
        test_metrics = self.evaluate_test()
        self.metrics_history['test_loss'].append(test_metrics['loss'])
        self.metrics_history['test_accuracy'].append(test_metrics['accuracy'])
        
        self.logger.info(f"Final Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return test_metrics
