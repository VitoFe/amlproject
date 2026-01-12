import argparse
import yaml
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import get_cifar100_datasets, get_dataloaders
from src.models.dino_vit import create_dino_vit
from src.training.centralized import CentralizedTrainer
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.utils.visualization import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Centralized Training on CIFAR-100')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'none'], help='LR scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')

    parser.add_argument('--freeze-backbone', action='store_true', 
                       help='Freeze DINO backbone (linear probe)')
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--experiment-name', type=str, default='centralized', help='Experiment name')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='federated-task-arithmetic')
    
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    parser.add_argument('--hp-search', action='store_true', help='Run hyperparameter search')
    parser.add_argument('--hp-epochs', type=int, default=10, help='Epochs per HP trial')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    config.update({
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'seed': args.seed
    })
    all_results = []
    
    for run in range(args.num_runs):
        run_seed = args.seed + run
        set_seed(run_seed)
        
        experiment_name = f"{args.experiment_name}_run{run}" if args.num_runs > 1 else args.experiment_name
        
        logger = setup_logging(
            log_dir=args.log_dir,
            experiment_name=experiment_name,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_config=config
        )
        
        logger.info(f"Starting run {run + 1}/{args.num_runs} with seed {run_seed}")
        logger.info(f"Device: {args.device}")
        
        logger.info("Loading dataset...")
        train_dataset, val_dataset, test_dataset = get_cifar100_datasets(
            data_dir=args.data_dir,
            val_split=args.val_split,
            seed=run_seed
        )
        
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        model = create_dino_vit(
            num_classes=100,
            freeze_backbone=args.freeze_backbone,
            device=args.device
        )
        
        param_counts = model.count_parameters()
        logger.info(f"model params: Total={param_counts['total']:,}, Trainable={param_counts['trainable']:,}")
        
        trainer = CentralizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=args.device,
            experiment_name=experiment_name
        )
        
        if args.hp_search:
            logger.info("hyperparameter search...")
            hp_results = trainer.hyperparameter_search(
                lr_range=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                epochs_per_trial=args.hp_epochs
            )
            logger.info(f"Best HP: LR={hp_results['best_lr']}")
            
            config['learning_rate'] = hp_results['best_lr']
            trainer = CentralizedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=args.device,
                experiment_name=experiment_name
            )
        
        test_metrics = trainer.train(
            resume=True,
            save_every=args.save_every,
            use_wandb=args.use_wandb
        )
        
        all_results.append(test_metrics)
        
        plot_training_curves(
            trainer.get_metrics_history(),
            title=f'Centralized Training - {experiment_name}',
            save_path=f'{args.log_dir}/{experiment_name}_curves.png'
        )
    
    if args.num_runs > 1:
        accuracies = [r['accuracy'] for r in all_results]
        losses = [r['loss'] for r in all_results]
        
        print("\n" + "="*50)
        print("AGGREGATE RESULTS")
        print("="*50)
        print(f"Test Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Test Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    
    return all_results


if __name__ == '__main__':
    main()
