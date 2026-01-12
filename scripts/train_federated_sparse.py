import argparse
import yaml
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import get_cifar100_datasets, get_dataloaders
from src.models.dino_vit import create_dino_vit
from src.training.federated_sparse import FederatedSparseTrainer
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.utils.visualization import plot_training_curves, plot_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Sparse Fine-tuning on CIFAR-100')
    
    parser.add_argument('--num-clients', type=int, default=100, help='Total number of clients (K)')
    parser.add_argument('--participation-rate', type=float, default=0.1, help='Client participation rate (C)')
    parser.add_argument('--local-steps', type=int, default=4, help='Local training steps per client (J)')
    parser.add_argument('--num-rounds', type=int, default=500, help='Communication rounds')
    parser.add_argument('--lr', type=float, default=0.01, help='Client learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    parser.add_argument('--sharding', type=str, default='iid', 
                       choices=['iid', 'non_iid'], help='Data sharding strategy')
    parser.add_argument('--nc', type=int, default=10, help='Classes per client for non-IID')
    
    parser.add_argument('--sparsity-ratio', type=float, default=0.9,
                       help='Fraction of parameters to mask (freeze)')
    parser.add_argument('--calibration-rounds', type=int, default=5,
                       help='Number of rounds for Fisher averaging')
    parser.add_argument('--mask-strategy', type=str, default='least_sensitive',
                       choices=['least_sensitive', 'most_sensitive', 
                               'lowest_magnitude', 'highest_magnitude', 'random'],
                       help='Gradient mask calibration strategy')
    parser.add_argument('--fisher-samples', type=int, default=512,
                       help='Samples for Fisher computation')
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--experiment-name', type=str, default='federated_sparse', help='Experiment name')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N rounds')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='federated-task-arithmetic')
    
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    parser.add_argument('--compare-strategies', action='store_true',
                       help='Compare all mask strategies')
    parser.add_argument('--sparsity-experiment', action='store_true',
                       help='Run experiment varying sparsity ratios')
    parser.add_argument('--sparsity-values', type=float, nargs='+', 
                       default=[0.5, 0.7, 0.9, 0.95, 0.99],
                       help='Sparsity ratios for sparsity experiment')
    
    return parser.parse_args()


def run_single_experiment(args, sparsity_ratio=None, mask_strategy=None, run=0):
    run_seed = args.seed + run
    set_seed(run_seed)
    
    config = {
        'num_clients': args.num_clients,
        'participation_rate': args.participation_rate,
        'local_steps': args.local_steps,
        'num_rounds': args.num_rounds,
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'seed': run_seed,
        'sharding': {
            'strategy': args.sharding,
            'nc': args.nc
        },
        'sparse': {
            'sparsity_ratio': sparsity_ratio or args.sparsity_ratio,
            'calibration_rounds': args.calibration_rounds,
            'mask_strategy': mask_strategy or args.mask_strategy,
            'fisher_samples': args.fisher_samples
        }
    }
    
    sharding_str = 'iid' if args.sharding == 'iid' else f'noniid_nc{args.nc}'
    strategy = mask_strategy or args.mask_strategy
    sparsity = sparsity_ratio or args.sparsity_ratio
    experiment_name = f"{args.experiment_name}_{sharding_str}_{strategy}_s{sparsity}"
    if args.num_runs > 1:
        experiment_name += f"_run{run}"
    
    logger = setup_logging(
        log_dir=args.log_dir,
        experiment_name=experiment_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=config
    )
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Device: {args.device}")
    
    logger.info("Loading CIFAR-100 dataset...")
    train_dataset, val_dataset, test_dataset = get_cifar100_datasets(
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=run_seed
    )
    
    _, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    logger.info("Creating DINO ViT-S/16 model...")
    model = create_dino_vit(
        num_classes=100,
        freeze_backbone=False,
        device=args.device
    )
    
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: Total={param_counts['total']:,}, Trainable={param_counts['trainable']:,}")
    
    trainer = FederatedSparseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=args.device,
        experiment_name=experiment_name
    )
    
    test_metrics = trainer.train(
        resume=True,
        save_every=args.save_every,
        use_wandb=args.use_wandb,
        calibrate_masks=True
    )
    
    plot_training_curves(
        trainer.get_metrics_history(),
        title=f'Federated Sparse Training - {experiment_name}',
        save_path=f'{args.log_dir}/{experiment_name}_curves.png'
    )
    
    return test_metrics, trainer.best_val_accuracy


def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    if args.compare_strategies:
        print("\n" + "="*60)
        print("MASK STRATEGY COMPARISON EXPERIMENT")
        print("="*60)
        
        strategies = [
            'least_sensitive',
            'most_sensitive',
            'lowest_magnitude',
            'highest_magnitude',
            'random'
        ]
        
        results = {}
        
        for strategy in strategies:
            print(f"\nRunning with strategy: {strategy}")
            run_results = []
            
            for run in range(args.num_runs):
                metrics, best_val = run_single_experiment(args, mask_strategy=strategy, run=run)
                run_results.append(metrics['accuracy'])
            
        results[strategy] = {
                'test_accuracy': np.mean(run_results),
                'test_accuracy_std': np.std(run_results)
            }
        
        print("\n" + "="*60)
        print("RESULTS: Mask Strategy Comparison")
        print("="*60)
        print(f"{'Strategy':<25} {'Test Accuracy':<20}")
        print("-"*45)
        for strategy, r in sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            print(f"{strategy:<25} {r['test_accuracy']:.4f} ± {r['test_accuracy_std']:.4f}")
        
        plot_comparison(
            {s: {'test_accuracy': r['test_accuracy']} for s, r in results.items()},
            metric='test_accuracy',
            title='Mask Strategy Comparison',
            save_path=f'{args.log_dir}/strategy_comparison.png'
        )
        
        return results
    
    if args.sparsity_experiment:
        print("\n" + "="*60)
        print("SPARSITY EXPERIMENT: Varying Sparsity Ratio")
        print("="*60)
        
        results = {}
        
        for sparsity in args.sparsity_values:
            print(f"\nRunning with sparsity={sparsity}")
            run_results = []
            
            for run in range(args.num_runs):
                metrics, _ = run_single_experiment(args, sparsity_ratio=sparsity, run=run)
                run_results.append(metrics['accuracy'])
            
        results[sparsity] = {
                'test_accuracy': np.mean(run_results),
                'test_accuracy_std': np.std(run_results)
            }
        
        print("\n" + "="*60)
        print("RESULTS: Sparsity Experiment")
        print("="*60)
        print(f"{'Sparsity':<15} {'Test Accuracy':<20}")
        print("-"*35)
        for sparsity, r in results.items():
            print(f"{sparsity:<15} {r['test_accuracy']:.4f} ± {r['test_accuracy_std']:.4f}")
        
        return results
    
    all_results = []
    
    for run in range(args.num_runs):
        metrics, _ = run_single_experiment(args, run=run)
        all_results.append(metrics)
    
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
