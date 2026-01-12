import argparse
import yaml
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.dataset import get_cifar100_datasets, get_dataloaders
from src.models.dino_vit import create_dino_vit
from src.training.federated import FederatedTrainer
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.utils.visualization import plot_training_curves, plot_heterogeneity_comparison


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning on CIFAR-100')
    
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
    
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--experiment-name', type=str, default='federated', help='Experiment name')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N rounds')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='federated-task-arithmetic')
    
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    parser.add_argument('--heterogeneity-experiment', action='store_true',
                       help='Run experiment varying Nc values')
    parser.add_argument('--local-steps-experiment', action='store_true',
                       help='Run experiment varying local steps J')
    parser.add_argument('--nc-values', type=int, nargs='+', default=[1, 5, 10, 50],
                       help='Nc values for heterogeneity experiment')
    parser.add_argument('--j-values', type=int, nargs='+', default=[4, 8, 16],
                       help='J values for local steps experiment')
    
    return parser.parse_args()


def run_single_experiment(args, nc=None, local_steps=None, run=0):
    """Run a single federated learning experiment."""
    run_seed = args.seed + run
    set_seed(run_seed)
    
    config = {
        'num_clients': args.num_clients,
        'participation_rate': args.participation_rate,
        'local_steps': local_steps or args.local_steps,
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
            'nc': nc or args.nc
        }
    }
    
    sharding_str = 'iid' if args.sharding == 'iid' else f'noniid_nc{nc or args.nc}'
    experiment_name = f"{args.experiment_name}_{sharding_str}_J{local_steps or args.local_steps}"
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
    
    trainer = FederatedTrainer(
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
        use_wandb=args.use_wandb
    )
    
    plot_training_curves(
        trainer.get_metrics_history(),
        title=f'Federated Training - {experiment_name}',
        save_path=f'{args.log_dir}/{experiment_name}_curves.png'
    )
    
    return test_metrics


def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    if args.heterogeneity_experiment:
        print("\n" + "="*60)
        print("HETEROGENEITY EXPERIMENT: Varying Nc")
        print("="*60)
        
        args.sharding = 'non_iid'
        results = {}
        
        for nc in args.nc_values:
            print(f"\nRunning with Nc={nc}")
            run_results = []
            
            for run in range(args.num_runs):
                metrics = run_single_experiment(args, nc=nc, run=run)
                run_results.append(metrics['accuracy'])
            
        results[nc] = {
                'test_accuracy': np.mean(run_results),
                'test_accuracy_std': np.std(run_results)
            }
        
        print("\n" + "="*60)
        print("RESULTS: Heterogeneity Experiment")
        print("="*60)
        print(f"{'Nc':<10} {'Test Accuracy':<20} {'Std':<10}")
        print("-"*40)
        for nc, r in results.items():
            print(f"{nc:<10} {r['test_accuracy']:.4f} ± {r['test_accuracy_std']:.4f}")
        
        return results
    
    if args.local_steps_experiment:
        print("\n" + "="*60)
        print("LOCAL STEPS EXPERIMENT: Varying J")
        print("="*60)
        
        results = {}
        
        for j in args.j_values:
            print(f"\nRunning with J={j}")
            
            adjusted_rounds = args.num_rounds * args.local_steps // j
            original_rounds = args.num_rounds
            args.num_rounds = adjusted_rounds
            
            run_results = []
            for run in range(args.num_runs):
                metrics = run_single_experiment(args, local_steps=j, run=run)
                run_results.append(metrics['accuracy'])
            
            args.num_rounds = original_rounds
            
            results[j] = {
                'test_accuracy': np.mean(run_results),
                'test_accuracy_std': np.std(run_results)
            }
        
        print("\n" + "="*60)
        print("RESULTS: Local Steps Experiment")
        print("="*60)
        print(f"{'J':<10} {'Test Accuracy':<20} {'Std':<10}")
        print("-"*40)
        for j, r in results.items():
            print(f"{j:<10} {r['test_accuracy']:.4f} ± {r['test_accuracy_std']:.4f}")
        
        return results
    
    all_results = []
    
    for run in range(args.num_runs):
        metrics = run_single_experiment(args, run=run)
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
