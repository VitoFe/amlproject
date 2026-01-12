"""
runs all required experiments for the project:
1. Centralized baseline
2. FedAvg with IID sharding
3. FedAvg with non-IID sharding (varying Nc)
4. FedAvg with varying local steps (J)
5. Federated Sparse (Task Arithmetic) experiments
6. Mask strategy comparison (Extension)

python scripts/run_all_experiments.py --quick  # Quick test run
python scripts/run_all_experiments.py --full   # Full experiments
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from src.data.dataset import get_cifar100_datasets, get_dataloaders
from src.models.dino_vit import create_dino_vit
from src.training.centralized import CentralizedTrainer
from src.training.federated import FederatedTrainer
from src.training.federated_sparse import FederatedSparseTrainer
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.visualization import plot_comparison, plot_heterogeneity_comparison


def aggregate_results(results: list, key: str = 'accuracy') -> dict:
    """Aggregate results across multiple runs."""
    values = [r[key] for r in results]
    return {
        'mean_accuracy': np.mean(values),
        'std_accuracy': np.std(values),
        'all_results': results
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Run All FL Experiments')
    
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test run with reduced parameters')
    parser.add_argument('--full', action='store_true', 
                       help='Full experiments (long runtime)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-runs', type=int, default=3, 
                       help='Number of independent runs per experiment')
    
    # Select specific experiments
    parser.add_argument('--only-centralized', action='store_true')
    parser.add_argument('--only-federated', action='store_true')
    parser.add_argument('--only-sparse', action='store_true')
    parser.add_argument('--only-extension', action='store_true')
    
    return parser.parse_args()


def get_experiment_params(quick=False):
    """Get experiment parameters based on mode."""
    if quick:
        return {
            'centralized_epochs': 5,
            'federated_rounds': 20,
            'num_clients': 10,
            'participation_rate': 0.5,
            'local_steps': 2,
            'batch_size': 64,
            'nc_values': [5, 10],
            'j_values': [2, 4],
            'sparsity_values': [0.5, 0.9],
            'num_runs': 1
        }
    else:
        return {
            'centralized_epochs': 50,
            'federated_rounds': 500,
            'num_clients': 100,
            'participation_rate': 0.1,
            'local_steps': 4,
            'batch_size': 64,
            'nc_values': [1, 5, 10, 50],
            'j_values': [4, 8, 16],
            'sparsity_values': [0.5, 0.7, 0.9, 0.95],
            'num_runs': 3
        }


def run_centralized_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run centralized baseline experiment."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Centralized Baseline")
    logger.info("="*60)
    
    results = []
    
    for run in range(params['num_runs']):
        set_seed(args.seed + run)
        
        model = create_dino_vit(num_classes=100, device=args.device)
        
        config = {
            'epochs': params['centralized_epochs'],
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir
        }
        
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], 
            shuffle=True, num_workers=2
        )
        
        trainer = CentralizedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=args.device,
            experiment_name=f'centralized_run{run}'
        )
        
        test_metrics = trainer.train(resume=False)
        results.append(test_metrics)
        
        logger.info(f"Run {run+1}: Test Accuracy = {test_metrics['accuracy']:.4f}")
    
    return aggregate_results(results)


def run_federated_iid_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run FedAvg with IID sharding."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: FedAvg with IID Sharding")
    logger.info("="*60)
    
    results = []
    
    for run in range(params['num_runs']):
        set_seed(args.seed + run)
        
        model = create_dino_vit(num_classes=100, device=args.device)
        
        config = {
            'num_clients': params['num_clients'],
            'participation_rate': params['participation_rate'],
            'local_steps': params['local_steps'],
            'num_rounds': params['federated_rounds'],
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'batch_size': params['batch_size'],
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir,
            'seed': args.seed + run,
            'sharding': {'strategy': 'iid', 'nc': 100}
        }
        
        trainer = FederatedTrainer(
            model=model,
            train_dataset=train_dataset,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=args.device,
            experiment_name=f'fedavg_iid_run{run}'
        )
        
        test_metrics = trainer.train(resume=False)
        results.append(test_metrics)
        
        logger.info(f"Run {run+1}: Test Accuracy = {test_metrics['accuracy']:.4f}")
    
    return aggregate_results(results)


def run_heterogeneity_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run FedAvg with non-IID sharding, varying Nc."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: FedAvg with Non-IID Sharding (Varying Nc)")
    logger.info("="*60)
    
    results = {}
    
    for nc in params['nc_values']:
        logger.info(f"\nTesting Nc = {nc}")
        nc_results = []
        
        for run in range(params['num_runs']):
            set_seed(args.seed + run)
            
            model = create_dino_vit(num_classes=100, device=args.device)
            
            config = {
                'num_clients': params['num_clients'],
                'participation_rate': params['participation_rate'],
                'local_steps': params['local_steps'],
                'num_rounds': params['federated_rounds'],
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'batch_size': params['batch_size'],
                'checkpoint_dir': args.checkpoint_dir,
                'log_dir': args.log_dir,
                'seed': args.seed + run,
                'sharding': {'strategy': 'non_iid', 'nc': nc}
            }
            
            trainer = FederatedTrainer(
                model=model,
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=args.device,
                experiment_name=f'fedavg_noniid_nc{nc}_run{run}'
            )
            
            test_metrics = trainer.train(resume=False)
            nc_results.append(test_metrics['accuracy'])
        
        results[nc] = {
            'mean_accuracy': np.mean(nc_results),
            'std_accuracy': np.std(nc_results)
        }
        logger.info(f"Nc={nc}: {results[nc]['mean_accuracy']:.4f} ± {results[nc]['std_accuracy']:.4f}")
    
    return results


def run_local_steps_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run FedAvg with varying local steps J."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 4: FedAvg with Varying Local Steps (J)")
    logger.info("="*60)
    
    results = {}
    
    for j in params['j_values']:
        logger.info(f"\nTesting J = {j}")
        j_results = []
        
        adjusted_rounds = params['federated_rounds'] * params['local_steps'] // j
        
        for run in range(params['num_runs']):
            set_seed(args.seed + run)
            
            model = create_dino_vit(num_classes=100, device=args.device)
            
            config = {
                'num_clients': params['num_clients'],
                'participation_rate': params['participation_rate'],
                'local_steps': j,
                'num_rounds': adjusted_rounds,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'batch_size': params['batch_size'],
                'checkpoint_dir': args.checkpoint_dir,
                'log_dir': args.log_dir,
                'seed': args.seed + run,
                'sharding': {'strategy': 'iid', 'nc': 100}
            }
            
            trainer = FederatedTrainer(
                model=model,
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=args.device,
                experiment_name=f'fedavg_iid_J{j}_run{run}'
            )
            
            test_metrics = trainer.train(resume=False)
            j_results.append(test_metrics['accuracy'])
        
        results[j] = {
            'mean_accuracy': np.mean(j_results),
            'std_accuracy': np.std(j_results)
        }
        logger.info(f"J={j}: {results[j]['mean_accuracy']:.4f} ± {results[j]['std_accuracy']:.4f}")
    
    return results


def run_sparse_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run Federated Sparse Fine-tuning with varying sparsity."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 5: Federated Sparse Fine-tuning (Varying Sparsity)")
    logger.info("="*60)
    
    results = {}
    
    for sparsity in params['sparsity_values']:
        logger.info(f"\nTesting Sparsity = {sparsity}")
        sparsity_results = []
        
        for run in range(params['num_runs']):
            set_seed(args.seed + run)
            
            model = create_dino_vit(num_classes=100, device=args.device)
            
            config = {
                'num_clients': params['num_clients'],
                'participation_rate': params['participation_rate'],
                'local_steps': params['local_steps'],
                'num_rounds': params['federated_rounds'],
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'batch_size': params['batch_size'],
                'checkpoint_dir': args.checkpoint_dir,
                'log_dir': args.log_dir,
                'seed': args.seed + run,
                'sharding': {'strategy': 'iid', 'nc': 100},
                'sparse': {
                    'sparsity_ratio': sparsity,
                    'calibration_rounds': 5,
                    'mask_strategy': 'least_sensitive',
                    'fisher_samples': 512
                }
            }
            
            trainer = FederatedSparseTrainer(
                model=model,
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=args.device,
                experiment_name=f'fedavg_sparse_s{sparsity}_run{run}'
            )
            
            test_metrics = trainer.train(resume=False)
            sparsity_results.append(test_metrics['accuracy'])
        
        results[sparsity] = {
            'mean_accuracy': np.mean(sparsity_results),
            'std_accuracy': np.std(sparsity_results)
        }
        logger.info(f"Sparsity={sparsity}: {results[sparsity]['mean_accuracy']:.4f} ± {results[sparsity]['std_accuracy']:.4f}")
    
    return results


def run_mask_strategy_experiment(args, params, train_dataset, val_loader, test_loader):
    """Run Extension: Compare mask strategies."""
    logger = get_logger('main')
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 6 (Extension): Mask Strategy Comparison")
    logger.info("="*60)
    
    strategies = [
        'least_sensitive',
        'most_sensitive',
        'lowest_magnitude',
        'highest_magnitude',
        'random'
    ]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\nTesting Strategy: {strategy}")
        strategy_results = []
        
        for run in range(params['num_runs']):
            set_seed(args.seed + run)
            
            model = create_dino_vit(num_classes=100, device=args.device)
            
            config = {
                'num_clients': params['num_clients'],
                'participation_rate': params['participation_rate'],
                'local_steps': params['local_steps'],
                'num_rounds': params['federated_rounds'],
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'batch_size': params['batch_size'],
                'checkpoint_dir': args.checkpoint_dir,
                'log_dir': args.log_dir,
                'seed': args.seed + run,
                'sharding': {'strategy': 'iid', 'nc': 100},
                'sparse': {
                    'sparsity_ratio': 0.9,
                    'calibration_rounds': 5,
                    'mask_strategy': strategy,
                    'fisher_samples': 512
                }
            }
            
            trainer = FederatedSparseTrainer(
                model=model,
                train_dataset=train_dataset,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=args.device,
                experiment_name=f'fedavg_sparse_{strategy}_run{run}'
            )
            
            test_metrics = trainer.train(resume=False)
            strategy_results.append(test_metrics['accuracy'])
        
        results[strategy] = {
            'mean_accuracy': np.mean(strategy_results),
            'std_accuracy': np.std(strategy_results)
        }
        logger.info(f"{strategy}: {results[strategy]['mean_accuracy']:.4f} ± {results[strategy]['std_accuracy']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    if not args.quick and not args.full:
        print("Please specify either --quick (test) or --full (complete experiments)")
        return
    
    params = get_experiment_params(quick=args.quick)
    if args.num_runs:
        params['num_runs'] = args.num_runs
    
    # Setup logging
    logger = setup_logging(
        log_dir=args.log_dir,
        experiment_name='all_experiments'
    )
    
    logger.info(f"Starting experiments with {'quick' if args.quick else 'full'} parameters")
    logger.info(f"Device: {args.device}")
    logger.info(f"Parameters: {json.dumps(params, indent=2)}")
    
    logger.info("Loading CIFAR-100 dataset...")
    set_seed(args.seed)
    train_dataset, val_dataset, test_dataset = get_cifar100_datasets(
        data_dir=args.data_dir,
        val_split=0.1,
        seed=args.seed
    )
    
    _, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=params['batch_size'],
        num_workers=2
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Store all results
    all_results = {}
    
    # Run experiments
    run_all = not (args.only_centralized or args.only_federated or 
                   args.only_sparse or args.only_extension)
    
    if run_all or args.only_centralized:
        all_results['centralized'] = run_centralized_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
    
    if run_all or args.only_federated:
        all_results['federated_iid'] = run_federated_iid_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
        all_results['heterogeneity'] = run_heterogeneity_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
        all_results['local_steps'] = run_local_steps_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
    
    if run_all or args.only_sparse:
        all_results['sparse'] = run_sparse_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
    
    if run_all or args.only_extension:
        all_results['mask_strategies'] = run_mask_strategy_experiment(
            args, params, train_dataset, val_loader, test_loader
        )
    
    # Save final results
    results_path = Path(args.log_dir) / 'all_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(all_results), f, indent=2)
    
    logger.info(f"\nAll results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if 'centralized' in all_results:
        r = all_results['centralized']
        print(f"Centralized Baseline: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    
    if 'federated_iid' in all_results:
        r = all_results['federated_iid']
        print(f"FedAvg (IID): {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    
    if 'heterogeneity' in all_results:
        print("FedAvg (Non-IID):")
        for nc, r in all_results['heterogeneity'].items():
            print(f"  Nc={nc}: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    
    if 'mask_strategies' in all_results:
        print("Mask Strategies:")
        for strategy, r in sorted(all_results['mask_strategies'].items(), 
                                   key=lambda x: x[1]['mean_accuracy'], reverse=True):
            print(f"  {strategy}: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
    
    return all_results


if __name__ == '__main__':
    main()
