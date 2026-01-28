"""
python main.py --help
python main.py train --mode centralized --epochs 50
python main.py train --mode federated --num-clients 100
python main.py train --mode sparse --sparsity-ratio 0.9
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Federated Learning with Task Arithmetic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    # Centralized baseline
    python main.py train --mode centralized --epochs 50
    
    # Federated learning (IID)
    python main.py train --mode federated --sharding iid
    
    # Federated learning (non-IID)
    python main.py train --mode federated --sharding non_iid --nc 5
    
    # Sparse fine-tuning
    python main.py train --mode sparse --sparsity-ratio 0.9
    
    # Run all experiments
    python main.py run-all --quick
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["centralized", "federated", "sparse"],
        help="Training mode",
    )
    train_parser.add_argument("--config", type=str, help="Path to config file")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--num-clients", type=int, default=100)
    train_parser.add_argument("--num-rounds", type=int, default=500)
    train_parser.add_argument("--local-steps", type=int, default=4)
    train_parser.add_argument(
        "--sharding", type=str, default="iid", choices=["iid", "non_iid"]
    )
    train_parser.add_argument("--nc", type=int, default=10)
    train_parser.add_argument("--sparsity-ratio", type=float, default=0.9)
    train_parser.add_argument("--mask-strategy", type=str, default="least_sensitive")
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--seed", type=int, default=42)

    # Run all experiments command
    runall_parser = subparsers.add_parser("run-all", help="Run all experiments")
    runall_parser.add_argument("--quick", action="store_true", help="Quick test run")
    runall_parser.add_argument("--full", action="store_true", help="Full experiments")
    runall_parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        if args.mode == "centralized":
            from scripts.train_centralized import main as train_main

            sys.argv = [
                "train_centralized.py",
                "--epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
            train_main()
        elif args.mode == "federated":
            from scripts.train_federated import main as train_main

            sys.argv = [
                "train_federated.py",
                "--num-clients",
                str(args.num_clients),
                "--num-rounds",
                str(args.num_rounds),
                "--local-steps",
                str(args.local_steps),
                "--sharding",
                args.sharding,
                "--nc",
                str(args.nc),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
            train_main()
        elif args.mode == "sparse":
            from scripts.train_federated_sparse import main as train_main

            sys.argv = [
                "train_federated_sparse.py",
                "--num-clients",
                str(args.num_clients),
                "--num-rounds",
                str(args.num_rounds),
                "--sparsity-ratio",
                str(args.sparsity_ratio),
                "--mask-strategy",
                args.mask_strategy,
                "--device",
                args.device,
                "--seed",
                str(args.seed),
            ]
            train_main()

    elif args.command == "run-all":
        from scripts.run_all_experiments import main as run_all_main

        sys.argv = ["run_all_experiments.py", "--device", args.device]
        if args.quick:
            sys.argv.append("--quick")
        elif args.full:
            sys.argv.append("--full")
        run_all_main()


if __name__ == "__main__":
    main()
