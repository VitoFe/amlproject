# Federated Learning Under the Lens of Task Arithmetic

A comprehensive implementation of Federated Learning with Task Arithmetic techniques for the course project. This codebase explores how sparse fine-tuning methods based on parameter sensitivity can be applied to federated learning scenarios to mitigate challenges like client drift and interference during model aggregation.

This project investigates the intersection of **Federated Learning (FL)** and **Task Arithmetic**:

1. **Federated Learning**: A distributed learning paradigm where multiple clients train models locally and a central server aggregates their updates
2. **Task Arithmetic**: Model editing techniques that merge fine-tuned models through arithmetic operations on model weights
3. **Sparse Fine-tuning**: Updating only a subset of parameters to minimize interference during model merging

## Key Research Questions

- How does data heterogeneity (non-IID) affect federated learning performance?
- Can sparse fine-tuning techniques reduce interference during FedAvg aggregation?
- Which gradient mask calibration strategies work best for federated sparse fine-tuning?

## Architecture & Design

### Project Structure

```
stuffaml/
├── configs/                    # Configuration files
│   └── default.yaml           # Default experiment configuration
├── scripts/                    # Executable experiment scripts
│   ├── train_centralized.py   # Centralized baseline training
│   ├── train_federated.py     # FedAvg experiments
│   ├── train_federated_sparse.py  # Sparse fine-tuning experiments
│   └── run_all_experiments.py # Run complete experiment suite
├── src/                        # Source code
│   ├── data/                  # Data loading and sharding
│   │   ├── dataset.py         # CIFAR-100 dataset handling
│   │   └── sharding.py        # IID/non-IID data partitioning
│   ├── models/                # Model definitions
│   │   └── dino_vit.py        # DINO ViT-S/16 classifier
│   ├── optimizers/            # Custom optimizers
│   │   ├── sparse_sgdm.py     # Sparse SGD with Momentum
│   │   └── fisher.py          # Fisher Information & mask calibration
│   ├── training/              # Training loops
│   │   ├── base.py            # Abstract base trainer
│   │   ├── centralized.py     # Standard centralized training
│   │   ├── federated.py       # FedAvg implementation
│   │   └── federated_sparse.py # Sparse federated training
│   └── utils/                 # Utilities
│       ├── checkpointing.py   # Checkpoint management
│       ├── logging.py         # Logging setup
│       ├── metrics.py         # Metric computation
│       ├── seed.py            # Reproducibility
│       └── visualization.py   # Plotting functions
├── checkpoints/               # Saved model checkpoints
├── logs/                      # Training logs and plots
└── data/                      # Downloaded datasets
```

### Design Principles (DRY)

The codebase follows the **Don't Repeat Yourself (DRY)** principle:

1. **Base Trainer Pattern**: All training paradigms (centralized, federated, sparse) inherit from `BaseTrainer`, sharing common functionality for:

   - Model evaluation
   - Checkpoint management
   - Metrics tracking
   - Learning rate scheduling

2. **Modular Optimizers**: The `SparseSGDM` optimizer extends standard SGD with gradient masking, enabling the same optimizer code for both dense and sparse training.

3. **Unified Data Pipeline**: Data loading, transforms, and sharding strategies are implemented once and reused across all experiments.

4. **Configuration-Driven**: YAML configuration files allow running different experiments without code changes.

## Methodology

### 1. Centralized Baseline

Standard training on CIFAR-100 using DINO ViT-S/16:

- Pre-trained backbone from Facebook Research
- Classification head fine-tuned on CIFAR-100
- Cosine annealing learning rate schedule
- Used as performance upper bound

### 2. Federated Averaging (FedAvg)

Implementation following [McMahan et al., 2017]:

- **K** clients with private data shards
- **C** fraction participate per round
- **J** local SGD steps before aggregation
- Weighted averaging based on dataset sizes

### 3. Data Heterogeneity Simulation

Two sharding strategies implemented:

- **IID**: Uniform random distribution across clients
- **Non-IID**: Each client has samples from only **Nc** classes

### 4. Sparse Fine-tuning (Task Arithmetic)

Based on [Iurada et al., 2025]:

**Gradient Mask Calibration**:

1. Compute parameter sensitivity using Fisher Information Matrix (diagonal approximation)
2. Average Fisher over multiple calibration rounds
3. Select parameters based on sensitivity scores

**Mask Strategies**:

- `least_sensitive`: Update low-sensitivity parameters (recommended for reducing interference)
- `most_sensitive`: Update high-sensitivity parameters
- `lowest_magnitude`: Update small-magnitude weights
- `highest_magnitude`: Update large-magnitude weights
- `random`: Random parameter selection (baseline)

**SparseSGDM Optimizer**:

- Extends standard SGDM with gradient masking
- Masked parameters receive zero gradient updates
- Momentum buffers maintained only for unmasked parameters

### 5. Fisher Information Computation

The Fisher Information Matrix measures parameter sensitivity:

```
F_ii = E[(∂L/∂θ_i)²]
```

Computed by:

1. Forward pass on calibration data
2. Backward pass to get gradients
3. Square and accumulate gradients
4. Average over samples

High Fisher values indicate parameters that significantly affect the loss → high sensitivity.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd stuffaml

# Create virtual environment with uv
uv venv
uv sync

# Or install dependencies manually
uv add torch torchvision timm wandb tqdm numpy matplotlib scikit-learn pyyaml
```

### Running Experiments

#### Quick Test Run

```bash
# Run abbreviated experiments to verify setup
uv run python scripts/run_all_experiments.py --quick
```

#### Individual Experiments

```bash
# Centralized baseline
uv run python scripts/train_centralized.py --epochs 50 --lr 0.001

# FedAvg with IID data
uv run python scripts/train_federated.py --num-clients 100 --participation-rate 0.1 --local-steps 4

# FedAvg with non-IID data (Nc=5 classes per client)
uv run python scripts/train_federated.py --sharding non_iid --nc 5

# Heterogeneity experiment (vary Nc)
uv run python scripts/train_federated.py --heterogeneity-experiment --nc-values 1 5 10 50

# Local steps experiment (vary J)
uv run python scripts/train_federated.py --local-steps-experiment --j-values 4 8 16

# Sparse fine-tuning
uv run python scripts/train_federated_sparse.py --sparsity-ratio 0.9 --mask-strategy least_sensitive

# Compare all mask strategies
uv run python scripts/train_federated_sparse.py --compare-strategies
```

#### Full Experiment Suite

```bash
# Run all experiments (long runtime - use on Colab or with GPU)
uv run python scripts/run_all_experiments.py --full --num-runs 3
```

### Google Colab Setup

```python
# Install dependencies
!pip install torch torchvision timm wandb tqdm numpy matplotlib scikit-learn pyyaml

# Clone repository
!git clone <repository-url>
%cd stuffaml

# Run experiments
!python scripts/run_all_experiments.py --quick --device cuda
```

## 📊 Experiments

### Base Experimentation

| Experiment         | Description                           | Parameters                           |
| ------------------ | ------------------------------------- | ------------------------------------ |
| Centralized        | Standard training baseline            | 50 epochs, LR=0.001, cosine schedule |
| FedAvg IID         | Federated with IID sharding           | K=100, C=0.1, J=4                    |
| FedAvg Non-IID     | Federated with non-IID (Nc=1,5,10,50) | Same as above                        |
| Local Steps        | Effect of J on convergence            | J=4,8,16 with adjusted rounds        |
| Sparse Fine-tuning | Task arithmetic in FL                 | Sparsity=0.9, least_sensitive        |

### Extension: Mask Strategy Comparison

Compares different strategies for selecting which parameters to update:

1. **Least Sensitive** (default): Updates parameters with low Fisher Information, minimizing interference during aggregation
2. **Most Sensitive**: Updates high-sensitivity parameters
3. **Lowest Magnitude**: Updates parameters with small absolute values
4. **Highest Magnitude**: Updates parameters with large absolute values
5. **Random**: Baseline with random parameter selection

## Output Files

After running experiments:

```
logs/
├── centralized_*.log          # Training logs
├── centralized_*_curves.png   # Training curves
├── fedavg_*.log
├── strategy_comparison.png    # Extension results
└── all_results.json           # Aggregated results

checkpoints/
├── centralized_latest.pt      # Latest checkpoint
├── centralized_epoch*.pt      # Epoch checkpoints
└── ...
```

## Configuration

Edit `configs/default.yaml` or pass command-line arguments:

```yaml
# Key configuration options
federated:
  num_clients: 100 # K: Total clients
  participation_rate: 0.1 # C: Fraction per round
  local_steps: 4 # J: Local SGD steps
  num_rounds: 500 # Communication rounds

sharding:
  strategy: "non_iid" # iid or non_iid
  nc: 10 # Classes per client

sparse:
  sparsity_ratio: 0.9 # Fraction to freeze
  calibration_rounds: 5 # Fisher averaging rounds
  mask_strategy: "least_sensitive"
```

## References

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
2. Iurada et al., "Efficient Model Editing with Task-Localized Sparse Fine-tuning", ICLR 2025
3. Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
4. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers (DINO)", ICCV 2021

## Contributing

This is a course project. For questions or issues, contact the project maintainers.

## License

This project is for educational purposes as part of a university course.
