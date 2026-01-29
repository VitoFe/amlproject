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

High Fisher values indicate parameters that significantly affect the loss -> high sensitivity.

## Quick Start

Make sure you have uv installed before proceeding ( `pip install uv` )

### Installation

```bash
git clone <repository-url>
cd stuffaml
uv sync
```

### Running Experiments

#### Quick Test Run

```bash
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
uv run python scripts/run_all_experiments.py --full --num-runs 3
```

## Configuration

Edit `configs/default.yaml` or pass command-line arguments.

## License

This project is for educational purposes as part of a university course.
