# Project Approach

Documentation of the implementation approach for each component of the project.

---

## Required vs ADDITIONs

| Component              | Required by Instructions                           | Our Enhancements                               |
| ---------------------- | -------------------------------------------------- | ---------------------------------------------- |
| **Model**              | DINO ViT-S/16 on CIFAR-100                         | Dropout, layer freezing, label smoothing       |
| **Optimizer**          | SGDM                                               | Weight decay tuning, warmup                    |
| **LR Scheduler**       | Cosine annealing (suggested)                       | Configurable scheduler types                   |
| **Data Split**         | Create validation set                              | Configurable split ratio                       |
| **IID Sharding**       | Required                                           | -                                              |
| **Non-IID Sharding**   | Nc parameter                                       | Sharding statistics logging                    |
| **FedAvg**             | K=100, C=0.1, J=4                                  | Mixed precision (AMP), LR scheduling per round |
| **SparseSGDM**         | Extend SGDM with gradient masks                    | Momentum buffer handling with masks            |
| **Fisher Information** | Calibrate in multiple rounds                       | Configurable samples per round                 |
| **Mask Strategies**    | Least-sensitive (base), 4 alternatives (extension) | Unified strategy interface                     |
| **Checkpointing**      | Required (Colab interruptions)                     | Auto-pruning, best-K tracking                  |
| **Logging**            | "Experiment logging" mentioned                     | W&B integration, structured logs               |
| **Visualization**      | Report plots required                              | Automated plot generation                      |
| **Early Stopping**     | Not required                                       | Added for efficiency                           |
| **Multiple Runs**      | "Multiple independent runs"                        | Aggregation utilities                          |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Model Choice: DINO ViT-S/16](#2-model-choice-dino-vit-s16)
3. [Centralized Baseline](#3-centralized-baseline)
4. [Federated Learning (FedAvg)](#4-federated-learning-fedavg)
5. [Data Sharding Strategies](#5-data-sharding-strategies)
6. [Task Arithmetic & Sparse Fine-tuning](#6-task-arithmetic--sparse-fine-tuning)
7. [Extension: Mask Strategy Comparison](#7-extension-mask-strategy-comparison)
8. [Engineering Decisions](#8-engineering-decisions)

---

## 1. Architecture Overview

> **REQUIRED**: Modular, organized codebase with version control  
> **ADDITION**: Abstract base classes, DRY design patterns

This codebase is organized into modular components that separate concerns: data handling, model definitions, training logic, and utilities. This design allows us to easily switch between centralized, federated, and sparse training modes while reusing common functionality.

The `BaseTrainer` abstract class provides common functionality (optimizer creation, evaluation, checkpointing) that is inherited by specialized trainers, we write the epoch loop once and customize it in subclasses.

---

## 2. Model Choice

> **REQUIRED**: DINO ViT-S/16 pretrained model on CIFAR-100  
> **ADDITION**: Dropout regularization, partial layer freezing, label smoothing

We use a Vision Transformer (ViT) pretrained with DINO on ImageNet. This model learns strong visual features through self-supervision, meaning it can represent images well without needing labeled data during pretraining. We add a simple classification layer on top and fine-tune it for CIFAR-100.

### Technical Details

- **Architecture**: ViT-S/16 (Small variant, 16×16 patch size, 384-dim embeddings, 12 transformer blocks)
- **Parameters**: ~21M total, with most in the frozen backbone
- **Loading**: Via `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`

We add regularization techniques to prevent overfitting on the relatively small CIFAR-100:

- **Dropout** (0.1-0.3) before the classification head
- **Label smoothing** (0.1) in the cross-entropy loss
- **Layer freezing**: To freeze early transformer blocks
- **Weight decay**: L2 regularization in the optimizer

The `create_dino_vit()` function centralizes model creation with configurable regularization.

---

## 3. Centralized Baseline

> **REQUIRED**: Train centralized baseline with SGDM, cosine annealing, hyperparameter search  
> **ADDITION**: Early stopping, configurable schedulers, hyperparameter search utilities

Before experimenting with federated learning, we train a "centralized" model where all data is accessible at once.

### Technical Details

**Configuration:**

- **Optimizer**: SGD with Momentum (0.9), weight decay (1e-4)
- **Learning Rate**: 0.001 with cosine annealing scheduler
- **Epochs**: 50 (with early stopping, patience=10)
- **Batch Size**: 64

**Implementation:**

1. **Validation Split**: We create a 10% validation split from the training data for hyperparameter tuning, since CIFAR-100 doesn't provide one.
2. **Early Stopping**: Monitors validation accuracy to prevent overfitting and save compute.
3. **Checkpointing**: Saves model state periodically and keeps best N checkpoints.

The `CentralizedTrainer` extends `BaseTrainer` and implements a standard epoch-based training loop.

---

## 4. Federated Learning (FedAvg)

> **REQUIRED**: Implement FedAvg [McMahan et al., 2017] with K=100, C=0.1, J=4, sequential simulation  
> **ADDITION**: Mixed precision (AMP), round-based LR scheduling, sparse evaluation

Federated Averaging (FedAvg) is the foundational algorithm for federated learning. Instead of collecting all data centrally, we simulate a scenario where data is distributed across 100 clients. In each round, a subset of clients updates their local models, and the server averages these updates to improve the global model.

This sequential simulation on a single GPU produces mathematically identical results to true parallel training.

### Technical Details

**Algorithm (per communication round):**

1. **Client Selection**: Randomly select `C×K` clients (C=0.1, K=100, so 10 clients per round)
2. **Local Training**: Each selected client performs `J` local SGD steps (J=4 default)
3. **Aggregation**: Server computes weighted average of client models

**Parameters:**
| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Total clients | K | 100 | Simulates 100 edge devices |
| Participation rate | C | 0.1 | 10% of clients train per round |
| Local steps | J | 4 | Steps before sending update back |
| Communication rounds | T | 500 | Total server-client exchanges |

**Implementation:**

```python
def _client_update(self, client_id, global_state):
    self.model.load_state_dict(global_state)
    optimizer = SGD(self.model.parameters(), lr=self.client_lr)

    for step in range(self.local_steps):
        inputs, targets = next(client_data)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    return self.model.state_dict()

def _aggregate(self, client_states, client_weights):
    # Weighted average based on dataset sizes
    aggregated = {}
    for key in client_states[0]:
        aggregated[key] = sum(w * s[key] for w, s in zip(weights, states))
    return aggregated
```

**Performance Optimizations:**

- **Mixed Precision (AMP)**: Reduces GPU memory and speeds up computation
- **LR Scheduling**: Cosine annealing adapted for round-based training
- **Sparse Evaluation**: Evaluate every N rounds instead of every round

---

## 5. Data Sharding Strategies

> **REQUIRED**: IID sharding + Non-IID sharding with Nc={1,5,10,50}  
> **ADDITION**: Sharding statistics logging, flexible Nc configuration

In reality, data on different devices isn't uniformly distributed. We simulate this "statistical heterogeneity" by controlling how we split data among clients:

- **IID (Independent & Identically Distributed)**: Each client gets a random mix of all 100 classes the ideal, easy case.
- **Non-IID**: Each client only sees `Nc` classes. With Nc=1, each client only has images from one class an extreme case that makes training very challenging.

### Technical Details

**IID Sharding** (`_iid_sharding`):

```python
# Shuffle all indices and split evenly
np.random.shuffle(indices)
client_splits = np.array_split(indices, num_clients)
# Result: each client has ~450 samples across all 100 classes
```

**Non-IID Sharding** (`_non_iid_sharding`):

```python
# 1. Group samples by class
class_indices = {c: samples_of_class_c for c in range(100)}

# 2. Assign Nc classes to each client
for client in clients:
    client.classes = select_nc_classes()

# 3. Distribute samples from those classes
for client in clients:
    for class_c in client.classes:
        client.samples.extend(class_indices[c][portion])
```

**Heterogeneity Levels Tested:**
| Nc | Classes per Client | Heterogeneity | Expected Effect |
|----|-------------------|---------------|-----------------|
| 100 | All | None (IID) | Best performance |
| 50 | Half | Low | Minor degradation |
| 10 | Few | Medium | Notable drop |
| 5 | Very few | High | Significant drop |
| 1 | One only | Extreme | Severe degradation |

The key phenomenon is **client drift**: when clients train on very different data distributions, their local updates diverge, and averaging them produces a suboptimal global model.

---

## 6. Task Arithmetic & Sparse Fine-tuning

> **REQUIRED**: Fisher Information for sensitivity, SparseSGDM optimizer, multi-round calibration  
> **ADDITION**: Configurable Fisher samples, mask merging utilities

The core idea from recent research (Iurada et al., 2025) is that not all parameters are equally important for learning new tasks. Some parameters are "low-sensitivity" changing them doesn't affect the model's existing knowledge much. By only updating these parameters, we can learn new information while minimizing interference between different clients.

### Technical Details

**Step 1: Compute Parameter Sensitivity (Fisher Information)**

The Fisher Information approximates how much the loss changes when we modify each parameter:

```python
def compute_fisher_information(model, dataloader, criterion):
    fisher = {name: zeros_like(p) for name, p in model.named_parameters()}

    for inputs, targets in dataloader:
        loss = criterion(model(inputs), targets)
        loss.backward()

        for name, p in model.named_parameters():
            # Fisher ≈ E[gradient²]
            fisher[name] += p.grad ** 2

    return {name: f / n_samples for name, f in fisher.items()}
```

**Step 2: Create Binary Masks**

Based on sensitivity scores, we decide which parameters to update (mask=1) or freeze (mask=0):

```python
def calibrate_gradient_mask(model, sensitivity_scores, sparsity_ratio=0.9):
    # Flatten all scores
    all_scores = concat([s.flatten() for s in sensitivity_scores])

    # Find threshold that keeps (1 - sparsity_ratio) of parameters
    # For sparsity_ratio=0.9, we freeze 90% and update 10%
    threshold = quantile(all_scores, 1 - sparsity_ratio)

    # Least-sensitive: update parameters BELOW threshold
    masks = {name: (scores <= threshold).float() for name, scores in sensitivity_scores}
    return masks
```

**Step 3: SparseSGDM Optimizer**

Our custom optimizer applies the mask during the update step:

```python
class SparseSGDM(Optimizer):
    def step(self):
        for p in params:
            # Standard SGD with momentum
            d_p = p.grad + weight_decay * p
            momentum_buffer = m * momentum_buffer + (1-dampening) * d_p

            # Apply mask: only update where mask == 1
            if param_name in gradient_masks:
                d_p = d_p * gradient_masks[param_name]

            p -= lr * d_p
```

**Key Hyperparameters:**

- **Sparsity Ratio** (0.5-0.99): What fraction of parameters to freeze. Higher = fewer updates = less interference but potentially less learning.
- **Calibration Rounds** (1-10): How many passes to average Fisher scores. More rounds = more stable masks.

---

## 7. Extension: Mask Strategy Comparison

> **REQUIRED (Guided Extension)**: Compare least-sensitive with: most-sensitive, lowest-magnitude, highest-magnitude, random  
> **ADDITION**: Unified strategy enum, automated comparison experiments

The original paper uses "least-sensitive" parameters, but what if we tried other strategies? Maybe the most important parameters should be updated, or we should use simpler heuristics like parameter magnitude. This extension compares five different masking strategies to understand which works best for federated learning.

### Technical Details

**Strategies Implemented:**

| Strategy            | Mask Criterion                        | Intuition                                 |
| ------------------- | ------------------------------------- | ----------------------------------------- |
| `least_sensitive`   | Keep params with LOW Fisher scores    | Update "safe" params that won't interfere |
| `most_sensitive`    | Keep params with HIGH Fisher scores   | Update the most important params first    |
| `lowest_magnitude`  | Keep params with LOW absolute values  | Small weights might be less important     |
| `highest_magnitude` | Keep params with HIGH absolute values | Update the strongest connections          |
| `random`            | Random selection                      | Baseline comparison                       |

**Implementation:**

```python
def _create_masks_from_scores(scores, sparsity_ratio, strategy):
    all_scores = flatten(scores)

    if strategy in ['least_sensitive', 'lowest_magnitude']:
        # Keep parameters with scores BELOW threshold
        threshold = quantile(all_scores, 1 - sparsity_ratio)
        masks = {name: (score <= threshold) for name, score in scores}
    else:  # most_sensitive, highest_magnitude
        # Keep parameters with scores ABOVE threshold
        threshold = quantile(all_scores, sparsity_ratio)
        masks = {name: (score >= threshold) for name, score in scores}

    return masks
```

**Expected Findings:**
Based on the literature, we expect:

1. **Least-sensitive** to perform best (minimizes interference)
2. **Random** to be surprisingly competitive (baseline)
3. **Most-sensitive/Highest-magnitude** to perform worse (more interference)

---

## 8. Engineering Decisions

> **REQUIRED**: Checkpointing (Colab recovery), experiment logging, reproducibility  
> **ADDITION**: W&B integration, auto-pruning checkpoints, visualization tools, early stopping

We implemented robust checkpointing (to survive Colab disconnections), comprehensive logging, and visualization tools. The code follows best practices like separation of concerns, type hints, and minimal code duplication.

### Technical Details

**Checkpointing System:**

- Saves every N epochs/rounds + on new best validation accuracy
- Stores model, optimizer, scheduler, metrics, and config
- `CheckpointManager` automatically prunes old checkpoints, keeping best K

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': {'val_accuracy': 0.75, ...},
    'config': {...}
}
torch.save(checkpoint, 'checkpoint_epoch50.pt')
```

**Logging:**

- Dual logging to console and file
- Optional Weights & Biases integration
- Structured format: `timestamp | level | message`

**Reproducibility:**

- All random seeds are set via `set_seed(42)`
- Covers: Python, NumPy, PyTorch, CUDA
