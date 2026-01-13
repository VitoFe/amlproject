# Project Approach: Federated Learning Under the Lens of Task Arithmetic

This document explains our implementation approach for each component of the project, providing both high-level intuition for newcomers and technical depth for practitioners.

---

## Legend: Required vs Our Additions

Throughout this document, we use the following markers:

| Marker              | Meaning                                   |
| ------------------- | ----------------------------------------- |
| 📋 **REQUIRED**     | Explicitly specified in `instructions.md` |
| 🚀 **OUR ADDITION** | Enhancement we added beyond requirements  |

### Quick Reference: What Was Required vs What We Added

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

> 📋 **REQUIRED**: Modular, well-organized codebase with version control  
> 🚀 **OUR ADDITION**: Abstract base classes, DRY design patterns

### High-Level Explanation

Our codebase is organized into modular components that separate concerns: data handling, model definitions, training logic, and utilities. This design allows us to easily switch between centralized, federated, and sparse training modes while reusing common functionality.

### Technical Details

```
src/
├── data/           # Dataset loading and client sharding
│   ├── dataset.py      # CIFAR-100 loading, train/val/test splits
│   └── sharding.py     # IID and non-IID client partitioning
├── models/
│   └── dino_vit.py     # DINO ViT wrapper with classification head
├── training/
│   ├── base.py         # Abstract trainer with shared logic
│   ├── centralized.py  # Standard training loop
│   ├── federated.py    # FedAvg implementation
│   └── federated_sparse.py  # Sparse fine-tuning variant
├── optimizers/
│   ├── fisher.py       # Fisher Information & mask calibration
│   └── sparse_sgdm.py  # SGDM with gradient masking
└── utils/              # Logging, checkpointing, visualization
```

The `BaseTrainer` abstract class provides common functionality (optimizer creation, evaluation, checkpointing) that is inherited by specialized trainers. This follows the DRY principle—we write the epoch loop once and customize it in subclasses.

---

## 2. Model Choice: DINO ViT-S/16

> 📋 **REQUIRED**: Use DINO ViT-S/16 pretrained model on CIFAR-100  
> 🚀 **OUR ADDITION**: Dropout regularization, partial layer freezing, label smoothing

### High-Level Explanation

We use a Vision Transformer (ViT) pretrained with DINO on ImageNet. This model learns strong visual features through self-supervision, meaning it can represent images well without needing labeled data during pretraining. We add a simple classification layer on top and fine-tune it for CIFAR-100.

### Technical Details

- **Architecture**: ViT-S/16 (Small variant, 16×16 patch size, 384-dim embeddings, 12 transformer blocks)
- **Parameters**: ~21M total, with most in the frozen backbone
- **Loading**: Via `torch.hub.load('facebookresearch/dino:main', 'dino_vits16')`

We add regularization techniques to prevent overfitting on the relatively small CIFAR-100:

- **Dropout** (0.1-0.3) before the classification head
- **Label smoothing** (0.1) in the cross-entropy loss
- **Layer freezing**: Option to freeze early transformer blocks (e.g., first 6 of 12)
- **Weight decay**: L2 regularization in the optimizer

The `create_dino_vit()` function centralizes model creation with configurable dropout and layer freezing:

```python
model = create_dino_vit(
    num_classes=100,
    freeze_backbone=False,  # Fine-tune all layers
    dropout=0.1,           # Regularization
    freeze_layers=6        # Optional: freeze first N transformer blocks
)
```

---

## 3. Centralized Baseline

> 📋 **REQUIRED**: Train centralized baseline with SGDM, cosine annealing, hyperparameter search  
> 🚀 **OUR ADDITION**: Early stopping, configurable schedulers, hyperparameter search utilities

### High-Level Explanation

Before experimenting with federated learning, we train a "centralized" model where all data is accessible at once—the standard deep learning setup. This gives us an upper bound on what accuracy we can achieve, since federated learning introduces challenges that typically reduce performance.

### Technical Details

**Training Configuration:**

- **Optimizer**: SGD with Momentum (0.9), weight decay (1e-4)
- **Learning Rate**: 0.001 with cosine annealing scheduler
- **Epochs**: 50 (with early stopping, patience=10)
- **Batch Size**: 64

**Key Implementation Points:**

1. **Validation Split**: We create a 10% validation split from the training data for hyperparameter tuning, since CIFAR-100 doesn't provide one.
2. **Early Stopping**: Monitors validation accuracy to prevent overfitting and save compute.
3. **Checkpointing**: Saves model state periodically and keeps best N checkpoints.

The `CentralizedTrainer` extends `BaseTrainer` and implements a standard epoch-based training loop:

```python
for epoch in range(epochs):
    train_loss, train_acc = self._train_epoch(optimizer)
    val_metrics = self.evaluate_val()

    if scheduler:
        scheduler.step()

    if early_stopping(val_metrics['accuracy']):
        break
```

**Expected Results**: With proper hyperparameter tuning, the centralized baseline should achieve **75-80% test accuracy** on CIFAR-100.

---

## 4. Federated Learning (FedAvg)

> 📋 **REQUIRED**: Implement FedAvg [McMahan et al., 2017] with K=100, C=0.1, J=4, sequential simulation  
> 🚀 **OUR ADDITION**: Mixed precision (AMP), round-based LR scheduling, sparse evaluation

### High-Level Explanation

Federated Averaging (FedAvg) is the foundational algorithm for federated learning. Instead of collecting all data centrally, we simulate a scenario where data is distributed across 100 clients (e.g., mobile devices). In each round, a subset of clients updates their local models, and the server averages these updates to improve the global model.

The key insight is that this _sequential simulation_ on a single GPU produces mathematically identical results to true parallel training—we just can't run clients simultaneously.

### Technical Details

**Algorithm (per communication round):**

1. **Client Selection**: Randomly select `C×K` clients (C=0.1, K=100 → 10 clients per round)
2. **Local Training**: Each selected client performs `J` local SGD steps (J=4 default)
3. **Aggregation**: Server computes weighted average of client models

**Key Parameters:**
| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Total clients | K | 100 | Simulates 100 edge devices |
| Participation rate | C | 0.1 | 10% of clients train per round |
| Local steps | J | 4 | Steps before sending update back |
| Communication rounds | T | 500 | Total server-client exchanges |

**Implementation Highlights:**

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

**Expected Results**: FedAvg with IID sharding should achieve **60-70% accuracy**, somewhat lower than centralized due to the distributed nature.

---

## 5. Data Sharding Strategies

> 📋 **REQUIRED**: IID sharding + Non-IID sharding with Nc={1,5,10,50}  
> 🚀 **OUR ADDITION**: Sharding statistics logging, flexible Nc configuration

### High-Level Explanation

In reality, data on different devices isn't uniformly distributed—a phone in Italy might have different image types than one in Japan. We simulate this "statistical heterogeneity" by controlling how we split data among clients:

- **IID (Independent & Identically Distributed)**: Each client gets a random mix of all 100 classes—the ideal, easy case.
- **Non-IID**: Each client only sees `Nc` classes. With Nc=1, each client only has images from one class—an extreme case that makes training very challenging.

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

> 📋 **REQUIRED**: Fisher Information for sensitivity, SparseSGDM optimizer, multi-round calibration  
> 🚀 **OUR ADDITION**: Configurable Fisher samples, mask merging utilities

### High-Level Explanation

The core idea from recent research (Iurada et al., 2025) is that not all parameters are equally important for learning new tasks. Some parameters are "low-sensitivity"—changing them doesn't affect the model's existing knowledge much. By only updating these parameters, we can learn new information while minimizing interference between different clients.

Think of it like editing a document: instead of rewriting everything, we identify which paragraphs can be safely modified without changing the document's core meaning.

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

> 📋 **REQUIRED (Guided Extension)**: Compare least-sensitive with: most-sensitive, lowest-magnitude, highest-magnitude, random  
> 🚀 **OUR ADDITION**: Unified strategy enum, automated comparison experiments

### High-Level Explanation

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

> 📋 **REQUIRED**: Checkpointing (Colab recovery), experiment logging, reproducibility  
> 🚀 **OUR ADDITION**: W&B integration, auto-pruning checkpoints, visualization tools, early stopping

### High-Level Explanation

Good engineering is crucial for ML research. We implemented robust checkpointing (to survive Colab disconnections), comprehensive logging, and visualization tools. The code follows best practices like separation of concerns, type hints, and minimal code duplication.

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

**Performance Optimizations Applied:**
| Optimization | Benefit | Where Used |
|--------------|---------|------------|
| Mixed Precision (AMP) | 2x faster training | FederatedTrainer |
| LR Scheduling | Better convergence | All trainers |
| Early Stopping | Saves compute | All trainers |
| Sparse Evaluation | 5-10x fewer evals | FederatedTrainer |
| Pin Memory | Faster data loading | All DataLoaders |

---

## Summary: How Components Connect

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT FLOW                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load CIFAR-100 → create train/val/test splits           │
│                         │                                    │
│  2. Create DINO ViT ────┼─── Centralized Baseline           │
│                         │         │                          │
│  3. Shard data ─────────┼─── FedAvg (IID)                   │
│     (IID / non-IID)     │         │                          │
│                         │    FedAvg (non-IID, vary Nc)       │
│                         │         │                          │
│  4. Calibrate masks ────┼─── Sparse FedAvg                  │
│     (Fisher Information)│         │                          │
│                         │    Compare mask strategies         │
│                         │         │                          │
│  5. Evaluate & Compare ─┴─────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Each experiment builds on the previous one, with shared infrastructure (checkpointing, logging, model creation) enabling consistent and reproducible results across all configurations.
