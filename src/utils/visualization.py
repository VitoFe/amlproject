import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def _save_figure(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to path if provided, creating parent directories if needed."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    loss_keys = [k for k in metrics_history if 'loss' in k.lower()]
    acc_keys = [k for k in metrics_history if 'accuracy' in k.lower() or 'acc' in k.lower()]
    
    ax = axes[0, 0]
    for key in loss_keys:
        if metrics_history[key]:
            ax.plot(metrics_history[key], label=key)
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for key in acc_keys:
        if metrics_history[key]:
            ax.plot(metrics_history[key], label=key)
    ax.set_xlabel('Epoch/Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        ax.plot(metrics_history['train_loss'], label='Train')
        ax.plot(metrics_history['val_loss'], label='Validation')
        ax.set_xlabel('Epoch/Round')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No train/val loss data', ha='center', va='center')
        ax.set_axis_off()
    
    ax = axes[1, 1]
    if 'learning_rate' in metrics_history:
        ax.plot(metrics_history['learning_rate'])
        ax.set_xlabel('Epoch/Round')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center')
        ax.set_axis_off()
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "test_accuracy",
    title: str = "Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison bar chart across different methods/settings.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]
    
    bars = ax.bar(methods, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(methods))))
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Method')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_heterogeneity_comparison(
    results: Dict[str, Dict[int, float]],
    nc_values: List[int],
    metric: str = "test_accuracy",
    title: str = "Effect of Data Heterogeneity",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison across different heterogeneity levels (Nc).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(nc_values))
    width = 0.8 / len(results)
    
    for i, (method, values) in enumerate(results.items()):
        method_values = [values.get(nc, 0) for nc in nc_values]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, method_values, width, label=method)
    
    ax.set_xlabel('Number of Classes per Client (Nc)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(nc) for nc in nc_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_local_steps_comparison(
    results: Dict[int, Dict[str, float]],
    local_steps: List[int],
    metric: str = "test_accuracy",
    title: str = "Effect of Local Steps",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison across different numbers of local steps (J).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = [results.get(j, {}).get(metric, 0) for j in local_steps]
    
    ax.plot(local_steps, values, 'o-', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Local Steps (J)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig


def plot_sparsity_comparison(
    results: Dict[float, Dict[str, float]],
    sparsity_ratios: List[float],
    metric: str = "test_accuracy",
    title: str = "Effect of Sparsity Ratio",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot comparison across different sparsity ratios.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    values = [results.get(s, {}).get(metric, 0) for s in sparsity_ratios]
    
    ax.plot(sparsity_ratios, values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    
    ax.set_xlabel('Sparsity Ratio (fraction of params frozen)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, save_path)
    
    return fig
