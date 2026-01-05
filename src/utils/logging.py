"""
Logging Utilities

Provides consistent logging setup for experiments with support
for both console and file logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import wandb


def setup_logging(
    log_dir: str = "./logs",
    experiment_name: str = "experiment",
    level: int = logging.INFO,
    use_wandb: bool = False,
    wandb_project: str = "federated-task-arithmetic",
    wandb_config: Optional[dict] = None
) -> logging.Logger:
    """
    Setup logging for an experiment.
    
    Creates both console and file handlers with consistent formatting.
    Optionally initializes Weights & Biases for experiment tracking.
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        level: Logging level
        use_wandb: Whether to initialize W&B
        wandb_project: W&B project name
        wandb_config: Configuration dict for W&B
    
    Returns:
        Configured logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"{experiment_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Initialize W&B if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config=wandb_config
        )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "experiment") -> logging.Logger:
    """
    Get existing logger or create a basic one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Add basic handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    step: int,
    prefix: str = "",
    use_wandb: bool = False
) -> None:
    """
    Log metrics to console, file, and optionally W&B.
    
    Args:
        logger: Logger instance
        metrics: Dict of metric name -> value
        step: Current step/epoch/round
        prefix: Prefix for metric names
        use_wandb: Whether to log to W&B
    """
    # Format metrics for logging
    metric_strs = [f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}" 
                   for k, v in metrics.items()]
    logger.info(f"Step {step} | " + " | ".join(metric_strs))
    
    # Log to W&B
    if use_wandb and wandb.run is not None:
        wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
