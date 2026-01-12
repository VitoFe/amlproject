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
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f"{experiment_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=experiment_name,
            config=wandb_config
        )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "experiment") -> logging.Logger:
    logger = logging.getLogger(name)
    
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
    metric_strs = [f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}" 
                   for k, v in metrics.items()]
    logger.info(f"Step {step} | " + " | ".join(metric_strs))
    if use_wandb and wandb.run is not None:
        wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
