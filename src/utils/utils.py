"""Utility functions for the Bull/Bear Flag Detector pipeline.

Common helper functions used across the project.
"""
from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> None:
    """Setup basic logging configuration.
    
    Args:
        log_file: Optional file path for file logging
        level: Logging level
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def setup_logger(
    name: str = "bullflag",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Set up a logger that outputs to stdout (for Docker capture) and optionally to file.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for file logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Format with timestamp, level, and message for clarity
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(funcName)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (stdout for Docker capture)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_separator(logger: logging.Logger, char: str = "=", length: int = 70) -> None:
    """Log a visual separator line."""
    logger.info(char * length)


def log_header(logger: logging.Logger, title: str, char: str = "=") -> None:
    """Log a header with separator lines."""
    log_separator(logger, char)
    logger.info(title.upper())
    log_separator(logger, char)


def log_config(logger: logging.Logger, config: Dict[str, Any], title: str = "CONFIGURATION") -> None:
    """Log configuration parameters in a formatted way.
    
    Args:
        logger: Logger instance
        config: Dictionary of configuration parameters
        title: Section title
    """
    log_header(logger, title)
    for key, value in config.items():
        logger.info(f"  {key:<30} : {value}")
    log_separator(logger, "-")


def log_model_summary(
    logger: logging.Logger,
    model: torch.nn.Module,
    input_shape: tuple,
) -> None:
    """Log model architecture summary with parameter counts.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        input_shape: Example input shape (without batch dimension)
    """
    log_header(logger, "MODEL ARCHITECTURE")
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Input shape: {input_shape}")
    logger.info("")
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        trainable_params += trainable
        logger.info(f"  {name:<25} : {params:>10,} params ({trainable:>10,} trainable)")
    
    logger.info("")
    logger.info(f"{'Total parameters':<25} : {total_params:>10,}")
    logger.info(f"{'Trainable parameters':<25} : {trainable_params:>10,}")
    logger.info(f"{'Non-trainable parameters':<25} : {total_params - trainable_params:>10,}")
    log_separator(logger, "-")


def log_epoch(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: Optional[float] = None,
    val_acc: Optional[float] = None,
    lr: Optional[float] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Log training progress for one epoch.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss (optional)
        val_acc: Validation accuracy (optional)
        lr: Current learning rate (optional)
        extra_metrics: Additional metrics to log
    """
    msg = f"Epoch [{epoch:>4}/{total_epochs}]"
    msg += f" | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
    
    if val_loss is not None:
        msg += f" | Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        msg += f" | Val Acc: {val_acc:.4f}"
    if lr is not None:
        msg += f" | LR: {lr:.6f}"
    
    if extra_metrics:
        for name, value in extra_metrics.items():
            msg += f" | {name}: {value:.4f}"
    
    logger.info(msg)


def log_evaluation(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    title: str = "FINAL EVALUATION",
) -> None:
    """Log evaluation metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of evaluation metrics
        title: Section title
    """
    log_header(logger, title)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:<25} : {value:.4f}")
        elif isinstance(value, (list, tuple)):
            logger.info(f"  {key}:")
            for item in value:
                logger.info(f"    {item}")
        else:
            logger.info(f"  {key:<25} : {value}")
    
    log_separator(logger, "=")


def log_confusion_matrix(
    logger: logging.Logger,
    cm: Any,  # np.ndarray
    labels: list,
) -> None:
    """Log confusion matrix in a readable format.
    
    Args:
        logger: Logger instance
        cm: Confusion matrix (numpy array)
        labels: Class labels
    """
    logger.info("Confusion Matrix:")
    
    # Header
    header = "         " + " ".join(f"{l[:8]:>8}" for l in labels)
    logger.info(header)
    
    # Rows
    for i, row in enumerate(cm):
        row_str = f"{labels[i][:8]:<8}" + " ".join(f"{v:>8}" for v in row)
        logger.info(row_str)
    
    logger.info("")


def log_class_report(
    logger: logging.Logger,
    report: str,
) -> None:
    """Log classification report."""
    logger.info("Classification Report:")
    for line in report.split("\n"):
        logger.info(f"  {line}")


def get_device() -> torch.device:
    """Get the best available device (GPU if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: Path,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint
