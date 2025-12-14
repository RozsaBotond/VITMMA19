"""Model training script for Bull/Bear Flag Detector.

This script defines the model architecture and runs the training loop:
1. Loads preprocessed data (X.npy, Y.npy)
2. Splits into train/val/test sets
3. Creates data loaders
4. Initializes model and optimizer
5. Trains with early stopping
6. Saves best model checkpoint

Usage:
    python src/02-training.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    setup_logger, log_header, log_separator, log_config,
    log_model_summary, log_epoch, get_device, save_checkpoint
)
from models.lstm_v1.model import LSTMv1
from models.lstm_v1 import config as model_config


logger = setup_logger("training")


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # First epoch, save model
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True  # Improved, save model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data."""
    logger.info(f"Loading data from: {config.DATA_DIR}")
    
    X = np.load(config.X_FILE)
    Y = np.load(config.Y_FILE)
    
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  Y shape: {Y.shape}")
    logger.info(f"  Unique labels: {np.unique(Y)}")
    
    return X, Y


def create_data_loaders(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders."""
    
    # Split into train+val and test
    test_ratio = 1.0 - train_ratio - val_ratio
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=random_seed, stratify=Y
    )
    
    # Split train+val into train and val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_ratio_adjusted,
        random_state=random_seed, stratify=Y_trainval
    )
    
    logger.info("Data split:")
    logger.info(f"  Train: {len(Y_train)} samples")
    logger.info(f"  Val:   {len(Y_val)} samples")
    logger.info(f"  Test:  {len(Y_test)} samples")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.LongTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)
    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.LongTensor(Y_test)
    
    # Create datasets
    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_ds = TensorDataset(X_val_t, Y_val_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(Y_batch).sum().item()
        total += Y_batch.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(Y_batch).sum().item()
        total += Y_batch.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train() -> None:
    """Main training function."""
    log_header(logger, "TRAINING PIPELINE")
    
    # Log configuration
    training_config = {
        **config.get_config_dict(),
        **model_config.get_config(),
    }
    log_config(logger, training_config, "Training Configuration")
    
    # Device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    log_separator(logger, "-")
    X, Y = load_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, Y,
        batch_size=model_config.BATCH_SIZE,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        random_seed=config.RANDOM_SEED,
    )
    
    # Create model
    log_separator(logger, "-")
    model = LSTMv1(
        input_size=config.WINDOW_SIZE,
        num_features=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_size=model_config.HIDDEN_SIZE,
        num_layers=model_config.NUM_LAYERS,
        dropout=model_config.LSTM_DROPOUT,
        bidirectional=model_config.BIDIRECTIONAL,
        fc_hidden_sizes=model_config.FC_HIDDEN_SIZES,
        fc_dropout=model_config.FC_DROPOUT,
    )
    model = model.to(device)
    
    # Log model summary
    log_model_summary(logger, model, (config.WINDOW_SIZE, config.NUM_FEATURES))
    
    # Loss function with class weights for imbalanced data
    class_counts = np.bincount(Y)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Using weighted CrossEntropyLoss")
    logger.info(f"  Class weights: {class_weights.cpu().numpy()}")
    
    # Optimizer
    if model_config.OPTIMIZER.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_config.LEARNING_RATE,
            weight_decay=model_config.WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.LEARNING_RATE,
            weight_decay=model_config.WEIGHT_DECAY,
        )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=model_config.LR_PATIENCE,
        factor=model_config.LR_FACTOR,
        min_lr=model_config.LR_MIN,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=model_config.EARLY_STOPPING_PATIENCE,
        min_delta=model_config.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )
    
    # Training loop
    log_header(logger, "TRAINING PROGRESS")
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    for epoch in range(1, model_config.EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log progress
        log_epoch(
            logger, epoch, model_config.EPOCHS,
            train_loss, train_acc, val_loss, val_acc, current_lr
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping check
        if early_stopping(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save best model
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_loss": val_loss, "val_acc": val_acc},
                    config.BEST_MODEL_PATH
                )
                logger.info(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    log_separator(logger, "=")
    logger.info(f"Training complete!")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {config.BEST_MODEL_PATH}")
    log_separator(logger, "=")


if __name__ == "__main__":
    train()
