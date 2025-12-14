"""Incremental Model Training with Hyperparameter Optimization.

Trains multiple versions of each model architecture with different hyperparameters.
Includes:
- Statistical baseline (non-neural)
- LSTM versions (v2a, v2b, v2c)
- Transformer versions (v1a, v1b, v1c)
- Hierarchical versions (v1a, v1b)

All models use learning rate scheduling (cosine annealing with warmup).

Usage:
    python src/train_incremental_hp.py
    python src/train_incremental_hp.py --models lstm_v2a transformer_v1a
    python src/train_incremental_hp.py --models baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import setup_logger, set_seed, get_device
from normalization import OHLCScaler
from augmentation import TimeSeriesAugmenter, balance_dataset_with_augmentation

# Import models
from models.lstm_v2.model import SeqLabelingLSTM
from models.transformer.model import SeqTransformer
from models.cnn1d.model import SeqCNN1D
from models.cnn_lstm.model import SeqCNNLSTM
from models.baseline.statistical import StatisticalBaseline

logger = setup_logger("train_incremental_hp")


# ============================================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================================

# LSTM v2 versions - varying hidden size, layers, dropout, LR
LSTM_CONFIGS = {
    "lstm_v2a": {
        "name": "LSTM v2a (baseline config)",
        "description": "Baseline LSTM: 2 layers, 128 hidden, moderate dropout",
        "model_params": {
            "input_size": 4,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.3,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 10,
        },
    },
    "lstm_v2b": {
        "name": "LSTM v2b (heavy regularization)",
        "description": "Strong regularization: high dropout, weight decay, smaller model",
        "model_params": {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.5,  # High dropout
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-3,  # Strong L2 regularization
            "epochs": 200,
            "batch_size": 16,
            "scheduler": "cosine_warm_restarts",
            "warmup_epochs": 15,
            "T_0": 50,
        },
    },
    "lstm_v2c": {
        "name": "LSTM v2c (minimal capacity)",
        "description": "Very small LSTM: 1 layer, 32 hidden, prevents overfitting",
        "model_params": {
            "input_size": 4,
            "hidden_size": 32,
            "num_layers": 1,
            "num_classes": 7,
            "dropout": 0.4,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,  # Moderate L2
            "epochs": 150,
            "batch_size": 32,
            "scheduler": "cosine",
            "warmup_epochs": 10,
        },
    },
    "lstm_v2d": {
        "name": "LSTM v2d (focal loss tuned)",
        "description": "Focal loss γ=3, aggressive class weighting, regularized",
        "model_params": {
            "input_size": 4,
            "hidden_size": 48,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.4,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 8e-4,
            "weight_decay": 1e-3,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 10,
            "focal_loss": True,
            "focal_gamma": 3.0,
        },
    },
    "lstm_v2e": {
        "name": "LSTM v2e (conservative - NO class weights)",
        "description": "No class weighting, forces model to predict None more often",
        "model_params": {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.3,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 10,
            "no_class_weights": True,  # Key: don't use class weights
        },
    },
    "lstm_v2f": {
        "name": "LSTM v2f (inverse class weights)",
        "description": "Penalize false alarms more than missed detections",
        "model_params": {
            "input_size": 4,
            "hidden_size": 64,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.3,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 10,
            "inverse_class_weights": True,  # Penalize None class less
        },
    },
}

# Transformer versions
TRANSFORMER_CONFIGS = {
    "transformer_v1a": {
        "name": "Transformer v1a (baseline)",
        "description": "Baseline Transformer: 3 layers, 4 heads, d=64",
        "model_params": {
            "input_size": 4,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 3,
            "num_classes": 7,
            "dropout": 0.2,
            "dim_feedforward": 256,
        },
        "train_params": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 20,
        },
    },
    "transformer_v1b": {
        "name": "Transformer v1b (deeper)",
        "description": "Deeper Transformer: 4 layers, 4 heads, d=64",
        "model_params": {
            "input_size": 4,
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 4,
            "num_classes": 7,
            "dropout": 0.3,
            "dim_feedforward": 256,
        },
        "train_params": {
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
            "epochs": 200,
            "batch_size": 16,
            "scheduler": "cosine_warm_restarts",
            "warmup_epochs": 25,
            "T_0": 40,
        },
    },
    "transformer_v1c": {
        "name": "Transformer v1c (wider)",
        "description": "Wider Transformer: 2 layers, 8 heads, d=128",
        "model_params": {
            "input_size": 4,
            "d_model": 128,
            "nhead": 8,
            "num_encoder_layers": 2,
            "num_classes": 7,
            "dropout": 0.2,
            "dim_feedforward": 512,
        },
        "train_params": {
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "epochs": 150,
            "batch_size": 8,
            "scheduler": "one_cycle",
            "warmup_epochs": 15,
        },
    },
}

# Hierarchical LSTM versions (trained as end-to-end 7-class in this script)
HIERARCHICAL_CONFIGS = {
    "hierarchical_v1a": {
        "name": "Hierarchical v1a (baseline)",
        "description": "Larger LSTM tuned for 7-class hierarchical pattern",
        "model_params": {
            "input_size": 4,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 7,
            "dropout": 0.3,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "epochs": 100,
            "batch_size": 16,
            "scheduler": "cosine",
            "warmup_epochs": 10,
        },
    },
    "hierarchical_v1b": {
        "name": "Hierarchical v1b (larger)",
        "description": "Larger LSTM with more capacity",
        "model_params": {
            "input_size": 4,
            "hidden_size": 192,
            "num_layers": 3,
            "num_classes": 7,
            "dropout": 0.4,
            "bidirectional": True,
        },
        "train_params": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "epochs": 150,
            "batch_size": 16,
            "scheduler": "reduce_on_plateau",
            "warmup_epochs": 15,
        },
    },
}

# Combine all configs
ALL_CONFIGS = {
    **LSTM_CONFIGS,
    **TRANSFORMER_CONFIGS,
    **HIERARCHICAL_CONFIGS,
}


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 10,
    steps_per_epoch: int = 1,
    **kwargs
) -> Tuple[Any, str]:
    """Create learning rate scheduler with warmup.
    
    Returns:
        Tuple of (scheduler, update_mode) where update_mode is 'step' or 'epoch'
    """
    
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    if scheduler_type == "cosine":
        # Cosine annealing after warmup
        base_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )
        warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
        return ChainedScheduler(warmup_scheduler, base_scheduler, warmup_epochs), "epoch"
    
    elif scheduler_type == "cosine_warm_restarts":
        T_0 = kwargs.get("T_0", 50)
        return CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=T_0, 
            T_mult=2,
            eta_min=1e-6
        ), "epoch"
    
    elif scheduler_type == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_epochs / epochs,
            anneal_strategy='cos',
        ), "step"
    
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
        ), "metric"
    
    else:
        # Default: cosine
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6), "epoch"


class ChainedScheduler:
    """Chain warmup scheduler with main scheduler."""
    
    def __init__(self, warmup_scheduler, main_scheduler, warmup_epochs):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, metric=None):
        if self.current_epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()
        self.current_epoch += 1
    
    def get_last_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return self.warmup_scheduler.get_last_lr()
        return self.main_scheduler.get_last_lr()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load sequence data."""
    X = np.load(config.DATA_DIR / "X_seq.npy")
    Y = np.load(config.DATA_DIR / "Y_seq.npy")
    with open(config.DATA_DIR / "metadata_seq.json") as f:
        metadata = json.load(f)
    logger.info(f"Loaded data: X={X.shape}, Y={Y.shape}")
    return X, Y, metadata


def prepare_data(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 16,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare train/val/test dataloaders."""
    
    # Stratified split based on dominant label
    n_samples = len(X)
    dominant_labels = []
    for y in Y:
        non_zero_mask = y > 0
        if non_zero_mask.sum() > 0:
            unique, counts = np.unique(y[non_zero_mask], return_counts=True)
            dominant_labels.append(unique[np.argmax(counts)])
        else:
            dominant_labels.append(0)
    dominant_labels = np.array(dominant_labels)
    
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=dominant_labels
    )
    temp_labels = dominant_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # Normalize
    scaler = OHLCScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Augment
    if augment:
        augmenter = TimeSeriesAugmenter(seed=42)
        X_train_aug, Y_train_aug = balance_dataset_with_augmentation(
            X_train_norm, Y_train, augmenter=augmenter, seed=42
        )
        logger.info(f"  Augmented: {len(X_train_aug)} train samples")
    else:
        X_train_aug, Y_train_aug = X_train_norm, Y_train
    
    # Class weights
    unique, counts = np.unique(Y_train_aug, return_counts=True)
    total = len(Y_train_aug.flatten())
    class_weights = torch.tensor([total / (len(unique) * c) for c in counts], dtype=torch.float32)
    class_weights = torch.clamp(class_weights, max=5.0)
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_aug, dtype=torch.float32),
            torch.tensor(Y_train_aug, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_norm, dtype=torch.float32),
            torch.tensor(Y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
    )
    
    return train_loader, val_loader, test_loader, class_weights, X_train_norm, Y_train


# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_neural_model(
    model_key: str,
    model_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a neural network model."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_cfg['name']}")
    logger.info(f"Description: {model_cfg['description']}")
    logger.info(f"{'='*60}")
    
    model_params = model_cfg["model_params"]
    train_params = model_cfg["train_params"]
    
    # Create model
    if "lstm" in model_key and "hierarchical" not in model_key:
        model = SeqLabelingLSTM(**model_params)
    elif "transformer" in model_key:
        model = SeqTransformer(**model_params)
    elif "hierarchical" in model_key:
        # Hierarchical uses simplified LSTM for direct 7-class prediction
        # Full hierarchical training is in 03-training-hierarchical.py
        model = SeqLabelingLSTM(**{**model_params, "num_classes": 7})
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
    )
    
    # Scheduler
    scheduler, sched_mode = create_scheduler(
        optimizer,
        train_params["scheduler"],
        train_params["epochs"],
        train_params.get("warmup_epochs", 10),
        len(train_loader),
        T_0=train_params.get("T_0", 50),
    )
    
    # Handle class weights options
    if train_params.get("no_class_weights", False):
        # No class weights - model will naturally predict majority class more
        loss_weights = None
        logger.info("  Using NO class weights (conservative mode)")
    elif train_params.get("inverse_class_weights", False):
        # Inverse: make "None" class have HIGHER weight (penalize false alarms)
        inverse_weights = 1.0 / (class_weights + 0.1)
        inverse_weights = inverse_weights / inverse_weights.sum() * len(inverse_weights)
        loss_weights = inverse_weights.to(device)
        logger.info("  Using INVERSE class weights (penalize false alarms)")
    else:
        loss_weights = class_weights.to(device)
    
    # Loss - support focal loss for better rare class handling
    if train_params.get("focal_loss", False):
        from models.lstm_v2.model import FocalLoss
        gamma = train_params.get("focal_gamma", 2.0)
        criterion = FocalLoss(
            gamma=gamma,
            weight=loss_weights,
            label_smoothing=0.1,
        )
        logger.info(f"  Using Focal Loss (γ={gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # Training loop
    early_stopping = EarlyStopping(patience=30)
    best_f1 = 0.0
    best_state = None
    history = {"train_loss": [], "val_f1": [], "lr": []}
    
    start_time = time.time()
    
    for epoch in range(train_params["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            
            output_flat = output.reshape(-1, output.shape[-1])
            target_flat = Y_batch.reshape(-1)
            
            loss = criterion(output_flat, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if sched_mode == "step":
                scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                preds = output.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(Y_batch.numpy().flatten())
        
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # LR scheduling
        current_lr = optimizer.param_groups[0]['lr']
        if sched_mode == "epoch":
            scheduler.step()
        elif sched_mode == "metric":
            scheduler.step(val_f1)
        
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict().copy()
        
        if epoch % 20 == 0 or epoch == train_params["epochs"] - 1:
            logger.info(f"  Epoch {epoch:3d}: loss={train_loss:.4f}, val_f1={val_f1:.4f}, lr={current_lr:.2e}")
        
        if early_stopping(val_f1):
            logger.info(f"  Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Load best and test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            preds = output.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(Y_batch.numpy().flatten())
    
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)
    
    # Detection metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pattern_mask_true = all_labels > 0
    pattern_mask_pred = all_preds > 0
    
    detection_rate = pattern_mask_pred[pattern_mask_true].mean() if pattern_mask_true.sum() > 0 else 0
    false_alarm = pattern_mask_pred[~pattern_mask_true].mean() if (~pattern_mask_true).sum() > 0 else 0
    
    logger.info(f"\n  Test Results:")
    logger.info(f"    Accuracy: {test_acc:.4f}")
    logger.info(f"    F1 (weighted): {test_f1:.4f}")
    logger.info(f"    Detection Rate: {detection_rate:.4f}")
    logger.info(f"    False Alarm: {false_alarm:.4f}")
    logger.info(f"    Training Time: {training_time:.1f}s")
    
    # Save model
    save_dir = Path(__file__).parent.parent / "models" / model_key
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_dir / "best_model.pth")
    
    with open(save_dir / "config.json", "w") as f:
        json.dump({
            "model_params": model_params,
            "train_params": train_params,
            "results": {
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "detection_rate": detection_rate,
                "false_alarm": false_alarm,
                "n_params": n_params,
                "training_time": training_time,
            }
        }, f, indent=2)
    
    return {
        "name": model_cfg["name"],
        "n_params": n_params,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "detection_rate": detection_rate,
        "false_alarm": false_alarm,
        "training_time": training_time,
        "best_val_f1": best_f1,
    }


def train_baseline(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train and evaluate statistical baseline."""
    
    logger.info(f"\n{'='*60}")
    logger.info("Training: Statistical Baseline")
    logger.info("Description: Trend + volatility-based rule detection")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    baseline = StatisticalBaseline(
        window_size=256,
        trend_window=20,
        volatility_window=14,
        consolidation_window=10,
        min_trend_strength=0.5,
        volatility_threshold=0.7,
    )
    
    baseline.fit(X_train, Y_train)
    
    predictions = baseline.predict(X_test)
    
    training_time = time.time() - start_time
    
    # Metrics
    all_preds = predictions.flatten()
    all_labels = Y_test.flatten()
    
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_acc = accuracy_score(all_labels, all_preds)
    
    pattern_mask_true = all_labels > 0
    pattern_mask_pred = all_preds > 0
    detection_rate = pattern_mask_pred[pattern_mask_true].mean() if pattern_mask_true.sum() > 0 else 0
    false_alarm = pattern_mask_pred[~pattern_mask_true].mean() if (~pattern_mask_true).sum() > 0 else 0
    
    logger.info(f"\n  Test Results:")
    logger.info(f"    Accuracy: {test_acc:.4f}")
    logger.info(f"    F1 (weighted): {test_f1:.4f}")
    logger.info(f"    Detection Rate: {detection_rate:.4f}")
    logger.info(f"    False Alarm: {false_alarm:.4f}")
    
    # Save
    save_dir = Path(__file__).parent.parent / "models" / "baseline"
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(str(save_dir / "baseline_stats.json"))
    
    return {
        "name": "Statistical Baseline",
        "n_params": baseline.get_num_parameters(),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "detection_rate": detection_rate,
        "false_alarm": false_alarm,
        "training_time": training_time,
        "best_val_f1": 0.0,  # No validation for baseline
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train models with HP optimization")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline"] + list(ALL_CONFIGS.keys()),
        default=["baseline"] + list(ALL_CONFIGS.keys()),
        help="Models to train",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Load data
    X, Y, metadata = load_data()
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader, class_weights, X_train_raw, Y_train_raw = prepare_data(
        X, Y, args.batch_size, not args.no_augment
    )
    
    # For baseline, we need raw data
    scaler = OHLCScaler()
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    X_train_raw = X[train_idx]
    Y_train_raw = Y[train_idx]
    X_test_raw = X[test_idx]
    Y_test_raw = Y[test_idx]
    
    results = []
    
    for model_key in args.models:
        if model_key == "baseline":
            result = train_baseline(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw)
        else:
            cfg = ALL_CONFIGS[model_key].copy()
            if args.epochs:
                cfg["train_params"]["epochs"] = args.epochs
            result = train_neural_model(
                model_key, cfg, train_loader, val_loader, test_loader, class_weights, device
            )
        
        result["key"] = model_key
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<30} {'Params':>10} {'Accuracy':>10} {'F1':>10} {'Det.Rate':>10}")
    logger.info("-" * 80)
    
    for r in sorted(results, key=lambda x: x["test_f1"], reverse=True):
        logger.info(
            f"{r['name']:<30} {r['n_params']:>10,} {r['test_accuracy']:>10.4f} "
            f"{r['test_f1']:>10.4f} {r['detection_rate']:>10.4f}"
        )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent.parent / "models" / f"hp_comparison_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
