"""Incremental Model Training Script.

This script trains models with proper train/val/test splits and monitors
for overfitting. Supports:
- Statistical baseline
- Minimal LSTM
- Incremental LSTM versions

Logs training and validation curves to detect overfitting.

Usage:
    python src/train_incremental.py --model minimal_lstm
    python src/train_incremental.py --model statistical_baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    setup_logger, log_header, log_separator, log_config,
    log_model_summary, log_epoch, get_device, save_checkpoint
)

logger = setup_logger("incremental_training")


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def load_data_splits() -> Tuple[np.ndarray, ...]:
    """Load data and split into train/val/test."""
    X = np.load(config.X_FILE)
    Y = np.load(config.Y_FILE)
    
    # Train/val/test split
    test_ratio = 1.0 - config.TRAIN_RATIO - config.VAL_RATIO
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=test_ratio, random_state=config.RANDOM_SEED, stratify=Y
    )
    
    val_ratio_adj = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_ratio_adj,
        random_state=config.RANDOM_SEED, stratify=Y_trainval
    )
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def train_statistical_baseline(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
) -> Dict:
    """Train and evaluate statistical baseline."""
    from models.statistical_baseline.model import StatisticalBaseline
    
    log_header(logger, "STATISTICAL BASELINE")
    
    model = StatisticalBaseline(classifier="logistic_regression")
    model.fit(X_train, Y_train)
    
    train_acc = model.score(X_train, Y_train)
    val_acc = model.score(X_val, Y_val)
    test_acc = model.score(X_test, Y_test)
    
    train_f1 = f1_score(Y_train, model.predict(X_train), average="macro")
    val_f1 = f1_score(Y_val, model.predict(X_val), average="macro")
    test_f1 = f1_score(Y_test, model.predict(X_test), average="macro")
    
    logger.info(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")
    logger.info(f"Val Accuracy:   {val_acc:.4f} | F1: {val_f1:.4f}")
    logger.info(f"Test Accuracy:  {test_acc:.4f} | F1: {test_f1:.4f}")
    
    return {
        "model": "Statistical Baseline",
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "test_f1": test_f1,
    }


def train_minimal_lstm(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    X_test: np.ndarray, Y_test: np.ndarray,
    hidden_size: int = 4,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    patience: int = 20,
    save_path: Optional[Path] = None,
) -> Dict:
    """Train minimal LSTM with overfitting monitoring."""
    from models.lstm_v3.model import MinimalLSTM
    
    log_header(logger, f"MINIMAL LSTM (hidden={hidden_size})")
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Create model
    model = MinimalLSTM(
        input_size=config.WINDOW_SIZE,
        num_features=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_size=hidden_size,
    )
    model = model.to(device)
    
    total_params, trainable_params = model.count_parameters()
    logger.info(f"Parameters: {trainable_params:,}")
    
    # Data loaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(Y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(Y_val)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    
    # History
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }
    
    best_val_loss = float("inf")
    best_epoch = 0
    
    log_separator(logger, "-")
    logger.info("Training...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (outputs.argmax(1) == Y_batch).sum().item()
            train_total += Y_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(1) == Y_batch).sum().item()
                val_total += Y_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
        
        # Early stopping & best model
        if early_stopping(val_loss):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if save_path:
                    save_checkpoint(model, optimizer, epoch, 
                                   {"val_loss": val_loss, "val_acc": val_acc},
                                   save_path)
        
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Test evaluation
    log_separator(logger, "-")
    logger.info(f"Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # Load best model if saved
    if save_path and save_path.exists():
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    Y_test_t = torch.LongTensor(Y_test).to(device)
    
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(1).cpu().numpy()
    
    test_acc = accuracy_score(Y_test, test_preds)
    test_f1 = f1_score(Y_test, test_preds, average="macro")
    
    logger.info(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # Overfitting analysis
    log_separator(logger, "-")
    gap = history["train_acc"][-1] - history["val_acc"][-1]
    logger.info(f"Train-Val accuracy gap: {gap:.4f}")
    if gap > 0.1:
        logger.info("⚠ Overfitting detected! Consider regularization.")
    elif gap < 0.02:
        logger.info("✓ Good generalization. Model may be underfitting.")
    else:
        logger.info("✓ Reasonable generalization.")
    
    return {
        "model": f"Minimal LSTM (h={hidden_size})",
        "num_params": trainable_params,
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1],
        "test_acc": test_acc,
        "test_f1": test_f1,
        "history": history,
        "best_epoch": best_epoch,
    }


def plot_training_curves(results: Dict, save_path: Path) -> None:
    """Plot training and validation curves."""
    history = results["history"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(len(history["train_loss"]))
    
    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - {results['model']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.axhline(y=1/6, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy - {results['model']}")
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Incremental model training")
    parser.add_argument(
        "--model",
        choices=["minimal_lstm", "statistical_baseline", "both"],
        default="both",
        help="Model to train",
    )
    parser.add_argument("--hidden-size", type=int, default=4, help="LSTM hidden size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()
    
    log_header(logger, "INCREMENTAL MODEL TRAINING")
    
    # Load data
    logger.info("Loading data...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_splits()
    
    logger.info(f"Train: {len(Y_train)} samples")
    logger.info(f"Val:   {len(Y_val)} samples")
    logger.info(f"Test:  {len(Y_test)} samples")
    logger.info(f"Random baseline: {1/config.NUM_CLASSES:.4f}")
    log_separator(logger, "-")
    
    all_results = {}
    
    # Statistical baseline
    if args.model in ["statistical_baseline", "both"]:
        all_results["statistical"] = train_statistical_baseline(
            X_train, Y_train, X_val, Y_val, X_test, Y_test
        )
        log_separator(logger, "=")
    
    # Minimal LSTM
    if args.model in ["minimal_lstm", "both"]:
        save_path = config.MODELS_DIR / "minimal_lstm" / "best_model.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_results["minimal_lstm"] = train_minimal_lstm(
            X_train, Y_train, X_val, Y_val, X_test, Y_test,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            save_path=save_path,
        )
        
        # Plot curves
        plot_path = Path("log") / f"training_minimal_lstm_h{args.hidden_size}.png"
        plot_path.parent.mkdir(exist_ok=True)
        plot_training_curves(all_results["minimal_lstm"], plot_path)
        log_separator(logger, "=")
    
    # Summary
    log_header(logger, "RESULTS SUMMARY")
    
    logger.info(f"{'Model':<30} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'F1':<8}")
    logger.info("-" * 70)
    
    for name, r in all_results.items():
        logger.info(
            f"{r['model']:<30} | "
            f"{r['train_acc']:.4f}   | "
            f"{r['val_acc']:.4f}   | "
            f"{r['test_acc']:.4f}   | "
            f"{r.get('test_f1', 0):.4f}"
        )
    
    logger.info("-" * 70)
    logger.info(f"{'Random Baseline':<30} | {1/6:.4f}   | {1/6:.4f}   | {1/6:.4f}   | -")
    
    log_separator(logger, "=")


if __name__ == "__main__":
    main()
