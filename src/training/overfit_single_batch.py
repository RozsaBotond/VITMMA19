"""Overfit on a Single Batch - Incremental Modeling Step 2.

This script tests if the model architecture can learn by attempting to
perfectly memorize a small batch of data. If the model cannot reach
near-zero training loss on 32 samples, something is wrong.

Steps:
1. Load a single batch (32 samples)
2. Disable all regularization
3. Train until loss ≈ 0 or accuracy = 100%
4. If it fails → model too small, bug in code, or data issue

Usage:
    python src/overfit_single_batch.py --model minimal_lstm
    python src/overfit_single_batch.py --model statistical_baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import setup_logger, log_header, log_separator, get_device

logger = setup_logger("overfit_test")


def load_single_batch(batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single batch of data."""
    X = np.load(config.X_FILE)
    Y = np.load(config.Y_FILE)
    
    # Take first batch_size samples (or stratified sample)
    if len(X) >= batch_size:
        # Try to get at least one sample per class
        indices = []
        for label in range(config.NUM_CLASSES):
            label_indices = np.where(Y == label)[0]
            if len(label_indices) > 0:
                n_samples = max(1, batch_size // config.NUM_CLASSES)
                indices.extend(label_indices[:n_samples].tolist())
        
        # Fill remaining with random samples
        remaining = batch_size - len(indices)
        if remaining > 0:
            other_indices = [i for i in range(len(X)) if i not in indices]
            indices.extend(other_indices[:remaining])
        
        indices = indices[:batch_size]
    else:
        indices = list(range(len(X)))
    
    return X[indices], Y[indices]


def test_statistical_baseline(X: np.ndarray, Y: np.ndarray) -> Dict:
    """Test statistical baseline on single batch."""
    from models.statistical_baseline.model import StatisticalBaseline
    
    logger.info("Testing Statistical Baseline...")
    
    model = StatisticalBaseline(classifier="logistic_regression")
    model.fit(X, Y)
    
    train_acc = model.score(X, Y)
    predictions = model.predict(X)
    
    return {
        "model": "Statistical Baseline",
        "train_accuracy": train_acc,
        "predictions": predictions,
        "can_overfit": train_acc > 0.9,
    }


def test_minimal_lstm(
    X: np.ndarray,
    Y: np.ndarray,
    max_epochs: int = 1000,
    target_loss: float = 0.01,
) -> Dict:
    """Test minimal LSTM on single batch - try to overfit."""
    from models.lstm_v3.model import MinimalLSTM
    
    logger.info("Testing Minimal LSTM...")
    
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Create model
    model = MinimalLSTM(
        input_size=config.WINDOW_SIZE,
        num_features=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        hidden_size=4,
    )
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    logger.info(f"Model parameters: {trainable_params:,}")
    
    # Prepare data
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.LongTensor(Y).to(device)
    
    # Training setup - no regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    accuracies = []
    
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, Y_tensor)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            acc = (preds == Y_tensor).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(acc)
        
        if epoch % 100 == 0 or epoch == max_epochs - 1:
            logger.info(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Acc: {acc:.4f}")
        
        # Early success
        if loss.item() < target_loss and acc > 0.99:
            logger.info(f"SUCCESS! Overfit achieved at epoch {epoch}")
            break
    
    final_loss = losses[-1]
    final_acc = accuracies[-1]
    
    return {
        "model": "Minimal LSTM",
        "num_params": trainable_params,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "losses": losses,
        "accuracies": accuracies,
        "can_overfit": final_acc > 0.9,
    }


def plot_overfit_results(results: Dict, save_path: Path) -> None:
    """Plot the overfitting curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(len(results["losses"]))
    
    # Loss curve
    ax1.plot(epochs, results["losses"], "b-", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss - {results['model']}")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, results["accuracies"], "g-", linewidth=1)
    ax2.axhline(y=1/6, color="r", linestyle="--", label="Random (16.7%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Training Accuracy - {results['model']}")
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Overfit on a single batch test")
    parser.add_argument(
        "--model",
        choices=["minimal_lstm", "statistical_baseline", "both"],
        default="both",
        help="Model to test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for overfitting test",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum epochs for LSTM overfitting",
    )
    args = parser.parse_args()
    
    log_header(logger, "OVERFIT ON SINGLE BATCH TEST")
    
    # Load data
    logger.info(f"Loading {args.batch_size} samples...")
    X, Y = load_single_batch(args.batch_size)
    logger.info(f"X shape: {X.shape}")
    logger.info(f"Y shape: {Y.shape}")
    logger.info(f"Label distribution: {np.bincount(Y)}")
    log_separator(logger, "-")
    
    results = {}
    
    # Test statistical baseline
    if args.model in ["statistical_baseline", "both"]:
        results["statistical"] = test_statistical_baseline(X, Y)
        logger.info(f"Statistical Baseline Accuracy: {results['statistical']['train_accuracy']:.4f}")
        log_separator(logger, "-")
    
    # Test minimal LSTM
    if args.model in ["minimal_lstm", "both"]:
        results["minimal_lstm"] = test_minimal_lstm(X, Y, args.max_epochs)
        
        # Save plot
        plot_path = Path("log") / "overfit_minimal_lstm.png"
        plot_path.parent.mkdir(exist_ok=True)
        plot_overfit_results(results["minimal_lstm"], plot_path)
        log_separator(logger, "-")
    
    # Summary
    log_header(logger, "SUMMARY")
    
    logger.info(f"Random baseline accuracy: {1/config.NUM_CLASSES:.4f} (1/{config.NUM_CLASSES})")
    
    if "statistical" in results:
        r = results["statistical"]
        status = "✓ PASS" if r["can_overfit"] else "✗ FAIL"
        logger.info(f"Statistical Baseline: {r['train_accuracy']:.4f} - {status}")
    
    if "minimal_lstm" in results:
        r = results["minimal_lstm"]
        status = "✓ PASS" if r["can_overfit"] else "✗ FAIL"
        logger.info(f"Minimal LSTM ({r['num_params']} params): {r['final_accuracy']:.4f} - {status}")
        logger.info(f"  Final loss: {r['final_loss']:.6f}")
    
    log_separator(logger, "=")
    
    # Recommendation
    if results.get("minimal_lstm", {}).get("can_overfit"):
        logger.info("✓ Model can learn! Ready for train/val split training.")
    else:
        logger.info("✗ Model cannot overfit batch. Consider:")
        logger.info("  - Increasing hidden_size")
        logger.info("  - Check for data issues")
        logger.info("  - Verify loss function")


if __name__ == "__main__":
    main()
