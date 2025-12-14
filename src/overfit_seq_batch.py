#!/usr/bin/env python3
"""Overfit test for Sequence Labeling LSTM.

Step 2 of incremental modeling: verify the model can overfit a small batch.
If the model can't overfit 32 samples, it's too small or has bugs.

Usage:
    python -m src.overfit_seq_batch --batch-size 32 --epochs 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_DIR, DEVICE, SEED
from src.utils import set_seed
from src.metrics import compute_all_metrics
from models.seq_lstm.model import SeqLabelingLSTM, compute_class_weights
from models.seq_lstm.config import CONFIG


def overfit_batch(
    batch_size: int = 32,
    epochs: int = 200,
    hidden_size: int = CONFIG["hidden_size"],
    num_layers: int = CONFIG["num_layers"],
    learning_rate: float = 1e-3,
    target_acc: float = 0.95,
    seed: int = SEED,
):
    """Try to overfit a small batch.
    
    Args:
        batch_size: Number of samples to try overfitting
        epochs: Maximum epochs
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate
        target_acc: Target accuracy to stop at
        seed: Random seed
    """
    set_seed(seed)
    
    print("=" * 60)
    print("SEQUENCE LABELING OVERFIT TEST")
    print("=" * 60)
    
    # Load data
    X = np.load(DATA_DIR / "X_seq.npy")
    Y = np.load(DATA_DIR / "Y_seq.npy")
    
    print(f"Full dataset: X={X.shape}, Y={Y.shape}")
    
    # Take small batch
    indices = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
    X_batch = X[indices]
    Y_batch = Y[indices]
    
    print(f"Overfit batch: X={X_batch.shape}, Y={Y_batch.shape}")
    
    # Check label distribution in batch
    unique, counts = np.unique(Y_batch, return_counts=True)
    print("\nLabel distribution in batch:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} ({100*count/Y_batch.size:.1f}%)")
    
    # Create tensors
    X_t = torch.FloatTensor(X_batch).to(DEVICE)
    Y_t = torch.LongTensor(Y_batch).to(DEVICE)
    
    # Compute class weights with balanced mode for sparse patterns
    class_weights = compute_class_weights(
        Y_t.cpu(), 
        num_classes=7,
        max_weight=CONFIG.get("max_class_weight", 10.0),
        balance_mode="balanced",
    ).to(DEVICE)
    print(f"\nClass weights (balanced): {class_weights.tolist()}")
    
    # Create model
    model = SeqLabelingLSTM(
        input_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=7,
        dropout=0.0,  # No dropout for overfit test
        bidirectional=False,
    ).to(DEVICE)
    
    print(f"\nModel: SeqLabelingLSTM")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {model.num_parameters:,}")
    
    # Loss and optimizer - use capped class weights for balance
    # Cap weights at 5x to avoid instability with missing classes
    capped_weights = class_weights.clamp(max=5.0)
    capped_weights = capped_weights / capped_weights.mean()  # Re-normalize
    print(f"  Capped weights: {capped_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=capped_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)
    
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Forward
        logits = model(X_t)  # (batch, seq_len, 7)
        logits_flat = logits.view(-1, 7)
        targets_flat = Y_t.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == Y_t).float().mean().item()
            
            # Per-class accuracy
            none_mask = Y_t == 0
            pattern_mask = Y_t > 0
            
            none_acc = (preds[none_mask] == Y_t[none_mask]).float().mean().item() if none_mask.any() else 0
            pattern_acc = (preds[pattern_mask] == Y_t[pattern_mask]).float().mean().item() if pattern_mask.any() else 0
        
        if acc > best_acc:
            best_acc = acc
        
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                f"Acc: {acc:.4f} | None: {none_acc:.4f} | Pattern: {pattern_acc:.4f}"
            )
        
        # Early success
        if acc >= target_acc:
            print(f"\n✓ Target accuracy {target_acc:.0%} reached at epoch {epoch}!")
            break
    
    print("-" * 60)
    print(f"\nFinal Results:")
    print(f"  Best accuracy: {best_acc:.4f}")
    
    # Compute full metrics
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    metrics = compute_all_metrics(Y_batch, preds, probs)
    print(metrics.summary())
    
    if best_acc >= target_acc:
        print("\n✓ OVERFIT TEST PASSED")
        print(f"  Model can learn from {batch_size} samples")
        return True
    else:
        print("\n✗ OVERFIT TEST FAILED")
        print(f"  Model could not reach {target_acc:.0%} accuracy")
        print("  Consider: larger hidden_size, more layers, or check data")
        return False


def main():
    parser = argparse.ArgumentParser(description="Overfit test for sequence labeling")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=CONFIG["hidden_size"])
    parser.add_argument("--num-layers", type=int, default=CONFIG["num_layers"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    success = overfit_batch(
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        target_acc=args.target_acc,
        seed=args.seed,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
