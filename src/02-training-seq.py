#!/usr/bin/env python3
"""Training script for Sequence Labeling models.

This script trains per-timestep sequence labeling models on the prepared data.

Supports three modes:
- binary: 2 classes (None vs Pattern) - best for sparse pattern detection
- 3class: 3 classes (None, Bearish, Bullish) - good balance of simplicity/detail  
- 7class: 7 classes (None + 6 pattern types) - requires more data

Usage:
    python -m src.02-training-seq --mode 3class --epochs 200
    python -m src.02-training-seq --mode binary --epochs 100

Features:
    - Per-timestep cross-entropy loss with balanced class weights
    - Comprehensive sequence labeling metrics
    - Early stopping on validation F1 (macro)
    - Checkpointing and logging
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Local imports
from src.config import (
    DATA_DIR, 
    MODELS_DIR, 
    LOG_DIR,
    DEVICE, 
    SEED,
)
from src.utils import setup_logging, set_seed
from src.metrics import compute_all_metrics, SequenceMetrics
from models.lstm_v2.model import SeqLabelingLSTM, compute_class_weights
from models.lstm_v2.config import CONFIG as SEQ_LSTM_CONFIG, CONFIG_7CLASS, CONFIG_BINARY, CLASS_3_NAMES, CLASS_7_NAMES


def convert_labels(Y: np.ndarray, mode: str = "3class") -> np.ndarray:
    """Convert 7-class labels to binary or 3-class.
    
    Args:
        Y: Labels with values 0-6
        mode: "binary", "3class", or "7class"
        
    Returns:
        Converted labels
    """
    if mode == "binary":
        # 0 = None, 1 = Any pattern
        return (Y > 0).astype(np.int64)
    elif mode == "3class":
        # 0 = None, 1 = Bearish (1-3), 2 = Bullish (4-6)
        Y_new = np.zeros_like(Y)
        Y_new[(Y >= 1) & (Y <= 3)] = 1
        Y_new[(Y >= 4) & (Y <= 6)] = 2
        return Y_new
    else:  # 7class
        return Y


def get_sample_dominant_class(Y: np.ndarray) -> np.ndarray:
    """Get dominant (non-zero) class for each sample.
    
    For stratified splitting - classifies each sample by its primary pattern.
    """
    sample_classes = np.zeros(len(Y), dtype=np.int64)
    for i, y in enumerate(Y):
        non_zero = y[y > 0]
        if len(non_zero) > 0:
            # Most common non-zero label
            unique, counts = np.unique(non_zero, return_counts=True)
            sample_classes[i] = unique[counts.argmax()]
        else:
            sample_classes[i] = 0
    return sample_classes


def stratified_split_with_all_classes(
    X: np.ndarray, 
    Y: np.ndarray,
    num_classes: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data ensuring all classes are represented in each split.
    
    Uses stratified splitting based on dominant class per sample.
    Ensures at least 1 sample of each class in train/val/test.
    """
    np.random.seed(seed)
    
    sample_classes = get_sample_dominant_class(Y)
    
    # Count samples per class
    unique_classes = np.unique(sample_classes)
    class_counts = {c: (sample_classes == c).sum() for c in unique_classes}
    
    print("\nSamples per class (before split):")
    for c in range(num_classes):
        count = class_counts.get(c, 0)
        status = "" if count >= 3 else " (NEED MORE FOR ALL SPLITS)" if count > 0 else " (MISSING)"
        print(f"  {c}: {count}{status}")
    
    # For classes with few samples, manually ensure 1 in each split
    train_indices = []
    val_indices = []
    test_indices = []
    remaining_indices = []
    
    for c in unique_classes:
        if c == 0:  # Skip None class - will be in all splits naturally
            continue
        class_indices = np.where(sample_classes == c)[0]
        n = len(class_indices)
        
        if n >= 3:
            # Ensure at least 1 in val and test
            np.random.shuffle(class_indices)
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[1])
            remaining_indices.extend(class_indices[2:])
        elif n == 2:
            # Put 1 in val, 1 in test, none in train (will be augmented)
            np.random.shuffle(class_indices)
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[1])
        elif n == 1:
            # Duplicate to val and test, original to train
            train_indices.append(class_indices[0])
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[0])
    
    # Add all class 0 (None) samples to remaining
    none_indices = np.where(sample_classes == 0)[0]
    remaining_indices.extend(none_indices.tolist())
    
    # Split remaining indices
    remaining_indices = np.array(remaining_indices)
    if len(remaining_indices) > 0:
        np.random.shuffle(remaining_indices)
        
        n_remaining = len(remaining_indices)
        n_test = max(1, int(n_remaining * test_size))
        n_val = max(1, int(n_remaining * val_size))
        n_train = n_remaining - n_test - n_val
        
        train_indices.extend(remaining_indices[:n_train].tolist())
        val_indices.extend(remaining_indices[n_train:n_train+n_val].tolist())
        test_indices.extend(remaining_indices[n_train+n_val:].tolist())
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    # Extract splits
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Val:   {len(X_val)} ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test:  {len(X_test)} ({100*len(X_test)/len(X):.1f}%)")
    
    # Verify all pattern classes in val/test
    val_classes = np.unique(get_sample_dominant_class(Y_val))
    test_classes = np.unique(get_sample_dominant_class(Y_test))
    print(f"\nClasses in val:  {sorted(val_classes.tolist())}")
    print(f"Classes in test: {sorted(test_classes.tolist())}")
    
    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
    }


def load_sequence_data_splits(
    data_dir: Path, 
    mode: str = "3class",
    balance: bool = True,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """Load raw data and perform splitting, normalization, and augmentation.
    
    Pipeline:
    1. Load X_seq.npy, Y_seq.npy (raw extracted windows)
    2. Convert labels based on mode (binary/3class/7class)
    3. Split with stratification ensuring all classes in each split
    4. Fit StandardScaler on training data, apply to all
    5. Augment training set to balance classes
    
    Args:
        data_dir: Directory containing X_seq.npy and Y_seq.npy
        mode: "binary", "3class", or "7class"
        balance: Whether to augment training set for class balance
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        
    Returns:
        splits dict: {'train': (X, Y), 'val': (X, Y), 'test': (X, Y)}
        metadata dict
    """
    # Check for raw data files
    if not (data_dir / "X_seq.npy").exists():
        raise FileNotFoundError(
            f"Raw data not found in {data_dir}. "
            "Run: python -m src.prepare.prepare_data_sequence --data-dir data"
        )
    
    logging.info("Loading raw sequence data...")
    X = np.load(data_dir / "X_seq.npy")
    Y = np.load(data_dir / "Y_seq.npy")
    
    # Load metadata
    with open(data_dir / "metadata_seq.json") as f:
        metadata = json.load(f)
    
    logging.info(f"Loaded: {X.shape[0]} samples, window size {X.shape[1]}")
    
    # Convert labels based on mode FIRST (before splitting)
    Y = convert_labels(Y, mode)
    
    # Determine number of classes
    if mode == "binary":
        num_classes = 2
        class_names = ["None", "Pattern"]
    elif mode == "3class":
        num_classes = 3
        class_names = CLASS_3_NAMES
    else:
        num_classes = 7
        class_names = CLASS_7_NAMES
    
    logging.info(f"Mode: {mode} ({num_classes} classes)")
    
    # STEP 1: Split data with stratification
    logging.info("\n" + "=" * 60)
    logging.info("SPLITTING DATA")
    logging.info("=" * 60)
    
    splits = stratified_split_with_all_classes(
        X, Y,
        num_classes=num_classes,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    
    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']
    
    # STEP 2: Fit StandardScaler on training data
    logging.info("\n" + "=" * 60)
    logging.info("FITTING StandardScaler ON TRAINING DATA")
    logging.info("=" * 60)
    
    from src.normalization import OHLCScaler
    
    scaler = OHLCScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    logging.info(f"Fitted on {len(X_train)} training samples")
    logging.info(f"  Mean (OHLC): {scaler.mean_}")
    logging.info(f"  Std (OHLC):  {scaler.scale_}")
    
    # Save scaler for inference
    scaler_path = data_dir / "scaler.pkl"
    scaler.save(scaler_path)
    
    # STEP 3: Augment training set if requested
    if balance:
        logging.info("\n" + "=" * 60)
        logging.info("AUGMENTING TRAINING SET")
        logging.info("=" * 60)
        
        from src.augmentation import balance_dataset_with_augmentation
        X_train, Y_train = balance_dataset_with_augmentation(
            X_train, Y_train,
            target_samples_per_class=None,  # Use max class count
            seed=seed
        )
        metadata["balanced"] = True
    else:
        metadata["balanced"] = False
    
    # Update splits with processed data
    splits = {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
    }
    
    # Update metadata
    metadata["mode"] = mode
    metadata["num_classes"] = num_classes
    metadata["class_names"] = class_names
    metadata["n_train"] = len(X_train)
    metadata["n_val"] = len(X_val)
    metadata["n_test"] = len(X_test)
    metadata["scaler_mean"] = scaler.mean_.tolist()
    metadata["scaler_std"] = scaler.scale_.tolist()
    
    return splits, metadata


def create_dataloaders_from_splits(
    splits: dict,
    num_classes: int,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create dataloaders from pre-split data.
    
    Args:
        splits: Dict with 'train', 'val', 'test' keys containing (X, Y) tuples
        num_classes: Number of classes (2, 3, or 7)
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']
    
    logging.info(f"Train: {len(X_train)} samples")
    logging.info(f"Val:   {len(X_val)} samples")
    logging.info(f"Test:  {len(X_test)} samples")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.LongTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)
    X_test_t = torch.FloatTensor(X_test)
    Y_test_t = torch.LongTensor(Y_test)
    
    # Compute class weights from training data using balanced mode
    class_weights = compute_class_weights(
        Y_train_t, 
        num_classes=num_classes, 
        max_weight=SEQ_LSTM_CONFIG.get("max_class_weight", 10.0),
        balance_mode="balanced",
    )
    logging.info(f"Class weights (balanced): {class_weights.tolist()}")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: (batch, seq_len, num_classes)
        logits = model(X_batch)
        
        # Reshape for loss: (batch * seq_len, num_classes) vs (batch * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = Y_batch.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        
        # Accuracy
        preds = logits_flat.argmax(dim=-1)
        correct += (preds == targets_flat).sum().item()
        total += targets_flat.numel()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_full_metrics: bool = False,
    num_classes: int = 3,
    class_names: Optional[list] = None,
) -> Tuple[float, float, Optional[SequenceMetrics]]:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        criterion: Loss function
        device: Device
        compute_full_metrics: Whether to compute all sequence labeling metrics
        num_classes: Number of output classes
        class_names: List of class names for metrics reporting
        
    Returns:
        avg_loss, accuracy, optional full metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        
        logits = model(X_batch)
        
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = Y_batch.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        total_loss += loss.item() * X_batch.size(0)
        
        preds = logits.argmax(dim=-1)
        correct += (preds == Y_batch).sum().item()
        total += Y_batch.numel()
        
        if compute_full_metrics:
            all_preds.append(preds.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    
    full_metrics = None
    if compute_full_metrics and all_preds:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        y_proba = np.concatenate(all_probs, axis=0)
        full_metrics = compute_all_metrics(
            y_true, y_pred, y_proba, 
            num_classes=num_classes, 
            class_names=class_names
        )
    
    return avg_loss, accuracy, full_metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    config: dict,
    device: torch.device,
    save_dir: Path,
    class_names: Optional[list] = None,
) -> Dict:
    """Full training loop with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        class_weights: Class weights for loss function
        config: Training configuration
        device: Device to use
        save_dir: Directory to save checkpoints
        class_names: List of class names for metrics reporting
    
    Returns:
        Dictionary with training history and final metrics
    """
    num_classes = config.get("num_classes", 3)
    
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Scheduler
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    
    logging.info("=" * 60)
    logging.info("TRAINING STARTED")
    logging.info("=" * 60)
    
    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, val_metrics = evaluate(
            model, val_loader, criterion, device, 
            compute_full_metrics=True,
            num_classes=num_classes,
            class_names=class_names,
        )
        val_f1 = val_metrics.f1_macro if val_metrics else 0.0
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        # Logging
        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
            f"LR: {lr:.2e}"
        )
        
        # Early stopping check
        if val_f1 > best_val_f1 + config.get("min_delta", 1e-4):
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "config": config,
            }, save_dir / "best_model.pth")
            
            logging.info(f"  -> New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 15):
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate on test set
    logging.info("=" * 60)
    logging.info("EVALUATING BEST MODEL ON TEST SET")
    logging.info("=" * 60)
    
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_metrics = evaluate(
        model, test_loader, criterion, device, 
        compute_full_metrics=True,
        num_classes=num_classes,
        class_names=class_names,
    )
    
    logging.info(f"\nTest Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    if test_metrics:
        logging.info(test_metrics.summary())
    
    # Save final results
    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_metrics": test_metrics.to_dict() if test_metrics else {},
        "history": history,
        "config": config,
    }
    
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train sequence labeling model")
    parser.add_argument("--model", type=str, default="seq_lstm", choices=["seq_lstm"])
    parser.add_argument("--mode", type=str, default="3class", choices=["binary", "3class", "7class"],
                        help="Classification mode: binary (2), 3class (3), or 7class (7)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    
    # Determine number of classes based on mode
    mode_to_classes = {"binary": 2, "3class": 3, "7class": 7}
    num_classes = mode_to_classes[args.mode]
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = MODELS_DIR / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = LOG_DIR / f"train_{args.model}_{args.mode}_{timestamp}.log"
    setup_logging(log_file)
    
    logging.info(f"Model: {args.model}")
    logging.info(f"Mode: {args.mode} ({num_classes} classes)")
    logging.info(f"Data dir: {data_dir}")
    logging.info(f"Save dir: {save_dir}")
    logging.info(f"Device: {DEVICE}")
    
    # Load pre-split data (train already augmented if --balance was used during prep)
    splits, metadata = load_sequence_data_splits(data_dir, mode=args.mode)
    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']
    
    logging.info(f"Loaded pre-split data:")
    logging.info(f"  Train: {X_train.shape} (may be augmented)")
    logging.info(f"  Val:   {X_val.shape} (original)")
    logging.info(f"  Test:  {X_test.shape} (original)")
    logging.info(f"Class names: {metadata.get('class_names', [])}")
    logging.info(f"Balanced: {metadata.get('balanced', False)}")
    
    # Log training label distribution
    unique, counts = np.unique(Y_train, return_counts=True)
    logging.info("Training label distribution:")
    for label, count in zip(unique, counts):
        pct = 100 * count / Y_train.size
        class_name = metadata.get('class_names', [])[label] if label < len(metadata.get('class_names', [])) else f"Class {label}"
        logging.info(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Get config based on mode
    if args.mode == "7class":
        config = CONFIG_7CLASS.copy()
    elif args.mode == "binary":
        config = CONFIG_BINARY.copy()
    else:
        config = SEQ_LSTM_CONFIG.copy()
    
    config["num_classes"] = num_classes
    config["mode"] = args.mode
    
    # Override with CLI args
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.hidden_size:
        config["hidden_size"] = args.hidden_size
    if args.num_layers:
        config["num_layers"] = args.num_layers
    
    logging.info(f"Config: {config}")
    
    # Create dataloaders from pre-split data
    train_loader, val_loader, test_loader, class_weights = create_dataloaders_from_splits(
        splits, num_classes=num_classes, batch_size=config["batch_size"]
    )
    
    # Create model
    model = SeqLabelingLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=num_classes,
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    
    logging.info(f"Model parameters: {model.num_parameters:,}")
    
    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        config=config,
        device=DEVICE,
        save_dir=save_dir,
        class_names=metadata.get("class_names"),
    )
    
    logging.info("=" * 60)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
