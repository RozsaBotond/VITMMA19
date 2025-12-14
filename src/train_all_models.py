"""Multi-model training script for sequence labeling.

Trains and compares multiple model architectures:
- SeqLabelingLSTM: Bidirectional LSTM
- SeqCNN1D: 1D CNN with dilated convolutions
- SeqTransformer: Transformer encoder
- SeqCNNLSTM: CNN-LSTM hybrid

Usage:
    python src/train_all_models.py
    python src/train_all_models.py --models lstm cnn1d
    python src/train_all_models.py --models transformer --epochs 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import setup_logger, set_seed, get_device
from normalization import OHLCScaler
from augmentation import TimeSeriesAugmenter, balance_dataset_with_augmentation

# Import all model architectures
from models.lstm_v2.model import SeqLabelingLSTM
from models.lstm_v2 import config as lstm_config

from models.cnn1d.model import SeqCNN1D
from models.cnn1d import config as cnn_config

from models.transformer.model import SeqTransformer
from models.transformer import config as transformer_config

from models.cnn_lstm.model import SeqCNNLSTM
from models.cnn_lstm import config as cnn_lstm_config


logger = setup_logger("train_all_models")


# Model registry
MODEL_REGISTRY = {
    "lstm_v2": {
        "class": SeqLabelingLSTM,
        "config": lstm_config.CONFIG,
        "name": "LSTM_v2 (SeqLabeling)",
    },
    "cnn1d": {
        "class": SeqCNN1D,
        "config": cnn_config.CONFIG,
        "name": "CNN1D",
    },
    "transformer": {
        "class": SeqTransformer,
        "config": transformer_config.CONFIG,
        "name": "Transformer",
    },
    "cnn_lstm": {
        "class": SeqCNNLSTM,
        "config": cnn_lstm_config.CONFIG,
        "name": "CNN_LSTM",
    },
}


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def load_data() -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load sequence data."""
    X_path = config.DATA_DIR / "X_seq.npy"
    Y_path = config.DATA_DIR / "Y_seq.npy"
    meta_path = config.DATA_DIR / "metadata_seq.json"
    
    logger.info(f"Loading data from {config.DATA_DIR}")
    
    X = np.load(X_path)
    Y = np.load(Y_path)
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    logger.info(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y, metadata


def prepare_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    metadata: dict,
    batch_size: int = 16,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Prepare train/val/test dataloaders with normalization and augmentation."""
    from sklearn.model_selection import train_test_split
    
    # Check if split indices exist in metadata, otherwise create stratified split
    if "split_indices" in metadata:
        train_idx = np.array(metadata["split_indices"]["train"])
        val_idx = np.array(metadata["split_indices"]["val"])
        test_idx = np.array(metadata["split_indices"]["test"])
    else:
        # Create stratified split based on dominant label
        n_samples = len(X)
        
        # Get dominant label for each sample (for stratification)
        dominant_labels = []
        for y in Y:
            # Find the non-zero label that appears most
            non_zero_mask = y > 0
            if non_zero_mask.sum() > 0:
                # Get most common non-zero label
                unique, counts = np.unique(y[non_zero_mask], return_counts=True)
                dominant_labels.append(unique[np.argmax(counts)])
            else:
                dominant_labels.append(0)
        dominant_labels = np.array(dominant_labels)
        
        # Stratified split: 70% train, 15% val, 15% test
        indices = np.arange(n_samples)
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=dominant_labels
        )
        
        # Split temp into val and test
        temp_labels = dominant_labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        logger.info(f"  Created stratified split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # Normalize
    scaler = OHLCScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Augment training data
    if augment:
        augmenter = TimeSeriesAugmenter(seed=42)
        X_train_aug, Y_train_aug = balance_dataset_with_augmentation(
            X_train_norm, Y_train, augmenter=augmenter, seed=42
        )
        logger.info(f"  Train augmented: {X_train_aug.shape[0]} samples")
    else:
        X_train_aug, Y_train_aug = X_train_norm, Y_train
    
    # Compute class weights
    unique, counts = np.unique(Y_train_aug, return_counts=True)
    total = len(Y_train_aug.flatten())
    class_weights = torch.tensor([total / (len(unique) * c) for c in counts], dtype=torch.float32)
    class_weights = torch.clamp(class_weights, max=5.0)  # Cap weights
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_aug, dtype=torch.float32),
        torch.tensor(Y_train_aug, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_norm, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.long),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_weights


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    model_config: dict,
    model_name: str,
) -> Tuple[nn.Module, dict]:
    """Train a single model."""
    
    epochs = model_config.get("epochs", 200)
    lr = model_config.get("learning_rate", 0.001)
    weight_decay = model_config.get("weight_decay", 1e-4)
    patience = model_config.get("patience", 30)
    label_smoothing = model_config.get("label_smoothing", 0.1)
    
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_f1 = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    
    logger.info(f"\nTraining {model_name}...")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Reshape for loss: (batch * seq_len, num_classes) vs (batch * seq_len,)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            Y_flat = Y_batch.reshape(-1)
            
            loss = criterion(outputs_flat, Y_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                outputs = model(X_batch)
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                Y_flat = Y_batch.reshape(-1)
                
                loss = criterion(outputs_flat, Y_flat)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(Y_batch.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
        
        # Logging
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}: "
                       f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")
        
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_f1):
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    logger.info(f"  Training completed in {elapsed:.1f}s, best val_f1={best_val_f1:.4f}")
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict:
    """Evaluate model on test set."""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(Y_batch.numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    
    # Detection rate (pattern vs no-pattern)
    pattern_mask = all_labels > 0
    if pattern_mask.sum() > 0:
        detection_rate = (all_preds[pattern_mask] > 0).mean()
    else:
        detection_rate = 0.0
    
    # False alarm rate
    no_pattern_mask = all_labels == 0
    if no_pattern_mask.sum() > 0:
        false_alarm_rate = (all_preds[no_pattern_mask] > 0).mean()
    else:
        false_alarm_rate = 0.0
    
    results = {
        "model": model_name,
        "accuracy": float(accuracy),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
    }
    
    logger.info(f"\n{model_name} Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
    logger.info(f"  F1 (macro): {f1_macro:.4f}")
    logger.info(f"  Detection Rate: {detection_rate:.4f}")
    logger.info(f"  False Alarm Rate: {false_alarm_rate:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Models to train"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all models")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 70)
    logger.info("Multi-Model Training Script")
    logger.info(f"Models: {args.models}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    
    # Load data
    X, Y, metadata = load_data()
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader, class_weights = prepare_dataloaders(
        X, Y, metadata,
        batch_size=args.batch_size,
        augment=not args.no_augment,
    )
    
    # Train each model
    all_results = []
    
    for model_key in args.models:
        model_info = MODEL_REGISTRY[model_key]
        model_class = model_info["class"]
        model_config = model_info["config"].copy()
        model_name = model_info["name"]
        
        # Override epochs if specified
        if args.epochs is not None:
            model_config["epochs"] = args.epochs
        
        # Initialize model
        model = model_class(
            input_size=4,
            num_classes=7,
            seq_len=256,
        )
        
        # Train
        model, history = train_model(
            model, train_loader, val_loader, class_weights,
            device, model_config, model_name,
        )
        
        # Evaluate
        results = evaluate_model(model, test_loader, device, model_name)
        results["history"] = history
        all_results.append(results)
        
        # Save model checkpoint
        save_path = config.MODELS_DIR / model_key / "best_model.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "results": results,
        }, save_path)
        logger.info(f"  Saved to {save_path}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Accuracy':<10} {'F1 (weighted)':<14} {'F1 (macro)':<12} {'Detection':<10} {'False Alarm':<12}")
    logger.info("-" * 70)
    
    for r in sorted(all_results, key=lambda x: x["f1_weighted"], reverse=True):
        logger.info(
            f"{r['model']:<20} {r['accuracy']:<10.4f} {r['f1_weighted']:<14.4f} "
            f"{r['f1_macro']:<12.4f} {r['detection_rate']:<10.4f} {r['false_alarm_rate']:<12.4f}"
        )
    
    # Save comparison results
    results_path = config.MODELS_DIR / f"comparison_results_{timestamp}.json"
    with open(results_path, "w") as f:
        # Remove history for JSON (too verbose)
        save_results = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
        json.dump(save_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
