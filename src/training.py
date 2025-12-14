"""Main training script for the Bull/Bear Flag Detector.

This script trains and compares multiple model architectures based on
predefined best hyperparameters. It handles data loading, preprocessing,
augmentation, model training, and evaluation, with verbose logging.

Models trained:
- SeqLabelingLSTM (lstm_v2)
- SeqTransformer (transformer)
- HierarchicalClassifier (hierarchical_v1)
- StatisticalBaseline (baseline)

Usage:
    python src/training.py
    python src/training.py --models lstm_v2 transformer
    python src/training.py --models hierarchical_v1 --epochs 100 --log-level DEBUG
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

# Add src to path for imports. This assumes src is added to PYTHONPATH
# or that main.py (which handles path fixing) is the entry point.
# _fix_sys_path removed to satisfy ruff E402.

from utils.config import AppConfig
from utils.utils import (
    setup_logger, set_seed, get_device,
    log_header, log_config, log_model_summary, log_epoch,
    log_evaluation, evaluate_model
)
from utils.normalization import OHLCScaler
from utils.augmentation import TimeSeriesAugmenter, balance_dataset_with_augmentation

# Import all model architectures
from models.lstm_v2.model import SeqLabelingLSTM
from models.cnn1d.model import SeqCNN1D
from models.transformer.model import SeqTransformer
from models.cnn_lstm.model import SeqCNNLSTM
from models.hierarchical_v1.model import HierarchicalClassifier
from models.baseline.statistical import StatisticalBaseline

# Global logger instance, configured in main
logger = logging.getLogger(__name__)

# --- Best Hyperparameters ---
BEST_HPARAMS = {
    "lstm_v2": {
        'hidden_size': 96, 'num_layers': 2, 'dropout': 0.5, 'bidirectional': True,
        'learning_rate': 0.00106, 'weight_decay': 0.00057, 'batch_size': 16,
        'scheduler': 'cosine', 'use_class_weights': False
    },
    "transformer": {
        'd_model': 128, 'nhead': 4, 'num_encoder_layers': 6, 'dim_feedforward': 128,
        'dropout': 0.5, 'learnable_pe': False, 'learning_rate': 0.00093,
        'weight_decay': 4.4e-06, 'batch_size': 8
    },
    "hierarchical_v1": {
        'learning_rate': 0.000288, 'weight_decay': 0.000149, 'batch_size': 32,
        'scheduler': 'cosine', 'grad_clip': 0.5, 's1_hidden_size': 192,
        's1_num_layers': 2, 's1_dropout': 0.5, 's1_bidirectional': True,
        's2_hidden_size': 64, 's2_num_layers': 3, 's2_dropout': 0.3,
        's2_bidirectional': False, 'use_class_weights': True, 'class_weight_scale': 0.6
    },
    "baseline": {} # No hyperparameters for baseline
}


# Model registry
MODEL_REGISTRY = {
    "lstm_v2": {"class": SeqLabelingLSTM, "name": "LSTM_v2 (SeqLabeling)"},
    "cnn1d": {"class": SeqCNN1D, "name": "CNN1D"},
    "transformer": {"class": SeqTransformer, "name": "Transformer"},
    "cnn_lstm": {"class": SeqCNNLSTM, "name": "CNN_LSTM"},
    "hierarchical_v1": {"class": HierarchicalClassifier, "name": "Hierarchical LSTM"},
    "baseline": {"class": StatisticalBaseline, "name": "Statistical Baseline"},
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


def load_split_data(config: AppConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed and split data."""
    log_header(logger, "LOADING SPLIT DATA")
    X_train = np.load(config.data_dir / "X_train.npy")
    Y_train = np.load(config.data_dir / "Y_train.npy")
    X_val = np.load(config.data_dir / "X_val.npy")
    Y_val = np.load(config.data_dir / "Y_val.npy")
    X_test = np.load(config.data_dir / "X_test.npy")
    Y_test = np.load(config.data_dir / "Y_test.npy")
    
    logger.info(f"Loaded train data: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"Loaded val data: X={X_val.shape}, Y={Y_val.shape}")
    logger.info(f"Loaded test data: X={X_test.shape}, Y={Y_test.shape}")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def prepare_dataloaders(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    batch_size: int = 16,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Prepare train/val dataloaders with normalization and augmentation."""
    log_header(logger, "DATALOADER PREPARATION")
    
    scaler = OHLCScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    logger.info("Data normalized using OHLCScaler (fitted on training data).")
    
    if augment:
        augmenter = TimeSeriesAugmenter(seed=42)
        X_train_aug, Y_train_aug = balance_dataset_with_augmentation(X_train_norm, Y_train, augmenter=augmenter, seed=42)
        logger.info(f"Train data augmented. New size: {X_train_aug.shape[0]} samples")
    else:
        X_train_aug, Y_train_aug = X_train_norm, Y_train
    
    unique, counts = np.unique(Y_train_aug, return_counts=True)
    total = len(Y_train_aug.flatten())
    class_weights = torch.tensor([total / (len(unique) * c) for c in counts], dtype=torch.float32)
    class_weights = torch.clamp(class_weights, max=5.0)
    logger.info(f"Calculated class weights (capped at 5.0): {class_weights.numpy().round(2)}")
    
    train_dataset = TensorDataset(torch.tensor(X_train_aug, dtype=torch.float32), torch.tensor(Y_train_aug, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Dataloaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)} samples.")
    return train_loader, val_loader, class_weights

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    config: AppConfig,
    model_name: str,
    hparams: Dict[str, Any]
) -> Tuple[nn.Module, dict]:
    """Train a single model with verbose logging."""
    log_header(logger, f"TRAINING: {model_name}")

    # Override config with best hparams
    config.epochs = hparams.get("epochs", config.epochs)
    config.learning_rate = hparams.get("learning_rate", config.learning_rate)
    config.weight_decay = hparams.get("weight_decay", config.weight_decay)
    
    model_config_dict = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "patience": config.early_stopping_patience,
        "label_smoothing": 0.1
    }
    log_config(logger, model_config_dict, title=f"{model_name} CONFIG")
    log_config(logger, hparams, title=f"{model_name} BEST HYPERPARAMETERS")


    model = model.to(device)
    log_model_summary(logger, model, input_shape=(config.window_size, 4))
    
    class_weights = class_weights.to(device) if hparams.get("use_class_weights", True) else None
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=model_config_dict["label_smoothing"])
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    best_val_f1 = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            outputs_flat, Y_flat = outputs.reshape(-1, outputs.size(-1)), Y_batch.reshape(-1)
            
            loss = criterion(outputs_flat, Y_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=-1).cpu().numpy().flatten())
            train_labels.extend(Y_batch.cpu().numpy().flatten())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                outputs = model(X_batch)
                outputs_flat, Y_flat = outputs.reshape(-1, outputs.size(-1)), Y_batch.reshape(-1)
                
                loss = criterion(outputs_flat, Y_flat)
                val_loss += loss.item()
                
                val_preds.extend(outputs.argmax(dim=-1).cpu().numpy().flatten())
                val_labels.extend(Y_batch.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            logger.debug(f"New best validation F1: {best_val_f1:.4f} at epoch {epoch+1}")
        
        log_epoch(
            logger, epoch + 1, config.epochs, train_loss, train_acc,
            val_loss=val_loss, val_acc=val_acc, lr=optimizer.param_groups[0]['lr'],
            extra_metrics={"Val F1": val_f1}
        )
        
        scheduler.step()
        
        if early_stopping(val_f1):
            logger.info(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation F1.")
            break
    
    elapsed = time.time() - start_time
    logger.info(f"Training for {model_name} completed in {elapsed:.1f}s. Best validation F1: {best_val_f1:.4f}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


def train_main(**kwargs):
    """Main function to drive model training and comparison."""
    # Parse arguments provided as kwargs or from sys.argv if not present
    parser = argparse.ArgumentParser(description="Train and compare multiple models.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys()), help="Models to train.")
    parser.add_argument("--epochs", type=int, help="Override epochs for all models.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation.")
    parser.add_argument("--log-file", type=str, default="log/run.log", help="Path to the log file.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    
    # Use parse_known_args to handle args already parsed by main.py
    args, unknown = parser.parse_known_args(args=sys.argv[1:])
    
    # Override args with kwargs where available
    for key, value in kwargs.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)

    # --- Setup ---
    log_level = getattr(logging, args.log_level.upper())
    setup_logger(log_file=Path(args.log_file), level=log_level, name=__name__)
    
    config = AppConfig()
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    set_seed(config.random_seed)
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_header(logger, "MULTI-MODEL TRAINING & COMPARISON")
    log_config(logger, {
        "Selected Models": args.models,
        "Epochs": config.epochs,
        "Device": device,
        "Log Level": args.log_level,
        "Augmentation": not args.no_augment
    })

    # --- Data Loading & Preparation ---
    X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw = load_split_data(config)
    
    # --- Training & Evaluation Loop ---
    all_results = []
    for model_key in args.models:
        model_info = MODEL_REGISTRY[model_key]
        
        if model_key == "baseline":
            log_header(logger, "TRAINING: Statistical Baseline")
            baseline_model = StatisticalBaseline()
            baseline_model.fit(X_train_raw, Y_train_raw)
            preds = baseline_model.predict(X_test_raw)
            
            metrics = {
                "accuracy": float(accuracy_score(Y_test_raw.flatten(), preds.flatten())),
                "f1_weighted": float(f1_score(Y_test_raw.flatten(), preds.flatten(), average="weighted")),
                "f1_macro": float(f1_score(Y_test_raw.flatten(), preds.flatten(), average="macro")),
            }
            log_evaluation(logger, metrics, title="Baseline Test Set Metrics")
            all_results.append({"model_key": "baseline", **metrics})
            continue

        hparams = BEST_HPARAMS.get(model_key, {})
        batch_size = hparams.get("batch_size", config.batch_size)

        train_loader, val_loader, class_weights = prepare_dataloaders(
            X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, batch_size=batch_size, augment=not args.no_augment
        )
        
        # Initialize model with hyperparameters
        model_class = model_info["class"]
        if model_key == "transformer":
            model_params = {k: v for k, v in hparams.items() if k in ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'dropout', 'learnable_pe']}
            model = model_class(input_size=4, num_classes=7, seq_len=config.window_size, **model_params)
        elif model_key == "hierarchical_v1":
            stage1_config = {
                'input_size': 4, 'hidden_size': hparams['s1_hidden_size'], 'num_layers': hparams['s1_num_layers'],
                'dropout': hparams['s1_dropout'], 'bidirectional': hparams['s1_bidirectional']
            }
            stage2_config = {
                'input_size': 4, 'hidden_size': hparams['s2_hidden_size'], 'num_layers': hparams['s2_num_layers'],
                'dropout': hparams['s2_dropout'], 'bidirectional': hparams['s2_bidirectional']
            }
            model = model_class(stage1_config=stage1_config, stage2_config=stage2_config)
        else: # lstm_v2 and others
            model_params = {k: v for k, v in hparams.items() if k in ['hidden_size', 'num_layers', 'dropout', 'bidirectional']}
            model = model_class(input_size=4, num_classes=7, seq_len=config.window_size, **model_params)

        test_loader = DataLoader(TensorDataset(torch.tensor(X_test_raw, dtype=torch.float32), torch.tensor(Y_test_raw, dtype=torch.long)), batch_size=batch_size)
        trained_model, history = train_model(
            model, train_loader, val_loader, class_weights, device, config, model_info["name"], hparams
        )
        
        eval_results = evaluate_model(trained_model, test_loader, device, model_info["name"])
        eval_results["model_key"] = model_key
        all_results.append(eval_results)
        
        save_path = config.models_dir / model_key / "best_model.pth"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": trained_model.state_dict(), "config": {**vars(config), **hparams}}, save_path)
        logger.info(f"Saved best model checkpoint to {save_path}")

    # --- Summary ---
    log_header(logger, "OVERALL COMPARISON SUMMARY")
    logger.info(f"{ 'Model':<25} {'Accuracy':<12} {'F1 (Weighted)':<15} {'F1 (Macro)':<12} {'Detection Rate':<16} {'False Alarm Rate':<18}")
    logger.info("-" * 100)
    
    for r in sorted(all_results, key=lambda x: x.get("f1_weighted", 0), reverse=True):
        logger.info(
            f"{MODEL_REGISTRY[r['model_key']]['name']:<25} "
            f"{r.get('accuracy', 0):<12.4f} {r.get('f1_weighted', 0):<15.4f} {r.get('f1_macro', 0):<12.4f} "
            f"{r.get('detection_rate', 0):<16.4f} {r.get('false_alarm_rate', 0):<18.4f}")
    
    results_path = config.log_dir / f"comparison_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFinal comparison results saved to {results_path}")


if __name__ == "__main__":
    train_main()
