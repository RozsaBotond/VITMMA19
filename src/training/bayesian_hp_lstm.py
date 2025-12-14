"""Bayesian Hyperparameter Optimization using Optuna.

Fine-tunes LSTM v2e (best performing model) using Bayesian optimization.
Uses Tree-structured Parzen Estimator (TPE) sampler with pruning.

Usage:
    python src/bayesian_hp_optim.py --n_trials 50
    python src/bayesian_hp_optim.py --n_trials 100 --timeout 3600
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import setup_logger, set_seed, get_device
from normalization import OHLCScaler
from augmentation import TimeSeriesAugmenter, balance_dataset_with_augmentation

# Import models
from models.lstm_v2.model import SeqLabelingLSTM

logger = setup_logger("bayesian_hp_optim")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Load and prepare data for training."""
    
    # Load data
    X = np.load(config.DATA_DIR / "X_seq.npy")
    Y = np.load(config.DATA_DIR / "Y_seq.npy")
    logger.info(f"Loaded data: X={X.shape}, Y={Y.shape}")
    
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
    
    # Augment training data
    augmenter = TimeSeriesAugmenter(seed=42)
    X_train_aug, Y_train_aug = balance_dataset_with_augmentation(
        X_train_norm, Y_train, augmenter=augmenter, seed=42
    )
    logger.info(f"Augmented: {len(X_train_aug)} train samples")
    
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
    
    return train_loader, val_loader, test_loader


# ============================================================================
# OBJECTIVE FUNCTION FOR OPTUNA
# ============================================================================

class Objective:
    """Optuna objective function for LSTM hyperparameter optimization."""
    
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_epochs: int = 100,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a single trial."""
        
        # Sample hyperparameters - EXPANDED SEARCH SPACE
        # Larger hidden sizes and more layers
        hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64, 96, 128, 192, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.05)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        
        # Learning rate with wider range
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
        
        # Scheduler choice - more options
        scheduler_type = trial.suggest_categorical(
            "scheduler", ["cosine", "one_cycle", "reduce_on_plateau", "step"]
        )
        
        # Gradient clipping
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0, step=0.5)
        
        # Whether to use any class weighting (key insight from lstm_v2e)
        use_class_weights = trial.suggest_categorical("use_class_weights", [False, True])
        class_weight_scale = 1.0
        if use_class_weights:
            class_weight_scale = trial.suggest_float("class_weight_scale", 0.1, 1.0, step=0.1)
        
        # Create model
        model = SeqLabelingLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=7,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(self.device)
        
        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Scheduler - now includes step scheduler
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        elif scheduler_type == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                epochs=self.max_epochs,
                steps_per_epoch=len(self.train_loader),
            )
        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        else:  # reduce_on_plateau
            scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
            )
        
        # Loss function - no class weights performed best in v2e
        if use_class_weights:
            # Compute scaled class weights
            all_labels = []
            for _, Y_batch in self.train_loader:
                all_labels.extend(Y_batch.numpy().flatten())
            unique, counts = np.unique(all_labels, return_counts=True)
            total = len(all_labels)
            weights = torch.tensor(
                [total / (len(unique) * c) for c in counts],
                dtype=torch.float32
            )
            weights = torch.clamp(weights * class_weight_scale, max=5.0)
            criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Training loop with pruning
        best_val_f1 = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(self.max_epochs):
            # Train
            model.train()
            for X_batch, Y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                output_flat = output.reshape(-1, output.shape[-1])
                target_flat = Y_batch.reshape(-1)
                
                loss = criterion(output_flat, target_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                if scheduler_type == "one_cycle":
                    scheduler.step()
            
            # Validate
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, Y_batch in self.val_loader:
                    X_batch = X_batch.to(self.device)
                    output = model(X_batch)
                    preds = output.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds.flatten())
                    all_labels.extend(Y_batch.numpy().flatten())
            
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Update scheduler
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "reduce_on_plateau":
                scheduler.step(val_f1)
            elif scheduler_type == "step":
                scheduler.step()
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Report intermediate value for pruning
            trial.report(val_f1, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_f1


def evaluate_best_params(
    params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 150,
) -> Dict[str, float]:
    """Train final model with best params and evaluate on test set."""
    
    logger.info(f"\n{'='*60}")
    logger.info("Training final model with best hyperparameters")
    logger.info(f"{'='*60}")
    logger.info(f"Parameters: {json.dumps(params, indent=2)}")
    
    # Create model
    model = SeqLabelingLSTM(
        input_size=4,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        num_classes=7,
        dropout=params["dropout"],
        bidirectional=params["bidirectional"],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    
    # Scheduler
    scheduler_type = params["scheduler"]
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=params["learning_rate"],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
        )
    
    # Loss
    if params.get("use_class_weights", False):
        all_labels = []
        for _, Y_batch in train_loader:
            all_labels.extend(Y_batch.numpy().flatten())
        unique, counts = np.unique(all_labels, return_counts=True)
        total = len(all_labels)
        weights = torch.tensor(
            [total / (len(unique) * c) for c in counts],
            dtype=torch.float32
        )
        scale = params.get("class_weight_scale", 1.0)
        weights = torch.clamp(weights * scale, max=5.0)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training
    best_f1 = 0.0
    best_state = None
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        model.train()
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
            
            if scheduler_type == "one_cycle":
                scheduler.step()
        
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
        
        if scheduler_type == "cosine":
            scheduler.step()
        elif scheduler_type == "reduce_on_plateau":
            scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch + 1}: val_f1={val_f1:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Evaluate on test set
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
    
    # Compute detection and false alarm rates
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    actual_patterns = all_labels > 0
    pred_patterns = all_preds > 0
    
    detection_rate = (pred_patterns & actual_patterns).sum() / max(1, actual_patterns.sum())
    false_alarm = (pred_patterns & ~actual_patterns).sum() / max(1, (~actual_patterns).sum())
    
    results = {
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm),
        "best_val_f1": float(best_f1),
        "n_params": n_params,
    }
    
    logger.info(f"\nFinal Results:")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Test F1:       {test_f1:.4f}")
    logger.info(f"  Detection:     {detection_rate:.4f}")
    logger.info(f"  False Alarm:   {false_alarm:.4f}")
    
    return results, best_state


def main():
    parser = argparse.ArgumentParser(description="Bayesian HP Optimization with Optuna")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("Loading data...")
    logger.info("=" * 60)
    train_loader, val_loader, test_loader = load_and_prepare_data(batch_size=16)
    
    # Create Optuna study
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting Bayesian Optimization ({args.n_trials} trials)")
    logger.info("=" * 60)
    
    # Use TPE sampler (Bayesian) with Hyperband pruner
    sampler = TPESampler(seed=args.seed)
    pruner = HyperbandPruner(min_resource=10, max_resource=args.epochs, reduction_factor=3)
    
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lstm_v2_bayesian_optimization",
    )
    
    # Enqueue lstm_v2e params as starting point (our best known config)
    study.enqueue_trial({
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "scheduler": "cosine",
        "use_class_weights": False,
    })
    
    # Create objective
    objective = Objective(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        max_epochs=args.epochs,
    )
    
    # Run optimization
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Trials completed: {len(study.trials)}")
    logger.info(f"Trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    logger.info(f"\nBest trial:")
    logger.info(f"  Value (val_f1): {study.best_trial.value:.4f}")
    logger.info(f"  Params: {json.dumps(study.best_trial.params, indent=4)}")
    
    # Show top 5 trials
    logger.info("\nTop 5 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )[:5]
    for i, trial in enumerate(sorted_trials):
        logger.info(f"  {i+1}. F1={trial.value:.4f} - hidden={trial.params['hidden_size']}, "
                   f"layers={trial.params['num_layers']}, lr={trial.params['learning_rate']:.2e}, "
                   f"weights={trial.params.get('use_class_weights', False)}")
    
    # Train final model with best params
    logger.info("\n" + "=" * 60)
    logger.info("Training final model with best hyperparameters...")
    logger.info("=" * 60)
    
    results, best_state = evaluate_best_params(
        params=study.best_trial.params,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=150,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best model
    model_dir = Path("models/lstm_v2g")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "best_model.pth")
    
    # Save config
    config_data = {
        "name": "LSTM v2g (Bayesian optimized)",
        "description": "Hyperparameters found via Bayesian optimization with Optuna",
        "best_params": study.best_trial.params,
        "results": results,
        "optimization": {
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_val_f1": study.best_trial.value,
            "elapsed_seconds": elapsed,
        },
        "timestamp": timestamp,
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Also save study results
    with open(f"models/bayesian_optim_{timestamp}.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"\nSaved best model to {model_dir}")
    logger.info(f"Config: {model_dir / 'config.json'}")
    
    # Print comparison with baseline
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH PREVIOUS BEST (LSTM v2e)")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<20} {'LSTM v2e':<15} {'LSTM v2g (Bayes)':<15}")
    logger.info("-" * 50)
    logger.info(f"{'F1 (weighted)':<20} {'0.5972':<15} {results['test_f1']:<15.4f}")
    logger.info(f"{'Accuracy':<20} {'0.6592':<15} {results['test_accuracy']:<15.4f}")
    logger.info(f"{'Detection Rate':<20} {'0.3791':<15} {results['detection_rate']:<15.4f}")
    logger.info(f"{'False Alarm':<20} {'0.0447':<15} {results['false_alarm_rate']:<15.4f}")


if __name__ == "__main__":
    main()
