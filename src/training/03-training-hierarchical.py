#!/usr/bin/env python3
"""Training script for Hierarchical classification.

Two-stage training:
1. Train Stage 1: Direction classifier (None/Bearish/Bullish)
2. Train Stage 2: Subtype classifier (Normal/Wedge/Pennant) on pattern regions only

Usage:
    python -m src.03-training-hierarchical --stage 1  # Train stage 1
    python -m src.03-training-hierarchical --stage 2  # Train stage 2
    python -m src.03-training-hierarchical --stage both  # Train both
    python -m src.03-training-hierarchical --evaluate  # Evaluate combined model
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATA_DIR, MODELS_DIR, LOG_DIR, DEVICE, SEED
from src.utils import setup_logging, set_seed
from src.metrics import compute_all_metrics
from models.lstm_v2.model import SeqLabelingLSTM, compute_class_weights
from models.hierarchical_v1.config import (
    CONFIG_STAGE1,
    CONFIG_STAGE2,
    STAGE1_CLASSES,
    STAGE2_CLASSES,
)
from models.hierarchical_v1.model import (
    HierarchicalClassifier,
    convert_labels_to_stage1,
    convert_labels_to_stage2,
    get_pattern_mask,
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional)
        reduction: 'none', 'mean', or 'sum'
        label_smoothing: Label smoothing factor (0-1)
    """
    
    def __init__(
        self, 
        gamma: float = 2.0, 
        alpha: torch.Tensor = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (N, C)
            targets: Class indices (N,)
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_sample_dominant_class(Y: np.ndarray) -> np.ndarray:
    """Get dominant (non-zero) class for each sample."""
    sample_classes = np.zeros(len(Y), dtype=np.int64)
    for i, y in enumerate(Y):
        non_zero = y[y > 0]
        if len(non_zero) > 0:
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
    """Split data ensuring all classes are represented in each split."""
    np.random.seed(seed)
    
    sample_classes = get_sample_dominant_class(Y)
    unique_classes = np.unique(sample_classes)
    
    train_indices = []
    val_indices = []
    test_indices = []
    remaining_indices = []
    
    for c in unique_classes:
        if c == 0:
            continue
        class_indices = np.where(sample_classes == c)[0]
        n = len(class_indices)
        
        if n >= 3:
            np.random.shuffle(class_indices)
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[1])
            remaining_indices.extend(class_indices[2:])
        elif n == 2:
            np.random.shuffle(class_indices)
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[1])
        elif n == 1:
            train_indices.append(class_indices[0])
            val_indices.append(class_indices[0])
            test_indices.append(class_indices[0])
    
    none_indices = np.where(sample_classes == 0)[0]
    remaining_indices.extend(none_indices.tolist())
    
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
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return {
        'train': (X[train_indices], Y[train_indices]),
        'val': (X[val_indices], Y[val_indices]),
        'test': (X[test_indices], Y[test_indices]),
    }


def load_and_prepare_data(
    data_dir: Path,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load data and prepare for hierarchical training.
    
    Returns splits with both 7-class and hierarchical labels.
    """
    from src.normalization import OHLCScaler
    from src.augmentation import balance_dataset_with_augmentation
    
    # Load raw data
    X = np.load(data_dir / "X_seq.npy")
    Y = np.load(data_dir / "Y_seq.npy")
    
    logging.info(f"Loaded: {X.shape[0]} samples")
    
    # Split with stratification
    splits = stratified_split_with_all_classes(
        X, Y,
        num_classes=7,
        test_size=0.15,
        val_size=0.15,
        seed=seed,
    )
    
    X_train, Y_train = splits['train']
    X_val, Y_val = splits['val']
    X_test, Y_test = splits['test']
    
    # Normalize
    scaler = OHLCScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    scaler.save(data_dir / "scaler.pkl")
    
    # Augment training set
    X_train, Y_train = balance_dataset_with_augmentation(
        X_train, Y_train,
        target_samples_per_class=None,
        seed=seed
    )
    
    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test),
    }


def train_stage1(
    splits: Dict,
    save_dir: Path,
    config: Dict = None,
) -> SeqLabelingLSTM:
    """Train Stage 1: Direction classifier (None/Bearish/Bullish)."""
    config = config or CONFIG_STAGE1
    
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING STAGE 1: Direction Classifier")
    logging.info("=" * 60)
    
    X_train, Y_train_7class = splits['train']
    X_val, Y_val_7class = splits['val']
    
    # Convert to 3-class labels
    Y_train = convert_labels_to_stage1(Y_train_7class)
    Y_val = convert_labels_to_stage1(Y_val_7class)
    
    logging.info(f"Train: {len(X_train)} samples")
    logging.info(f"Val: {len(X_val)} samples")
    
    # Log class distribution
    for name, Y in [("Train", Y_train), ("Val", Y_val)]:
        counts = np.bincount(Y.flatten(), minlength=3)
        total = counts.sum()
        logging.info(f"{name} distribution:")
        for i, cls_name in enumerate(STAGE1_CLASSES):
            logging.info(f"  {cls_name}: {counts[i]} ({100*counts[i]/total:.1f}%)")
    
    # Create dataloaders
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.LongTensor(Y_train)
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)
    
    class_weights = compute_class_weights(
        Y_train_t, 
        num_classes=3,
        max_weight=config.get("max_class_weight", 10.0),
        balance_mode="balanced",
    )
    logging.info(f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, Y_val_t),
        batch_size=config["batch_size"],
        shuffle=False,
    )
    
    # Create model
    model = SeqLabelingLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=3,
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    ).to(DEVICE)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    if config.get("focal_loss", False):
        criterion = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=class_weights.to(DEVICE),
            reduction='mean',
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        logging.info(f"Using Focal Loss (gamma={config.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE),
            label_smoothing=config.get("label_smoothing", 0.0),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits.view(-1, 3), Y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == Y_batch).sum().item()
            train_total += Y_batch.numel()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                
                logits = model(X_batch)
                loss = criterion(logits.view(-1, 3), Y_batch.view(-1))
                val_loss += loss.item()
                
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(Y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = (all_preds == all_labels).mean()
        
        logging.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_f1': val_f1,
            }, save_dir / "stage1_best.pth")
            logging.info(f"  -> New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(save_dir / "stage1_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def train_stage2(
    splits: Dict,
    save_dir: Path,
    config: Dict = None,
) -> SeqLabelingLSTM:
    """Train Stage 2: Subtype classifier (Normal/Wedge/Pennant).
    
    Only trains on pattern regions (where Y_7class > 0).
    """
    config = config or CONFIG_STAGE2
    
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING STAGE 2: Subtype Classifier")
    logging.info("=" * 60)
    
    X_train, Y_train_7class = splits['train']
    X_val, Y_val_7class = splits['val']
    
    # Convert to 3-class subtype labels
    Y_train = convert_labels_to_stage2(Y_train_7class)
    Y_val = convert_labels_to_stage2(Y_val_7class)
    
    # Get pattern masks
    mask_train = get_pattern_mask(Y_train_7class)
    mask_val = get_pattern_mask(Y_val_7class)
    
    logging.info(f"Train: {len(X_train)} samples, {mask_train.sum()} pattern timesteps")
    logging.info(f"Val: {len(X_val)} samples, {mask_val.sum()} pattern timesteps")
    
    # Log class distribution (pattern regions only)
    for name, Y, mask in [("Train", Y_train, mask_train), ("Val", Y_val, mask_val)]:
        pattern_labels = Y[mask]
        counts = np.bincount(pattern_labels, minlength=3)
        total = counts.sum()
        logging.info(f"{name} subtype distribution (pattern regions):")
        for i, cls_name in enumerate(STAGE2_CLASSES):
            logging.info(f"  {cls_name}: {counts[i]} ({100*counts[i]/total:.1f}%)")
    
    # Create dataloaders - use full sequences but mask loss
    X_train_t = torch.FloatTensor(X_train)
    Y_train_t = torch.LongTensor(Y_train)
    mask_train_t = torch.BoolTensor(mask_train)
    
    X_val_t = torch.FloatTensor(X_val)
    Y_val_t = torch.LongTensor(Y_val)
    mask_val_t = torch.BoolTensor(mask_val)
    
    # Compute class weights from pattern regions only
    pattern_labels = Y_train[mask_train]
    counts = np.bincount(pattern_labels, minlength=3).astype(float)
    total = counts.sum()
    class_weights = total / (3 * counts + 1e-6)
    class_weights = np.clip(class_weights, 1.0, config.get("max_class_weight", 5.0))
    class_weights = torch.FloatTensor(class_weights)
    logging.info(f"Class weights: {class_weights.tolist()}")
    
    train_dataset = TensorDataset(X_train_t, Y_train_t, mask_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t, mask_val_t)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )
    
    # Create model
    model = SeqLabelingLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=3,
        dropout=config["dropout"],
        bidirectional=config.get("bidirectional", False),
    ).to(DEVICE)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss (no reduction - we'll mask manually)
    if config.get("focal_loss", False):
        criterion = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=class_weights.to(DEVICE),
            reduction='none',
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        logging.info(f"Using Focal Loss (gamma={config.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(DEVICE),
            reduction='none',
            label_smoothing=config.get("label_smoothing", 0.0),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
    )
    
    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, Y_batch, mask_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)  # (batch, seq, 3)
            
            # Compute loss only on pattern regions
            loss_all = criterion(logits.view(-1, 3), Y_batch.view(-1))
            loss_all = loss_all.view_as(mask_batch)
            
            if mask_batch.sum() > 0:
                loss = loss_all[mask_batch].mean()
            else:
                loss = loss_all.mean() * 0  # No patterns in batch
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy on pattern regions only
            preds = logits.argmax(dim=-1)
            if mask_batch.sum() > 0:
                train_correct += (preds[mask_batch] == Y_batch[mask_batch]).sum().item()
                train_total += mask_batch.sum().item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / max(train_total, 1)
        
        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, Y_batch, mask_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                mask_batch = mask_batch.to(DEVICE)
                
                logits = model(X_batch)
                loss_all = criterion(logits.view(-1, 3), Y_batch.view(-1))
                loss_all = loss_all.view_as(mask_batch)
                
                if mask_batch.sum() > 0:
                    val_loss += loss_all[mask_batch].mean().item()
                    
                    preds = logits.argmax(dim=-1)
                    all_preds.append(preds[mask_batch].cpu().numpy())
                    all_labels.append(Y_batch[mask_batch].cpu().numpy())
        
        val_loss /= len(val_loader)
        
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            from sklearn.metrics import f1_score
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_acc = (all_preds == all_labels).mean()
        else:
            val_f1 = 0.0
            val_acc = 0.0
        
        logging.info(
            f"Epoch {epoch+1:3d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_f1': val_f1,
            }, save_dir / "stage2_best.pth")
            logging.info(f"  -> New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load(save_dir / "stage2_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_hierarchical(
    splits: Dict,
    save_dir: Path,
) -> Dict:
    """Evaluate combined hierarchical model on test set."""
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATING HIERARCHICAL MODEL")
    logging.info("=" * 60)
    
    X_test, Y_test = splits['test']
    
    # Load both stages with correct bidirectional setting
    stage1 = SeqLabelingLSTM(
        input_size=CONFIG_STAGE1["input_size"],
        hidden_size=CONFIG_STAGE1["hidden_size"],
        num_layers=CONFIG_STAGE1["num_layers"],
        num_classes=3,
        dropout=0.0,
        bidirectional=CONFIG_STAGE1.get("bidirectional", False),
    ).to(DEVICE)
    
    stage2 = SeqLabelingLSTM(
        input_size=CONFIG_STAGE2["input_size"],
        hidden_size=CONFIG_STAGE2["hidden_size"],
        num_layers=CONFIG_STAGE2["num_layers"],
        num_classes=3,
        dropout=0.0,
        bidirectional=CONFIG_STAGE2.get("bidirectional", False),
    ).to(DEVICE)
    
    stage1_ckpt = torch.load(save_dir / "stage1_best.pth")
    stage2_ckpt = torch.load(save_dir / "stage2_best.pth")
    
    stage1.load_state_dict(stage1_ckpt['model_state_dict'])
    stage2.load_state_dict(stage2_ckpt['model_state_dict'])
    
    stage1.eval()
    stage2.eval()
    
    logging.info(f"Loaded Stage 1 (val F1: {stage1_ckpt['val_f1']:.4f})")
    logging.info(f"Loaded Stage 2 (val F1: {stage2_ckpt['val_f1']:.4f})")
    
    # Create hierarchical model
    hierarchical = HierarchicalClassifier()
    hierarchical.stage1 = stage1
    hierarchical.stage2 = stage2
    hierarchical.eval()
    
    # Run inference
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    
    with torch.no_grad():
        stage1_preds, stage2_preds, final_preds = hierarchical.predict(X_test_t)
    
    stage1_preds = stage1_preds.cpu().numpy()
    stage2_preds = stage2_preds.cpu().numpy()
    final_preds = final_preds.cpu().numpy()
    
    # Evaluate Stage 1 (3-class)
    Y_test_stage1 = convert_labels_to_stage1(Y_test)
    from sklearn.metrics import classification_report, f1_score
    
    logging.info("\n" + "-" * 40)
    logging.info("STAGE 1 RESULTS (Direction)")
    logging.info("-" * 40)
    logging.info("\n" + classification_report(
        Y_test_stage1.flatten(),
        stage1_preds.flatten(),
        target_names=STAGE1_CLASSES,
    ))
    
    stage1_f1 = f1_score(Y_test_stage1.flatten(), stage1_preds.flatten(), average='macro')
    
    # Evaluate Stage 2 (on pattern regions only)
    mask = get_pattern_mask(Y_test)
    Y_test_stage2 = convert_labels_to_stage2(Y_test)
    
    logging.info("\n" + "-" * 40)
    logging.info("STAGE 2 RESULTS (Subtype, pattern regions only)")
    logging.info("-" * 40)
    logging.info("\n" + classification_report(
        Y_test_stage2[mask],
        stage2_preds[mask],
        target_names=STAGE2_CLASSES,
    ))
    
    stage2_f1 = f1_score(Y_test_stage2[mask], stage2_preds[mask], average='macro')
    
    # Evaluate combined (7-class)
    logging.info("\n" + "-" * 40)
    logging.info("COMBINED 7-CLASS RESULTS")
    logging.info("-" * 40)
    
    class_names_7 = [
        "None", "Bearish Normal", "Bearish Wedge", "Bearish Pennant",
        "Bullish Normal", "Bullish Wedge", "Bullish Pennant"
    ]
    logging.info("\n" + classification_report(
        Y_test.flatten(),
        final_preds.flatten(),
        target_names=class_names_7,
        zero_division=0,
    ))
    
    # Compute sequence metrics
    metrics = compute_all_metrics(
        Y_test.flatten(),
        final_preds.flatten(),
        num_classes=7,
    )
    
    logging.info("\n" + "=" * 60)
    logging.info("SEQUENCE LABELING METRICS")
    logging.info("=" * 60)
    logging.info(f"\nAccuracy:         {metrics.accuracy:.4f}")
    logging.info(f"F1 (macro):       {metrics.f1_macro:.4f}")
    logging.info(f"Detection Rate:   {metrics.detection_rate:.4f}")
    logging.info(f"False Alarm Rate: {metrics.false_alarm_rate:.4f}")
    
    # Save results
    results = {
        'stage1_f1': float(stage1_f1),
        'stage2_f1': float(stage2_f1),
        'combined_accuracy': float(metrics.accuracy),
        'combined_f1_macro': float(metrics.f1_macro),
        'detection_rate': float(metrics.detection_rate),
    }
    
    with open(save_dir / "hierarchical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical classifier")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Which stage to train (1, 2, or both)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate combined model",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    save_dir = MODELS_DIR / "hierarchical"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = LOG_DIR / f"hierarchical_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(log_file)
    
    logging.info("=" * 60)
    logging.info("HIERARCHICAL CLASSIFICATION TRAINING")
    logging.info("=" * 60)
    logging.info(f"Data dir: {args.data_dir}")
    logging.info(f"Save dir: {save_dir}")
    logging.info(f"Device: {DEVICE}")
    
    # Load data
    data_dir = Path(args.data_dir)
    splits = load_and_prepare_data(data_dir, seed=args.seed)
    
    # Train
    if args.stage in ["1", "both"]:
        train_stage1(splits, save_dir)
    
    if args.stage in ["2", "both"]:
        train_stage2(splits, save_dir)
    
    # Always evaluate if both stages trained
    if args.stage == "both" or args.evaluate:
        evaluate_hierarchical(splits, save_dir)
    
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
