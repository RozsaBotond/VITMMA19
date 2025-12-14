"""Model evaluation script for Bull/Bear Flag Detector.

This script evaluates the trained model on the test set:
1. Loads the best saved model
2. Evaluates on test data
3. Generates metrics: accuracy, precision, recall, F1-score
4. Produces confusion matrix
5. Per-class analysis

Usage:
    python src/03-evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    setup_logger, log_header, log_separator, log_evaluation,
    log_confusion_matrix, log_class_report, get_device, load_checkpoint
)
from models.lstm_v1.model import LSTMv1
from models.lstm_v1 import config as model_config

logger = setup_logger("evaluation")


def load_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load and split data to get test set."""
    logger.info(f"Loading data from: {config.DATA_DIR}")
    
    X = np.load(config.X_FILE)
    Y = np.load(config.Y_FILE)
    
    # Reproduce the same split as training
    test_ratio = 1.0 - config.TRAIN_RATIO - config.VAL_RATIO
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X, Y, test_size=test_ratio,
        random_state=config.RANDOM_SEED, stratify=Y
    )
    
    logger.info(f"  Test set size: {len(Y_test)} samples")
    return X_test, Y_test


def load_model(device: torch.device) -> nn.Module:
    """Load the best trained model."""
    logger.info(f"Loading model from: {config.BEST_MODEL_PATH}")
    
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
    
    checkpoint = load_checkpoint(config.BEST_MODEL_PATH, model)
    model = model.to(device)
    model.eval()
    
    logger.info(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Checkpoint metrics: {checkpoint.get('metrics', {})}")
    
    return model


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get model predictions and probabilities."""
    model.eval()
    
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    
    return predictions, probabilities


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Dict[int, str],
) -> Dict:
    """Compute all evaluation metrics."""
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    labels = sorted(label_names.keys())
    target_names = [label_names[i] for i in labels]
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    
    # Per-class metrics
    per_class = {}
    for label_idx in labels:
        mask = y_true == label_idx
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == label_idx).mean()
            per_class[label_names[label_idx]] = {
                "support": int(mask.sum()),
                "accuracy": float(class_acc),
            }
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class": per_class,
    }


def evaluate() -> Dict:
    """Main evaluation function."""
    log_header(logger, "MODEL EVALUATION")
    
    # Device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load test data
    X_test, Y_test = load_test_data()
    
    # Load model
    log_separator(logger, "-")
    model = load_model(device)
    
    # Get predictions
    log_separator(logger, "-")
    logger.info("Running inference on test set...")
    predictions, probabilities = get_predictions(model, X_test, device)
    
    # Compute metrics
    log_separator(logger, "-")
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(Y_test, predictions, config.LABEL_NAMES)
    
    # Log results
    log_header(logger, "EVALUATION RESULTS")
    
    logger.info("Overall Metrics:")
    logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    logger.info(f"  F1-score (macro):   {metrics['f1_macro']:.4f}")
    logger.info("")
    logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    logger.info(f"  F1-score (weighted):  {metrics['f1_weighted']:.4f}")
    
    log_separator(logger, "-")
    
    # Confusion matrix
    label_names_list = [config.LABEL_NAMES[i] for i in sorted(config.LABEL_NAMES.keys())]
    log_confusion_matrix(logger, metrics["confusion_matrix"], label_names_list)
    
    log_separator(logger, "-")
    
    # Classification report
    log_class_report(logger, metrics["classification_report"])
    
    log_separator(logger, "-")
    
    # Per-class analysis
    logger.info("Per-Class Analysis:")
    for class_name, class_metrics in metrics["per_class"].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Support: {class_metrics['support']}")
        logger.info(f"    Accuracy: {class_metrics['accuracy']:.4f}")
    
    log_separator(logger, "=")
    logger.info("Evaluation complete!")
    log_separator(logger, "=")
    
    return metrics


if __name__ == "__main__":
    evaluate()
