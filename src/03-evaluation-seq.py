#!/usr/bin/env python3
"""Evaluation script for Sequence Labeling models.

This script evaluates trained models on the test set with comprehensive metrics.

Usage:
    python -m src.03-evaluation-seq --model seq_lstm --checkpoint models/seq_lstm/best_model.pth
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, MODELS_DIR, LOG_DIR, DEVICE, SEED
from src.utils import setup_logging, set_seed
from src.metrics import compute_all_metrics, LABEL_NAMES


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a trained model from checkpoint."""
    from models.seq_lstm.model import SeqLabelingLSTM
    from models.seq_lstm.config import CONFIG
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint format
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", CONFIG)
    else:
        # Just state_dict
        state_dict = checkpoint
        config = CONFIG
    
    model = SeqLabelingLSTM(
        input_size=config.get("input_size", 4),
        hidden_size=config.get("hidden_size", 64),
        num_layers=config.get("num_layers", 2),
        num_classes=config.get("num_classes", 7),
        dropout=0.0,  # No dropout for eval
        bidirectional=config.get("bidirectional", False),
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
):
    """Evaluate model on test data."""
    from src.metrics import compute_all_metrics
    
    model.eval()
    
    all_preds = []
    all_probs = []
    
    n_samples = len(X_test)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_x = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            logits = model(batch_x)
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            all_preds.append(preds)
            all_probs.append(probs)
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_proba = np.concatenate(all_probs, axis=0)
    
    metrics = compute_all_metrics(Y_test, y_pred, y_proba)
    
    return metrics, y_pred, y_proba


def online_inference_demo(
    model: torch.nn.Module,
    X_sample: np.ndarray,
    device: torch.device,
):
    """Demonstrate online (streaming) inference."""
    logging.info("\n" + "=" * 60)
    logging.info("ONLINE INFERENCE DEMO")
    logging.info("=" * 60)
    
    model.eval()
    model.reset_hidden()
    
    # Convert to tensor
    x = torch.FloatTensor(X_sample).unsqueeze(0).to(device)  # (1, seq_len, 4)
    
    predictions = []
    
    with torch.no_grad():
        for t in range(x.size(1)):
            x_t = x[:, t:t+1, :]  # (1, 1, 4)
            logits = model.forward_online(x_t)
            pred = logits.argmax(dim=-1).item()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Show first 50 predictions
    logging.info(f"First 50 timestep predictions: {predictions[:50].tolist()}")
    
    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    logging.info("\nPrediction distribution:")
    for label, count in zip(unique, counts):
        name = LABEL_NAMES.get(label, f"Class {label}")
        logging.info(f"  {name}: {count} ({100*count/len(predictions):.1f}%)")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate sequence labeling model")
    parser.add_argument("--model", type=str, default="seq_lstm")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--demo-online", action="store_true", help="Demo online inference")
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"eval_{args.model}_{timestamp}.log"
    setup_logging(log_file)
    
    logging.info("=" * 60)
    logging.info("SEQUENCE LABELING MODEL EVALUATION")
    logging.info("=" * 60)
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = MODELS_DIR / args.model / "best_model.pth"
    
    logging.info(f"Model: {args.model}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"Device: {DEVICE}")
    
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load data
    X = np.load(data_dir / "X_seq.npy")
    Y = np.load(data_dir / "Y_seq.npy")
    logging.info(f"Data: X={X.shape}, Y={Y.shape}")
    
    # Split to get test set (same split as training)
    _, X_temp, _, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=args.seed)
    _, X_test, _, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=args.seed)
    
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Load model
    model, config = load_model(checkpoint_path, DEVICE)
    logging.info(f"Model config: {config}")
    
    # Evaluate
    metrics, y_pred, y_proba = evaluate_model(model, X_test, Y_test, DEVICE)
    
    # Log results
    logging.info(metrics.summary())
    
    # Save predictions
    results_dir = MODELS_DIR / args.model
    np.save(results_dir / "test_predictions.npy", y_pred)
    np.save(results_dir / "test_probabilities.npy", y_proba)
    
    # Save metrics
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logging.info(f"\nSaved predictions and metrics to {results_dir}")
    
    # Demo online inference
    if args.demo_online:
        online_inference_demo(model, X_test[0], DEVICE)
    
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
