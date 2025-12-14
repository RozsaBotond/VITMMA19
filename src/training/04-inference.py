"""Inference script for Bull/Bear Flag Detector.

This script runs the trained model on new, unseen data:
1. Loads the best saved model
2. Processes input data (OHLC sequences)
3. Generates predictions with confidence scores
4. Outputs results in a readable format

Usage:
    python src/04-inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import (
    setup_logger, log_header, log_separator, get_device, load_checkpoint
)
from models.lstm_v1.model import LSTMv1
from models.lstm_v1 import config as model_config

logger = setup_logger("inference")


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
    
    logger.info(f"  Model loaded successfully")
    
    return model


@torch.no_grad()
def predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on input data.
    
    Args:
        model: Trained model
        X: Input data of shape (n_samples, window_size, features)
        device: Torch device
        
    Returns:
        Tuple of (predictions, confidences, probabilities)
    """
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    outputs = model(X_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    
    predictions = outputs.argmax(dim=1).cpu().numpy()
    confidences = probabilities.max(dim=1).values.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    return predictions, confidences, probabilities


def format_prediction(
    idx: int,
    prediction: int,
    confidence: float,
    probabilities: np.ndarray,
    label_names: Dict[int, str],
) -> Dict:
    """Format a single prediction for output."""
    return {
        "sample_id": idx,
        "predicted_class": label_names[prediction],
        "predicted_label": int(prediction),
        "confidence": float(confidence),
        "probabilities": {
            label_names[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
    }


def run_inference_demo() -> None:
    """Run inference on a sample from the dataset for demonstration."""
    log_header(logger, "INFERENCE PIPELINE")
    
    # Device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    log_separator(logger, "-")
    model = load_model(device)
    
    # Load some test data for demonstration
    log_separator(logger, "-")
    logger.info("Loading sample data for demonstration...")
    
    X = np.load(config.X_FILE)
    Y = np.load(config.Y_FILE)
    
    # Take a few samples for demo
    n_demo_samples = min(10, len(X))
    demo_indices = np.random.choice(len(X), n_demo_samples, replace=False)
    X_demo = X[demo_indices]
    Y_demo = Y[demo_indices]
    
    logger.info(f"  Running inference on {n_demo_samples} samples")
    
    # Run inference
    log_separator(logger, "-")
    predictions, confidences, probabilities = predict(model, X_demo, device)
    
    # Display results
    log_header(logger, "INFERENCE RESULTS")
    
    correct = 0
    for i, (pred, conf, probs, true_label) in enumerate(
        zip(predictions, confidences, probabilities, Y_demo)
    ):
        pred_name = config.LABEL_NAMES[pred]
        true_name = config.LABEL_NAMES[true_label]
        is_correct = pred == true_label
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        logger.info(f"Sample {i+1}:")
        logger.info(f"  True label:      {true_name}")
        logger.info(f"  Predicted:       {pred_name} (confidence: {conf:.2%}) {status}")
        
        # Top 3 probabilities
        top3_idx = np.argsort(probs)[-3:][::-1]
        logger.info(f"  Top predictions:")
        for rank, idx in enumerate(top3_idx, 1):
            logger.info(f"    {rank}. {config.LABEL_NAMES[idx]}: {probs[idx]:.2%}")
        logger.info("")
    
    # Summary
    log_separator(logger, "-")
    demo_accuracy = correct / n_demo_samples
    logger.info(f"Demo Results: {correct}/{n_demo_samples} correct ({demo_accuracy:.1%} accuracy)")
    
    log_separator(logger, "=")
    logger.info("Inference complete!")
    logger.info("")
    logger.info("For production use:")
    logger.info("  1. Load your OHLC data as numpy array (window_size, 4)")
    logger.info("  2. Normalize each window to [0, 1] range")
    logger.info("  3. Call predict(model, data, device)")
    logger.info("  4. Use predictions and confidence scores")
    log_separator(logger, "=")


if __name__ == "__main__":
    run_inference_demo()
