"""Inference script for the Bull/Bear Flag Detector.

This script loads a trained model and runs inference on a sample
of data, demonstrating how to use the model for predictions.

Usage:
    python src/inference.py --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for imports. This assumes src is added to PYTHONPATH
# or that main.py (which handles path fixing) is the entry point.
# _fix_sys_path removed to satisfy ruff E402.

from utils.config import AppConfig
from utils.utils import setup_logger, set_seed, get_device, load_checkpoint, log_header
from utils.normalization import OHLCScaler

# Import all model architectures
from models.lstm_v2.model import SeqLabelingLSTM
from models.cnn1d.model import SeqCNN1D
from models.transformer.model import SeqTransformer
from models.cnn_lstm.model import SeqCNNLSTM
from models.hierarchical_v1.model import HierarchicalClassifier

logger = logging.getLogger(__name__)

# --- Best Hyperparameters (used for model reconstruction) ---
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
}

MODEL_REGISTRY = {
    "lstm_v2": {"class": SeqLabelingLSTM, "name": "LSTM_v2 (SeqLabeling)"},
    "cnn1d": {"class": SeqCNN1D, "name": "CNN1D"},
    "transformer": {"class": SeqTransformer, "name": "Transformer"},
    "cnn_lstm": {"class": SeqCNNLSTM, "name": "CNN_LSTM"},
    "hierarchical_v1": {"class": HierarchicalClassifier, "name": "Hierarchical LSTM"},
}

def inference_main(**kwargs):
    """Main function for model inference."""
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument("--model-key", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Key of the model to use for inference.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--log-file", type=str, default="log/inference_run.log", help="Path to the log file.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    
    args, unknown = parser.parse_known_args(args=sys.argv[1:])
    
    for key, value in kwargs.items():
        if hasattr(args, key) and value is not None:
            setattr(args, key, value)

    log_level = getattr(logging, args.log_level.upper())
    setup_logger(log_file=Path(args.log_file), level=log_level, name=__name__)

    config = AppConfig()
    set_seed(config.random_seed)
    device = get_device()

    model_info = MODEL_REGISTRY[args.model_key]
    model_class = model_info["class"]
    model_name = model_info["name"]
    hparams = BEST_HPARAMS.get(args.model_key, {}) # Use best hparams for reconstruction

    logger.info(f"Loading model '{model_name}' from {args.checkpoint_path}")
    model, _ = load_checkpoint(
        path=args.checkpoint_path,
        model_class=model_class,
        hparams=hparams,
        num_classes=config.num_classes,
        input_size=config.num_features,
        seq_len=config.window_size,
        device=device
    )
    model.to(device)
    model.eval()
    logger.info(f"Model '{model_name}' loaded successfully.")

    # Load a sample from the test set for inference
    X_test_path = config.data_dir / "X_test.npy"
    if not X_test_path.exists():
        logger.error(f"Test data not found at {X_test_path}. Please run data preprocessing first.")
        sys.exit(1)
        
    X_test = np.load(X_test_path)
    sample_idx = 0
    sample_data = X_test[sample_idx:sample_idx+1] # Keep batch dimension

    # Normalize sample
    scaler = OHLCScaler()
    # In a real scenario, the scaler from training should be used.
    # Here, we fit on the sample itself for demonstration.
    sample_norm = scaler.fit_transform(sample_data)
    
    # Convert to tensor
    sample_tensor = torch.tensor(sample_norm, dtype=torch.float32).to(device)
    
    log_header(logger, "INFERENCE")
    logger.info(f"Running inference on sample {sample_idx} from the test set.")
    
    with torch.no_grad():
        outputs = model(sample_tensor)
        predictions = outputs.argmax(dim=-1).cpu().numpy()
        
    logger.info(f"Input shape: {sample_tensor.shape}")
    logger.info(f"Output predictions shape: {predictions.shape}")
    logger.info(f"Predictions for first 10 timesteps: {predictions[0, :10]}")
    
    # You can add more logic here to visualize or further process the predictions.

if __name__ == "__main__":
    inference_main()
