"""Configuration settings for the Bull/Bear Flag Detector.

This module contains all hyperparameters and paths used in the pipeline.
Modify these values to experiment with different configurations.
"""
import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
# Detect if running in Docker or locally
if Path("/app/data").exists():
    # Docker environment
    PROJECT_ROOT = Path("/app")
    DATA_DIR = Path("/app/data")
    MODELS_DIR = Path("/app/models")
    LOG_DIR = Path("/app/log")
else:
    # Local development
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOG_DIR = PROJECT_ROOT / "log"

# Data files
RAW_DATA_DIR = DATA_DIR / "raw_data"
LABELS_FILE = DATA_DIR / "cleaned_labels_merged.json"
X_FILE = DATA_DIR / "X.npy"
Y_FILE = DATA_DIR / "Y.npy"
METADATA_FILE = DATA_DIR / "metadata.json"

# Model checkpoints
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
WINDOW_SIZE = 256  # Number of OHLC bars per sample (from window size selection analysis)
FEATURES = ["open", "high", "low", "close"]  # OHLC features
NUM_FEATURES = len(FEATURES)

# Label mapping
LABEL_MAP = {
    "Bearish Normal": 0,
    "Bearish Wedge": 1,
    "Bearish Pennant": 2,
    "Bullish Normal": 3,
    "Bullish Wedge": 4,
    "Bullish Pennant": 5,
}
NUM_CLASSES = len(LABEL_MAP)
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# General training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# Learning rate scheduler
LR_SCHEDULER = "reduce_on_plateau"  # Options: "reduce_on_plateau", "cosine", "step"
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# LSTM model parameters
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True

# Fully connected layers after LSTM
FC_HIDDEN_SIZES = [128, 64]
FC_DROPOUT = 0.5

# =============================================================================
# DEVICE
# =============================================================================
USE_CUDA = True  # Use GPU if available
USE_MPS = True   # Use Apple MPS if available

import torch
if USE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif USE_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Alias for RANDOM_SEED
SEED = RANDOM_SEED

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
LOG_INTERVAL = 1  # Log every N epochs

# =============================================================================
# INFERENCE
# =============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for predictions


def get_config_dict() -> dict:
    """Return all configuration as a dictionary for logging."""
    return {
        # Paths
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        # Data
        "window_size": WINDOW_SIZE,
        "num_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        # Training
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_scheduler": LR_SCHEDULER,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        # Data split
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        # Model
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "lstm_num_layers": LSTM_NUM_LAYERS,
        "lstm_dropout": LSTM_DROPOUT,
        "lstm_bidirectional": LSTM_BIDIRECTIONAL,
        "fc_hidden_sizes": FC_HIDDEN_SIZES,
        "fc_dropout": FC_DROPOUT,
    }
