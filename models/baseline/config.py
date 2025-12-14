"""Configuration for MLP Baseline Model.

Simple Multi-Layer Perceptron baseline model that flattens the OHLC sequence
and classifies it. Serves as a reference point for more sophisticated models.
"""

# Model identification
MODEL_NAME = "MLP Baseline"
MODEL_VERSION = "1.0"
MODEL_DESCRIPTION = "Simple MLP that flattens the OHLC sequence. No temporal modeling."

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
MODEL_TYPE = "mlp_baseline"
HIDDEN_SIZES = [512, 256, 128]
DROPOUT = 0.5
ACTIVATION = "relu"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"

# Learning rate scheduler
LR_SCHEDULER = "reduce_on_plateau"
LR_PATIENCE = 10
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "val_loss"


def get_config() -> dict:
    """Return all configuration as a dictionary."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer": OPTIMIZER,
        "lr_scheduler": LR_SCHEDULER,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }
