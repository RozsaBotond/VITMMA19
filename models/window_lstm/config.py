"""Configuration for LSTM v1 Model.

Bidirectional LSTM for flag pattern detection. First sequence model that
captures temporal patterns in OHLC data.
"""

# Model identification
MODEL_NAME = "LSTM v1"
MODEL_VERSION = "1.0"
MODEL_DESCRIPTION = "Bidirectional LSTM with FC classifier for flag detection"

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
MODEL_TYPE = "lstm_v1"

# LSTM parameters
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
BIDIRECTIONAL = True

# Classifier head
FC_HIDDEN_SIZES = [128, 64]
FC_DROPOUT = 0.5

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adamw"

# Learning rate scheduler
LR_SCHEDULER = "reduce_on_plateau"
LR_PATIENCE = 5
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "val_loss"


def get_config() -> dict:
    """Return all configuration as a dictionary."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "lstm_dropout": LSTM_DROPOUT,
        "bidirectional": BIDIRECTIONAL,
        "fc_hidden_sizes": FC_HIDDEN_SIZES,
        "fc_dropout": FC_DROPOUT,
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
