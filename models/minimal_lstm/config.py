"""Configuration for Minimal LSTM Model.

This is the SMALLEST LSTM that can overfit a 32-sample batch.
Found via incremental testing:
- hidden=4, 1-layer: 190 params → cannot overfit (62% max)
- hidden=32, 1-layer: 5K params → cannot overfit (90% max)
- hidden=24, 2-layer: 7.8K params → CAN overfit (100%)! ✓

Random baseline for 6 classes = 16.7% accuracy.
"""

# Model identification
MODEL_NAME = "Minimal LSTM"
MODEL_VERSION = "1.0"
MODEL_DESCRIPTION = "Smallest LSTM that can overfit - ~7.8K parameters"

# =============================================================================
# MODEL ARCHITECTURE (MINIMAL THAT CAN LEARN)
# =============================================================================
MODEL_TYPE = "minimal_lstm"

# LSTM parameters - found via incremental testing
HIDDEN_SIZE = 24         # Minimum that can overfit with 2 layers
NUM_LAYERS = 2           # 2 layers needed for enough capacity
LSTM_DROPOUT = 0.0       # No dropout for overfit test
BIDIRECTIONAL = False    # Unidirectional to minimize params

# No FC layers - direct LSTM output to classifier
FC_HIDDEN_SIZES = []     # Empty - just linear projection
FC_DROPOUT = 0.0         # No dropout

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
EPOCHS = 200             # Enough for convergence
BATCH_SIZE = 32          # Standard batch
LEARNING_RATE = 1e-2     # Higher LR for small model
WEIGHT_DECAY = 0.0       # No regularization initially
OPTIMIZER = "adam"

# Learning rate scheduler
LR_SCHEDULER = "reduce_on_plateau"
LR_PATIENCE = 10
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 30
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
    }


def calculate_params() -> dict:
    """Calculate the number of parameters in the minimal LSTM.
    
    LSTM parameter count formula:
    - For each gate (input, forget, cell, output):
      - Weights from input: hidden_size × input_size
      - Weights from hidden: hidden_size × hidden_size  
      - Biases: hidden_size
    - Total per layer: 4 × (input_size × hidden_size + hidden_size² + hidden_size)
    
    Output layer: hidden_size × num_classes + num_classes
    """
    input_size = 4  # OHLC
    num_classes = 6
    
    # LSTM params: 4 gates × (input weights + recurrent weights + biases)
    lstm_params = 4 * (input_size * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE)
    
    # Output layer params
    output_params = HIDDEN_SIZE * num_classes + num_classes
    
    total = lstm_params + output_params
    
    return {
        "lstm_params": lstm_params,
        "output_params": output_params,
        "total_params": total,
        "vs_random_baseline": "16.7% (1/6 classes)",
    }
