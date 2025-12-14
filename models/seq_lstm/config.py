"""Configuration for Sequence Labeling LSTM.

This model outputs per-timestep predictions (seq_len, num_classes)
instead of window-level classification.

Tested configurations:
- Binary (pattern vs none): hidden=128, num_layers=2, ~200K params → 99%+ overfit
- 3-class (None/Bearish/Bullish): hidden=128, num_layers=2, ~200K params → 99%+ overfit
- 7-class: Requires more data or hierarchical approach
"""

# Class mappings
LABEL_MAP = {
    "none": 0,
    "bearish_normal": 1,
    "bearish_wedge": 2,
    "bearish_pennant": 3,
    "bullish_normal": 4,
    "bullish_wedge": 5,
    "bullish_pennant": 6,
}

# 3-class mapping: 0=None, 1=Bearish (1-3), 2=Bullish (4-6)
CLASS_3_NAMES = ["None", "Bearish", "Bullish"]

# 7-class names
CLASS_7_NAMES = ["None", "Bearish Normal", "Bearish Wedge", "Bearish Pennant", 
                 "Bullish Normal", "Bullish Wedge", "Bullish Pennant"]


CONFIG = {
    # Model architecture
    "input_size": 4,        # OHLC features
    "hidden_size": 128,     # Larger for better capacity (was 64)
    "num_layers": 2,        # 2-layer LSTM
    "num_classes": 3,       # Default to 3-class (None/Bearish/Bullish)
    "dropout": 0.2,         # Moderate regularization
    "bidirectional": False, # Single direction for online inference
    
    # Training
    "learning_rate": 1e-3,
    "batch_size": 16,
    "epochs": 200,          # More epochs for convergence
    "weight_decay": 1e-5,
    "use_class_weights": True,
    "max_class_weight": 10.0,
    
    # Early stopping
    "patience": 20,         # More patience with cosine schedule
    "min_delta": 1e-4,
    
    # Scheduler
    "scheduler": "cosine",
    "warmup_epochs": 5,
    
    # Loss
    "label_smoothing": 0.05,  # Light regularization
    "focal_loss": False,
    "focal_gamma": 2.0,
    
    # Model info
    "name": "seq_lstm",
    "description": "LSTM for per-timestep sequence labeling",
}

# Binary configuration (for pattern detection)
CONFIG_BINARY = {
    **CONFIG,
    "num_classes": 2,
    "name": "seq_lstm_binary",
    "description": "Binary LSTM: pattern vs no-pattern detection",
}

# 7-class configuration (for detailed pattern classification)
CONFIG_7CLASS = {
    **CONFIG,
    "num_classes": 7,
    "hidden_size": 128,
    "epochs": 300,  # Need more epochs for 7-class
    "patience": 50,  # Much more patience for noisy val set with few samples
    "learning_rate": 5e-4,  # Lower LR for stability
    "name": "seq_lstm_7class",
    "description": "7-class LSTM: detailed pattern type classification",
}
