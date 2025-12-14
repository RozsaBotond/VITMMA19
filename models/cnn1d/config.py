"""Configuration for 1D CNN models.

Two variants:
- CNN1D: Window classification (one label per sequence)
- SeqCNN1D: Sequence labeling (one label per timestep)
"""

CONFIG = {
    "name": "cnn1d",
    "description": "1D Convolutional Neural Network for time series",
    
    # Model architecture
    "input_size": 4,           # OHLC features
    "num_classes": 7,          # 7-class output
    "seq_len": 256,            # Window size
    
    # CNN architecture
    "channels": [32, 64, 128, 256],  # Channel progression
    "kernel_sizes": [7, 5, 3, 3],    # Kernel sizes per layer
    "pool_sizes": [2, 2, 2, 2],       # Max pool sizes
    "use_batch_norm": True,
    "dropout": 0.3,
    
    # For sequence labeling variant
    "use_dilated_conv": True,         # Dilated convolutions for larger receptive field
    "dilation_rates": [1, 2, 4, 8],   # Dilation rates
    
    # Training
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 200,
    "weight_decay": 1e-4,
    "use_class_weights": True,
    "patience": 30,
    "min_delta": 0.0001,
    
    # Scheduler
    "scheduler": "cosine",
    "scheduler_params": {
        "T_max": 200,
        "eta_min": 1e-6,
    },
    
    # Loss
    "label_smoothing": 0.1,
}

# Lighter version for quick experiments
CONFIG_LIGHT = {
    **CONFIG,
    "name": "cnn1d_light",
    "channels": [16, 32, 64],
    "kernel_sizes": [5, 3, 3],
    "pool_sizes": [2, 2, 2],
    "epochs": 100,
    "patience": 20,
}
