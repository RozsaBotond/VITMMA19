"""Configuration for Transformer models.

Two variants:
- TransformerClassifier: Window classification (one label per sequence)
- SeqTransformer: Sequence labeling (one label per timestep)
"""

CONFIG = {
    "name": "transformer",
    "description": "Transformer encoder for time series classification",
    
    # Model architecture
    "input_size": 4,           # OHLC features
    "num_classes": 7,          # 7-class output
    "seq_len": 256,            # Window size
    
    # Transformer architecture (Optimized 2025-12-14)
    "d_model": 128,            # Model dimension
    "nhead": 2,                # Number of attention heads
    "num_encoder_layers": 6,   # Number of transformer encoder layers
    "dim_feedforward": 128,    # Feedforward dimension
    "dropout": 0.1,
    
    # Positional encoding
    "max_len": 512,            # Max sequence length for positional encoding
    "learnable_pe": False,     # Use learnable positional embeddings
    
    # Classification head
    "pool_type": "cls",        # "cls" (CLS token), "mean", or "last"
    
    # Training (Optimized 2025-12-14)
    "learning_rate": 7.3e-4,
    "batch_size": 32,
    "epochs": 200,
    "weight_decay": 1.6e-7,
    "use_class_weights": True,
    "patience": 30,
    "min_delta": 0.0001,
    
    # Scheduler
    "scheduler": "cosine",
    "scheduler_params": {
        "T_max": 200,
        "eta_min": 1e-6,
    },
    
    # Warmup
    "warmup_epochs": 10,
    
    # Loss
    "label_smoothing": 0.1,
}

# Smaller version for quick experiments
CONFIG_SMALL = {
    **CONFIG,
    "name": "transformer_small",
    "d_model": 32,
    "nhead": 2,
    "num_encoder_layers": 2,
    "dim_feedforward": 128,
    "epochs": 100,
    "patience": 20,
}