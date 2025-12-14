"""Configuration for CNN-LSTM hybrid models.

Two variants:
- CNNLSTM: Window classification (one label per sequence)
- SeqCNNLSTM: Sequence labeling (one label per timestep)

Architecture combines:
- CNN for local feature extraction
- LSTM for temporal modeling
"""

CONFIG = {
    "name": "cnn_lstm",
    "description": "CNN-LSTM hybrid for time series classification",
    
    # Model architecture
    "input_size": 4,           # OHLC features
    "num_classes": 7,          # 7-class output
    "seq_len": 256,            # Window size
    
    # CNN feature extractor
    "cnn_channels": [32, 64],        # CNN channel progression
    "cnn_kernel_sizes": [5, 3],      # Kernel sizes
    "cnn_pool_sizes": [2, 2],        # Pooling (reduces seq_len)
    "use_batch_norm": True,
    "cnn_dropout": 0.2,
    
    # LSTM temporal model
    "lstm_hidden_size": 64,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.3,
    "bidirectional": True,
    
    # Classifier
    "fc_dropout": 0.4,
    
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

# Smaller version for quick experiments
CONFIG_SMALL = {
    **CONFIG,
    "name": "cnn_lstm_small",
    "cnn_channels": [16, 32],
    "lstm_hidden_size": 32,
    "lstm_num_layers": 1,
    "epochs": 100,
    "patience": 20,
}
