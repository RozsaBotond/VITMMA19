"""Configuration settings for the Bull/Bear Flag Detector.

This module contains all hyperparameters and paths used in the pipeline.
Modify these values to experiment with different configurations.
"""
from pathlib import Path
import torch

class AppConfig:
    def __init__(self):
        # =============================================================================
        # PATHS
        # =============================================================================
        # Detect if running in Docker or locally
        if Path("/app/data").exists():
            # Docker environment
            self.project_root = Path("/app")
            self.data_dir = Path("/app/data")
            self.models_dir = Path("/app/models")
            self.log_dir = Path("/app/log")
        else:
            # Local development
            self.project_root = Path(__file__).parent.parent.parent
            self.data_dir = self.project_root / "data"
            self.models_dir = self.project_root / "models"
            self.log_dir = self.project_root / "log"

        # Data files
        self.raw_data_dir = self.data_dir / "raw_data"
        self.labels_file = self.data_dir / "cleaned_labels_merged.json"
        self.x_file = self.data_dir / "X.npy"
        self.y_file = self.data_dir / "Y.npy"
        self.metadata_file = self.data_dir / "metadata.json"

        # Model checkpoints
        self.best_model_path = self.models_dir / "best_model.pth"
        self.checkpoint_dir = self.models_dir / "checkpoints"

        # =============================================================================
        # DATA PREPROCESSING
        # =============================================================================
        self.window_size = 256
        self.features = ["open", "high", "low", "close"]
        self.num_features = len(self.features)

        # Label mapping
        self.label_map = {
            "Bearish Normal": 0, "Bearish Wedge": 1, "Bearish Pennant": 2,
            "Bullish Normal": 3, "Bullish Wedge": 4, "Bullish Pennant": 5,
        }
        self.num_classes = len(self.label_map)
        self.label_names = {v: k for k, v in self.label_map.items()}

        # =============================================================================
        # TRAINING HYPERPARAMETERS
        # =============================================================================
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4

        # Early stopping
        self.early_stopping_patience = 15
        self.early_stopping_min_delta = 1e-4

        # Learning rate scheduler
        self.lr_scheduler = "reduce_on_plateau"
        self.lr_patience = 5
        self.lr_factor = 0.5
        self.lr_min = 1e-6

        # Data split
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.random_seed = 42

        # =============================================================================
        # MODEL ARCHITECTURE
        # =============================================================================
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2
        self.lstm_dropout = 0.3
        self.lstm_bidirectional = True
        self.fc_hidden_sizes = [128, 64]
        self.fc_dropout = 0.5

        # =============================================================================
        # DEVICE
        # =============================================================================
        self.use_cuda = True
        self.use_mps = True
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        self.seed = self.random_seed

        # =============================================================================
        # LOGGING
        # =============================================================================
        self.log_level = "INFO"
        self.log_interval = 1

        # =============================================================================
        # INFERENCE
        # =============================================================================
        self.confidence_threshold = 0.5

    def get_config_dict(self) -> dict:
        """Return all configuration as a dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Create a single instance of the config to be used throughout the app
config = AppConfig()