"""CNN-LSTM Hybrid Model for Time Series Classification.

Two model variants:
- CNNLSTM: Window classification (single label per sequence)
- SeqCNNLSTM: Sequence labeling (label per timestep)

Architecture:
    Input -> CNN (feature extraction) -> LSTM (temporal) -> Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from .config import CONFIG


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting local features from time series."""
    
    def __init__(
        self,
        input_size: int,
        channels: list,
        kernel_sizes: list,
        pool_sizes: list,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        in_ch = input_size
        
        for out_ch, k_size, p_size in zip(channels, kernel_sizes, pool_sizes):
            # Conv -> BN -> ReLU -> Pool
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k_size, padding=k_size // 2))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if p_size > 1:
                layers.append(nn.MaxPool1d(p_size))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            
        self.layers = nn.Sequential(*layers)
        self.output_channels = channels[-1]
        
        # Calculate output sequence length reduction
        self.pool_factor = 1
        for p in pool_sizes:
            self.pool_factor *= p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.
        
        Args:
            x: Input (batch, features, seq_len) in Conv1d format
            
        Returns:
            Features (batch, out_channels, reduced_seq_len)
        """
        return self.layers(x)


class CNNLSTM(BaseModel):
    """CNN-LSTM hybrid for window classification.
    
    Uses CNN to extract local features, then LSTM to model
    temporal dependencies, and finally classifies the whole sequence.
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        cnn_channels: List of CNN channel sizes
        cnn_kernel_sizes: List of CNN kernel sizes
        cnn_pool_sizes: List of CNN pool sizes
        cnn_dropout: CNN dropout rate
        use_batch_norm: Use batch normalization in CNN
        lstm_hidden_size: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout rate
        bidirectional: Use bidirectional LSTM
        fc_dropout: Classifier dropout rate
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        cnn_pool_sizes: list = None,
        cnn_dropout: float = CONFIG["cnn_dropout"],
        use_batch_norm: bool = CONFIG["use_batch_norm"],
        lstm_hidden_size: int = CONFIG["lstm_hidden_size"],
        lstm_num_layers: int = CONFIG["lstm_num_layers"],
        lstm_dropout: float = CONFIG["lstm_dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        fc_dropout: float = CONFIG["fc_dropout"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        cnn_channels = cnn_channels or CONFIG["cnn_channels"]
        cnn_kernel_sizes = cnn_kernel_sizes or CONFIG["cnn_kernel_sizes"]
        cnn_pool_sizes = cnn_pool_sizes or CONFIG["cnn_pool_sizes"]
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "cnn_channels": cnn_channels,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "cnn_pool_sizes": cnn_pool_sizes,
            "cnn_dropout": cnn_dropout,
            "use_batch_norm": use_batch_norm,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "lstm_dropout": lstm_dropout,
            "bidirectional": bidirectional,
            "fc_dropout": fc_dropout,
        }
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_size=input_size,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            pool_sizes=cnn_pool_sizes,
            dropout=cnn_dropout,
            use_batch_norm=use_batch_norm,
        )
        
        # LSTM temporal model
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Classifier
        lstm_output_size = lstm_hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_output_size // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Extract CNN features
        x = self.cnn(x)  # (batch, channels, reduced_seq_len)
        
        # Transpose to (batch, reduced_seq_len, channels) for LSTM
        x = x.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state (concatenate both directions if bidirectional)
        if self.num_directions == 2:
            # Concatenate forward and backward final states
            h_n = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        else:
            h_n = h_n[-1, :, :]
        
        # Classify
        out = self.classifier(h_n)
        
        return out
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams


class SeqCNNLSTM(BaseModel):
    """CNN-LSTM hybrid for sequence labeling.
    
    Uses CNN to extract local features, then bidirectional LSTM
    to model temporal dependencies, and classifies each timestep.
    
    Note: Due to CNN pooling, upsampling is needed to restore sequence length.
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        cnn_channels: List of CNN channel sizes
        cnn_kernel_sizes: List of CNN kernel sizes
        cnn_pool_sizes: List of CNN pool sizes
        cnn_dropout: CNN dropout rate
        use_batch_norm: Use batch normalization in CNN
        lstm_hidden_size: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout rate
        bidirectional: Use bidirectional LSTM
        fc_dropout: Classifier dropout rate
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        cnn_pool_sizes: list = None,
        cnn_dropout: float = CONFIG["cnn_dropout"],
        use_batch_norm: bool = CONFIG["use_batch_norm"],
        lstm_hidden_size: int = CONFIG["lstm_hidden_size"],
        lstm_num_layers: int = CONFIG["lstm_num_layers"],
        lstm_dropout: float = CONFIG["lstm_dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        fc_dropout: float = CONFIG["fc_dropout"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        cnn_channels = cnn_channels or CONFIG["cnn_channels"]
        cnn_kernel_sizes = cnn_kernel_sizes or CONFIG["cnn_kernel_sizes"]
        cnn_pool_sizes = cnn_pool_sizes or CONFIG["cnn_pool_sizes"]
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "cnn_channels": cnn_channels,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "cnn_pool_sizes": cnn_pool_sizes,
            "cnn_dropout": cnn_dropout,
            "use_batch_norm": use_batch_norm,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "lstm_dropout": lstm_dropout,
            "bidirectional": bidirectional,
            "fc_dropout": fc_dropout,
        }
        
        self.seq_len = seq_len
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_size=input_size,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            pool_sizes=cnn_pool_sizes,
            dropout=cnn_dropout,
            use_batch_norm=use_batch_norm,
        )
        
        # Calculate pool factor for upsampling
        self.pool_factor = self.cnn.pool_factor
        
        # LSTM temporal model
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Per-timestep classifier (applied after upsampling)
        lstm_output_size = lstm_hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(lstm_output_size // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Extract CNN features
        x = self.cnn(x)  # (batch, channels, reduced_seq_len)
        
        # Transpose to (batch, reduced_seq_len, channels) for LSTM
        x = x.transpose(1, 2)  # (batch, reduced_seq_len, channels)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)  # (batch, reduced_seq_len, hidden * num_directions)
        
        # Upsample back to original sequence length
        # Transpose for interpolation: (batch, hidden, reduced_seq_len)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = F.interpolate(lstm_out, size=seq_len, mode='linear', align_corners=False)
        # Transpose back: (batch, seq_len, hidden)
        lstm_out = lstm_out.transpose(1, 2)
        
        # Per-timestep classification
        out = self.classifier(lstm_out)  # (batch, seq_len, num_classes)
        
        return out
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams
