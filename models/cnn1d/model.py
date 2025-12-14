"""1D Convolutional Neural Network for Time Series Classification.

Two model variants:
- CNN1D: Window classification (single label per sequence)
- SeqCNN1D: Sequence labeling (label per timestep) using dilated convolutions

Architecture:
    Input (batch, seq_len, 4) -> Conv1D blocks -> [Global Pool / Upsample] -> Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from .config import CONFIG


class ConvBlock(nn.Module):
    """Convolutional block: Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        dilation: int = 1,
    ):
        super().__init__()
        
        # Calculate padding to maintain sequence length (for dilated conv)
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CNN1D(BaseModel):
    """1D CNN for window classification (one label per sequence).
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        channels: List of channel sizes for each conv layer
        kernel_sizes: List of kernel sizes for each conv layer
        pool_sizes: List of pool sizes for each conv layer
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        channels: list = None,
        kernel_sizes: list = None,
        pool_sizes: list = None,
        dropout: float = CONFIG["dropout"],
        use_batch_norm: bool = CONFIG["use_batch_norm"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        channels = channels or CONFIG["channels"]
        kernel_sizes = kernel_sizes or CONFIG["kernel_sizes"]
        pool_sizes = pool_sizes or CONFIG["pool_sizes"]
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "pool_sizes": pool_sizes,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
        }
        
        # Build convolutional layers
        layers = []
        in_ch = input_size
        for out_ch, k_size, p_size in zip(channels, kernel_sizes, pool_sizes):
            layers.append(ConvBlock(
                in_ch, out_ch,
                kernel_size=k_size,
                pool_size=p_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            ))
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after all pooling
        total_pool = 1
        for p in pool_sizes:
            total_pool *= p
        final_seq_len = seq_len // total_pool
        
        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, num_classes),
        )
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classify
        x = self.classifier(x)
        
        return x


class DilatedConvBlock(nn.Module):
    """Dilated convolution block for sequence labeling (maintains sequence length)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if channel sizes match
        self.residual = (in_channels == out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.residual else None
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        if identity is not None:
            x = x + identity
            
        return x


class SeqCNN1D(BaseModel):
    """1D CNN for sequence labeling (one label per timestep).
    
    Uses dilated convolutions to maintain sequence length while
    expanding the receptive field.
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        channels: List of channel sizes
        kernel_sizes: List of kernel sizes
        dilation_rates: List of dilation rates
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        channels: list = None,
        kernel_sizes: list = None,
        dilation_rates: list = None,
        dropout: float = CONFIG["dropout"],
        use_batch_norm: bool = CONFIG["use_batch_norm"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        channels = channels or CONFIG["channels"]
        kernel_sizes = kernel_sizes or CONFIG["kernel_sizes"]
        dilation_rates = dilation_rates or CONFIG.get("dilation_rates", [1] * len(channels))
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "channels": channels,
            "kernel_sizes": kernel_sizes,
            "dilation_rates": dilation_rates,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
        }
        
        # Build dilated convolutional layers (no pooling, maintains sequence length)
        layers = []
        in_ch = input_size
        for i, (out_ch, k_size, dilation) in enumerate(zip(channels, kernel_sizes, dilation_rates)):
            layers.append(DilatedConvBlock(
                in_ch, out_ch,
                kernel_size=k_size,
                dilation=dilation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            ))
            in_ch = out_ch
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Per-timestep classifier
        self.classifier = nn.Conv1d(channels[-1], num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions
        x = self.conv_layers(x)
        
        # Per-timestep classification
        x = self.classifier(x)
        
        # Transpose back to (batch, seq_len, num_classes)
        x = x.transpose(1, 2)
        
        return x
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the network."""
        rf = 1
        for k, d in zip(self.hparams["kernel_sizes"], self.hparams["dilation_rates"]):
            rf += (k - 1) * d
        return rf
