"""Minimal LSTM Model.

The SMALLEST LSTM that can overfit a 32-sample batch.
Found via incremental testing - need 2 layers with hidden=24.

Architecture:
    Input (batch, 256, 4)  
    -> LSTM (hidden=24, 2 layers, unidirectional)
    -> Last hidden state (batch, 24)
    -> Linear (24 -> 6 classes)

Total: ~7,800 parameters
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseModel


class MinimalLSTM(BaseModel):
    """Minimal LSTM for flag detection - smallest architecture that can learn.
    
    Found via incremental testing:
    - hidden=4, 1-layer: 190 params → 62% max (cannot overfit)
    - hidden=24, 2-layer: 7,830 params → 100% (can overfit) ✓
    """
    
    def __init__(
        self,
        input_size: int = 256,
        num_features: int = 4,
        num_classes: int = 6,
        hidden_size: int = 24,
        num_layers: int = 2,
        **kwargs,
    ):
        """Initialize Minimal LSTM.
        
        Args:
            input_size: Sequence length (window size)
            num_features: Number of input features (4 for OHLC)
            num_classes: Number of output classes
            hidden_size: LSTM hidden size (default: 24)
            num_layers: Number of LSTM layers (default: 2)
        """
        super().__init__(input_size, num_features, num_classes)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Minimal LSTM - 2 layers, unidirectional
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        
        # Direct output - no hidden layers
        self.output = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights for better gradient flow
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # LSTM forward
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (1, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        hidden = h_n[-1]  # (batch, hidden_size)
        
        # Output projection
        return self.output(hidden)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "input_size": self.input_size,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
    
    def get_name(self) -> str:
        return "Minimal LSTM"
    
    def get_description(self) -> str:
        total, trainable = self.count_parameters()
        return f"Minimal LSTM (h={self.hidden_size}, L={self.num_layers}, {trainable} params)"


__all__ = ["MinimalLSTM"]
