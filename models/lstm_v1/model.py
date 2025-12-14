"""LSTM v1 - Basic Bidirectional LSTM for Flag Detection.

This is the first sequence model for flag pattern detection.
Uses a bidirectional LSTM to capture temporal patterns in OHLC data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from base import BaseModel


class LSTMv1(BaseModel):
    """Bidirectional LSTM model for flag pattern detection.
    
    Architecture:
        Input (batch, seq_len, features)
        -> Bidirectional LSTM
        -> Last hidden state (concatenated from both directions)
        -> FC layers with ReLU and Dropout
        -> Output (batch, num_classes)
    """
    
    def __init__(
        self,
        input_size: int = 256,
        num_features: int = 4,
        num_classes: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_hidden_sizes: List[int] = [128, 64],
        fc_dropout: float = 0.5,
        **kwargs,
    ):
        """Initialize LSTM v1.
        
        Args:
            input_size: Sequence length (window size)
            num_features: Number of input features (4 for OHLC)
            num_classes: Number of output classes
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: LSTM dropout (between layers)
            bidirectional: Use bidirectional LSTM
            fc_hidden_sizes: Classifier hidden layer sizes
            fc_dropout: Classifier dropout
        """
        super().__init__(input_size, num_features, num_classes)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_dropout = dropout
        self.bidirectional = bidirectional
        self.fc_hidden_sizes = fc_hidden_sizes
        self.fc_dropout = fc_dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Calculate LSTM output size
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        
        # Build classifier head
        fc_layers = []
        prev_size = lstm_out_size
        
        for fc_size in fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout),
            ])
            prev_size = fc_size
        
        # Output layer
        fc_layers.append(nn.Linear(prev_size, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # LSTM forward
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Get last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            # h_n[-2] is forward, h_n[-1] is backward (from last layer)
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        # Classifier
        return self.classifier(hidden)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "input_size": self.input_size,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.lstm_dropout,
            "bidirectional": self.bidirectional,
            "fc_hidden_sizes": self.fc_hidden_sizes,
            "fc_dropout": self.fc_dropout,
        }
    
    def get_name(self) -> str:
        return "LSTM v1"
    
    def get_description(self) -> str:
        direction = "Bidirectional" if self.bidirectional else "Unidirectional"
        return f"{direction} LSTM ({self.num_layers} layers, {self.hidden_size} hidden)"


__all__ = ["LSTMv1"]
