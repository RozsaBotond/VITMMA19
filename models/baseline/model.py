"""Simple MLP Baseline Model.

A basic Multi-Layer Perceptron that flattens the OHLC sequence and
classifies it. This model intentionally ignores temporal structure
and serves as a baseline to compare against sequence-aware models.
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


class MLPBaseline(BaseModel):
    """Simple MLP baseline model.
    
    Architecture:
        Input (batch, seq_len, features)
        -> Flatten (batch, seq_len * features)
        -> FC layers with ReLU and Dropout
        -> Output (batch, num_classes)
    """
    
    def __init__(
        self,
        input_size: int = 256,
        num_features: int = 4,
        num_classes: int = 6,
        hidden_sizes: List[int] = [512, 256, 128],
        dropout: float = 0.5,
        **kwargs,
    ):
        """Initialize MLP Baseline.
        
        Args:
            input_size: Sequence length (window size)
            num_features: Number of input features (4 for OHLC)
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__(input_size, num_features, num_classes)
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout
        
        # Build layers
        flat_size = input_size * num_features
        layers = []
        
        prev_size = flat_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Flatten: (batch, seq_len, features) -> (batch, seq_len * features)
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "input_size": self.input_size,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout_rate,
        }
    
    def get_name(self) -> str:
        return "MLP Baseline"
    
    def get_description(self) -> str:
        return "Simple MLP that flattens OHLC sequence (no temporal modeling)"


# Export for easy import
__all__ = ["MLPBaseline"]
