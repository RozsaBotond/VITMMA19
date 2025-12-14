"""Transformer Models for Time Series Classification.

Two model variants:
- TransformerClassifier: Window classification (single label per sequence)
- SeqTransformer: Sequence labeling (label per timestep)

Based on the vanilla Transformer encoder architecture with positional encoding.
"""

import math
import torch
import torch.nn as nn

from models.base import BaseModel
from .config import CONFIG


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """Transformer encoder for window classification.
    
    Uses a CLS token (or pooling) to aggregate sequence information
    for single-label classification.
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        pool_type: Pooling type ("cls", "mean", "last")
        learnable_pe: Use learnable positional embeddings
        max_len: Maximum sequence length for positional encoding
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        d_model: int = CONFIG["d_model"],
        nhead: int = CONFIG["nhead"],
        num_encoder_layers: int = CONFIG["num_encoder_layers"],
        dim_feedforward: int = CONFIG["dim_feedforward"],
        dropout: float = CONFIG["dropout"],
        pool_type: str = CONFIG["pool_type"],
        learnable_pe: bool = CONFIG["learnable_pe"],
        max_len: int = CONFIG["max_len"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "pool_type": pool_type,
            "learnable_pe": learnable_pe,
            "max_len": max_len,
        }
        
        self.pool_type = pool_type
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # CLS token (if using cls pooling)
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Positional encoding
        if learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size = x.size(0)
        
        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add CLS token if using cls pooling
        if self.pool_type == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Pool sequence
        if self.pool_type == "cls":
            x = x[:, 0, :]  # Take CLS token output
        elif self.pool_type == "mean":
            x = x.mean(dim=1)  # Mean pooling
        elif self.pool_type == "last":
            x = x[:, -1, :]  # Take last position
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Classify
        x = self.classifier(x)
        
        return x
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams


class SeqTransformer(BaseModel):
    """Transformer encoder for sequence labeling.
    
    Produces a prediction for each timestep in the input sequence.
    
    Args:
        input_size: Number of input features (4 for OHLC)
        num_classes: Number of output classes
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        learnable_pe: Use learnable positional embeddings
        max_len: Maximum sequence length for positional encoding
    """
    
    def __init__(
        self,
        input_size: int = CONFIG["input_size"],
        num_classes: int = CONFIG["num_classes"],
        seq_len: int = CONFIG["seq_len"],
        d_model: int = CONFIG["d_model"],
        nhead: int = CONFIG["nhead"],
        num_encoder_layers: int = CONFIG["num_encoder_layers"],
        dim_feedforward: int = CONFIG["dim_feedforward"],
        dropout: float = CONFIG["dropout"],
        learnable_pe: bool = CONFIG["learnable_pe"],
        max_len: int = CONFIG["max_len"],
    ):
        super().__init__(
            input_size=seq_len,
            num_features=input_size,
            num_classes=num_classes,
        )
        
        # Store hyperparameters
        self.hparams = {
            "input_size": input_size,
            "num_classes": num_classes,
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "learnable_pe": learnable_pe,
            "max_len": max_len,
        }
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        if learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Per-timestep classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Logits of shape (batch, seq_len, num_classes)
        """
        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Per-timestep classification
        x = self.classifier(x)  # (batch, seq_len, num_classes)
        
        return x
    
    def get_config(self) -> dict:
        """Return model configuration."""
        return self.hparams
