"""Base model class for Bull/Bear Flag Detector.

This module defines the abstract base class for all models in the project.
All models (baselines, incremental versions) should inherit from this class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all flag detection models.
    
    All models must implement:
        - forward(): The forward pass
        - get_config(): Returns model configuration dict
    
    Optional overrides:
        - get_name(): Human-readable model name
        - get_description(): Brief description of the model
    """
    
    def __init__(
        self,
        input_size: int,
        num_features: int,
        num_classes: int,
        **kwargs,
    ):
        """Initialize the base model.
        
        Args:
            input_size: Sequence length (window size)
            num_features: Number of input features (4 for OHLC)
            num_classes: Number of output classes
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.input_size = input_size
        self.num_features = num_features
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Output tensor of shape (batch, num_classes) - logits
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary.
        
        Returns:
            Dictionary of model hyperparameters
        """
        pass
    
    def get_name(self) -> str:
        """Get human-readable model name."""
        return self.__class__.__name__
    
    def get_description(self) -> str:
        """Get brief description of the model."""
        return f"{self.get_name()} model for flag pattern detection"
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count model parameters.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def save(self, path: Path, optimizer: Optional[torch.optim.Optimizer] = None, **extra) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer to save state
            **extra: Additional items to save (e.g., epoch, metrics)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_config(),
            "model_class": self.__class__.__name__,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint.update(extra)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    @classmethod
    def load(
        cls,
        path: Path,
        map_location: str = "cpu",
        **override_config,
    ) -> Tuple["BaseModel", Dict[str, Any]]:
        """Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint
            map_location: Device to load the model to
            **override_config: Override config values
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        config = checkpoint.get("model_config", {})
        config.update(override_config)
        
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model, checkpoint
    
    def summary(self) -> str:
        """Get a summary string of the model architecture."""
        total, trainable = self.count_parameters()
        
        lines = [
            f"Model: {self.get_name()}",
            f"Description: {self.get_description()}",
            f"Input: ({self.input_size}, {self.num_features})",
            f"Output: {self.num_classes} classes",
            f"Parameters: {total:,} ({trainable:,} trainable)",
            "",
            "Configuration:",
        ]
        
        for key, value in self.get_config().items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
