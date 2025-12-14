"""Models package for Bull/Bear Flag Detector.

This package contains:
- base.py: Abstract base class for all models
- baseline/: MLP baseline models (simple architectures for comparison)
- lstm_v1/: First LSTM implementation (window classification)
- lstm_v2/: Sequence labeling LSTM (per-timestep predictions)
- lstm_v3/: Minimal LSTM (smallest trainable architecture)
- cnn1d/: 1D Convolutional Networks
- transformer/: Transformer encoder models
- cnn_lstm/: CNN-LSTM hybrid models
- hierarchical_v1/: Two-stage hierarchical classifier
- statistical_baseline/: Non-DL baseline models
"""

from .base import BaseModel

__all__ = ["BaseModel"]
