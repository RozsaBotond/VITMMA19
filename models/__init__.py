"""Models package for Bull/Bear Flag Detector.

This package contains:
- base.py: Abstract base class for all models
- baseline/: Baseline models (simple architectures for comparison)
- lstm_v1/: First LSTM implementation
- lstm_v2/: Improved LSTM with bidirectional layers
- ... (add more as you experiment)
"""

from .base import BaseModel

__all__ = ["BaseModel"]
