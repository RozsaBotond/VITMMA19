"""1D CNN model for sequence classification."""

from .model import CNN1D, SeqCNN1D
from .config import CONFIG

__all__ = ["CNN1D", "SeqCNN1D", "CONFIG"]
