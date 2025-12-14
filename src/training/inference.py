"""Inference utilities for sequence labeling model.

This module provides a clean interface for running inference:
1. Loads the trained StandardScaler for consistent normalization
2. Loads the trained model
3. Generates predictions with confidence scores

Usage:
------
from src.inference import SequenceLabelingInference

# Initialize
inferencer = SequenceLabelingInference(
    model_path="models/seq_lstm/best_model.pth",
    scaler_path="data/scaler.pkl",  # Or defaults to models/seq_lstm/scaler.pkl
    mode="7class"
)

# Run inference on raw OHLC data
predictions, confidences = inferencer.predict(ohlc_window)
# ohlc_window: (seq_len, 4) unnormalized OHLC data

# Get pattern regions
patterns = inferencer.get_pattern_regions(ohlc_window)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Import normalization utilities
from src.normalization import OHLCScaler


# Class names for different modes
CLASS_NAMES = {
    "binary": ["None", "Pattern"],
    "3class": ["None", "Bearish", "Bullish"],
    "7class": ["None", "Bearish Normal", "Bearish Wedge", "Bearish Pennant",
               "Bullish Normal", "Bullish Wedge", "Bullish Pennant"],
}

# Default scaler path
DEFAULT_SCALER_PATH = Path("data/scaler.pkl")


class SequenceLabelingInference:
    """Inference wrapper for sequence labeling models.
    
    Handles:
    - Model loading
    - Data normalization using fitted StandardScaler
    - Prediction generation
    - Post-processing (pattern extraction)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "models/seq_lstm/best_model.pth",
        scaler_path: Optional[Union[str, Path]] = None,
        mode: str = "7class",
        device: Optional[str] = None,
    ):
        """Initialize inference engine.
        
        Args:
            model_path: Path to saved model checkpoint
            scaler_path: Path to saved StandardScaler. Defaults to data/scaler.pkl
            mode: Classification mode ("binary", "3class", "7class")
            device: Device to run on ("cuda", "cpu", or None for auto)
        """
        self.model_path = Path(model_path)
        self.mode = mode
        self.class_names = CLASS_NAMES[mode]
        self.num_classes = len(self.class_names)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load scaler (fitted on training data)
        scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        if not scaler_path.exists():
            # Try alternate location
            scaler_path = Path("models/seq_lstm/scaler.pkl")
        self.scaler = OHLCScaler.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        from models.lstm_v2.model import SeqLabelingLSTM
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        config = checkpoint.get("config", {})
        
        model = SeqLabelingLSTM(
            input_size=config.get("input_size", 4),
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            num_classes=self.num_classes,
            dropout=config.get("dropout", 0.0),  # No dropout at inference
            bidirectional=config.get("bidirectional", False),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        ohlc: np.ndarray,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for OHLC data.
        
        Args:
            ohlc: OHLC data, shape (seq_len, 4) or (batch, seq_len, 4)
            normalize: Whether to apply StandardScaler normalization (set False if already normalized)
            
        Returns:
            Tuple of:
            - predictions: Per-timestep class predictions (seq_len,) or (batch, seq_len)
            - confidences: Per-timestep confidence scores (seq_len,) or (batch, seq_len)
        """
        # Handle single sample vs batch
        single_sample = ohlc.ndim == 2
        if single_sample:
            ohlc = ohlc[np.newaxis, ...]  # (1, seq_len, 4)
        
        # Normalize using fitted StandardScaler
        if normalize:
            ohlc = self.scaler.transform(ohlc)
        
        # Convert to tensor
        X = torch.FloatTensor(ohlc).to(self.device)
        
        # Run inference
        logits = self.model(X)  # (batch, seq_len, num_classes)
        probs = torch.softmax(logits, dim=-1)
        
        # Get predictions and confidences
        confidences, predictions = probs.max(dim=-1)
        
        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        # Remove batch dimension for single sample
        if single_sample:
            predictions = predictions[0]
            confidences = confidences[0]
        
        return predictions, confidences
    
    @torch.no_grad()
    def predict_proba(
        self,
        ohlc: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Get class probabilities for each timestep.
        
        Args:
            ohlc: OHLC data, shape (seq_len, 4) or (batch, seq_len, 4)
            normalize: Whether to apply StandardScaler normalization
            
        Returns:
            Probabilities of shape (seq_len, num_classes) or (batch, seq_len, num_classes)
        """
        single_sample = ohlc.ndim == 2
        if single_sample:
            ohlc = ohlc[np.newaxis, ...]
        
        if normalize:
            ohlc = self.scaler.transform(ohlc)
        
        X = torch.FloatTensor(ohlc).to(self.device)
        logits = self.model(X)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        if single_sample:
            probs = probs[0]
        
        return probs
    
    def get_pattern_regions(
        self,
        ohlc: np.ndarray,
        min_length: int = 5,
        confidence_threshold: float = 0.5,
    ) -> List[Dict]:
        """Extract pattern regions from predictions.
        
        Args:
            ohlc: OHLC data, shape (seq_len, 4)
            min_length: Minimum pattern length to report
            confidence_threshold: Minimum average confidence for pattern
            
        Returns:
            List of pattern dicts with:
            - 'start': Start index
            - 'end': End index
            - 'label': Class index
            - 'label_name': Class name
            - 'confidence': Average confidence in region
        """
        predictions, confidences = self.predict(ohlc, normalize=True)
        
        patterns = []
        current_label = 0
        start_idx = 0
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if pred != current_label:
                # End of previous pattern
                if current_label > 0:  # Not "None"
                    avg_conf = float(confidences[start_idx:i].mean())
                    length = i - start_idx
                    
                    if length >= min_length and avg_conf >= confidence_threshold:
                        patterns.append({
                            'start': start_idx,
                            'end': i,
                            'label': int(current_label),
                            'label_name': self.class_names[current_label],
                            'length': length,
                            'confidence': avg_conf,
                        })
                
                # Start of new pattern
                current_label = pred
                start_idx = i
        
        # Handle last pattern
        if current_label > 0:
            avg_conf = float(confidences[start_idx:].mean())
            length = len(predictions) - start_idx
            
            if length >= min_length and avg_conf >= confidence_threshold:
                patterns.append({
                    'start': start_idx,
                    'end': len(predictions),
                    'label': int(current_label),
                    'label_name': self.class_names[current_label],
                    'length': length,
                    'confidence': avg_conf,
                })
        
        return patterns


def quick_predict(
    ohlc: np.ndarray,
    model_path: str = "models/seq_lstm/best_model.pth",
    mode: str = "7class",
) -> Tuple[np.ndarray, np.ndarray]:
    """Quick prediction without persistent inference object.
    
    Args:
        ohlc: Raw OHLC data (seq_len, 4) - will be normalized automatically
        model_path: Path to model checkpoint
        mode: Classification mode
        
    Returns:
        (predictions, confidences) arrays
    """
    inferencer = SequenceLabelingInference(model_path, mode)
    return inferencer.predict(ohlc, normalize=True)


if __name__ == "__main__":
    # Demo
    print("Sequence Labeling Inference Demo")
    print("=" * 50)
    
    # Check if model exists
    model_path = Path("models/seq_lstm/best_model.pth")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train a model first with: python -m src.02-training-seq")
        exit(1)
    
    # Load some test data
    test_X = np.load("data/X_test.npy")
    test_Y = np.load("data/Y_test.npy")
    
    print(f"Loaded test data: {test_X.shape}")
    
    # Initialize inference
    inferencer = SequenceLabelingInference(model_path, mode="7class")
    print(f"Model loaded on {inferencer.device}")
    
    # Run inference on first sample
    sample = test_X[0]  # Already normalized during data prep, but we'll normalize anyway
    predictions, confidences = inferencer.predict(sample, normalize=False)
    
    print(f"\nSample shape: {sample.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predictions: {np.unique(predictions)}")
    print(f"Average confidence: {confidences.mean():.3f}")
    
    # Get pattern regions
    patterns = inferencer.get_pattern_regions(sample)
    print(f"\nDetected patterns: {len(patterns)}")
    for p in patterns:
        print(f"  {p['label_name']}: [{p['start']}, {p['end']}) conf={p['confidence']:.2f}")
