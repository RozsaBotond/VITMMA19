"""Normalization utilities for OHLC time series data.

This module provides consistent normalization for training and inference
using sklearn's StandardScaler.

Normalization Strategy:
=======================
We use **StandardScaler** which:
1. Fits on training data to learn mean and std per feature (OHLC)
2. Saves the scaler to disk for use at inference time
3. Applies the same transformation (z-score) at inference

This ensures no data leakage: we only use training statistics.

Usage:
------
# Training:
from src.normalization import OHLCScaler
scaler = OHLCScaler()
X_train_norm = scaler.fit_transform(X_train)
scaler.save("models/scaler.pkl")  # Save for inference

# Inference:
scaler = OHLCScaler.load("models/scaler.pkl")
X_test_norm = scaler.transform(X_test)
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.preprocessing import StandardScaler


# Default scaler path
DEFAULT_SCALER_PATH = Path("models/seq_lstm/scaler.pkl")


class OHLCScaler:
    """StandardScaler wrapper for 3D OHLC time series data.
    
    Handles reshaping of (batch, seq_len, features) data for sklearn's
    StandardScaler which expects 2D input.
    
    The scaler learns mean and std for each of the 4 OHLC features
    from the training data and applies the same transformation at inference.
    
    Example:
        # Training
        scaler = OHLCScaler()
        X_train_norm = scaler.fit_transform(X_train)  # (N, 256, 4)
        scaler.save("models/scaler.pkl")
        
        # Inference
        scaler = OHLCScaler.load("models/scaler.pkl")
        X_new_norm = scaler.transform(X_new)
    """
    
    def __init__(self):
        """Initialize with a fresh StandardScaler."""
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'OHLCScaler':
        """Fit the scaler on training data.
        
        Args:
            X: Training data of shape (batch, seq_len, 4) or (seq_len, 4)
            
        Returns:
            self
        """
        X_2d = self._reshape_to_2d(X)
        self.scaler.fit(X_2d)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            X: Data of shape (batch, seq_len, 4) or (seq_len, 4)
            
        Returns:
            Normalized data, same shape as input
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform. "
                             "Call fit() or load a saved scaler.")
        
        original_shape = X.shape
        X_2d = self._reshape_to_2d(X)
        X_normalized = self.scaler.transform(X_2d)
        return X_normalized.reshape(original_shape)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Training data of shape (batch, seq_len, 4) or (seq_len, 4)
            
        Returns:
            Normalized data, same shape as input
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale.
        
        Args:
            X: Normalized data of shape (batch, seq_len, 4) or (seq_len, 4)
            
        Returns:
            Data in original scale
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        
        original_shape = X.shape
        X_2d = self._reshape_to_2d(X)
        X_original = self.scaler.inverse_transform(X_2d)
        return X_original.reshape(original_shape)
    
    def _reshape_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Reshape 3D data to 2D for sklearn.
        
        (batch, seq_len, features) -> (batch * seq_len, features)
        """
        if X.ndim == 2:
            return X
        elif X.ndim == 3:
            batch, seq_len, features = X.shape
            return X.reshape(-1, features)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
    
    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the fitted scaler to disk.
        
        Args:
            path: Path to save the scaler. Defaults to models/seq_lstm/scaler.pkl
            
        Returns:
            Path where scaler was saved
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted scaler. Call fit() first.")
        
        path = Path(path) if path else DEFAULT_SCALER_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'mean': self.scaler.mean_,
                'std': self.scaler.scale_,
            }, f)
        
        print(f"Scaler saved to {path}")
        print(f"  Mean (OHLC): {self.scaler.mean_}")
        print(f"  Std (OHLC):  {self.scaler.scale_}")
        
        return path
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> 'OHLCScaler':
        """Load a fitted scaler from disk.
        
        Args:
            path: Path to saved scaler. Defaults to models/seq_lstm/scaler.pkl
            
        Returns:
            Loaded OHLCScaler instance
        """
        path = Path(path) if path else DEFAULT_SCALER_PATH
        
        if not path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {path}. "
                "Run training first to create the scaler."
            )
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.scaler = data['scaler']
        instance._is_fitted = True
        
        return instance
    
    @property
    def mean_(self) -> np.ndarray:
        """Get the mean values per feature."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler.mean_
    
    @property
    def scale_(self) -> np.ndarray:
        """Get the std values per feature."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler.scale_


# Legacy function for backward compatibility
def normalize_window(ohlc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Legacy per-window min-max normalization.
    
    DEPRECATED: Use OHLCScaler for StandardScaler-based normalization.
    
    This is kept for backward compatibility but new code should use OHLCScaler.
    """
    if ohlc.ndim == 2:
        min_val = ohlc.min()
        max_val = ohlc.max()
        if max_val - min_val < eps:
            return np.zeros_like(ohlc)
        return (ohlc - min_val) / (max_val - min_val)
    elif ohlc.ndim == 3:
        result = np.zeros_like(ohlc)
        for i in range(len(ohlc)):
            result[i] = normalize_window(ohlc[i], eps)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {ohlc.ndim}D")


def normalize_for_inference(ohlc: np.ndarray, eps: float = 1e-8):
    """Legacy function - DEPRECATED. Use OHLCScaler.load().transform() instead."""
    import warnings
    warnings.warn(
        "normalize_for_inference is deprecated. Use OHLCScaler.load().transform()",
        DeprecationWarning
    )
    return normalize_window(ohlc, eps), {'legacy': True}
