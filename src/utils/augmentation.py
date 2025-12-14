"""Data Augmentation for Time Series Sequence Labeling.

This module provides augmentation techniques for OHLC financial time series,
designed to balance underrepresented pattern classes by creating synthetic
variations of existing samples.

IMPORTANT: Normalization
========================
All augmentations assume data is ALREADY NORMALIZED to [0, 1] using per-window
min-max normalization (see src/normalization.py). Augmentations clip outputs
to [0, 1] to maintain valid normalized data.

Augmentation techniques:
- Gaussian noise injection
- Magnitude scaling (uniform and segment-wise)
- Time jittering (small temporal shifts)
- Trend injection (add/remove small trends)
- Volatility scaling (scale price ranges)

All augmentations preserve the label structure (per-timestep labels remain valid).
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    
    # Gaussian noise
    noise_std: float = 0.02  # Std dev of noise relative to price range
    
    # Magnitude scaling
    scale_range: Tuple[float, float] = (0.9, 1.1)  # Min/max scale factors
    
    # Time jittering (shift within labels)
    jitter_range: int = 3  # Max timesteps to shift
    
    # Trend injection
    trend_magnitude: float = 0.05  # Max trend slope
    
    # Volatility scaling
    volatility_range: Tuple[float, float] = (0.8, 1.2)  # Scale high-low range


class TimeSeriesAugmenter:
    """Augmenter for OHLC time series with per-timestep labels."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None, seed: int = 42):
        """Initialize augmenter.
        
        Args:
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(seed)
    
    def add_gaussian_noise(
        self, 
        X: np.ndarray, 
        noise_std: Optional[float] = None
    ) -> np.ndarray:
        """Add Gaussian noise to OHLC data.
        
        Adds small random perturbations while maintaining OHLC relationships:
        - High >= Open, Close
        - Low <= Open, Close
        
        Args:
            X: Input data (seq_len, 4) or (batch, seq_len, 4)
            noise_std: Noise standard deviation (uses config if None)
            
        Returns:
            Augmented data with same shape
        """
        if noise_std is None:
            noise_std = self.config.noise_std
        
        X_aug = X.copy()
        
        # Add noise
        noise = self.rng.normal(0, noise_std, X.shape)
        X_aug = X_aug + noise
        
        # Ensure OHLC consistency
        if X.ndim == 2:
            X_aug = self._fix_ohlc_consistency(X_aug)
        else:
            for i in range(len(X_aug)):
                X_aug[i] = self._fix_ohlc_consistency(X_aug[i])
        
        # Clip to valid range [0, 1] for normalized data
        X_aug = np.clip(X_aug, 0, 1)
        
        return X_aug
    
    def _fix_ohlc_consistency(self, X: np.ndarray) -> np.ndarray:
        """Fix OHLC consistency: High >= max(O,C), Low <= min(O,C).
        
        Args:
            X: Data of shape (seq_len, 4) with columns [O, H, L, C]
            
        Returns:
            Fixed data
        """
        X = X.copy()
        # Columns: 0=Open, 1=High, 2=Low, 3=Close
        open_close_max = np.maximum(X[:, 0], X[:, 3])
        open_close_min = np.minimum(X[:, 0], X[:, 3])
        
        # High must be >= max(Open, Close)
        X[:, 1] = np.maximum(X[:, 1], open_close_max)
        
        # Low must be <= min(Open, Close)
        X[:, 2] = np.minimum(X[:, 2], open_close_min)
        
        return X
    
    def scale_magnitude(
        self, 
        X: np.ndarray, 
        scale_factor: Optional[float] = None
    ) -> np.ndarray:
        """Scale the magnitude of price movements.
        
        Scales prices around their mean, making patterns more/less pronounced.
        
        Args:
            X: Input data (seq_len, 4) or (batch, seq_len, 4)
            scale_factor: Scaling factor (random from range if None)
            
        Returns:
            Scaled data
        """
        if scale_factor is None:
            scale_factor = self.rng.uniform(*self.config.scale_range)
        
        X_aug = X.copy()
        
        if X.ndim == 2:
            mean = X_aug.mean()
            X_aug = mean + (X_aug - mean) * scale_factor
            X_aug = self._fix_ohlc_consistency(X_aug)
        else:
            for i in range(len(X_aug)):
                mean = X_aug[i].mean()
                X_aug[i] = mean + (X_aug[i] - mean) * scale_factor
                X_aug[i] = self._fix_ohlc_consistency(X_aug[i])
        
        return np.clip(X_aug, 0, 1)
    
    def add_trend(
        self, 
        X: np.ndarray, 
        trend_slope: Optional[float] = None
    ) -> np.ndarray:
        """Add a linear trend to the data.
        
        Args:
            X: Input data (seq_len, 4) or (batch, seq_len, 4)
            trend_slope: Trend slope per timestep (random if None)
            
        Returns:
            Data with added trend
        """
        if trend_slope is None:
            trend_slope = self.rng.uniform(
                -self.config.trend_magnitude, 
                self.config.trend_magnitude
            )
        
        X_aug = X.copy()
        
        if X.ndim == 2:
            seq_len = X.shape[0]
            trend = np.linspace(0, trend_slope * seq_len, seq_len)[:, np.newaxis]
            X_aug = X_aug + trend
            X_aug = self._fix_ohlc_consistency(X_aug)
        else:
            seq_len = X.shape[1]
            trend = np.linspace(0, trend_slope * seq_len, seq_len)[:, np.newaxis]
            for i in range(len(X_aug)):
                X_aug[i] = X_aug[i] + trend
                X_aug[i] = self._fix_ohlc_consistency(X_aug[i])
        
        return np.clip(X_aug, 0, 1)
    
    def scale_volatility(
        self, 
        X: np.ndarray, 
        scale_factor: Optional[float] = None
    ) -> np.ndarray:
        """Scale the volatility (high-low range) of each candle.
        
        Args:
            X: Input data (seq_len, 4) or (batch, seq_len, 4)
            scale_factor: Volatility scaling factor (random if None)
            
        Returns:
            Data with scaled volatility
        """
        if scale_factor is None:
            scale_factor = self.rng.uniform(*self.config.volatility_range)
        
        X_aug = X.copy()
        
        if X.ndim == 2:
            X_aug = self._scale_volatility_single(X_aug, scale_factor)
        else:
            for i in range(len(X_aug)):
                X_aug[i] = self._scale_volatility_single(X_aug[i], scale_factor)
        
        return np.clip(X_aug, 0, 1)
    
    def _scale_volatility_single(
        self, 
        X: np.ndarray, 
        scale_factor: float
    ) -> np.ndarray:
        """Scale volatility for a single sample.
        
        Args:
            X: Data of shape (seq_len, 4) [O, H, L, C]
            scale_factor: Volatility scaling factor
            
        Returns:
            Scaled data
        """
        X = X.copy()
        
        # Calculate mid-price for each candle
        mid = (X[:, 1] + X[:, 2]) / 2  # (High + Low) / 2
        
        # Scale deviation from mid
        for col in range(4):
            X[:, col] = mid + (X[:, col] - mid) * scale_factor
        
        return self._fix_ohlc_consistency(X)
    
    def time_shift(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        shift: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shift the time series and labels together.
        
        Shifts the entire sequence left or right, padding with edge values.
        
        Args:
            X: Input data (seq_len, 4)
            Y: Labels (seq_len,)
            shift: Number of timesteps to shift (positive = right)
            
        Returns:
            Shifted (X, Y) tuple
        """
        if shift is None:
            shift = self.rng.randint(
                -self.config.jitter_range, 
                self.config.jitter_range + 1
            )
        
        if shift == 0:
            return X.copy(), Y.copy()
        
        X_aug = np.zeros_like(X)
        Y_aug = np.zeros_like(Y)
        
        if shift > 0:  # Shift right
            X_aug[shift:] = X[:-shift]
            X_aug[:shift] = X[0]  # Pad with first value
            Y_aug[shift:] = Y[:-shift]
            Y_aug[:shift] = Y[0]
        else:  # Shift left
            shift = -shift
            X_aug[:-shift] = X[shift:]
            X_aug[-shift:] = X[-1]  # Pad with last value
            Y_aug[:-shift] = Y[shift:]
            Y_aug[-shift:] = Y[-1]
        
        return X_aug, Y_aug
    
    def random_augment(
        self, 
        X: np.ndarray, 
        Y: np.ndarray,
        augment_types: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random combination of augmentations.
        
        Args:
            X: Input data (seq_len, 4)
            Y: Labels (seq_len,)
            augment_types: List of augmentation types to apply.
                Options: 'noise', 'scale', 'trend', 'volatility', 'shift'
                If None, randomly selects 1-3 augmentations.
                
        Returns:
            Augmented (X, Y) tuple
        """
        all_types = ['noise', 'scale', 'trend', 'volatility', 'shift']
        
        if augment_types is None:
            # Randomly select 1-3 augmentations
            n_augs = self.rng.randint(1, 4)
            augment_types = list(self.rng.choice(all_types, n_augs, replace=False))
        
        X_aug = X.copy()
        Y_aug = Y.copy()
        
        for aug_type in augment_types:
            if aug_type == 'noise':
                X_aug = self.add_gaussian_noise(X_aug)
            elif aug_type == 'scale':
                X_aug = self.scale_magnitude(X_aug)
            elif aug_type == 'trend':
                X_aug = self.add_trend(X_aug)
            elif aug_type == 'volatility':
                X_aug = self.scale_volatility(X_aug)
            elif aug_type == 'shift':
                X_aug, Y_aug = self.time_shift(X_aug, Y_aug)
        
        return X_aug, Y_aug


def balance_dataset_with_augmentation(
    X: np.ndarray,
    Y: np.ndarray,
    target_samples_per_class: Optional[int] = None,
    augmenter: Optional[TimeSeriesAugmenter] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance dataset by augmenting underrepresented classes.
    
    Each sample is associated with its dominant pattern class (non-zero label
    with most timesteps). Underrepresented classes are augmented to match
    the most common class.
    
    Args:
        X: Features (n_samples, seq_len, 4)
        Y: Labels (n_samples, seq_len)
        target_samples_per_class: Target number of samples per class.
            If None, uses the count of the most common class.
        augmenter: Augmenter instance (creates new one if None)
        seed: Random seed
        
    Returns:
        Balanced (X_balanced, Y_balanced) with augmented samples
    """
    if augmenter is None:
        augmenter = TimeSeriesAugmenter(seed=seed)
    
    n_samples, seq_len, n_features = X.shape
    
    # Determine dominant class for each sample (the pattern class with most timesteps)
    sample_classes = []
    for i in range(n_samples):
        labels_in_sample = Y[i]
        # Find the dominant pattern (non-zero) label
        unique, counts = np.unique(labels_in_sample, return_counts=True)
        
        # Exclude class 0 (None) when finding dominant pattern
        pattern_mask = unique > 0
        if pattern_mask.any():
            pattern_labels = unique[pattern_mask]
            pattern_counts = counts[pattern_mask]
            dominant_class = pattern_labels[np.argmax(pattern_counts)]
        else:
            dominant_class = 0  # All None
        
        sample_classes.append(dominant_class)
    
    sample_classes = np.array(sample_classes)
    
    # Count samples per class
    class_counts = {}
    for c in range(7):
        class_counts[c] = np.sum(sample_classes == c)
    
    print("Original samples per class:")
    for c, count in class_counts.items():
        print(f"  Class {c}: {count}")
    
    # Determine target count
    if target_samples_per_class is None:
        # Use max of pattern classes (1-6), not class 0
        pattern_counts = [class_counts[c] for c in range(1, 7)]
        target_samples_per_class = max(pattern_counts)
    
    print(f"\nTarget samples per pattern class: {target_samples_per_class}")
    
    # Collect augmented samples
    X_augmented = [X]
    Y_augmented = [Y]
    
    for target_class in range(1, 7):  # Augment pattern classes (1-6)
        current_count = class_counts[target_class]
        needed = target_samples_per_class - current_count
        
        if needed <= 0:
            continue
        
        # Get indices of samples belonging to this class
        class_indices = np.where(sample_classes == target_class)[0]
        
        if len(class_indices) == 0:
            print(f"  Warning: No samples for class {target_class}, skipping")
            continue
        
        print(f"  Class {target_class}: adding {needed} augmented samples")
        
        # Create augmented samples
        aug_X = []
        aug_Y = []
        
        for i in range(needed):
            # Randomly select a sample from this class
            idx = class_indices[i % len(class_indices)]
            
            # Apply random augmentation
            X_aug, Y_aug = augmenter.random_augment(X[idx], Y[idx])
            
            aug_X.append(X_aug)
            aug_Y.append(Y_aug)
        
        X_augmented.append(np.array(aug_X))
        Y_augmented.append(np.array(aug_Y))
    
    # Concatenate all
    X_balanced = np.concatenate(X_augmented, axis=0)
    Y_balanced = np.concatenate(Y_augmented, axis=0)
    
    print(f"\nBalanced dataset: {X_balanced.shape[0]} samples (was {n_samples})")
    
    return X_balanced, Y_balanced


def augment_on_the_fly(
    X: np.ndarray,
    Y: np.ndarray,
    augmenter: Optional[TimeSeriesAugmenter] = None,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply augmentation with probability p (for training batches).
    
    Args:
        X: Batch features (batch, seq_len, 4)
        Y: Batch labels (batch, seq_len)
        augmenter: Augmenter instance
        p: Probability of augmenting each sample
        
    Returns:
        (Augmented X, Y)
    """
    if augmenter is None:
        augmenter = TimeSeriesAugmenter()
    
    X_aug = X.copy()
    Y_aug = Y.copy()
    
    for i in range(len(X)):
        if augmenter.rng.random() < p:
            X_aug[i], Y_aug[i] = augmenter.random_augment(X[i], Y[i])
    
    return X_aug, Y_aug


if __name__ == "__main__":
    # Test augmentation
    print("Testing augmentation...")
    
    # Create dummy data
    X = np.random.rand(10, 256, 4).astype(np.float32)
    Y = np.zeros((10, 256), dtype=np.int64)
    
    # Add some pattern labels
    Y[0, 50:80] = 1
    Y[1, 100:130] = 2
    Y[2, 60:90] = 3
    Y[3, 70:100] = 4
    Y[4, 80:110] = 5
    Y[5, 90:120] = 6
    Y[6, 50:80] = 1
    Y[7, 100:130] = 2
    Y[8, 60:90] = 4
    Y[9, 70:100] = 5
    
    # Test balancing
    X_bal, Y_bal = balance_dataset_with_augmentation(X, Y)
    
    print(f"\nOriginal: {X.shape}")
    print(f"Balanced: {X_bal.shape}")
    
    # Verify augmented data
    print("\nAugmented data stats:")
    print(f"  X min: {X_bal.min():.4f}, max: {X_bal.max():.4f}")
    print(f"  Y unique: {np.unique(Y_bal)}")
