"""Statistical Baseline Model for Bull/Bear Flag Detection.

This baseline uses trend detection and volatility analysis based on
training data statistics. It serves as a non-neural-network comparison
to validate that deep learning models provide meaningful improvements.

Detection Strategy:
1. Compute trend direction using linear regression slope
2. Measure volatility via ATR (Average True Range)
3. Detect consolidation periods (low volatility after strong trend)
4. Classify patterns using rule-based thresholds from training stats
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression


@dataclass
class TrainingStatistics:
    """Statistics computed from training data."""
    bullish_slope_mean: float = 0.0
    bullish_slope_std: float = 1.0
    bearish_slope_mean: float = 0.0
    bearish_slope_std: float = 1.0
    pattern_volatility_mean: float = 0.0
    pattern_volatility_std: float = 1.0
    no_pattern_volatility_mean: float = 0.0
    no_pattern_volatility_std: float = 1.0
    consolidation_ratio_mean: float = 0.5
    consolidation_ratio_std: float = 0.2
    class_priors: np.ndarray = None
    
    def __post_init__(self):
        if self.class_priors is None:
            self.class_priors = np.ones(7) / 7


class StatisticalBaseline:
    """Statistical baseline using trend and volatility features.
    
    This model does NOT use neural networks. It computes handcrafted
    features based on trend, volatility, and consolidation detection.
    """
    
    CLASSES = [
        "None",
        "Bullish_Normal", "Bullish_Wedge", "Bullish_Pennant",
        "Bearish_Normal", "Bearish_Wedge", "Bearish_Pennant"
    ]
    
    def __init__(
        self,
        window_size: int = 256,
        trend_window: int = 20,
        volatility_window: int = 14,
        consolidation_window: int = 10,
        min_trend_strength: float = 0.5,
        volatility_threshold: float = 0.7,
    ):
        self.window_size = window_size
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.consolidation_window = consolidation_window
        self.min_trend_strength = min_trend_strength
        self.volatility_threshold = volatility_threshold
        self.stats = TrainingStatistics()
        self.is_fitted = False
    
    def compute_atr(self, ohlc: np.ndarray, window: int = 14) -> np.ndarray:
        """Compute Average True Range."""
        high, low, close = ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0]
        return np.convolve(tr, np.ones(window)/window, mode='same')
    
    def compute_slope(self, prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute rolling linear regression slope using vectorized numpy."""
        n = len(prices)
        slopes = np.zeros(n)
        
        # Vectorized slope computation using least squares formula
        # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        x = np.arange(window)
        sum_x = x.sum()
        sum_x2 = (x**2).sum()
        denom = window * sum_x2 - sum_x**2
        
        for i in range(window - 1, n):
            y = prices[i - window + 1:i + 1]
            sum_y = y.sum()
            sum_xy = (x * y).sum()
            slopes[i] = (window * sum_xy - sum_x * sum_y) / denom
        
        slopes[:window - 1] = slopes[window - 1]
        return slopes
    
    def compute_volatility_ratio(self, ohlc: np.ndarray, short: int = 10, long: int = 30) -> np.ndarray:
        """Compute ratio of short-term to long-term volatility."""
        atr_short = self.compute_atr(ohlc, short)
        atr_long = np.maximum(self.compute_atr(ohlc, long), 1e-8)
        return atr_short / atr_long
    
    def extract_features(self, ohlc: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features for a single sample."""
        close = ohlc[:, 3]
        return {
            'slope': self.compute_slope(close, self.trend_window),
            'atr': self.compute_atr(ohlc, self.volatility_window),
            'volatility_ratio': self.compute_volatility_ratio(ohlc, self.consolidation_window, self.trend_window * 2),
            'price_range': (ohlc[:, 1] - ohlc[:, 2]) / (np.abs(close) + 1e-8),
        }
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'StatisticalBaseline':
        """Compute statistics from training data."""
        print("Computing training statistics for baseline...")
        
        bullish_slopes, bearish_slopes = [], []
        pattern_volatility, no_pattern_volatility = [], []
        consolidation_ratios = []
        
        for i in range(len(X)):
            features = self.extract_features(X[i])
            labels = Y[i]
            
            for t in range(len(labels)):
                label = labels[t]
                if label == 0:
                    no_pattern_volatility.append(features['atr'][t])
                elif label in [1, 2, 3]:
                    bullish_slopes.append(features['slope'][t])
                    pattern_volatility.append(features['atr'][t])
                    consolidation_ratios.append(features['volatility_ratio'][t])
                elif label in [4, 5, 6]:
                    bearish_slopes.append(features['slope'][t])
                    pattern_volatility.append(features['atr'][t])
                    consolidation_ratios.append(features['volatility_ratio'][t])
        
        self.stats.bullish_slope_mean = np.mean(bullish_slopes) if bullish_slopes else 0.001
        self.stats.bullish_slope_std = np.std(bullish_slopes) if bullish_slopes else 0.001
        self.stats.bearish_slope_mean = np.mean(bearish_slopes) if bearish_slopes else -0.001
        self.stats.bearish_slope_std = np.std(bearish_slopes) if bearish_slopes else 0.001
        self.stats.pattern_volatility_mean = np.mean(pattern_volatility) if pattern_volatility else 0.01
        self.stats.pattern_volatility_std = np.std(pattern_volatility) if pattern_volatility else 0.01
        self.stats.no_pattern_volatility_mean = np.mean(no_pattern_volatility) if no_pattern_volatility else 0.01
        self.stats.no_pattern_volatility_std = np.std(no_pattern_volatility) if no_pattern_volatility else 0.01
        self.stats.consolidation_ratio_mean = np.mean(consolidation_ratios) if consolidation_ratios else 0.5
        self.stats.consolidation_ratio_std = np.std(consolidation_ratios) if consolidation_ratios else 0.2
        
        unique, counts = np.unique(Y.flatten(), return_counts=True)
        priors = np.zeros(7)
        for cls, count in zip(unique, counts):
            priors[int(cls)] = count
        self.stats.class_priors = priors / priors.sum()
        
        print(f"  Bullish slope: {self.stats.bullish_slope_mean:.4f} ± {self.stats.bullish_slope_std:.4f}")
        print(f"  Bearish slope: {self.stats.bearish_slope_mean:.4f} ± {self.stats.bearish_slope_std:.4f}")
        print(f"  Class priors: {self.stats.class_priors}")
        
        self.is_fitted = True
        return self
    
    def predict_sample(self, ohlc: np.ndarray) -> np.ndarray:
        """Predict labels for a single sample."""
        features = self.extract_features(ohlc)
        n = len(ohlc)
        predictions = np.zeros(n, dtype=np.int64)
        
        slope = features['slope']
        vol_ratio = features['volatility_ratio']
        slope_std = max(self.stats.bullish_slope_std, self.stats.bearish_slope_std, 1e-8)
        
        for t in range(n):
            is_consolidation = vol_ratio[t] < (
                self.stats.consolidation_ratio_mean + 
                self.stats.consolidation_ratio_std * self.volatility_threshold
            )
            
            trend_slope = np.mean(slope[max(0, t - self.trend_window):t + 1])
            normalized_trend = trend_slope / slope_std
            
            if abs(normalized_trend) < self.min_trend_strength:
                predictions[t] = 0
            elif is_consolidation:
                if normalized_trend > 0:
                    if vol_ratio[t] < self.stats.consolidation_ratio_mean * 0.5:
                        predictions[t] = 3
                    elif vol_ratio[t] < self.stats.consolidation_ratio_mean * 0.8:
                        predictions[t] = 2
                    else:
                        predictions[t] = 1
                else:
                    if vol_ratio[t] < self.stats.consolidation_ratio_mean * 0.5:
                        predictions[t] = 6
                    elif vol_ratio[t] < self.stats.consolidation_ratio_mean * 0.8:
                        predictions[t] = 5
                    else:
                        predictions[t] = 4
            else:
                predictions[t] = 0
        
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for multiple samples."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        predictions = np.zeros((len(X), X.shape[1]), dtype=np.int64)
        for i in range(len(X)):
            predictions[i] = self.predict_sample(X[i])
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        predictions = self.predict(X)
        n_samples, seq_len = predictions.shape
        proba = np.ones((n_samples, seq_len, 7)) * 0.05
        for i in range(n_samples):
            for t in range(seq_len):
                proba[i, t, predictions[i, t]] = 0.7
        return proba / proba.sum(axis=-1, keepdims=True)
    
    def get_num_parameters(self) -> int:
        return 14
    
    def save(self, path: str):
        import json
        stats_dict = {
            'bullish_slope_mean': self.stats.bullish_slope_mean,
            'bullish_slope_std': self.stats.bullish_slope_std,
            'bearish_slope_mean': self.stats.bearish_slope_mean,
            'bearish_slope_std': self.stats.bearish_slope_std,
            'pattern_volatility_mean': self.stats.pattern_volatility_mean,
            'pattern_volatility_std': self.stats.pattern_volatility_std,
            'no_pattern_volatility_mean': self.stats.no_pattern_volatility_mean,
            'no_pattern_volatility_std': self.stats.no_pattern_volatility_std,
            'consolidation_ratio_mean': self.stats.consolidation_ratio_mean,
            'consolidation_ratio_std': self.stats.consolidation_ratio_std,
            'class_priors': self.stats.class_priors.tolist(),
            'config': {
                'window_size': self.window_size,
                'trend_window': self.trend_window,
                'volatility_window': self.volatility_window,
                'consolidation_window': self.consolidation_window,
                'min_trend_strength': self.min_trend_strength,
                'volatility_threshold': self.volatility_threshold,
            }
        }
        with open(path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'StatisticalBaseline':
        import json
        with open(path, 'r') as f:
            stats_dict = json.load(f)
        config = stats_dict['config']
        model = cls(**config)
        model.stats.bullish_slope_mean = stats_dict['bullish_slope_mean']
        model.stats.bullish_slope_std = stats_dict['bullish_slope_std']
        model.stats.bearish_slope_mean = stats_dict['bearish_slope_mean']
        model.stats.bearish_slope_std = stats_dict['bearish_slope_std']
        model.stats.pattern_volatility_mean = stats_dict['pattern_volatility_mean']
        model.stats.pattern_volatility_std = stats_dict['pattern_volatility_std']
        model.stats.no_pattern_volatility_mean = stats_dict['no_pattern_volatility_mean']
        model.stats.no_pattern_volatility_std = stats_dict['no_pattern_volatility_std']
        model.stats.consolidation_ratio_mean = stats_dict['consolidation_ratio_mean']
        model.stats.consolidation_ratio_std = stats_dict['consolidation_ratio_std']
        model.stats.class_priors = np.array(stats_dict['class_priors'])
        model.is_fitted = True
        return model
