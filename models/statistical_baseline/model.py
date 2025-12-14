"""Statistical Baseline Model.

Uses handcrafted features based on technical analysis principles:
- Trend detection (pole phase)
- Volatility/consolidation (flag phase)
- Pattern shape features

This is NOT a neural network - it's a traditional ML baseline.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Tuple, Optional


class StatisticalBaseline:
    """Statistical baseline using trend and volatility features.
    
    Bull/Bear flags have a characteristic structure:
    1. Strong trend (the "pole") - high directional movement
    2. Consolidation (the "flag") - lower volatility, slight counter-trend
    
    We extract features that capture this pattern.
    """
    
    def __init__(
        self,
        classifier: str = "logistic_regression",
        **kwargs,
    ):
        """Initialize the statistical baseline.
        
        Args:
            classifier: "logistic_regression" or "random_forest"
        """
        self.classifier_type = classifier
        self.scaler = StandardScaler()
        
        if classifier == "logistic_regression":
            self.model = LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 1000),
                class_weight="balanced",
                random_state=42,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                class_weight="balanced",
                random_state=42,
            )
        
        self.is_fitted = False
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract handcrafted features from OHLC data.
        
        Args:
            X: OHLC data of shape (n_samples, seq_len, 4)
               Features are [open, high, low, close]
               
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        n_samples = X.shape[0]
        features = []
        
        for i in range(n_samples):
            sample = X[i]  # (seq_len, 4)
            open_prices = sample[:, 0]
            high_prices = sample[:, 1]
            low_prices = sample[:, 2]
            close_prices = sample[:, 3]
            
            # 1. Trend direction (1=bullish, 0=bearish)
            price_change = close_prices[-1] - close_prices[0]
            trend_direction = 1.0 if price_change > 0 else 0.0
            
            # 2. Trend strength (normalized price change)
            price_range = high_prices.max() - low_prices.min()
            trend_strength = abs(price_change) / (price_range + 1e-8)
            
            # 3. Volatility (std of returns)
            returns = np.diff(close_prices) / (close_prices[:-1] + 1e-8)
            volatility = np.std(returns)
            
            # 4. Average range ratio
            ranges = (high_prices - low_prices) / (close_prices + 1e-8)
            avg_range_ratio = np.mean(ranges)
            
            # 5. Average body ratio (how much of the candle is body vs wicks)
            bodies = np.abs(close_prices - open_prices)
            candle_ranges = high_prices - low_prices + 1e-8
            avg_body_ratio = np.mean(bodies / candle_ranges)
            
            # 6. Consolidation ratio (compare volatility in first vs second half)
            mid = len(close_prices) // 2
            vol_first = np.std(returns[:mid]) if mid > 1 else 0
            vol_second = np.std(returns[mid:]) if mid > 1 else 0
            consolidation_ratio = vol_second / (vol_first + 1e-8)
            
            # 7. Slope in first half (pole)
            slope_first = (close_prices[mid] - close_prices[0]) / (mid + 1e-8)
            
            # 8. Slope in second half (flag)
            slope_second = (close_prices[-1] - close_prices[mid]) / (len(close_prices) - mid + 1e-8)
            
            features.append([
                trend_direction,
                trend_strength,
                volatility,
                avg_range_ratio,
                avg_body_ratio,
                consolidation_ratio,
                slope_first,
                slope_second,
            ])
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatisticalBaseline":
        """Fit the model.
        
        Args:
            X: OHLC data of shape (n_samples, seq_len, 4)
            y: Labels of shape (n_samples,)
        """
        features = self.extract_features(X)
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.
        
        Args:
            X: OHLC data of shape (n_samples, seq_len, 4)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        features = self.extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: OHLC data of shape (n_samples, seq_len, 4)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        features = self.extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        features = self.extract_features(X)
        features_scaled = self.scaler.transform(features)
        return self.model.score(features_scaled, y)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "classifier": self.classifier_type,
            "n_features": 8,
        }
    
    def get_name(self) -> str:
        return "Statistical Baseline"
    
    def get_description(self) -> str:
        return f"Trend + Volatility features with {self.classifier_type}"


__all__ = ["StatisticalBaseline"]
