"""
Baseline Model Trainer

Trains a simple baseline model (Logistic Regression) for bull/bear flag pattern
classification. Uses hand-crafted statistical features extracted from OHLC segments.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


def extract_statistical_features(segment: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a single OHLC segment.
    
    Features extracted per column (open, high, low, close):
    - Mean
    - Standard deviation
    - Min
    - Max
    - Range (max - min)
    - First value
    - Last value
    - Trend (last - first)
    
    Additional features:
    - Segment length
    - Average candle size (high - low)
    - Average body size (|close - open|)
    - Bullish ratio (% of candles where close > open)
    
    Args:
        segment: OHLC numpy array with shape (n_timesteps, 4).
        
    Returns:
        Feature vector.
    """
    if segment.ndim != 2 or segment.shape[1] != 4 or segment.shape[0] == 0:
        # Return zeros for invalid segments
        return np.zeros(45)
    
    features = []
    
    # Per-column statistics (4 columns x 8 features = 32 features)
    for col_idx in range(4):
        col = segment[:, col_idx]
        features.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
            np.max(col) - np.min(col),  # Range
            col[0],                       # First value
            col[-1],                      # Last value
            col[-1] - col[0],             # Trend
        ])
    
    # Segment length (1 feature)
    features.append(len(segment))
    
    # Average candle size: high - low (1 feature)
    candle_sizes = segment[:, 1] - segment[:, 2]  # high - low
    features.append(np.mean(candle_sizes))
    
    # Average body size: |close - open| (1 feature)
    body_sizes = np.abs(segment[:, 3] - segment[:, 0])  # |close - open|
    features.append(np.mean(body_sizes))
    
    # Bullish ratio (1 feature)
    bullish = segment[:, 3] > segment[:, 0]  # close > open
    features.append(np.mean(bullish))
    
    # Price volatility metrics (4 features)
    close = segment[:, 3]
    if len(close) > 1:
        returns = np.diff(close) / close[:-1]
        features.extend([
            np.mean(returns),
            np.std(returns),
            np.min(returns),
            np.max(returns),
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Normalized range features (4 features)
    for col_idx in range(4):
        col = segment[:, col_idx]
        if np.mean(col) != 0:
            features.append((np.max(col) - np.min(col)) / np.mean(col))
        else:
            features.append(0)
    
    # Trend features - linear regression slope (1 feature)
    if len(close) > 1:
        x = np.arange(len(close))
        slope = np.polyfit(x, close, 1)[0]
        features.append(slope)
    else:
        features.append(0)
    
    return np.array(features)


def extract_features_batch(data: np.ndarray) -> np.ndarray:
    """
    Extract features from a batch of segments.
    
    Args:
        data: Object array of OHLC segments.
        
    Returns:
        Feature matrix with shape (n_samples, n_features).
    """
    features_list = []
    for segment in data:
        features = extract_statistical_features(segment)
        features_list.append(features)
    return np.array(features_list)


class BaselineClassifier:
    """
    Baseline classifier for bull/bear flag pattern detection.
    
    Uses statistical features and Logistic Regression or Random Forest.
    """
    
    def __init__(
        self,
        model_type: str = 'logistic',
        random_state: int = 42
    ):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'logistic' or 'random_forest'.
            random_state: Random seed.
        """
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'BaselineClassifier':
        """
        Train the classifier.
        
        Args:
            X_train: Training data (object array of segments).
            y_train: Training labels.
            X_val: Optional validation data.
            y_val: Optional validation labels.
            
        Returns:
            self
        """
        print(f"Training {self.model_type} classifier...")
        print(f"  Training samples: {len(X_train)}")
        
        # Extract features
        X_train_features = extract_features_batch(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Train model
        self.model.fit(X_train_scaled, y_train_encoded)
        self.is_fitted = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train_encoded, train_pred)
        print(f"  Training accuracy: {train_acc:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred, val_acc = self.evaluate(X_val, y_val, verbose=False)
            print(f"  Validation accuracy: {val_acc:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new data.
        
        Args:
            X: Data (object array of segments).
            
        Returns:
            Predicted labels (strings).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_features = extract_features_batch(X)
        X_scaled = self.scaler.transform(X_features)
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Data (object array of segments).
            
        Returns:
            Probability matrix.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_features = extract_features_batch(X)
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test data.
            y: True labels.
            verbose: Whether to print classification report.
            
        Returns:
            Tuple of (predictions, accuracy).
        """
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        
        if verbose:
            print("\nClassification Report:")
            print(classification_report(
                y, y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0
            ))
        
        return y_pred, acc
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (for Random Forest only).
        
        Returns:
            Dict mapping feature names to importance scores.
        """
        if self.model_type != 'random_forest':
            return None
        
        feature_names = self._get_feature_names()
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))
    
    def _get_feature_names(self) -> List[str]:
        """Generate feature names for interpretability."""
        names = []
        columns = ['open', 'high', 'low', 'close']
        stats = ['mean', 'std', 'min', 'max', 'range', 'first', 'last', 'trend']
        
        for col in columns:
            for stat in stats:
                names.append(f'{col}_{stat}')
        
        names.extend([
            'segment_length',
            'avg_candle_size',
            'avg_body_size',
            'bullish_ratio',
            'return_mean',
            'return_std',
            'return_min',
            'return_max',
        ])
        
        for col in columns:
            names.append(f'{col}_norm_range')
        
        names.append('price_slope')
        
        return names


def train_baseline_model(
    data_path: str,
    labels_path: str,
    model_type: str = 'logistic',
    test_size: float = 0.2,
    random_state: int = 42
) -> BaselineClassifier:
    """
    Train a baseline model from numpy files.
    
    Args:
        data_path: Path to data numpy file.
        labels_path: Path to labels numpy file.
        model_type: 'logistic' or 'random_forest'.
        test_size: Fraction of data for testing.
        random_state: Random seed.
        
    Returns:
        Trained classifier.
    """
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    print(f"  Loaded {len(data)} samples with {len(np.unique(labels))} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )
    
    # Train model
    classifier = BaselineClassifier(model_type=model_type, random_state=random_state)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    print("\nTest Set Evaluation:")
    classifier.evaluate(X_test, y_test)
    
    return classifier


def cross_validate_baseline(
    data_path: str,
    labels_path: str,
    model_type: str = 'logistic',
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Perform cross-validation on baseline model.
    
    Args:
        data_path: Path to data numpy file.
        labels_path: Path to labels numpy file.
        model_type: 'logistic' or 'random_forest'.
        cv: Number of cross-validation folds.
        random_state: Random seed.
        
    Returns:
        Dict with cross-validation scores.
    """
    # Load data
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    # Extract features
    X = extract_features_batch(data)
    
    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model
    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Cross-validation
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    
    results = {
        'mean_accuracy': np.mean(scores),
        'std_accuracy': np.std(scores),
        'min_accuracy': np.min(scores),
        'max_accuracy': np.max(scores),
    }
    
    print(f"\nCross-Validation Results ({cv}-fold):")
    print(f"  Mean accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
    print(f"  Range: [{results['min_accuracy']:.4f}, {results['max_accuracy']:.4f}]")
    
    return results


if __name__ == '__main__':
    import sys
    
    # Default paths
    data_path = 'processed/segments_no_handle_data.npy'
    labels_path = 'processed/segments_no_handle_labels.npy'
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        labels_path = sys.argv[2]
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Usage: python train_baseline_model.py <data.npy> [labels.npy]")
        sys.exit(1)
    
    # Train both model types
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    train_baseline_model(data_path, labels_path, model_type='logistic')
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    train_baseline_model(data_path, labels_path, model_type='random_forest')
