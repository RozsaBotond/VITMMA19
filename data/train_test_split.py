"""
Train/Test/Validation Splitter Module

Splits labeled segment data into train, validation, and test sets.
Supports stratified splitting to maintain label distribution.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from collections import Counter


@dataclass
class DataSplit:
    """Container for train/val/test split data."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    
    def summary(self) -> str:
        """Return a summary of the split sizes."""
        train_dist = dict(Counter(self.y_train))
        val_dist = dict(Counter(self.y_val))
        test_dist = dict(Counter(self.y_test))
        
        lines = [
            f"Train: {len(self.X_train)} samples - {train_dist}",
            f"Val:   {len(self.X_val)} samples - {val_dist}",
            f"Test:  {len(self.X_test)} samples - {test_dist}",
        ]
        return '\n'.join(lines)


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> DataSplit:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature array (can be object array with variable-length segments).
        y: Label array.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        random_state: Random seed for reproducibility.
        stratify: Whether to use stratified splitting.
        
    Returns:
        DataSplit object containing all splits.
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    stratify_y = y if stratify else None
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_y
    )
    
    # Second split: separate train and validation from remaining
    # Adjust val_ratio for the remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )


def load_and_split(
    data_path: str,
    labels_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> DataSplit:
    """
    Load data from numpy files and create train/val/test split.
    
    Args:
        data_path: Path to numpy file with segment data.
        labels_path: Path to numpy file with labels.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        random_state: Random seed.
        stratify: Whether to stratify by labels.
        
    Returns:
        DataSplit object.
    """
    X = np.load(data_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True)
    
    print(f"Loaded {len(X)} samples with {len(np.unique(y))} unique labels")
    
    return stratified_train_val_test_split(
        X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        stratify=stratify
    )


def save_split(
    split: DataSplit,
    output_dir: str,
    prefix: str = ''
) -> Dict[str, str]:
    """
    Save split data to numpy files.
    
    Args:
        split: DataSplit object.
        output_dir: Output directory.
        prefix: Optional prefix for filenames.
        
    Returns:
        Dict mapping split names to file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    prefix_str = f'{prefix}_' if prefix else ''
    
    paths = {}
    
    for name, (X, y) in [
        ('train', (split.X_train, split.y_train)),
        ('val', (split.X_val, split.y_val)),
        ('test', (split.X_test, split.y_test))
    ]:
        data_path = os.path.join(output_dir, f'{prefix_str}{name}_data.npy')
        labels_path = os.path.join(output_dir, f'{prefix_str}{name}_labels.npy')
        
        np.save(data_path, X)
        np.save(labels_path, y)
        
        paths[f'{name}_data'] = data_path
        paths[f'{name}_labels'] = labels_path
    
    return paths


def get_label_distribution(y: np.ndarray) -> Dict[str, int]:
    """Get the count of each label."""
    return dict(Counter(y))


def check_minimum_samples_per_class(
    y: np.ndarray,
    min_samples: int = 2
) -> Tuple[bool, List[str]]:
    """
    Check if all classes have minimum required samples for stratified split.
    
    Args:
        y: Label array.
        min_samples: Minimum samples required per class.
        
    Returns:
        Tuple of (is_valid, problematic_classes).
    """
    dist = get_label_distribution(y)
    problematic = [label for label, count in dist.items() if count < min_samples]
    return len(problematic) == 0, problematic


def create_balanced_subsample(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a balanced subsample with equal samples per class.
    
    Args:
        X: Feature array.
        y: Label array.
        samples_per_class: Number of samples per class. If None, uses minimum class count.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_balanced, y_balanced).
    """
    np.random.seed(random_state)
    
    unique_labels = np.unique(y)
    dist = get_label_distribution(y)
    
    if samples_per_class is None:
        samples_per_class = min(dist.values())
    
    indices = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        if len(label_indices) >= samples_per_class:
            selected = np.random.choice(label_indices, samples_per_class, replace=False)
        else:
            # Use all available samples if class has fewer than requested
            selected = label_indices
        indices.extend(selected)
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    return X[indices], y[indices]


class DatasetManager:
    """
    Manager class for handling dataset loading, splitting, and saving.
    
    Example usage:
        manager = DatasetManager()
        manager.load('segments_data.npy', 'segments_labels.npy')
        split = manager.create_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        manager.save('output/')
    """
    
    def __init__(self):
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.data_split: Optional[DataSplit] = None
    
    def load(self, data_path: str, labels_path: str) -> 'DatasetManager':
        """Load data from numpy files."""
        self.X = np.load(data_path, allow_pickle=True)
        self.y = np.load(labels_path, allow_pickle=True)
        print(f"Loaded {len(self.X)} samples")
        print(f"Label distribution: {get_label_distribution(self.y)}")
        return self
    
    def create_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ) -> DataSplit:
        """Create train/val/test split."""
        if self.X is None or self.y is None:
            raise ValueError("No data loaded. Call load() first.")
        
        # Check if stratified split is possible
        if stratify:
            is_valid, problematic = check_minimum_samples_per_class(self.y, min_samples=2)
            if not is_valid:
                print(f"Warning: Classes with <2 samples: {problematic}")
                print("Falling back to non-stratified split")
                stratify = False
        
        self.data_split = stratified_train_val_test_split(
            self.X, self.y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            stratify=stratify
        )
        
        print("\nSplit summary:")
        print(self.data_split.summary())
        
        return self.data_split
    
    def save(self, output_dir: str, prefix: str = '') -> Dict[str, str]:
        """Save the current split to files."""
        if self.data_split is None:
            raise ValueError("No split created. Call create_split() first.")
        
        paths = save_split(self.data_split, output_dir, prefix)
        print(f"\nSaved split to {output_dir}")
        return paths


if __name__ == '__main__':
    import sys
    import os
    
    # Default paths
    data_path = 'processed/segments_no_handle_data.npy'
    labels_path = 'processed/segments_no_handle_labels.npy'
    output_dir = 'processed'
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        labels_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Usage: python train_test_split.py <data.npy> [labels.npy] [output_dir]")
        sys.exit(1)
    
    # Use DatasetManager
    manager = DatasetManager()
    manager.load(data_path, labels_path)
    manager.create_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    manager.save(output_dir)
