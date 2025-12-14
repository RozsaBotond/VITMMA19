"""
Data Module - Main Entry Point

This module provides the main Data class for loading and processing
labeled bull/bear flag pattern data from time series.

The data pipeline consists of:
1. Label Parser: Reads Label Studio JSON exports
2. Segment Extractor: Extracts OHLC segments from CSV files
3. Train/Test Splitter: Creates stratified train/val/test splits

Example usage:
    from data import Data
    
    # Initialize with labels and data directory
    data = Data(
        labels_path='labels.json',
        data_dir='raw_data',
        ratios=(0.7, 0.15, 0.15)
    )
    
    # Access train/val/test data
    X_train, y_train = data.train_data, data.train_labels
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.model_selection import train_test_split
from collections import Counter

# Import submodules
from label_parser import (
    parse_label_studio_export,
    LabeledSegment,
    get_unique_labels,
    get_label_counts
)
from segment_extractor import (
    extract_all_segments,
    create_segment_arrays,
    ExtractedSegment
)
from train_test_split import (
    DataSplit,
    stratified_train_val_test_split,
    DatasetManager
)


# Type aliases
Ratios = Tuple[float, float, float]


class Data:
    """
    Main data processing class for bull/bear flag pattern detection.
    
    This class handles the complete pipeline from Label Studio JSON exports
    to ready-to-use train/val/test numpy arrays.
    
    Attributes:
        labels_path: Path to Label Studio JSON export
        data_dir: Directory containing CSV data files
        ratios: Tuple of (train, val, test) ratios summing to 1.0
        random_state: Random seed for reproducibility
        remove_handle: Whether to remove the flag pole from segments
    """
    
    def __init__(
        self,
        labels_path: str = "labels.json",
        data_dir: str = "./raw_data",
        ratios: Ratios = (0.7, 0.15, 0.15),
        random_state: int = 42,
        remove_handle: bool = True,
        min_segment_length: int = 5,
        file_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Data pipeline.
        
        Args:
            labels_path: Path to Label Studio JSON export file.
            data_dir: Directory containing the CSV timeseries files.
            ratios: Tuple of (train_ratio, val_ratio, test_ratio).
            random_state: Random seed for train/test splitting.
            remove_handle: Whether to remove flag pole using heuristic.
            min_segment_length: Minimum number of timesteps in a segment.
            file_mapping: Optional dict mapping label filenames to actual filenames.
        """
        # Validate inputs
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")
        
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.ratios = ratios
        self.random_state = random_state
        self.remove_handle = remove_handle
        self.min_segment_length = min_segment_length
        self.file_mapping = file_mapping or self._create_default_file_mapping()
        
        # Data containers
        self.labeled_segments: List[LabeledSegment] = []
        self.extracted_segments: List[ExtractedSegment] = []
        self.data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.split: Optional[DataSplit] = None
        
        # Convenience accessors
        self.train_data: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None
        self.val_labels: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        
        # Run the pipeline
        self._load_and_process()
    
    def _create_default_file_mapping(self) -> Dict[str, str]:
        """
        Create default file mapping for Label Studio exports.
        
        Label Studio often prefixes uploaded files with UUIDs.
        This mapping handles common patterns.
        """
        mapping = {}
        
        # Map limited data files to raw data files
        limited_to_raw = {
            'XAU_1h_data_limited.csv': 'XAU_1h_data.csv',
            'XAU_1m_data_limited.csv': 'XAU_1m_data.csv',
            'XAU_5m_data_limited.csv': 'XAU_5m_data.csv',
            'XAU_15m_data_limited.csv': 'XAU_15m_data.csv',
            'XAU_30m_data_limited.csv': 'XAU_30m_data.csv',
        }
        
        # Check which files exist in data_dir
        if os.path.exists(self.data_dir):
            for limited, raw in limited_to_raw.items():
                # Try limited first, then raw
                if os.path.exists(os.path.join(self.data_dir, limited)):
                    mapping[limited] = limited
                elif os.path.exists(os.path.join(self.data_dir, raw)):
                    mapping[limited] = raw
        
        return mapping
    
    def _load_and_process(self):
        """Run the complete data loading and processing pipeline."""
        # Step 1: Parse labels
        print(f"Loading labels from {self.labels_path}...")
        self.labeled_segments = parse_label_studio_export(self.labels_path)
        print(f"  Found {len(self.labeled_segments)} labeled segments")
        print(f"  Labels: {get_label_counts(self.labeled_segments)}")
        
        # Step 2: Extract segments from CSV files
        print(f"\nExtracting segments from {self.data_dir}...")
        self.extracted_segments = extract_all_segments(
            self.labeled_segments,
            self.data_dir,
            self.file_mapping
        )
        print(f"  Extracted {len(self.extracted_segments)} segments")
        
        # Step 3: Create numpy arrays
        print(f"\nCreating numpy arrays (remove_handle={self.remove_handle})...")
        self.data, self.labels = create_segment_arrays(
            self.extracted_segments,
            remove_handle=self.remove_handle,
            min_length=self.min_segment_length
        )
        print(f"  Final dataset: {len(self.data)} samples")
        
        # Step 4: Split data
        print(f"\nSplitting data (ratios={self.ratios})...")
        self._split_data()
    
    def _split_data(self):
        """Split data into train/val/test sets."""
        train_ratio, val_ratio, test_ratio = self.ratios
        
        # Check if stratified split is possible
        label_counts = Counter(self.labels)
        min_count = min(label_counts.values())
        stratify = min_count >= 2
        
        if not stratify:
            print("  Warning: Some classes have <2 samples, using non-stratified split")
        
        self.split = stratified_train_val_test_split(
            self.data, self.labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Set convenience accessors
        self.train_data = self.split.X_train
        self.train_labels = self.split.y_train
        self.val_data = self.split.X_val
        self.val_labels = self.split.y_val
        self.test_data = self.split.X_test
        self.test_labels = self.split.y_test
        
        print(f"  Train: {len(self.train_data)} samples")
        print(f"  Val:   {len(self.val_data)} samples")
        print(f"  Test:  {len(self.test_data)} samples")
    
    def get_label_encoder(self):
        """Get a fitted label encoder for the labels."""
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(self.labels)
        return encoder
    
    def get_unique_labels(self) -> List[str]:
        """Get list of unique labels."""
        return sorted(list(set(self.labels)))
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get count of each label."""
        return dict(Counter(self.labels))
    
    def save(self, output_dir: str, prefix: str = 'bullflag'):
        """
        Save processed data to numpy files.
        
        Args:
            output_dir: Output directory.
            prefix: Filename prefix.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full dataset
        np.save(os.path.join(output_dir, f'{prefix}_data.npy'), self.data)
        np.save(os.path.join(output_dir, f'{prefix}_labels.npy'), self.labels)
        
        # Save splits
        np.save(os.path.join(output_dir, f'{prefix}_train_data.npy'), self.train_data)
        np.save(os.path.join(output_dir, f'{prefix}_train_labels.npy'), self.train_labels)
        np.save(os.path.join(output_dir, f'{prefix}_val_data.npy'), self.val_data)
        np.save(os.path.join(output_dir, f'{prefix}_val_labels.npy'), self.val_labels)
        np.save(os.path.join(output_dir, f'{prefix}_test_data.npy'), self.test_data)
        np.save(os.path.join(output_dir, f'{prefix}_test_labels.npy'), self.test_labels)
        
        print(f"Saved all data to {output_dir}/")
    
    def summary(self) -> str:
        """Return a summary of the loaded data."""
        lines = [
            f"Bull/Bear Flag Dataset Summary",
            f"=" * 40,
            f"Labels file: {self.labels_path}",
            f"Data directory: {self.data_dir}",
            f"Total samples: {len(self.data)}",
            f"Remove handle: {self.remove_handle}",
            f"",
            f"Label distribution:",
        ]
        
        for label, count in sorted(self.get_label_distribution().items()):
            lines.append(f"  {label}: {count}")
        
        lines.extend([
            f"",
            f"Split ({self.ratios}):",
            f"  Train: {len(self.train_data)} samples",
            f"  Val:   {len(self.val_data)} samples",
            f"  Test:  {len(self.test_data)} samples",
        ])
        
        return '\n'.join(lines)
    
    def __repr__(self):
        return f"Data(samples={len(self.data)}, labels={len(self.get_unique_labels())})"


# Backward compatibility: keep the old interface working
def load_data_from_files(
    file_paths: List[str],
    ratios: Ratios = (0.8, 0.05, 0.15),
    random_state: int = 42
) -> Tuple[List, List, List]:
    """
    Legacy function for loading raw CSV files without labels.
    
    This is kept for backward compatibility with the old interface.
    For new code, use the Data class with Label Studio exports.
    
    Args:
        file_paths: List of CSV file paths.
        ratios: Tuple of (train, val, test) ratios.
        random_state: Random seed.
        
    Returns:
        Tuple of (train_data, val_data, test_data).
    """
    data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        df = pd.read_csv(file_path)
        data.append((df, file_path))
    
    train_ratio, val_ratio, test_ratio = ratios
    
    train_data, temp_data = train_test_split(
        data, test_size=(1 - train_ratio), random_state=random_state
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size), random_state=random_state
    )
    
    return train_data, val_data, test_data


if __name__ == '__main__':
    # Demo usage
    import sys
    
    labels_path = 'labels.json'
    data_dir = 'raw_data'
    
    if len(sys.argv) > 1:
        labels_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    
    try:
        data = Data(labels_path=labels_path, data_dir=data_dir)
        print("\n" + data.summary())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python data.py [labels.json] [data_dir]")
