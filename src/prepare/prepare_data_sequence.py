"""Prepare training data for SEQUENCE LABELING.

Problem Formulation:
====================
- Input: OHLC time series (seq_len, 4)
- Output: Label per timestep (seq_len,) where:
  - 0 = None (no pattern / background)
  - 1 = Bearish Normal
  - 2 = Bearish Wedge
  - 3 = Bearish Pennant
  - 4 = Bullish Normal
  - 5 = Bullish Wedge
  - 6 = Bullish Pennant

Data Pipeline:
==============
1. Load raw OHLC CSVs
2. Load labeled regions from JSON (start/end timestamps + label)
3. For each labeled region:
   - Extract window centered on the pattern
   - Create per-timestep labels:
     * Timesteps inside labeled region → pattern label (1-6)
     * Timesteps outside (padding) → 0 (None)
4. SPLIT first into train/val/test with stratification (all classes represented)
5. Augment ONLY the training set to balance classes
6. Save X_train.npy, X_val.npy, X_test.npy, Y_train.npy, Y_val.npy, Y_test.npy

Usage:
    python -m src.prepare.prepare_data_sequence --data-dir data --window-size 256
    python -m src.prepare.prepare_data_sequence --data-dir data --balance  # With augmentation
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Label mapping for sequence labeling (0 = None/Background)
LABEL_MAP = {
    "None": 0,           # Background - no pattern
    "Bearish Normal": 1,
    "Bearish Wedge": 2,
    "Bearish Pennant": 3,
    "Bullish Normal": 4,
    "Bullish Wedge": 5,
    "Bullish Pennant": 6,
}

# Reverse mapping for decoding
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)  # 7 classes (including None)


def get_sample_dominant_class(Y: np.ndarray) -> np.ndarray:
    """Get the dominant (non-zero) class for each sample.
    
    Args:
        Y: Labels array (n_samples, seq_len)
        
    Returns:
        Array of dominant class per sample (n_samples,)
    """
    n_samples = Y.shape[0]
    sample_classes = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        labels = Y[i]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Find dominant pattern (non-zero) label
        pattern_mask = unique > 0
        if pattern_mask.any():
            pattern_labels = unique[pattern_mask]
            pattern_counts = counts[pattern_mask]
            sample_classes[i] = pattern_labels[np.argmax(pattern_counts)]
        else:
            sample_classes[i] = 0  # All None
    
    return sample_classes


def stratified_split_with_all_classes(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data ensuring all classes are represented in EACH split (train, val, test).
    
    Strategy:
    1. GUARANTEE at least 1 sample per class in train, val, AND test
    2. For classes with < 3 samples, we MUST have at least 3 to split properly
    3. Remaining samples distributed proportionally
    
    Args:
        X: Features (n_samples, seq_len, 4)
        Y: Labels (n_samples, seq_len)
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed
        
    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (X, Y) tuple
    """
    np.random.seed(seed)
    
    # Get dominant class per sample
    sample_classes = get_sample_dominant_class(Y)
    n_samples = len(X)
    
    # Count samples per class
    unique_classes, counts = np.unique(sample_classes, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    
    print("\nSamples per class (before split):")
    for c in range(NUM_CLASSES):
        count = class_counts.get(c, 0)
        status = ""
        if count == 0:
            status = " (MISSING - will skip)"
        elif count < 3:
            status = f" (WARNING: need at least 3 for proper split, have {count})"
        print(f"  {c} ({LABEL_NAMES[c]}): {count}{status}")
    
    # Initialize index sets
    train_indices = []
    val_indices = []
    test_indices = []
    
    # For each class, ensure at least 1 sample in train, val, AND test
    for c in unique_classes:
        class_indices = np.where(sample_classes == c)[0]
        np.random.shuffle(class_indices)
        n_class = len(class_indices)
        
        if n_class < 3:
            # Not enough samples - CRITICAL: we need at least 3 to have 1 in each split
            # For now, prioritize: 1 train (for augmentation), then val, then test
            print(f"  WARNING: Class {c} ({LABEL_NAMES[c]}) has only {n_class} samples!")
            if n_class >= 1:
                train_indices.append(class_indices[0])  # Must have train for augmentation
            if n_class >= 2:
                val_indices.append(class_indices[1])    # Try to have val
            if n_class >= 3:
                test_indices.append(class_indices[2])   # Try to have test
            # If only 1-2 samples, val/test won't have this class
            continue
        
        # We have at least 3 samples - guarantee 1 in each split
        train_indices.append(class_indices[0])  # 1 for train
        val_indices.append(class_indices[1])    # 1 for val
        test_indices.append(class_indices[2])   # 1 for test
        
        if n_class > 3:
            # Distribute remaining samples proportionally
            remaining = class_indices[3:]
            n_remaining = len(remaining)
            
            # Calculate how many more for test and val
            n_test_extra = max(0, int(n_remaining * test_size / (test_size + val_size + (1 - test_size - val_size))))
            n_val_extra = max(0, int(n_remaining * val_size / (test_size + val_size + (1 - test_size - val_size))))
            
            test_indices.extend(remaining[:n_test_extra])
            val_indices.extend(remaining[n_test_extra:n_test_extra + n_val_extra])
            train_indices.extend(remaining[n_test_extra + n_val_extra:])
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_indices)} ({100*len(train_indices)/n_samples:.1f}%)")
    print(f"  Val:   {len(val_indices)} ({100*len(val_indices)/n_samples:.1f}%)")
    print(f"  Test:  {len(test_indices)} ({100*len(test_indices)/n_samples:.1f}%)")
    
    # Verify all classes in val and test
    val_classes = np.unique(sample_classes[val_indices])
    test_classes = np.unique(sample_classes[test_indices])
    
    print(f"\nClasses in val:  {sorted(val_classes)}")
    print(f"Classes in test: {sorted(test_classes)}")
    
    return {
        'train': (X[train_indices], Y[train_indices]),
        'val': (X[val_indices], Y[val_indices]),
        'test': (X[test_indices], Y[test_indices]),
    }


def load_ohlc_csv(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load OHLC CSV with robust timestamp parsing."""
    df = pd.read_csv(file_path, sep=sep)
    
    column_mapping = {
        "Date": "timestamp", "date": "timestamp",
        "Timestamp": "timestamp", "timestamp": "timestamp",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
    }
    df.rename(columns=column_mapping, inplace=True)
    
    if "timestamp" not in df.columns:
        df = pd.read_csv(file_path, sep=";")
        df.rename(columns=column_mapping, inplace=True)
    
    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column in {file_path}")
    
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    df.set_index("timestamp", inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df.sort_index(inplace=True)
    return df


def parse_ls_time(time_str: str) -> datetime:
    """Parse Label Studio time format 'YYYY-MM-DD HH:MM'."""
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M")


def normalize_ohlc(ohlc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """DEPRECATED: Use OHLCScaler instead.
    
    Legacy per-window min-max normalization.
    Kept for backward compatibility only.
    """
    import warnings
    warnings.warn("normalize_ohlc is deprecated. Use OHLCScaler instead.", DeprecationWarning)
    min_val = ohlc.min()
    max_val = ohlc.max()
    if max_val - min_val < eps:
        return np.zeros_like(ohlc)
    return (ohlc - min_val) / (max_val - min_val)


def extract_segment_with_labels(
    df: pd.DataFrame,
    start_str: str,
    end_str: str,
    label: str,
    window_size: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract a fixed-size segment with per-timestep labels.
    
    Args:
        df: DataFrame with OHLC data
        start_str: Start timestamp of labeled region
        end_str: End timestamp of labeled region
        label: Pattern label string
        window_size: Fixed output size
        
    Returns:
        Tuple of (OHLC array (window_size, 4), labels array (window_size,))
        Labels: 0 for padding (None), 1-6 for pattern timesteps
    """
    try:
        start_dt = pd.Timestamp(parse_ls_time(start_str), tz="UTC")
        end_dt = pd.Timestamp(parse_ls_time(end_str), tz="UTC")
    except ValueError:
        return None
    
    # Get indices for the labeled region
    start_idx = df.index.get_indexer([start_dt], method="nearest")[0]
    end_idx = df.index.get_indexer([end_dt], method="nearest")[0]
    
    if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
        return None
    
    # Length of the actual labeled region
    label_length = end_idx - start_idx + 1
    
    # Calculate center of the labeled region
    center_idx = (start_idx + end_idx) // 2
    
    # Extract window centered on the label
    half_window = window_size // 2
    extract_start = center_idx - half_window
    extract_end = extract_start + window_size
    
    # Handle edge cases
    if extract_start < 0:
        extract_start = 0
        extract_end = window_size
    if extract_end > len(df):
        extract_end = len(df)
        extract_start = max(0, extract_end - window_size)
    
    # Check if we have enough data
    if extract_end - extract_start < window_size:
        return None
    
    # Extract OHLC
    ohlc = df.iloc[extract_start:extract_end][["open", "high", "low", "close"]].values.astype(np.float32)
    
    # Create per-timestep labels
    labels = np.zeros(window_size, dtype=np.int64)  # All zeros (None) initially
    
    # Calculate where the labeled region falls within the window
    label_start_in_window = max(0, start_idx - extract_start)
    label_end_in_window = min(window_size, end_idx - extract_start + 1)
    
    # Set labels for the pattern region
    pattern_label = LABEL_MAP.get(label, 0)
    if pattern_label > 0:  # Only if it's a valid pattern (not None)
        labels[label_start_in_window:label_end_in_window] = pattern_label
    
    # NOTE: No normalization here - training script handles it
    # This ensures we fit scaler on training data only
    
    return ohlc, labels


def prepare_sequence_data(
    data_dir: str,
    labels_file: str = "cleaned_labels_merged.json",
    window_size: int = 256,
    output_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Prepare sequence labeling data from labeled regions.
    
    Extracts windows around labeled patterns and creates per-timestep labels.
    Outputs X_seq.npy and Y_seq.npy - training script handles splitting,
    normalization, and augmentation.
    
    Args:
        data_dir: Directory containing raw_data/ and labels file
        labels_file: Name of the labels JSON file
        window_size: Fixed window size for each segment
        output_dir: Output directory (defaults to data_dir)
        
    Returns:
        Dict with 'X', 'Y', 'metadata' keys
    """
    data_path = Path(data_dir)
    labels_path = data_path / labels_file
    raw_data_path = data_path / "raw_data"
    output_path = Path(output_dir) if output_dir else data_path
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path) as f:
        tasks = json.load(f)
    
    # Build CSV path map
    csv_map = {}
    for csv_file in raw_data_path.glob("*.csv"):
        csv_map[csv_file.name] = csv_file
    
    X_list = []
    Y_list = []
    metadata = {
        "window_size": window_size,
        "label_map": LABEL_MAP,
        "num_classes": NUM_CLASSES,
        "task_type": "sequence_labeling",
        "samples": [],
    }
    
    for task in tasks:
        csv_ref = task.get("data", {}).get("csv", "")
        if not csv_ref:
            continue
        
        csv_filename = os.path.basename(csv_ref)
        
        # Match to actual file
        matched_path = None
        for name, path in csv_map.items():
            if csv_filename.endswith(name) or name.endswith(csv_filename.split("-")[-1]):
                matched_path = path
                break
        
        if not matched_path:
            stripped = csv_filename.split("-", 1)[-1] if "-" in csv_filename else csv_filename
            if stripped in csv_map:
                matched_path = csv_map[stripped]
        
        if not matched_path:
            print(f"[WARN] Could not find CSV: {csv_filename}")
            continue
        
        # Load CSV
        try:
            df = load_ohlc_csv(str(matched_path))
        except Exception as e:
            print(f"[WARN] Failed to load {matched_path}: {e}")
            continue
        
        csv_name = matched_path.stem
        
        # Process each labeled result
        results = task.get("annotations", [{}])[0].get("result", [])
        for r in results:
            labels = r.get("value", {}).get("timeserieslabels", [])
            if not labels:
                continue
            
            label = labels[0]
            if label not in LABEL_MAP:
                print(f"[WARN] Unknown label: {label}")
                continue
            
            start_str = r.get("value", {}).get("start", "")
            end_str = r.get("value", {}).get("end", "")
            
            if not start_str or not end_str:
                continue
            
            # Extract segment with per-timestep labels
            result = extract_segment_with_labels(df, start_str, end_str, label, window_size)
            if result is None:
                continue
            
            ohlc, timestep_labels = result
            
            X_list.append(ohlc)
            Y_list.append(timestep_labels)
            
            # Calculate pattern coverage in this window
            pattern_timesteps = (timestep_labels > 0).sum()
            coverage_ratio = pattern_timesteps / window_size
            
            metadata["samples"].append({
                "csv": csv_name,
                "label": label,
                "start": start_str,
                "end": end_str,
                "pattern_timesteps": int(pattern_timesteps),
                "coverage_ratio": float(coverage_ratio),
            })
    
    if not X_list:
        raise ValueError("No valid segments extracted")
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.int64)
    
    print(f"\nExtracted {len(X)} samples total")
    
    # Print class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    sample_classes = get_sample_dominant_class(Y)
    for c in range(NUM_CLASSES):
        count = (sample_classes == c).sum()
        if count > 0:
            print(f"  {c} ({LABEL_NAMES[c]}): {count}")
    
    # Save raw data - training script will handle splitting, normalization, augmentation
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "X_seq.npy", X)
    np.save(output_path / "Y_seq.npy", Y)
    
    # Save metadata
    metadata["n_samples"] = len(X)
    with open(output_path / "metadata_seq.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print(f"  - X_seq.npy: {X.shape}")
    print(f"  - Y_seq.npy: {Y.shape}")
    print(f"  - metadata_seq.json")
    print("\nNote: Training script will handle splitting, normalization, and augmentation")
    
    return {
        'X': X,
        'Y': Y,
        'metadata': metadata,
    }


def print_stats(X: np.ndarray, Y: np.ndarray, metadata: Dict) -> None:
    """Print dataset statistics for sequence labeling."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    print(f"Total samples: {len(X)}")
    print(f"Window size: {metadata['window_size']}")
    print(f"Num classes: {metadata['num_classes']} (including None)")
    print()
    
    # Per-timestep label distribution
    print("Per-timestep label distribution:")
    print("-" * 50)
    all_labels = Y.flatten()
    total = len(all_labels)
    for label_name, label_idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        count = (all_labels == label_idx).sum()
        pct = 100.0 * count / total
        print(f"  {label_idx}: {label_name:<20} {count:>8} ({pct:>6.2f}%)")
    
    # Samples per class (by dominant pattern)
    print("\n" + "-" * 50)
    print("Samples per class (by dominant pattern):")
    sample_classes = get_sample_dominant_class(Y)
    for c in range(NUM_CLASSES):
        count = (sample_classes == c).sum()
        if count > 0:
            print(f"  {c} ({LABEL_NAMES[c]}): {count}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sequence labeling data (X_seq.npy, Y_seq.npy)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory containing raw_data/ and labels file",
    )
    parser.add_argument(
        "--labels-file",
        default="cleaned_labels_merged.json",
        help="Labels JSON file name",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Fixed window size for each segment",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to data-dir)",
    )
    args = parser.parse_args()
    
    result = prepare_sequence_data(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        window_size=args.window_size,
        output_dir=args.output_dir,
    )
    
    print_stats(result['X'], result['Y'], result['metadata'])


if __name__ == "__main__":
    main()
