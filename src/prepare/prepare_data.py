"""Prepare training data from labeled segments.

This script:
1. Loads verified labels from cleaned_labels_merged.json
2. Extracts OHLC segments with fixed window size (256 bars)
3. Saves as X.npy (features) and Y.npy (labels)

Usage:
    python -m src.prepare.prepare_data --data-dir data --window-size 256
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# Label mapping for classification
LABEL_MAP = {
    "Bearish Normal": 0,
    "Bearish Wedge": 1,
    "Bearish Pennant": 2,
    "Bullish Normal": 3,
    "Bullish Wedge": 4,
    "Bullish Pennant": 5,
}

# Reverse mapping for decoding
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


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
        # Try semicolon separator
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


def normalize_ohlc(ohlc: np.ndarray) -> np.ndarray:
    """Normalize OHLC data to [0, 1] range based on min-max of the segment."""
    min_val = ohlc.min()
    max_val = ohlc.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(ohlc)
    return (ohlc - min_val) / (max_val - min_val)


def extract_segment(
    df: pd.DataFrame,
    start_str: str,
    end_str: str,
    window_size: int,
) -> Optional[np.ndarray]:
    """Extract a fixed-size segment centered on the labeled region.
    
    Args:
        df: DataFrame with OHLC data
        start_str: Start timestamp string
        end_str: End timestamp string
        window_size: Fixed output size
        
    Returns:
        Normalized OHLC array of shape (window_size, 4) or None if extraction fails
    """
    try:
        start_dt = pd.Timestamp(parse_ls_time(start_str), tz="UTC")
        end_dt = pd.Timestamp(parse_ls_time(end_str), tz="UTC")
    except ValueError:
        return None
    
    # Get nearest indices
    start_idx = df.index.get_indexer([start_dt], method="nearest")[0]
    end_idx = df.index.get_indexer([end_dt], method="nearest")[0]
    
    if start_idx < 0 or end_idx < 0:
        return None
    
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
    
    # Normalize
    ohlc = normalize_ohlc(ohlc)
    
    return ohlc


def prepare_data(
    data_dir: str,
    labels_file: str = "cleaned_labels_merged.json",
    window_size: int = 256,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Prepare training data from labeled segments.
    
    Args:
        data_dir: Directory containing raw_data/ and labels file
        labels_file: Name of the labels JSON file
        window_size: Fixed window size for each segment
        output_dir: Output directory (defaults to data_dir)
        
    Returns:
        Tuple of (X, Y, metadata)
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
            
            # Extract segment
            ohlc = extract_segment(df, start_str, end_str, window_size)
            if ohlc is None:
                continue
            
            X_list.append(ohlc)
            Y_list.append(LABEL_MAP[label])
            metadata["samples"].append({
                "csv": csv_name,
                "label": label,
                "start": start_str,
                "end": end_str,
            })
    
    if not X_list:
        raise ValueError("No valid segments extracted")
    
    X = np.stack(X_list, axis=0)
    Y = np.array(Y_list, dtype=np.int64)
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "X.npy", X)
    np.save(output_path / "Y.npy", Y)
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return X, Y, metadata


def print_stats(X: np.ndarray, Y: np.ndarray, metadata: Dict) -> None:
    """Print dataset statistics."""
    print("=" * 60)
    print("TRAINING DATA PREPARED")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Window size: {metadata['window_size']}")
    print()
    print("Label distribution:")
    print("-" * 40)
    for label_name, label_idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        count = (Y == label_idx).sum()
        pct = 100.0 * count / len(Y)
        print(f"  {label_idx}: {label_name:<20} {count:>5} ({pct:>5.1f}%)")
    print("-" * 40)
    print(f"  Total: {len(Y):>26}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from labeled segments"
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
    
    X, Y, metadata = prepare_data(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        window_size=args.window_size,
        output_dir=args.output_dir,
    )
    
    print_stats(X, Y, metadata)
    print(f"\nSaved to: {args.output_dir or args.data_dir}")
    print("  - X.npy")
    print("  - Y.npy")
    print("  - metadata.json")


if __name__ == "__main__":
    main()
