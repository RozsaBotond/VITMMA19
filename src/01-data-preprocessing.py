"""Data preprocessing script for Bull/Bear Flag Detector.

This script handles data loading, cleaning, and transformation:
1. Loads raw OHLC data from CSV files
2. Loads labeled segments from cleaned_labels_merged.json
3. Extracts fixed-size windows centered on labeled regions
4. Normalizes OHLC data to [0, 1] range
5. Saves processed data as X.npy and Y.npy

Usage:
    python src/01-data-preprocessing.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import config
from utils import setup_logger, log_header, log_separator, log_config

logger = setup_logger("preprocessing")


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
    """Extract a fixed-size segment centered on the labeled region."""
    try:
        start_dt = pd.Timestamp(parse_ls_time(start_str), tz="UTC")
        end_dt = pd.Timestamp(parse_ls_time(end_str), tz="UTC")
    except ValueError:
        return None
    
    start_idx = df.index.get_indexer([start_dt], method="nearest")[0]
    end_idx = df.index.get_indexer([end_dt], method="nearest")[0]
    
    if start_idx < 0 or end_idx < 0:
        return None
    
    center_idx = (start_idx + end_idx) // 2
    half_window = window_size // 2
    extract_start = center_idx - half_window
    extract_end = extract_start + window_size
    
    if extract_start < 0:
        extract_start = 0
        extract_end = window_size
    if extract_end > len(df):
        extract_end = len(df)
        extract_start = max(0, extract_end - window_size)
    
    if extract_end - extract_start < window_size:
        return None
    
    ohlc = df.iloc[extract_start:extract_end][["open", "high", "low", "close"]].values.astype(np.float32)
    ohlc = normalize_ohlc(ohlc)
    
    return ohlc


def preprocess() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Main preprocessing function.
    
    Returns:
        Tuple of (X, Y, metadata) arrays
    """
    log_header(logger, "DATA PREPROCESSING")
    
    # Log configuration
    log_config(logger, {
        "data_dir": str(config.DATA_DIR),
        "labels_file": str(config.LABELS_FILE),
        "window_size": config.WINDOW_SIZE,
        "features": config.FEATURES,
        "num_classes": config.NUM_CLASSES,
    }, "Preprocessing Configuration")
    
    # Check if preprocessed data already exists
    if config.X_FILE.exists() and config.Y_FILE.exists():
        logger.info("Preprocessed data already exists. Loading from cache...")
        X = np.load(config.X_FILE)
        Y = np.load(config.Y_FILE)
        
        with open(config.METADATA_FILE) as f:
            metadata = json.load(f)
        
        logger.info(f"  Loaded X: {X.shape}")
        logger.info(f"  Loaded Y: {Y.shape}")
        log_separator(logger, "-")
        return X, Y, metadata
    
    # Load labels
    logger.info(f"Loading labels from: {config.LABELS_FILE}")
    with open(config.LABELS_FILE) as f:
        tasks = json.load(f)
    logger.info(f"  Found {len(tasks)} tasks")
    
    # Build CSV path map
    csv_map = {}
    for csv_file in config.RAW_DATA_DIR.glob("*.csv"):
        csv_map[csv_file.name] = csv_file
    logger.info(f"  Found {len(csv_map)} CSV files in raw_data/")
    
    # Extract segments
    X_list = []
    Y_list = []
    metadata = {
        "window_size": config.WINDOW_SIZE,
        "label_map": config.LABEL_MAP,
        "samples": [],
    }
    
    logger.info("Extracting labeled segments...")
    
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
            logger.warning(f"Could not find CSV: {csv_filename}")
            continue
        
        # Load CSV
        try:
            df = load_ohlc_csv(str(matched_path))
        except Exception as e:
            logger.warning(f"Failed to load {matched_path}: {e}")
            continue
        
        csv_name = matched_path.stem
        
        # Process each labeled result
        results = task.get("annotations", [{}])[0].get("result", [])
        for r in results:
            labels = r.get("value", {}).get("timeserieslabels", [])
            if not labels:
                continue
            
            label = labels[0]
            if label not in config.LABEL_MAP:
                logger.warning(f"Unknown label: {label}")
                continue
            
            start_str = r.get("value", {}).get("start", "")
            end_str = r.get("value", {}).get("end", "")
            
            if not start_str or not end_str:
                continue
            
            ohlc = extract_segment(df, start_str, end_str, config.WINDOW_SIZE)
            if ohlc is None:
                continue
            
            X_list.append(ohlc)
            Y_list.append(config.LABEL_MAP[label])
            metadata["samples"].append({
                "csv": csv_name,
                "label": label,
                "start": start_str,
                "end": end_str,
            })
    
    if not X_list:
        raise ValueError("No valid segments extracted!")
    
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.array(Y_list, dtype=np.int64)
    
    # Save preprocessed data
    logger.info("Saving preprocessed data...")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.X_FILE, X)
    np.save(config.Y_FILE, Y)
    
    with open(config.METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Log statistics
    log_header(logger, "DATASET STATISTICS")
    logger.info(f"X shape: {X.shape} (samples, timesteps, features)")
    logger.info(f"Y shape: {Y.shape}")
    logger.info("")
    logger.info("Label distribution:")
    
    for label_name, label_idx in sorted(config.LABEL_MAP.items(), key=lambda x: x[1]):
        count = (Y == label_idx).sum()
        pct = 100.0 * count / len(Y)
        logger.info(f"  {label_idx}: {label_name:<20} {count:>5} ({pct:>5.1f}%)")
    
    logger.info(f"\nTotal samples: {len(Y)}")
    log_separator(logger, "=")
    
    return X, Y, metadata


if __name__ == "__main__":
    preprocess()
