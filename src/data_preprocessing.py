"""Data preprocessing script for Bull/Bear Flag Detector.

This script handles data loading, cleaning, and transformation:
1. Loads raw OHLC data from CSV files (if not already processed)
2. Loads labeled segments from cleaned_labels_merged.json
3. Extracts fixed-size windows centered on labeled regions
4. Saves processed data as X_seq.npy and Y_seq.npy
5. Performs train/validation/test split and saves X/Y splits to disk.

Usage:
    python src/data_preprocessing.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path for imports. This assumes src is added to PYTHONPATH
# or that main.py (which handles path fixing) is the entry point.
# _fix_sys_path removed to satisfy ruff E402.

from utils.config import AppConfig
from utils.utils import setup_logger, log_header, log_separator, log_config
from utils.normalization import normalize_window as normalize_ohlc_segment

logger = setup_logger("preprocessing")
config = AppConfig()

# Label mapping from config
LABEL_MAP = config.label_map
LABEL_NAMES = config.label_names


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
    ohlc = normalize_ohlc_segment(ohlc) # Use the normalize_ohlc_segment from normalization.py
    
    return ohlc


def preprocess_data_main():
    """Main preprocessing function.
    Loads raw data, extracts segments, and performs train/val/test split.
    """
    log_header(logger, "DATA PREPROCESSING AND SPLITTING")
    
    # Log configuration
    log_config(logger, {
        "data_dir": str(config.data_dir),
        "labels_file": str(config.labels_file),
        "window_size": config.window_size,
        "features": config.features,
        "num_classes": config.num_classes,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "random_seed": config.random_seed,
    }, "Preprocessing Configuration")
    
    X_seq_path = config.data_dir / "X_seq.npy"
    Y_seq_path = config.data_dir / "Y_seq.npy"
    metadata_seq_path = config.data_dir / "metadata_seq.json"

    # --- Step 1: Extract segments from raw data ---
    if X_seq_path.exists() and Y_seq_path.exists():
        logger.info("Preprocessed sequence data already exists. Loading from cache...")
        X_seq = np.load(X_seq_path)
        Y_seq = np.load(Y_seq_path)
        with open(metadata_seq_path) as f:
            metadata_seq = json.load(f)
        logger.info(f"  Loaded X_seq: {X_seq.shape}")
        logger.info(f"  Loaded Y_seq: {Y_seq.shape}")
    else:
        logger.info(f"Loading labels from: {config.labels_file}")
        with open(config.labels_file) as f:
            tasks = json.load(f)
        logger.info(f"  Found {len(tasks)} tasks")
        
        csv_map = {csv_file.name: csv_file for csv_file in config.raw_data_dir.glob("*.csv")}
        logger.info(f"  Found {len(csv_map)} CSV files in raw_data/")
        
        X_list = []
        Y_list = []
        metadata_seq = {
            "window_size": config.window_size,
            "label_map": config.label_map,
            "samples": [],
        }
        
        logger.info("Extracting labeled segments...")
        for task in tasks:
            csv_ref = task.get("data", {}).get("csv", "")
            if not csv_ref:
                continue
            csv_filename = os.path.basename(csv_ref)
            
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
            
            try:
                df = load_ohlc_csv(str(matched_path))
            except Exception as e:
                logger.warning(f"Failed to load {matched_path}: {e}")
                continue
            
            csv_name = matched_path.stem
            results = task.get("annotations", [{}])[0].get("result", [])
            for r in results:
                labels = r.get("value", {}).get("timeserieslabels", [])
                if not labels:
                    continue
                label = labels[0]
                if label not in config.label_map:
                    logger.warning(f"Unknown label: {label}")
                    continue
                start_str = r.get("value", {}).get("start", "")
                end_str = r.get("value", {}).get("end", "")
                if not start_str or not end_str:
                    continue
                
                ohlc = extract_segment(df, start_str, end_str, config.window_size)
                if ohlc is None:
                    continue
                
                X_list.append(ohlc)
                Y_list.append(config.label_map[label])
                metadata_seq["samples"].append({
                    "csv": csv_name, "label": label, "start": start_str, "end": end_str,
                })
        
        if not X_list:
            raise ValueError("No valid segments extracted!")
        
        X_seq = np.stack(X_list, axis=0).astype(np.float32)
        Y_seq = np.array(Y_list, dtype=np.int64)
        
        config.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(X_seq_path, X_seq)
        np.save(Y_seq_path, Y_seq)
        with open(metadata_seq_path, "w") as f:
            json.dump(metadata_seq, f, indent=2)
        logger.info(f"Saved preprocessed sequence data: X_seq={X_seq.shape}, Y_seq={Y_seq.shape}")

    # --- Step 2: Perform train/val/test split ---
    log_header(logger, "DATA SPLITTING")
    n_samples = len(X_seq)
    
    # Get dominant label for each sample (for stratification)
    dominant_labels = [np.argmax(np.bincount(y[y>0])) if (y>0).any() else 0 for y in Y_seq]
    
    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(indices, test_size=(config.val_ratio + config.test_ratio), random_state=config.random_seed, stratify=dominant_labels)
    
    # Adjust test_size for the second split to maintain desired ratios
    test_size_for_second_split = config.test_ratio / (config.val_ratio + config.test_ratio)
    temp_labels = [dominant_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_size_for_second_split, random_state=config.random_seed, stratify=temp_labels)
    
    logger.info(f"Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Save split indices and data
    metadata_seq["split_indices"] = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    with open(metadata_seq_path, "w") as f:
        json.dump(metadata_seq, f, indent=2)
    logger.info(f"Split indices saved to {metadata_seq_path}")

    # Save X and Y splits
    np.save(config.data_dir / "X_train.npy", X_seq[train_idx])
    np.save(config.data_dir / "Y_train.npy", Y_seq[train_idx])
    np.save(config.data_dir / "X_val.npy", X_seq[val_idx])
    np.save(config.data_dir / "Y_val.npy", Y_seq[val_idx])
    np.save(config.data_dir / "X_test.npy", X_seq[test_idx])
    np.save(config.data_dir / "Y_test.npy", Y_seq[test_idx])
    logger.info("Saved X/Y train/val/test split files.")

    log_separator(logger, "=")

if __name__ == "__main__":
    preprocess_data_main()