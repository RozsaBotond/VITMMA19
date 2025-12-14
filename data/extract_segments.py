"""Extract labeled flag segments for training.

This script:
1. Loads merged labels from cleaned_labels_merged.json
2. Extracts each labeled segment with configurable padding
3. Saves segments as numpy arrays for training
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabeledSegment:
    """A labeled flag segment with metadata."""
    csv_name: str
    label: str
    direction: str  # "Bullish" or "Bearish"
    pattern_type: str  # "Normal", "Wedge", or "Pennant"
    start_idx: int
    end_idx: int
    ohlc: np.ndarray  # Shape: (length, 4) for OHLC
    

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


def parse_label(label_str: str) -> Tuple[str, str]:
    """Parse label like 'Bullish Normal' into (direction, pattern_type)."""
    parts = label_str.split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    return label_str, "Unknown"


def extract_segments(
    labels_path: str,
    data_dir: str,
    padding_bars: int = 15,
    min_length: int = 5,
) -> List[LabeledSegment]:
    """Extract all labeled segments from the merged labels JSON.
    
    Args:
        labels_path: Path to cleaned_labels_merged.json
        data_dir: Directory containing raw_data/*.csv
        padding_bars: Number of bars to add before and after segment
        min_length: Minimum segment length to include
    
    Returns:
        List of LabeledSegment objects
    """
    with open(labels_path) as f:
        tasks = json.load(f)
    
    # Build CSV path map
    raw_data_path = Path(data_dir) / "raw_data"
    csv_map = {}
    for csv_file in raw_data_path.glob("*.csv"):
        csv_map[csv_file.name] = csv_file
    
    segments: List[LabeledSegment] = []
    
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
        
        csv_key = matched_path.stem
        
        # Extract segments from labeled results
        results = task.get("annotations", [{}])[0].get("result", [])
        for r in results:
            labels = r.get("value", {}).get("timeserieslabels", [])
            if not labels:
                continue
            
            label = labels[0]
            start_str = r.get("value", {}).get("start", "")
            end_str = r.get("value", {}).get("end", "")
            
            if not start_str or not end_str:
                continue
            
            try:
                start_dt = parse_ls_time(start_str)
                end_dt = parse_ls_time(end_str)
            except ValueError:
                continue
            
            start_dt = pd.Timestamp(start_dt, tz="UTC")
            end_dt = pd.Timestamp(end_dt, tz="UTC")
            
            # Get nearest indices
            start_idx = df.index.get_indexer([start_dt], method="nearest")[0]
            end_idx = df.index.get_indexer([end_dt], method="nearest")[0]
            
            # Skip too short segments
            if end_idx - start_idx < min_length:
                continue
            
            # Apply padding
            padded_start = max(0, start_idx - padding_bars)
            padded_end = min(len(df), end_idx + padding_bars)
            
            # Extract OHLC
            ohlc = df.iloc[padded_start:padded_end][["open", "high", "low", "close"]].values.astype(np.float32)
            
            direction, pattern_type = parse_label(label)
            
            segments.append(LabeledSegment(
                csv_name=csv_key,
                label=label,
                direction=direction,
                pattern_type=pattern_type,
                start_idx=start_idx,
                end_idx=end_idx,
                ohlc=ohlc,
            ))
    
    return segments


def save_training_data(
    segments: List[LabeledSegment],
    output_dir: str,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Save segments as training data.
    
    If max_length is provided, segments are padded/truncated to that length.
    Otherwise, saves variable-length segments.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build label mapping
    unique_labels = sorted(set(s.label for s in segments))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Prepare arrays
    data_list = []
    labels_list = []
    metadata_list = []
    
    for seg in segments:
        data_list.append(seg.ohlc)
        labels_list.append(label_to_idx[seg.label])
        metadata_list.append({
            "csv": seg.csv_name,
            "label": seg.label,
            "direction": seg.direction,
            "pattern_type": seg.pattern_type,
            "start_idx": int(seg.start_idx),
            "end_idx": int(seg.end_idx),
            "length": len(seg.ohlc),
        })
    
    # If max_length specified, create fixed-size arrays
    if max_length:
        fixed_data = np.zeros((len(segments), max_length, 4), dtype=np.float32)
        for i, ohlc in enumerate(data_list):
            length = min(len(ohlc), max_length)
            fixed_data[i, :length] = ohlc[:length]
        np.save(output_path / "segments_data.npy", fixed_data)
    else:
        # Save as object array for variable lengths
        np.save(output_path / "segments_data.npy", np.array(data_list, dtype=object), allow_pickle=True)
    
    # Save labels
    np.save(output_path / "segments_labels.npy", np.array(labels_list, dtype=np.int32))
    
    # Save metadata and label mapping
    with open(output_path / "segments_metadata.json", "w") as f:
        json.dump({
            "label_mapping": label_to_idx,
            "segments": metadata_list,
        }, f, indent=2)
    
    stats = {
        "total_segments": len(segments),
        "label_distribution": {},
        "length_stats": {
            "min": min(len(s.ohlc) for s in segments),
            "max": max(len(s.ohlc) for s in segments),
            "mean": np.mean([len(s.ohlc) for s in segments]),
            "median": np.median([len(s.ohlc) for s in segments]),
        }
    }
    for label in unique_labels:
        stats["label_distribution"][label] = sum(1 for s in segments if s.label == label)
    
    return stats


def main():
    data_dir = Path(__file__).parent
    labels_path = data_dir / "cleaned_labels_merged.json"
    output_dir = data_dir / "processed_segments"
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return
    
    # Based on analysis:
    # - Median flag duration: 29 bars
    # - 50% padding recommended: ~15 bars
    padding_bars = 15
    
    print(f"Extracting segments with {padding_bars} bars padding...")
    segments = extract_segments(
        str(labels_path),
        str(data_dir),
        padding_bars=padding_bars,
        min_length=5,
    )
    
    print(f"\nExtracted {len(segments)} segments")
    
    # Stats by timeframe
    by_csv = {}
    for seg in segments:
        if seg.csv_name not in by_csv:
            by_csv[seg.csv_name] = []
        by_csv[seg.csv_name].append(seg)
    
    print("\nBy timeframe:")
    for csv_name, segs in sorted(by_csv.items()):
        print(f"  {csv_name}: {len(segs)} segments")
    
    # Stats by label
    by_label = {}
    for seg in segments:
        if seg.label not in by_label:
            by_label[seg.label] = 0
        by_label[seg.label] += 1
    
    print("\nBy label:")
    for label, count in sorted(by_label.items()):
        print(f"  {label}: {count}")
    
    # Save
    print(f"\nSaving to {output_dir}...")
    stats = save_training_data(segments, str(output_dir))
    
    print(f"\nDone!")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Length range: {stats['length_stats']['min']:.0f} - {stats['length_stats']['max']:.0f} bars")
    print(f"  Mean length: {stats['length_stats']['mean']:.1f} bars")


if __name__ == "__main__":
    main()
