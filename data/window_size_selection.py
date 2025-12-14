"""Window size selection for flag pattern detection using claspy.

This script:
1. Loads labeled flag segments from cleaned_labels_merged.json
2. Extracts the flag durations (in bars) from each labeled region
3. Runs claspy window size selection algorithms (SuSS, ACF) to determine optimal padding
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from claspy.window_size import suss, highest_autocorrelation, dominant_fourier_frequency


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


def get_flag_durations(labels_path: str, data_dir: str) -> Dict[str, List[int]]:
    """Extract flag durations (in bars) for each timeframe/CSV.
    
    Returns dict mapping CSV basename to list of durations.
    """
    with open(labels_path) as f:
        tasks = json.load(f)
    
    # Build CSV path map
    raw_data_path = Path(data_dir) / "raw_data"
    csv_map = {}
    for csv_file in raw_data_path.glob("*.csv"):
        csv_map[csv_file.name] = csv_file
    
    durations_by_csv: Dict[str, List[int]] = {}
    
    for task in tasks:
        csv_ref = task.get("data", {}).get("csv", "")
        if not csv_ref:
            continue
        
        csv_filename = os.path.basename(csv_ref)
        # Match to actual file (strip UUID prefix if present)
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
        if csv_key not in durations_by_csv:
            durations_by_csv[csv_key] = []
        
        # Extract durations from labeled results
        results = task.get("annotations", [{}])[0].get("result", [])
        for r in results:
            labels = r.get("value", {}).get("timeserieslabels", [])
            if not labels:
                continue  # Skip unlabeled
            
            start_str = r.get("value", {}).get("start", "")
            end_str = r.get("value", {}).get("end", "")
            if not start_str or not end_str:
                continue
            
            try:
                start_dt = parse_ls_time(start_str)
                end_dt = parse_ls_time(end_str)
            except ValueError:
                continue
            
            # Find indices in dataframe
            start_dt = pd.Timestamp(start_dt, tz="UTC")
            end_dt = pd.Timestamp(end_dt, tz="UTC")
            
            # Get nearest indices
            start_idx = df.index.get_indexer([start_dt], method="nearest")[0]
            end_idx = df.index.get_indexer([end_dt], method="nearest")[0]
            
            duration = end_idx - start_idx
            if duration > 0:
                durations_by_csv[csv_key].append(duration)
    
    return durations_by_csv


def run_window_size_selection(close_series: np.ndarray, max_window: int = 500) -> Dict[str, int]:
    """Run multiple window size selection algorithms.
    
    Returns dict with algorithm name -> suggested window size.
    """
    results = {}
    
    # Ensure we have enough data
    n = len(close_series)
    if n < 50:
        print(f"[WARN] Series too short ({n} points) for window size selection")
        return results
    
    # SuSS (Summary Statistics-based Window Size Selection)
    try:
        suss_result = suss(close_series)
        results["SuSS"] = suss_result
    except Exception as e:
        print(f"[WARN] SuSS failed: {e}")
    
    # ACF (Highest Autocorrelation)
    try:
        acf_result = highest_autocorrelation(close_series)
        results["ACF"] = acf_result
    except Exception as e:
        print(f"[WARN] ACF failed: {e}")
    
    # FFT (Dominant Fourier Frequency)
    try:
        fft_result = dominant_fourier_frequency(close_series)
        results["FFT"] = fft_result
    except Exception as e:
        print(f"[WARN] FFT failed: {e}")
    
    return results


def main():
    data_dir = Path(__file__).parent
    labels_path = data_dir / "cleaned_labels_merged.json"
    
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        return
    
    print("=" * 60)
    print("STEP 1: Analyzing labeled flag durations")
    print("=" * 60)
    
    durations = get_flag_durations(str(labels_path), str(data_dir))
    
    all_durations = []
    for csv_key, durs in sorted(durations.items()):
        if durs:
            all_durations.extend(durs)
            print(f"\n{csv_key}:")
            print(f"  Count: {len(durs)}")
            print(f"  Min: {min(durs)} bars")
            print(f"  Max: {max(durs)} bars")
            print(f"  Mean: {np.mean(durs):.1f} bars")
            print(f"  Median: {np.median(durs):.1f} bars")
            print(f"  Std: {np.std(durs):.1f} bars")
    
    if all_durations:
        print(f"\n--- ALL TIMEFRAMES COMBINED ---")
        print(f"  Total labels: {len(all_durations)}")
        print(f"  Min: {min(all_durations)} bars")
        print(f"  Max: {max(all_durations)} bars")
        print(f"  Mean: {np.mean(all_durations):.1f} bars")
        print(f"  Median: {np.median(all_durations):.1f} bars")
        print(f"  Std: {np.std(all_durations):.1f} bars")
        print(f"  25th percentile: {np.percentile(all_durations, 25):.1f} bars")
        print(f"  75th percentile: {np.percentile(all_durations, 75):.1f} bars")
    
    print("\n" + "=" * 60)
    print("STEP 2: Running claspy window size selection algorithms")
    print("=" * 60)
    
    # Run on each CSV's close series
    raw_data_path = data_dir / "raw_data"
    for csv_file in sorted(raw_data_path.glob("*.csv")):
        print(f"\n{csv_file.name}:")
        try:
            df = load_ohlc_csv(str(csv_file))
            close = df["close"].values.astype(float)
            
            results = run_window_size_selection(close)
            for algo, window in results.items():
                print(f"  {algo}: {window} bars")
        except Exception as e:
            print(f"  [ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("STEP 3: Padding recommendations")
    print("=" * 60)
    
    if all_durations:
        median_dur = np.median(all_durations)
        # Common approach: use 20-50% of pattern length as padding
        padding_20 = int(median_dur * 0.2)
        padding_50 = int(median_dur * 0.5)
        padding_100 = int(median_dur * 1.0)
        
        print(f"\nBased on median flag duration of {median_dur:.0f} bars:")
        print(f"  20% padding: {padding_20} bars (conservative)")
        print(f"  50% padding: {padding_50} bars (moderate)")
        print(f"  100% padding: {padding_100} bars (generous)")
        print(f"\nRecommended: Use 50% padding ({padding_50} bars) for balanced context")


if __name__ == "__main__":
    main()
