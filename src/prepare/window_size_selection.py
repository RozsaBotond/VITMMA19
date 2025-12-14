"""Window size selection for LSTM using claspy algorithms.

This module runs SuSS and FFT window size selection algorithms on OHLC data
to determine optimal sequence length for LSTM training.

Usage:
    python -m src.prepare.window_size_selection --data-dir data/raw_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from claspy.window_size import suss, dominant_fourier_frequency


def load_ohlc_csv(file_path: str, sep: str = ",") -> pd.DataFrame:
    """Load OHLC CSV with robust column detection."""
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
    
    return df


def compute_window_sizes(close_series: np.ndarray) -> Dict[str, Optional[int]]:
    """Compute window sizes using SuSS and FFT algorithms.
    
    Args:
        close_series: Array of close prices
        
    Returns:
        Dictionary with algorithm names as keys and window sizes as values
    """
    results = {}
    
    # SuSS (Summary Statistics-based Window Size Selection)
    try:
        results["SuSS"] = int(suss(close_series))
    except Exception:
        results["SuSS"] = None
    
    # FFT (Dominant Fourier Frequency)
    try:
        results["FFT"] = int(dominant_fourier_frequency(close_series))
    except Exception:
        results["FFT"] = None
    
    return results


def run_window_size_selection(data_dir: str) -> Dict[str, any]:
    """Run window size selection on all CSV files in directory.
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        Dictionary with results and recommendations
    """
    raw_data_path = Path(data_dir)
    csv_files = sorted(raw_data_path.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    results: List[Dict] = []
    
    for csv_file in csv_files:
        df = load_ohlc_csv(str(csv_file))
        close = df["close"].values.astype(float)
        
        window_sizes = compute_window_sizes(close)
        
        results.append({
            "dataset": csv_file.stem,
            **window_sizes,
        })
    
    # Compute averages
    suss_values = [r["SuSS"] for r in results if r["SuSS"] is not None]
    fft_values = [r["FFT"] for r in results if r["FFT"] is not None]
    
    suss_avg = np.mean(suss_values) if suss_values else None
    fft_avg = np.mean(fft_values) if fft_values else None
    
    # Compute per-dataset averages
    per_dataset_avg = []
    for r in results:
        valid_values = [v for v in [r["SuSS"], r["FFT"]] if v is not None]
        if valid_values:
            per_dataset_avg.append(np.mean(valid_values))
    
    grand_avg = np.mean(per_dataset_avg) if per_dataset_avg else None
    
    # SuSS-only average (more conservative)
    suss_only_avg = suss_avg
    
    return {
        "per_dataset": results,
        "averages": {
            "SuSS": suss_avg,
            "FFT": fft_avg,
        },
        "grand_average": grand_avg,
        "recommended_window_size": int(round(grand_avg)) if grand_avg else None,
        "conservative_window_size": int(round(suss_only_avg)) if suss_only_avg else None,
    }


def print_results(results: Dict) -> None:
    """Print formatted results."""
    print("=" * 60)
    print("WINDOW SIZE SELECTION RESULTS")
    print("=" * 60)
    print(f"{'Dataset':<35} {'SuSS':>10} {'FFT':>10}")
    print("-" * 60)
    
    for r in results["per_dataset"]:
        suss_str = str(r["SuSS"]) if r["SuSS"] is not None else "N/A"
        fft_str = str(r["FFT"]) if r["FFT"] is not None else "N/A"
        print(f"{r['dataset']:<35} {suss_str:>10} {fft_str:>10}")
    
    print("-" * 60)
    
    avgs = results["averages"]
    suss_avg_str = f"{avgs['SuSS']:.1f}" if avgs["SuSS"] else "N/A"
    fft_avg_str = f"{avgs['FFT']:.1f}" if avgs["FFT"] else "N/A"
    print(f"{'AVERAGE ACROSS DATASETS':<35} {suss_avg_str:>10} {fft_avg_str:>10}")
    
    print("=" * 60)
    print("\nPER-DATASET METHOD AVERAGE:")
    print("-" * 50)
    
    for r in results["per_dataset"]:
        valid_values = [v for v in [r["SuSS"], r["FFT"]] if v is not None]
        if valid_values:
            avg = np.mean(valid_values)
            print(f"  {r['dataset']:<35} {avg:>10.1f}")
    
    print("=" * 60)
    print(f"\n{'GRAND AVERAGE (SuSS + FFT):':<45} {results['grand_average']:>10.1f}")
    print(f"{'Suggested LSTM window size:':<45} {results['recommended_window_size']:>10}")
    print("=" * 60)
    print(f"\n{'[Conservative: SuSS only]':<45} {results['conservative_window_size']:>10}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run window size selection algorithms for LSTM sequence length"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw_data",
        help="Directory containing OHLC CSV files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the recommended window size",
    )
    args = parser.parse_args()
    
    results = run_window_size_selection(args.data_dir)
    
    if args.quiet:
        print(results["recommended_window_size"])
    else:
        print_results(results)


if __name__ == "__main__":
    main()
