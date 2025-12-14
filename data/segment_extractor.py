"""
Segment Extractor Module

Extracts time series segments from CSV files based on labeled annotations.
Creates numpy arrays suitable for machine learning training.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass

from label_parser import LabeledSegment, parse_label_studio_export


@dataclass
class ExtractedSegment:
    """A segment extracted from the time series data."""
    data: np.ndarray  # OHLC data as numpy array
    label: str        # Label string
    source_file: str  # Source CSV file
    start: str        # Start timestamp
    end: str          # End timestamp


def load_timeseries_csv(csv_path: str, sep: str = ',') -> pd.DataFrame:
    """
    Load a timeseries CSV file and parse timestamps.
    
    Args:
        csv_path: Path to the CSV file.
        sep: CSV separator (default: ',', use ';' for some raw data files).
        
    Returns:
        DataFrame with parsed timestamp index and OHLC columns.
    """
    df = pd.read_csv(csv_path, sep=sep)
    
    # Handle different column naming conventions
    column_mapping = {
        'Date': 'timestamp',
        'date': 'timestamp',
        'Timestamp': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Parse timestamp column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Ensure we have the required OHLC columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def extract_segment_from_df(
    df: pd.DataFrame,
    start_time: str,
    end_time: str
) -> Optional[np.ndarray]:
    """
    Extract a segment from a DataFrame based on start and end times.
    
    Args:
        df: DataFrame with datetime index and OHLC columns.
        start_time: Start timestamp string.
        end_time: End timestamp string.
        
    Returns:
        Numpy array with shape (n_timesteps, 4) containing OHLC data,
        or None if no data found in the range.
    """
    try:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        return None
    
    # Select data within the time range
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    segment_df = df.loc[mask, ['open', 'high', 'low', 'close']]
    
    if segment_df.empty:
        return None
    
    return segment_df.values


def extract_all_segments(
    labeled_segments: List[LabeledSegment],
    data_dir: str,
    file_mapping: Optional[dict] = None
) -> List[ExtractedSegment]:
    """
    Extract all segments from labeled annotations.
    
    Args:
        labeled_segments: List of LabeledSegment objects from label parser.
        data_dir: Directory containing the CSV data files.
        file_mapping: Optional dict mapping file_upload names to actual filenames.
        
    Returns:
        List of ExtractedSegment objects with numpy data.
    """
    extracted = []
    
    # Cache loaded DataFrames to avoid reloading
    df_cache = {}
    
    for seg in labeled_segments:
        # Determine the CSV file path
        csv_filename = seg.csv_path
        
        # Apply file mapping if provided
        if file_mapping and csv_filename in file_mapping:
            csv_filename = file_mapping[csv_filename]
        
        csv_path = os.path.join(data_dir, csv_filename)
        
        # Load DataFrame (with caching)
        if csv_path not in df_cache:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
                continue
            try:
                df_cache[csv_path] = load_timeseries_csv(csv_path)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue
        
        df = df_cache[csv_path]
        
        # Extract the segment
        data = extract_segment_from_df(df, seg.start, seg.end)
        
        if data is None or len(data) == 0:
            print(f"Warning: No data found for segment {seg.label} "
                  f"({seg.start} to {seg.end}) in {csv_filename}")
            continue
        
        extracted.append(ExtractedSegment(
            data=data,
            label=seg.label,
            source_file=csv_filename,
            start=seg.start,
            end=seg.end
        ))
    
    return extracted


def remove_handle_heuristic(
    segment_data: np.ndarray,
    label: str,
    search_fraction: float = 0.5
) -> np.ndarray:
    """
    Remove the "handle" (pole) part of a flag pattern using a heuristic.
    
    For bullish patterns: Find the highest point in the first half and cut from there.
    For bearish patterns: Find the lowest point in the first half and cut from there.
    
    Args:
        segment_data: OHLC numpy array with shape (n_timesteps, 4).
        label: Label string (should contain "Bullish" or "Bearish").
        search_fraction: Fraction of segment to search for handle end (default: 0.5).
        
    Returns:
        Numpy array with handle removed (the "flag" portion only).
    """
    if segment_data.size == 0:
        return segment_data
    
    search_len = int(len(segment_data) * search_fraction)
    if search_len == 0 and len(segment_data) > 0:
        search_len = 1
    
    search_area = segment_data[:search_len]
    
    if search_area.size == 0:
        return segment_data
    
    # Determine cut-off index based on pattern type
    cut_off_index = 0
    if 'Bullish' in label:
        # For bullish: handle ends at highest high
        cut_off_index = np.argmax(search_area[:, 1])  # column 1 = high
    elif 'Bearish' in label:
        # For bearish: handle ends at lowest low
        cut_off_index = np.argmin(search_area[:, 2])  # column 2 = low
    
    return segment_data[cut_off_index:]


def create_segment_arrays(
    extracted_segments: List[ExtractedSegment],
    remove_handle: bool = True,
    min_length: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create numpy arrays from extracted segments.
    
    Args:
        extracted_segments: List of ExtractedSegment objects.
        remove_handle: Whether to apply the handle removal heuristic.
        min_length: Minimum segment length to include.
        
    Returns:
        Tuple of (data_array, labels_array) where:
          - data_array is an object array of variable-length OHLC arrays
          - labels_array is a string array of labels
    """
    data_list = []
    labels_list = []
    
    for seg in extracted_segments:
        data = seg.data
        
        if remove_handle:
            data = remove_handle_heuristic(data, seg.label)
        
        if len(data) < min_length:
            print(f"Warning: Segment too short after processing ({len(data)} < {min_length}), skipping")
            continue
        
        data_list.append(data)
        labels_list.append(seg.label)
    
    # Use object array to handle variable-length segments
    data_array = np.array(data_list, dtype=object)
    labels_array = np.array(labels_list)
    
    return data_array, labels_array


def save_segments(
    data_array: np.ndarray,
    labels_array: np.ndarray,
    output_dir: str,
    prefix: str = 'segments'
) -> Tuple[str, str]:
    """
    Save segment arrays to numpy files.
    
    Args:
        data_array: Array of OHLC segment data.
        labels_array: Array of label strings.
        output_dir: Output directory.
        prefix: Filename prefix.
        
    Returns:
        Tuple of (data_path, labels_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(output_dir, f'{prefix}_data.npy')
    labels_path = os.path.join(output_dir, f'{prefix}_labels.npy')
    
    np.save(data_path, data_array)
    np.save(labels_path, labels_array)
    
    print(f"Saved {len(data_array)} segments:")
    print(f"  Data: {data_path}")
    print(f"  Labels: {labels_path}")
    
    return data_path, labels_path


def process_labels_to_segments(
    labels_json_path: str,
    data_dir: str,
    output_dir: str,
    remove_handle: bool = True,
    min_length: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: parse labels, extract segments, and save.
    
    Args:
        labels_json_path: Path to Label Studio JSON export.
        data_dir: Directory containing CSV data files.
        output_dir: Output directory for numpy files.
        remove_handle: Whether to remove the handle/pole from patterns.
        min_length: Minimum segment length.
        
    Returns:
        Tuple of (data_array, labels_array).
    """
    # Parse labels
    print(f"Parsing labels from {labels_json_path}...")
    labeled_segments = parse_label_studio_export(labels_json_path)
    print(f"Found {len(labeled_segments)} labeled segments")
    
    # Extract segments
    print(f"\nExtracting segments from {data_dir}...")
    extracted = extract_all_segments(labeled_segments, data_dir)
    print(f"Successfully extracted {len(extracted)} segments")
    
    # Create arrays
    print(f"\nCreating numpy arrays (remove_handle={remove_handle})...")
    data_array, labels_array = create_segment_arrays(
        extracted, 
        remove_handle=remove_handle,
        min_length=min_length
    )
    
    # Save
    suffix = 'no_handle' if remove_handle else 'with_handle'
    save_segments(data_array, labels_array, output_dir, f'segments_{suffix}')
    
    return data_array, labels_array


if __name__ == '__main__':
    import sys
    
    # Default paths
    labels_json = 'labels.json'
    data_dir = 'raw_data'
    output_dir = 'processed'
    
    if len(sys.argv) > 1:
        labels_json = sys.argv[1]
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    if not os.path.exists(labels_json):
        print(f"Error: Labels file not found: {labels_json}")
        print("Usage: python segment_extractor.py <labels.json> [data_dir] [output_dir]")
        sys.exit(1)
    
    # Process with handle removed
    process_labels_to_segments(
        labels_json, data_dir, output_dir,
        remove_handle=True, min_length=5
    )
    
    # Also process with handle included
    print("\n" + "="*60)
    process_labels_to_segments(
        labels_json, data_dir, output_dir,
        remove_handle=False, min_length=5
    )
