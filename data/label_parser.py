"""
Label Parser Module

Parses Label Studio JSON exports for bull/bear flag pattern annotations.
Extracts labeled time series segments with their corresponding labels.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LabeledSegment:
    """Represents a single labeled segment from the annotation."""
    file_upload: str  # Original filename from Label Studio
    csv_path: str     # Path to the actual CSV file
    start: str        # Start timestamp
    end: str          # End timestamp
    label: str        # Label type (e.g., "Bullish Normal", "Bearish Wedge")
    annotation_id: str # Unique annotation ID
    
    def __repr__(self):
        return f"LabeledSegment(label={self.label}, start={self.start}, end={self.end})"


def parse_label_studio_export(json_path: str) -> List[LabeledSegment]:
    """
    Parse a Label Studio JSON export file and extract all labeled segments.
    
    Args:
        json_path: Path to the Label Studio JSON export file.
        
    Returns:
        List of LabeledSegment objects containing annotation data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments: List[LabeledSegment] = []
    
    for task in data:
        file_upload = task.get('file_upload', '')
        csv_path = task.get('data', {}).get('csv', '')
        
        # Extract filename from csv_path if available
        if csv_path:
            # Remove the /data/upload/1/ prefix and UUID prefix
            csv_filename = os.path.basename(csv_path)
            # Remove UUID prefix (e.g., "8e2ef3a4-" from "8e2ef3a4-XAU_1h_data_limited.csv")
            if '-' in csv_filename and len(csv_filename.split('-')[0]) == 8:
                csv_filename = '-'.join(csv_filename.split('-')[1:])
        else:
            csv_filename = file_upload
            
        annotations = task.get('annotations', [])
        
        for annotation in annotations:
            results = annotation.get('result', [])
            
            for result in results:
                if result.get('type') == 'timeserieslabels':
                    value = result.get('value', {})
                    start = value.get('start', '')
                    end = value.get('end', '')
                    labels = value.get('timeserieslabels', [])
                    annotation_id = result.get('id', '')
                    
                    # Combine all labels into a single label string
                    if labels:
                        label = labels[0]  # Use the first label
                        
                        segment = LabeledSegment(
                            file_upload=file_upload,
                            csv_path=csv_filename,
                            start=start,
                            end=end,
                            label=label,
                            annotation_id=annotation_id
                        )
                        segments.append(segment)
    
    return segments


def get_unique_labels(segments: List[LabeledSegment]) -> List[str]:
    """Get unique labels from a list of labeled segments."""
    return sorted(list(set(seg.label for seg in segments)))


def get_label_counts(segments: List[LabeledSegment]) -> dict:
    """Get count of each label type."""
    counts = {}
    for seg in segments:
        counts[seg.label] = counts.get(seg.label, 0) + 1
    return counts


def filter_segments_by_label(
    segments: List[LabeledSegment], 
    labels: List[str]
) -> List[LabeledSegment]:
    """Filter segments to only include specified labels."""
    return [seg for seg in segments if seg.label in labels]


def filter_segments_by_file(
    segments: List[LabeledSegment], 
    filename: str
) -> List[LabeledSegment]:
    """Filter segments from a specific file."""
    return [seg for seg in segments if filename in seg.csv_path or filename in seg.file_upload]


def summarize_labels(json_path: str) -> None:
    """Print a summary of the labels in the JSON file."""
    segments = parse_label_studio_export(json_path)
    
    print(f"Total segments: {len(segments)}")
    print(f"\nUnique labels: {get_unique_labels(segments)}")
    print(f"\nLabel counts:")
    
    for label, count in sorted(get_label_counts(segments).items()):
        print(f"  {label}: {count}")
    
    print(f"\nFiles referenced:")
    files = sorted(list(set(seg.csv_path for seg in segments)))
    for f in files:
        count = len([s for s in segments if s.csv_path == f])
        print(f"  {f}: {count} segments")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = 'labels.json'
    
    if os.path.exists(json_path):
        summarize_labels(json_path)
    else:
        print(f"Error: File not found: {json_path}")
        print("Usage: python label_parser.py <path_to_labels.json>")
