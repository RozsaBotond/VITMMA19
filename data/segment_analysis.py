"""
Time Series Segmentation Module using ClaSPy

Uses ClaSP (Classification Score Profile) for automatic time series segmentation.
Also provides clustering and visualization tools for verifying segments.
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# ClaSPy imports
from claspy.segmentation import BinaryClaSPSegmentation


@dataclass
class SegmentInfo:
    """Information about a detected segment."""
    start_idx: int
    end_idx: int
    start_time: Optional[pd.Timestamp] = None
    end_time: Optional[pd.Timestamp] = None
    label: Optional[str] = None
    cluster_id: Optional[int] = None


def segment_timeseries_clasp(
    timeseries: np.ndarray,
    n_segments: Optional[int] = None,
    window_size: Optional[int] = None
) -> Tuple[List[int], Any]:
    """
    Segment a time series using ClaSP algorithm.
    
    Args:
        timeseries: 1D or 2D numpy array of time series data.
                   For OHLC, use close prices or a combination.
        n_segments: Expected number of segments (optional).
        window_size: Window size for ClaSP (optional, auto-detected if None).
        
    Returns:
        Tuple of (change_points, clasp_model).
    """
    # Initialize ClaSP
    clasp = BinaryClaSPSegmentation(
        n_segments=n_segments,
        window_size=window_size
    )
    
    # Fit and predict change points
    change_points = clasp.fit_predict(timeseries)
    
    return list(change_points), clasp


def segment_ohlc_data(
    df: pd.DataFrame,
    column: str = 'close',
    n_segments: Optional[int] = None,
    window_size: Optional[int] = None
) -> Tuple[List[SegmentInfo], Any]:
    """
    Segment OHLC data using ClaSP on a specified column.
    
    Args:
        df: DataFrame with OHLC data and datetime index.
        column: Column to use for segmentation (default: 'close').
        n_segments: Expected number of segments.
        window_size: Window size for ClaSP.
        
    Returns:
        Tuple of (list of SegmentInfo, clasp_model).
    """
    timeseries = df[column].values
    change_points, clasp = segment_timeseries_clasp(
        timeseries, n_segments, window_size
    )
    
    # Create segment info objects
    segments = []
    
    # Add start and end to change points for full segmentation
    all_points = [0] + list(change_points) + [len(df)]
    
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i + 1]
        
        segment = SegmentInfo(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=df.index[start_idx] if hasattr(df.index, '__getitem__') else None,
            end_time=df.index[end_idx - 1] if hasattr(df.index, '__getitem__') and end_idx <= len(df) else None
        )
        segments.append(segment)
    
    return segments, clasp


def extract_segment_features(segment_data: np.ndarray) -> np.ndarray:
    """
    Extract features from a segment for clustering.
    
    Args:
        segment_data: OHLC numpy array for the segment.
        
    Returns:
        Feature vector.
    """
    if segment_data.size == 0 or len(segment_data) < 2:
        return np.zeros(20)
    
    features = []
    
    # Basic statistics per column (if OHLC)
    if segment_data.ndim == 2 and segment_data.shape[1] >= 4:
        for col in range(4):
            col_data = segment_data[:, col]
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                col_data[-1] - col_data[0],  # Trend
            ])
    else:
        # 1D time series
        features.extend([
            np.mean(segment_data),
            np.std(segment_data),
            segment_data[-1] - segment_data[0],
        ])
    
    # Segment length
    features.append(len(segment_data))
    
    # Price range
    if segment_data.ndim == 2:
        price_range = np.max(segment_data) - np.min(segment_data)
    else:
        price_range = np.max(segment_data) - np.min(segment_data)
    features.append(price_range)
    
    # Volatility (std of returns)
    if segment_data.ndim == 2 and segment_data.shape[1] >= 4:
        close = segment_data[:, 3]
    else:
        close = segment_data
    
    if len(close) > 1:
        returns = np.diff(close) / (close[:-1] + 1e-10)
        features.append(np.std(returns))
        features.append(np.mean(returns))
    else:
        features.extend([0, 0])
    
    # Pad to fixed size
    while len(features) < 20:
        features.append(0)
    
    return np.array(features[:20])


def cluster_segments(
    segments_data: List[np.ndarray],
    n_clusters: int = 6,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, Any]:
    """
    Cluster segments based on extracted features.
    
    Args:
        segments_data: List of segment OHLC arrays.
        n_clusters: Number of clusters.
        method: Clustering method ('kmeans', 'hierarchical', 'dbscan').
        
    Returns:
        Tuple of (cluster_labels, clustering_model).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    
    # Extract features
    features = np.array([extract_segment_features(seg) for seg in segments_data])
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Cluster
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(features_scaled)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(features_scaled)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
        labels = model.fit_predict(features_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return labels, model


def visualize_segmentation(
    df: pd.DataFrame,
    segments: List[SegmentInfo],
    clasp_model: Optional[Any] = None,
    title: str = "Time Series Segmentation",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize segmented time series with colored regions.
    
    Args:
        df: DataFrame with OHLC data and datetime index.
        segments: List of SegmentInfo objects.
        clasp_model: Optional ClaSP model for score profile.
        title: Plot title.
        save_path: Path to save the figure.
        show: Whether to display the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Color palette for segments
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    
    fig, axes = plt.subplots(2 if clasp_model else 1, 1, 
                             figsize=(14, 8 if clasp_model else 5),
                             sharex=True)
    
    if not clasp_model:
        axes = [axes]
    
    ax1 = axes[0]
    
    # Plot price data
    if 'close' in df.columns:
        ax1.plot(df.index, df['close'], 'k-', linewidth=0.8, alpha=0.7)
    
    # Highlight segments
    y_min, y_max = ax1.get_ylim() if ax1.get_ylim()[0] != ax1.get_ylim()[1] else (df['close'].min(), df['close'].max())
    
    for i, seg in enumerate(segments):
        start = seg.start_idx
        end = seg.end_idx
        
        if hasattr(df.index, '__getitem__'):
            x_start = df.index[start]
            x_end = df.index[min(end - 1, len(df) - 1)]
        else:
            x_start = start
            x_end = end
        
        # Add colored background
        ax1.axvspan(x_start, x_end, alpha=0.3, color=colors[i % len(colors)],
                   label=f'Segment {i+1}' + (f' ({seg.label})' if seg.label else ''))
    
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot ClaSP score profile if available
    if clasp_model and len(axes) > 1:
        ax2 = axes[1]
        if hasattr(clasp_model, 'profile') and clasp_model.profile is not None:
            profile = clasp_model.profile
            ax2.plot(range(len(profile)), profile, 'b-', linewidth=0.8)
            ax2.set_ylabel('ClaSP Score')
            ax2.set_xlabel('Index')
            ax2.grid(True, alpha=0.3)
            
            # Mark change points
            if hasattr(clasp_model, 'change_points_'):
                for cp in clasp_model.change_points_:
                    ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_segment_comparison(
    segments_data: List[np.ndarray],
    labels: List[str],
    cluster_labels: Optional[np.ndarray] = None,
    n_cols: int = 4,
    title: str = "Segment Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize multiple segments side by side for comparison.
    
    Args:
        segments_data: List of OHLC segment arrays.
        labels: List of label strings for each segment.
        cluster_labels: Optional cluster assignments.
        n_cols: Number of columns in subplot grid.
        title: Overall title.
        save_path: Path to save figure.
        show: Whether to display.
        
    Returns:
        Matplotlib figure.
    """
    n_segments = len(segments_data)
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    
    # Color map for labels
    unique_labels = list(set(labels))
    label_colors = {label: plt.cm.tab10(i / len(unique_labels)) 
                   for i, label in enumerate(unique_labels)}
    
    for idx, (seg, label) in enumerate(zip(segments_data, labels)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if seg.ndim == 2 and seg.shape[1] >= 4:
            # Plot close prices
            ax.plot(seg[:, 3], color=label_colors[label], linewidth=1)
            # Add high/low as shaded region
            ax.fill_between(range(len(seg)), seg[:, 2], seg[:, 1], 
                          alpha=0.2, color=label_colors[label])
        else:
            ax.plot(seg, color=label_colors[label], linewidth=1)
        
        title_text = f"{label}"
        if cluster_labels is not None:
            title_text += f" (C{cluster_labels[idx]})"
        ax.set_title(title_text, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_segments, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_clusters(
    segments_data: List[np.ndarray],
    labels: List[str],
    cluster_labels: np.ndarray,
    title: str = "Clustered Segments",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize segments grouped by cluster.
    
    Args:
        segments_data: List of segment arrays.
        labels: Original labels.
        cluster_labels: Cluster assignments.
        title: Plot title.
        save_path: Path to save figure.
        show: Whether to display.
        
    Returns:
        Matplotlib figure.
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    # Color map for original labels
    unique_labels = list(set(labels))
    label_colors = {label: plt.cm.tab10(i / len(unique_labels)) 
                   for i, label in enumerate(unique_labels)}
    
    for cluster_idx, cluster_id in enumerate(unique_clusters):
        ax = axes[cluster_idx]
        
        # Get segments in this cluster
        mask = cluster_labels == cluster_id
        cluster_segments = [seg for seg, m in zip(segments_data, mask) if m]
        cluster_labels_orig = [lab for lab, m in zip(labels, mask) if m]
        
        # Plot each segment (normalized to same length for visualization)
        for seg, lab in zip(cluster_segments, cluster_labels_orig):
            if seg.ndim == 2 and seg.shape[1] >= 4:
                prices = seg[:, 3]  # Close prices
            else:
                prices = seg
            
            # Normalize to [0, 1] for comparison
            if np.max(prices) != np.min(prices):
                prices_norm = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            else:
                prices_norm = prices
            
            # Normalize x-axis to [0, 1]
            x = np.linspace(0, 1, len(prices_norm))
            ax.plot(x, prices_norm, color=label_colors[lab], alpha=0.6, linewidth=1)
        
        # Count labels in cluster
        label_counts = {}
        for lab in cluster_labels_orig:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        
        count_str = ", ".join([f"{k}: {v}" for k, v in sorted(label_counts.items())])
        ax.set_title(f"Cluster {cluster_id} ({len(cluster_segments)} segments): {count_str}", fontsize=10)
        ax.set_xlabel("Normalized Time")
        ax.set_ylabel("Normalized Price")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def interactive_segment_viewer(
    segments_data: List[np.ndarray],
    labels: List[str],
    save_dir: str = "segment_viewer",
    n_per_page: int = 16
) -> None:
    """
    Create multiple PNG files for visual inspection of segments.
    
    Args:
        segments_data: List of segment arrays.
        labels: Labels for each segment.
        save_dir: Directory to save PNG files.
        n_per_page: Number of segments per page.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    n_segments = len(segments_data)
    n_pages = (n_segments + n_per_page - 1) // n_per_page
    
    for page in range(n_pages):
        start_idx = page * n_per_page
        end_idx = min(start_idx + n_per_page, n_segments)
        
        page_segments = segments_data[start_idx:end_idx]
        page_labels = labels[start_idx:end_idx]
        
        save_path = os.path.join(save_dir, f"segments_page_{page + 1:03d}.png")
        
        visualize_segment_comparison(
            page_segments, page_labels,
            title=f"Segments {start_idx + 1}-{end_idx} of {n_segments}",
            save_path=save_path,
            show=False
        )
    
    print(f"Created {n_pages} visualization pages in {save_dir}/")


def run_full_analysis(
    data_path: str,
    labels_path: str,
    output_dir: str,
    n_clusters: int = 6
) -> Dict[str, Any]:
    """
    Run full segmentation and clustering analysis.
    
    Args:
        data_path: Path to numpy data file.
        labels_path: Path to numpy labels file.
        output_dir: Output directory for visualizations.
        n_clusters: Number of clusters.
        
    Returns:
        Dict with analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    print(f"  Loaded {len(data)} segments")
    
    # Cluster segments
    print(f"\nClustering segments into {n_clusters} clusters...")
    cluster_labels, cluster_model = cluster_segments(list(data), n_clusters)
    
    # Print cluster distribution
    print("\nCluster distribution:")
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_orig_labels = labels[mask]
        print(f"  Cluster {cluster_id}: {sum(mask)} segments")
        label_counts = {}
        for lab in cluster_orig_labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1
        for lab, count in sorted(label_counts.items()):
            print(f"    - {lab}: {count}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Segment comparison grid
    visualize_segment_comparison(
        list(data), list(labels), cluster_labels,
        title="All Segments with Cluster Labels",
        save_path=os.path.join(output_dir, "all_segments.png"),
        show=False
    )
    
    # 2. Cluster visualization
    visualize_clusters(
        list(data), list(labels), cluster_labels,
        title="Segments Grouped by Cluster",
        save_path=os.path.join(output_dir, "clusters.png"),
        show=False
    )
    
    # 3. Interactive viewer pages
    interactive_segment_viewer(
        list(data), list(labels),
        save_dir=os.path.join(output_dir, "pages"),
        n_per_page=16
    )
    
    # 4. Per-label visualization
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        label_data = data[mask]
        label_labels = labels[mask]
        label_clusters = cluster_labels[mask]
        
        safe_label = label.replace(" ", "_").replace("/", "_")
        visualize_segment_comparison(
            list(label_data), list(label_labels), label_clusters,
            title=f"Segments: {label}",
            save_path=os.path.join(output_dir, f"label_{safe_label}.png"),
            show=False,
            n_cols=min(4, len(label_data))
        )
    
    print(f"\nAll visualizations saved to {output_dir}/")
    
    return {
        'cluster_labels': cluster_labels,
        'cluster_model': cluster_model,
        'n_segments': len(data),
        'n_clusters': n_clusters,
        'output_dir': output_dir
    }


if __name__ == '__main__':
    import sys
    
    # Default paths
    data_path = 'processed/bullflag_data.npy'
    labels_path = 'processed/bullflag_labels.npy'
    output_dir = 'segment_analysis'
    n_clusters = 6
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        labels_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        n_clusters = int(sys.argv[4])
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print("Usage: python segment_analysis.py <data.npy> [labels.npy] [output_dir] [n_clusters]")
        sys.exit(1)
    
    run_full_analysis(data_path, labels_path, output_dir, n_clusters)
