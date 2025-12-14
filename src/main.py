"""
Bull/Bear Flag Pattern Detector - Main Entry Point

This project implements a deep learning system for detecting bull flag and bear flag
patterns in financial time series data.

Pattern Types:
- Bull Flag: Strong uptrend (pole) followed by consolidation (flag)
  - Normal: Parallel consolidation channel
  - Wedge: Narrowing consolidation
  - Pennant: Symmetric triangle consolidation

- Bear Flag: Strong downtrend (pole) followed by consolidation (flag)
  - Normal: Parallel consolidation channel  
  - Wedge: Narrowing consolidation
  - Pennant: Symmetric triangle consolidation

Usage:
    # Parse and summarize labels
    uv run python main.py labels labels.json
    
    # Extract segments from labeled data
    uv run python main.py extract labels.json data_dir output_dir
    
    # Split data into train/val/test
    uv run python main.py split data.npy labels.npy output_dir
    
    # Train baseline model
    uv run python main.py train data.npy labels.npy
    
    # Run full pipeline
    uv run python main.py pipeline labels.json data_dir output_dir
"""

import sys
import os

# Get the absolute path of the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Add data directory to path for imports (at the beginning to take priority)
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)


def print_help():
    """Print help message."""
    print(__doc__)
    print("\nCommands:")
    print("  labels <json_path>           - Summarize labels from Label Studio export")
    print("  extract <json> <data> <out>  - Extract segments from CSV files")
    print("  split <data> <labels> <out>  - Split data into train/val/test")
    print("  augment <in> <out> [amp] [seed] [copies] - Augment data with noise")
    print("  analyze <data> <labels> <out> [clusters] - Visualize and cluster segments")
    print("  verify <json> <data_dir>     - TUI for verifying labels with context")
    print("  train <data> <labels>        - Train baseline classifier")
    print("  pipeline <json> <data> <out> - Run complete pipeline")
    print("  help                         - Show this help message")


def cmd_labels(args):
    """Summarize labels from JSON file."""
    if len(args) < 1:
        print("Usage: uv run python main.py labels <labels.json>")
        return 1
    
    from label_parser import summarize_labels
    summarize_labels(args[0])
    return 0


def cmd_extract(args):
    """Extract segments from labeled data."""
    if len(args) < 3:
        print("Usage: uv run python main.py extract <labels.json> <data_dir> <output_dir>")
        return 1
    
    from segment_extractor import process_labels_to_segments
    
    labels_json = args[0]
    data_dir = args[1]
    output_dir = args[2]
    
    # Process with handle removed
    process_labels_to_segments(
        labels_json, data_dir, output_dir,
        remove_handle=True, min_length=5
    )
    
    # Also process with handle included
    print("\n" + "=" * 60)
    process_labels_to_segments(
        labels_json, data_dir, output_dir,
        remove_handle=False, min_length=5
    )
    
    return 0


def cmd_split(args):
    """Split data into train/val/test sets."""
    if len(args) < 3:
        print("Usage: uv run python main.py split <data.npy> <labels.npy> <output_dir>")
        return 1
    
    from train_test_split import DatasetManager
    
    data_path = args[0]
    labels_path = args[1]
    output_dir = args[2]
    
    manager = DatasetManager()
    manager.load(data_path, labels_path)
    manager.create_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    manager.save(output_dir)
    
    return 0


def cmd_augment(args):
    """Augment data with zero-mean Gaussian noise."""
    if len(args) < 2:
        print("Usage: uv run python main.py augment <input_dir> <output_dir> [amplitude] [seed] [num_copies]")
        print("\nOptions:")
        print("  input_dir   - Directory containing processed data (e.g., data/processed)")
        print("  output_dir  - Output directory for augmented data (e.g., data/processed_augmented)")
        print("  amplitude   - Noise amplitude as fraction of price range (default: 0.01)")
        print("  seed        - Random seed for reproducibility (default: 42)")
        print("  num_copies  - Number of augmented copies per sample (default: 1)")
        return 1
    
    from augment_data import augment_train_val_test
    
    input_dir = args[0]
    output_dir = args[1]
    noise_amplitude = float(args[2]) if len(args) > 2 else 0.01
    seed = int(args[3]) if len(args) > 3 else 42
    num_copies = int(args[4]) if len(args) > 4 else 1
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Noise amplitude: {noise_amplitude}")
    print(f"Random seed: {seed}")
    print(f"Augmented copies per sample: {num_copies}")
    print()
    
    augment_train_val_test(
        input_dir=input_dir,
        output_dir=output_dir,
        noise_amplitude=noise_amplitude,
        seed=seed,
        num_copies=num_copies
    )
    
    return 0


def cmd_analyze(args):
    """Analyze and visualize segments using clustering."""
    if len(args) < 3:
        print("Usage: uv run python main.py analyze <data.npy> <labels.npy> <output_dir> [n_clusters]")
        print("\nOptions:")
        print("  data.npy    - Path to segments data")
        print("  labels.npy  - Path to labels")
        print("  output_dir  - Output directory for visualizations")
        print("  n_clusters  - Number of clusters for K-means (default: 6)")
        return 1
    
    from segment_analysis import run_full_analysis
    
    data_path = args[0]
    labels_path = args[1]
    output_dir = args[2]
    n_clusters = int(args[3]) if len(args) > 3 else 6
    
    print(f"Data: {data_path}")
    print(f"Labels: {labels_path}")
    print(f"Output: {output_dir}")
    print(f"Clusters: {n_clusters}")
    print()
    
    run_full_analysis(
        data_path=data_path,
        labels_path=labels_path,
        output_dir=output_dir,
        n_clusters=n_clusters
    )
    
    return 0


def cmd_verify(args):
    """Launch TUI for verifying labels with context."""
    if len(args) < 2:
        print("Usage: uv run python main.py verify <labels.json> <data_dir> [output.json] [--tui]")
        print("\nOptions:")
        print("  labels.json  - Label Studio JSON export file")
        print("  data_dir     - Directory containing the source CSV files")
        print("  output.json  - Output file for verification results (optional)")
        print("  --tui        - Use text-based TUI instead of matplotlib (optional)")
        print("\nControls:")
        print("  Y = Mark correct, N = Mark incorrect")
        print("  J/→ = Next, K/← = Previous")
        print("  S = Save results")
        return 1
    
    labels_path = args[0]
    data_dir = args[1]
    
    # Check for --tui flag
    use_tui = '--tui' in args
    remaining_args = [a for a in args[2:] if a != '--tui']
    output_path = remaining_args[0] if remaining_args else None
    
    print(f"Labels: {labels_path}")
    print(f"Data directory: {data_dir}")
    if output_path:
        print(f"Output: {output_path}")
    
    if use_tui:
        print("\nLaunching TUI...")
        from label_verifier import run_verifier
        run_verifier(
            labels_path=labels_path,
            data_dir=data_dir,
            output_path=output_path
        )
    else:
        print("\nLaunching matplotlib viewer...")
        from label_verifier_mpl import run_verifier_mpl
        run_verifier_mpl(
            labels_path=labels_path,
            data_dir=data_dir,
            output_path=output_path
        )
    
    return 0


def cmd_train(args):
    """Train baseline classifier."""
    if len(args) < 2:
        print("Usage: uv run python main.py train <data.npy> <labels.npy>")
        return 1
    
    from train_baseline_model import train_baseline_model, cross_validate_baseline
    
    data_path = args[0]
    labels_path = args[1]
    
    # Train logistic regression
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    train_baseline_model(data_path, labels_path, model_type='logistic')
    
    # Train random forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    train_baseline_model(data_path, labels_path, model_type='random_forest')
    
    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    cross_validate_baseline(data_path, labels_path, model_type='logistic')
    
    return 0


def cmd_pipeline(args):
    """Run complete pipeline."""
    if len(args) < 3:
        print("Usage: uv run python main.py pipeline <labels.json> <data_dir> <output_dir>")
        return 1
    
    # Import from the data directory (already in sys.path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_module", os.path.join(DATA_DIR, "data.py"))
    data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_module)
    Data = data_module.Data
    
    labels_json = args[0]
    data_dir = args[1]
    output_dir = args[2]
    
    # Initialize and run pipeline
    print("=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)
    
    data = Data(
        labels_path=labels_json,
        data_dir=data_dir,
        ratios=(0.7, 0.15, 0.15),
        remove_handle=True
    )
    
    print("\n" + data.summary())
    
    # Save processed data
    data.save(output_dir)
    
    # Train baseline model
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)
    
    from train_baseline_model import BaselineClassifier
    
    classifier = BaselineClassifier(model_type='logistic')
    classifier.fit(
        data.train_data, data.train_labels,
        data.val_data, data.val_labels
    )
    
    print("\nTest Set Evaluation:")
    classifier.evaluate(data.test_data, data.test_labels)
    
    return 0


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_help()
        return 1
    
    command = sys.argv[1].lower()
    args = sys.argv[2:]
    
    commands = {
        'labels': cmd_labels,
        'extract': cmd_extract,
        'split': cmd_split,
        'augment': cmd_augment,
        'analyze': cmd_analyze,
        'verify': cmd_verify,
        'train': cmd_train,
        'pipeline': cmd_pipeline,
        'help': lambda _: print_help() or 0,
    }
    
    if command not in commands:
        print(f"Unknown command: {command}")
        print_help()
        return 1
    
    return commands[command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
