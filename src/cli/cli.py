import argparse
import sys
import os

# Ensure the 'src' directory is in the Python path for module discovery
def _fix_sys_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    src_dir = os.path.join(project_root, 'src')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

_fix_sys_path()


# Import functions from the restructured project
from src.data.augmentation import augment_train_val_test
from src.bullflag_detector.export_labelstudio import summarize_labels
from src.bullflag_detector.ui.candidate_tui import run_verifier
from src.bullflag_detector.ui.candidate_gui import run_verifier_mpl
from models.baseline.statistical import train_baseline_model, cross_validate_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Bull/Bear Flag Pattern Detector CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Labels command
    labels_parser = subparsers.add_parser("labels", help="Summarize labels from Label Studio export")
    labels_parser.add_argument("json_path", type=str, help="Path to the labels JSON file.")
    labels_parser.set_defaults(func=_cmd_labels)

    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Augment data with zero-mean Gaussian noise")
    augment_parser.add_argument("input_dir", type=str, help="Directory containing processed data (e.g., data/processed).")
    augment_parser.add_argument("output_dir", type=str, help="Output directory for augmented data (e.g., data/processed_augmented).")
    augment_parser.add_argument("--amplitude", type=float, default=0.01, help="Noise amplitude as fraction of price range (default: 0.01).")
    augment_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    augment_parser.add_argument("--num_copies", type=int, default=1, help="Number of augmented copies per sample (default: 1).")
    augment_parser.set_defaults(func=_cmd_augment)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Launch TUI for verifying labels with context")
    verify_parser.add_argument("labels_path", type=str, help="Label Studio JSON export file.")
    verify_parser.add_argument("data_dir", type=str, help="Directory containing the source CSV files.")
    verify_parser.add_argument("--output_path", type=str, help="Output file for verification results (optional).")
    verify_parser.add_argument("--tui", action="store_true", help="Use text-based TUI instead of matplotlib.")
    verify_parser.set_defaults(func=_cmd_verify)

    # Train command (baseline)
    train_parser = subparsers.add_parser("train", help="Train baseline classifier")
    train_parser.add_argument("data_path", type=str, help="Path to the data.npy file.")
    train_parser.add_argument("labels_path", type=str, help="Path to the labels.npy file.")
    train_parser.set_defaults(func=_cmd_train)
    
    # NOTE: The 'extract', 'split', and 'analyze' commands have been temporarily removed
    # because their underlying implementation files (segment_extractor.py,
    # train_test_split.py, segment_analysis.py) were not found in the project.
    # The 'pipeline' command was also removed as it depends on these missing files.

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def _cmd_labels(args):
    summarize_labels(args.json_path)

def _cmd_augment(args):
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Noise amplitude: {args.amplitude}")
    print(f"Random seed: {args.seed}")
    print(f"Augmented copies per sample: {args.num_copies}")
    print()
    augment_train_val_test(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        noise_amplitude=args.amplitude,
        seed=args.seed,
        num_copies=args.num_copies
    )

def _cmd_verify(args):
    print(f"Labels: {args.labels_path}")
    print(f"Data directory: {args.data_dir}")
    if args.output_path:
        print(f"Output: {args.output_path}")

    if args.tui:
        print("\nLaunching TUI...")
        run_verifier(
            labels_path=args.labels_path,
            data_dir=args.data_dir,
            output_path=args.output_path
        )
    else:
        print("\nLaunching matplotlib viewer...")
        run_verifier_mpl(
            labels_path=args.labels_path,
            data_dir=args.data_dir,
            output_path=args.output_path
        )

def _cmd_train(args):
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 60)
    train_baseline_model(args.data_path, args.labels_path, model_type='logistic')
    
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)
    train_baseline_model(args.data_path, args.labels_path, model_type='random_forest')
    
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    cross_validate_baseline(args.data_path, args.labels_path, model_type='logistic')


if __name__ == "__main__":
    main()
