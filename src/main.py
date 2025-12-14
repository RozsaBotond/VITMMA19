"""Main entry point for the Bull/Bear Flag Detector CLI.

This script provides a command-line interface to run the different stages
of the machine learning pipeline:
- Data Preprocessing
- Model Training
- Model Evaluation
- Inference

Usage:
    python src/main.py preprocess
    python src/main.py train --models lstm_v2 transformer
    python src/main.py evaluate --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth
    python src/main.py infer --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports to enable absolute imports from src.*
# This assumes src/main.py is run directly or src is added to PYTHONPATH
# If running through a wrapper script (e.g. run.sh), PYTHONPATH should be set there.
# For direct execution, we can add it here.
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent # Assuming src is directly under project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_dir) not in sys.path: # Add src itself
    sys.path.insert(0, str(current_dir))

# Import main functions from individual scripts
from data_preprocessing import preprocess_data_main
from training import train_main
from evaluation import evaluate_main
from inference import inference_main

def main():
    parser = argparse.ArgumentParser(
        description="Bull/Bear Flag Pattern Detector CLI.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Run data preprocessing and splitting.")
    preprocess_parser.set_defaults(func=preprocess_data_main)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train one or more models.")
    train_parser.add_argument("--models", nargs="+", help="Models to train.")
    train_parser.add_argument("--epochs", type=int, help="Override epochs for all models.")
    train_parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    train_parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation.")
    train_parser.add_argument("--log-file", type=str, default="log/run.log", help="Path to the log file.")
    train_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    train_parser.set_defaults(func=train_main) # Call train_main directly

    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model.")
    eval_parser.add_argument("--model-key", required=True, help="Key of the model to evaluate.")
    eval_parser.add_argument("--checkpoint-path", required=True, type=Path, help="Path to the model checkpoint.")
    eval_parser.add_argument("--log-file", type=str, default="log/evaluation_run.log", help="Path to the log file.")
    eval_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    eval_parser.set_defaults(func=evaluate_main) # Call evaluate_main directly

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference with a trained model.")
    infer_parser.add_argument("--model-key", required=True, help="Key of the model to use for inference.")
    infer_parser.add_argument("--checkpoint-path", required=True, type=Path, help="Path to the model checkpoint.")
    infer_parser.add_argument("--log-file", type=str, default="log/inference_run.log", help="Path to the log file.")
    infer_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    infer_parser.set_defaults(func=inference_main) # Call inference_main directly

    args = parser.parse_args()
    
    # Pass parsed arguments as kwargs to the respective main function
    # This avoids modifying sys.argv directly and makes functions cleaner.
    func_args = vars(args)
    func_to_call = func_args.pop('func')
    func_args.pop('command') # Remove the command itself

    # Dynamically extract parameters expected by the target function
    import inspect
    sig = inspect.signature(func_to_call)
    
    # Filter args to only include those expected by the function
    filtered_args = {k: v for k, v in func_args.items() if k in sig.parameters}

    func_to_call(**filtered_args)

if __name__ == "__main__":
    main()