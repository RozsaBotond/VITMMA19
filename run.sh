#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

# Set PYTHONPATH to include the src directory for modular imports
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/src

# Run each stage of the pipeline using uv run to ensure the correct environment
echo "[run.sh] Running data preprocessing..."
uv run python src/main.py preprocess

echo "[run.sh] Running model training..."
# Pass default models and logging options. Logs will be in log/run.log
uv run python src/main.py train --models lstm_v2 hierarchical_v1 transformer baseline --log-file log/run.log --log-level INFO

echo "[run.sh] Running model evaluation..."
# Evaluate the best performing model from training (e.g., lstm_v2)
# The checkpoint path should reflect where the training script saved the model.
uv run python src/main.py evaluate --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth --log-file log/evaluation_run.log --log-level INFO

echo "[run.sh] Running inference demo..."
uv run python src/main.py infer --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth --log-file log/inference_run.log --log-level INFO

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"