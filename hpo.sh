#!/usr/bin/env bash
# hpo.sh - run the hyperparameter optimization scripts

set -euo pipefail

echo "[hpo.sh] Starting hyperparameter optimization at $(date --iso-8601=seconds)"

# Set PYTHONPATH to include the project root for modular imports
export PYTHONPATH=$(pwd)

# Create log directory if it doesn't exist
mkdir -p log

# Run HPO scripts and tee output to log files
echo "[hpo.sh] Running Hyperparameter Optimization for LSTM..."
uv run python src/hpo/bayesian_hp_lstm.py --n-trials 50 | tee log/hpo_lstm_run.log

echo "[hpo.sh] Running Hyperparameter Optimization for Hierarchical LSTM..."
uv run python src/hpo/bayesian_hp_hierarchical.py --n-trials 50 | tee log/hpo_hierarchical_run.log

echo "[hpo.sh] Running Hyperparameter Optimization for Transformer..."
uv run python src/hpo/bayesian_hp_transformer.py --n-trials 50 | tee log/hpo_transformer_run.log

echo "[hpo.sh] Hyperparameter optimization finished at $(date --iso-8601=seconds)"
