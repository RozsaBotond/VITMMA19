#!/usr/bin/env bash
# run.sh - Run the full pipeline scripts in order
# This script is used by the Docker container to execute the main pipeline
# stages in sequence: preprocessing, training, evaluation, inference.
#
# Pipeline modes (set via PIPELINE_MODE env variable):
#   - LSTM_V1: Basic LSTM model (02-training.py, 03-evaluation.py)
#   - HIERARCHICAL: Two-stage hierarchical classifier (03-training-hierarchical.py)
#   - ALL_MODELS: Train and compare all model architectures (train_all_models.py)
#
# Default: ALL_MODELS

set -euo pipefail

PIPELINE_MODE=${PIPELINE_MODE:-ALL_MODELS}

echo "========================================================================"
echo "[run.sh] Bull/Bear Flag Detector Pipeline"
echo "[run.sh] Mode: ${PIPELINE_MODE}"
echo "[run.sh] Started at $(date --iso-8601=seconds)"
echo "========================================================================"
echo ""

echo "[run.sh] Step 1/4: Data Preprocessing"
echo "------------------------------------------------------------------------"
python src/01-data-preprocessing.py
echo ""

if [ "$PIPELINE_MODE" = "ALL_MODELS" ]; then
    echo "[run.sh] Step 2-3/4: Multi-Model Training & Evaluation"
    echo "------------------------------------------------------------------------"
    python src/train_all_models.py
    echo ""

elif [ "$PIPELINE_MODE" = "HIERARCHICAL" ]; then
    echo "[run.sh] Step 2/4: Hierarchical Model Training"
    echo "------------------------------------------------------------------------"
    python src/03-training-hierarchical.py
    echo ""

    echo "[run.sh] Step 3/4: Model Evaluation"
    echo "------------------------------------------------------------------------"
    python src/03-evaluation-seq.py
    echo ""
else
    echo "[run.sh] Step 2/4: Model Training"
    echo "------------------------------------------------------------------------"
    python src/02-training.py
    echo ""

    echo "[run.sh] Step 3/4: Model Evaluation"
    echo "------------------------------------------------------------------------"
    python src/03-evaluation.py
    echo ""
fi

echo "[run.sh] Step 4/4: Inference Demo"
echo "------------------------------------------------------------------------"
python src/04-inference.py
echo ""

echo "========================================================================"
echo "[run.sh] Pipeline completed successfully at $(date --iso-8601=seconds)"
echo "========================================================================"
"
