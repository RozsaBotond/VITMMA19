#!/usr/bin/env bash
# run.sh - Run the full pipeline scripts in order
# This script is used by the Docker container to execute the main pipeline
# stages in sequence: preprocessing, training, evaluation, inference.

set -euo pipefail

echo "========================================================================"
echo "[run.sh] Bull/Bear Flag Detector Pipeline"
echo "[run.sh] Started at $(date --iso-8601=seconds)"
echo "========================================================================"
echo ""

echo "[run.sh] Step 1/4: Data Preprocessing"
echo "------------------------------------------------------------------------"
python src/01-data-preprocessing.py
echo ""

echo "[run.sh] Step 2/4: Model Training"
echo "------------------------------------------------------------------------"
python src/02-training.py
echo ""

echo "[run.sh] Step 3/4: Model Evaluation"
echo "------------------------------------------------------------------------"
python src/03-evaluation.py
echo ""

echo "[run.sh] Step 4/4: Inference Demo"
echo "------------------------------------------------------------------------"
python src/04-inference.py
echo ""

echo "========================================================================"
echo "[run.sh] Pipeline completed successfully at $(date --iso-8601=seconds)"
echo "========================================================================"
