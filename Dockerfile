# Bull/Bear Flag Detector Dockerfile
# 
# This Dockerfile builds an image for training and evaluating the
# Bull/Bear Flag pattern detection model on OHLC financial data.
#
# Build: docker build -t dl-project .
# Run:   docker run -v /path/to/data:/app/data dl-project > log/run.log 2>&1
#
# Pipeline modes:
#   LSTM_V1 (default): Basic LSTM classifier
#   HIERARCHICAL: Two-stage hierarchical classifier
#
# Example with hierarchical mode:
#   docker run -e PIPELINE_MODE=HIERARCHICAL -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1

# Use PyTorch with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ src/
COPY models/ models/
COPY run.sh run.sh

# Create directories for data and logs (data will be mounted)
RUN mkdir -p /app/data /app/log

# Make run script executable
RUN chmod +x /app/run.sh

# Set Python path to include project root
ENV PYTHONPATH=/app:$PYTHONPATH

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# Default command: run the full pipeline
CMD ["bash", "/app/run.sh"]
