# Deep Learning Class (VITMMA19) Project Work# Bull-flag detector (VITMMA19)



## Project DetailsDeep-learning project for detecting **bull flag** and **bear flag** patterns in financial time series.



### Project Information## What’s in this repo



- **Selected Topic**: Bull-flag detector- `data/` – data prep, label parsing, segment extraction, verification tools

- **Student Name**: [Enter Your Name Here]- `src/bullflag_detector/` – model + detection utilities

- **Aiming for +1 Mark**: Yes- `models/` – trained weights (if any)



### Solution Description## Labeling workflow



This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.### 1) Verify existing Label Studio labels



**Problem**: Financial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using LSTM-based neural networks.You can verify an exported Label Studio JSON with either:



**Model Architecture**: - a Textual TUI: `main.py verify ... --tui`

- Bidirectional LSTM (2 layers, 128 hidden units) captures temporal dependencies in both directions- a matplotlib viewer: `main.py verify ...`

- Fully connected classifier head with dropout for regularization

- Class-weighted loss function to handle imbalanced label distribution### 2) Export *potential* flags for Label Studio (non-interactive)



**Training Methodology**:The interactive GUI labeler has been removed.

- Window size of 256 OHLC bars determined via claspy window size selection (SuSS + FFT)

- Min-max normalization per segmentInstead, use the batch exporter which:

- Stratified train/val/test split (70/15/15)

- AdamW optimizer with ReduceLROnPlateau scheduler- scans all CSVs under `data/raw_data/`

- Early stopping with patience of 15 epochs- detects *potential* flag candidates

- writes per-CSV **Label Studio import JSON** files (unlabeled regions)

**Classes** (6 total):

1. Bearish Normal - Standard bear flag patternRun:

2. Bearish Wedge - Bear flag with wedge formation

3. Bearish Pennant - Bear flag with pennant formation```bash

4. Bullish Normal - Standard bull flag patternuv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import

5. Bullish Wedge - Bull flag with wedge formation```

6. Bullish Pennant - Bull flag with pennant formation

Output example:

### Extra Credit Justification

- `data/labelstudio_import/XAU_1h_data_limited_potential_flags.json`

This project qualifies for extra credit due to:

Each task contains a proposed `start`/`end` span with `timeserieslabels: []` so you can select one of the 6 classes inside Label Studio.

1. **Data Acquisition & Labeling Pipeline**: Built a complete labeling workflow using Label Studio with custom export/import scripts for time series data, including automated candidate detection to speed up manual labeling.

## Notes

2. **Incremental Model Development**: Implemented baseline MLP model and LSTM v1 with clear architecture separation in the `models/` directory.

This candidate detector is heuristic on purpose. It’s meant to speed up *manual* labeling, not replace it.

3. **Advanced Evaluation**: Per-class metrics, confusion matrix, and weighted metrics to account for class imbalance.

4. **Window Size Selection**: Used claspy library's SuSS and FFT algorithms to determine optimal window size rather than arbitrary selection.

## Data Preparation

The data preparation process involves:

1. **Raw Data**: OHLC candlestick data for XAU (Gold) at multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h) stored in `data/raw_data/` as CSV files.

2. **Labeling**: Patterns were labeled using Label Studio. The verified labels are stored in `data/cleaned_labels_merged.json`.

3. **Preprocessing** (`src/01-data-preprocessing.py`):
   - Loads OHLC CSVs and labeled regions from JSON
   - Extracts fixed 256-bar windows centered on labeled patterns
   - Normalizes each window to [0, 1] range
   - Saves as `X.npy` (samples, 256, 4) and `Y.npy` (labels)

**To prepare data for training**:
```bash
# Data files should be placed in:
# - data/raw_data/*.csv (OHLC data)
# - data/cleaned_labels_merged.json (labels)
```

## Docker Instructions

### Build

```bash
docker build -t dl-project .
```

### Run

Mount your local data directory and capture logs:

```bash
docker run -v /absolute/path/to/your/data:/app/data dl-project > log/run.log 2>&1
```

Example with the project data:
```bash
docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1
```

The container executes the full pipeline:
1. Data preprocessing
2. Model training
3. Evaluation
4. Inference demo

## File Structure

```
.
├── src/                          # Source code
│   ├── 01-data-preprocessing.py  # Data loading, cleaning, preprocessing
│   ├── 02-training.py            # Model training with logging
│   ├── 03-evaluation.py          # Test set evaluation with metrics
│   ├── 04-inference.py           # Inference on new data
│   ├── config.py                 # Hyperparameters and paths
│   ├── utils.py                  # Logger and helper functions
│   ├── prepare/                  # Data preparation utilities
│   │   ├── prepare_data.py       # Segment extraction script
│   │   └── window_size_selection.py  # claspy window analysis
│   └── bullflag_detector/        # Detection and labeling tools
│
├── models/                       # Model definitions
│   ├── base.py                   # Abstract base model class
│   ├── baseline/                 # MLP baseline model
│   │   ├── __init__.py
│   │   ├── config.py             # Baseline hyperparameters
│   │   └── model.py              # MLPBaseline implementation
│   └── lstm_v1/                  # Bidirectional LSTM v1
│       ├── __init__.py
│       ├── config.py             # LSTM v1 hyperparameters
│       └── model.py              # LSTMv1 implementation
│
├── notebooks/                    # Jupyter notebooks (analysis)
│
├── data/                         # Data directory (mounted in Docker)
│   ├── raw_data/                 # OHLC CSV files
│   ├── cleaned_labels_merged.json  # Verified labels
│   ├── X.npy                     # Preprocessed features
│   ├── Y.npy                     # Preprocessed labels
│   └── metadata.json             # Dataset metadata
│
├── log/                          # Log files
│   └── run.log                   # Training run log
│
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── run.sh                        # Pipeline execution script
└── README.md                     # This file
```

## Training Configuration

Key hyperparameters (see `src/config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| EPOCHS | 100 | Maximum training epochs |
| BATCH_SIZE | 32 | Samples per batch |
| LEARNING_RATE | 0.001 | Initial learning rate |
| WINDOW_SIZE | 256 | OHLC bars per sample |
| LSTM_HIDDEN_SIZE | 128 | LSTM hidden dimension |
| LSTM_NUM_LAYERS | 2 | Number of LSTM layers |
| EARLY_STOPPING_PATIENCE | 15 | Epochs without improvement |

## Labeling Tools

The project includes tools for creating and verifying labels:

```bash
# Export potential flag candidates for Label Studio
uv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import

# Verify labels with TUI
python main.py verify --labels data/cleaned_labels_merged.json --tui
```
