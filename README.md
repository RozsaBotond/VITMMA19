# Bull Flag Detector (VITMMA19 - Deep Learning Project)# Bull Flag Detector (VITMMA19 - Deep Learning Project)# Deep Learning Class (VITMMA19) - Project Work# Deep Learning Class (VITMMA19) - Project Work# Deep Learning Class (VITMMA19) Project Work# Bull-flag detector (VITMMA19)



## Project Information



| Field | Value |## Project Information

|-------|-------|

| **Selected Topic** | Bull-flag detector |

| **Student Name** | Rozsa |

| **Aiming for +1 Mark** | Yes || Field | Value |## Project Information



---|-------|-------|



## Solution Description| **Selected Topic** | Bull-flag detector |



### Problem Statement| **Student Name** | Rozsa |



This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.| **Aiming for +1 Mark** | Yes || Field | Value |## Project Information



Financial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using multiple neural network architectures.



### Pattern Classes (7)---|-------|-------|



| Class | Description |

|-------|-------------|

| **None** | No pattern detected |## Solution Description| **Selected Topic** | Bull-flag detector |

| **Bullish Normal** | Standard bull flag with parallel consolidation |

| **Bullish Wedge** | Bull flag with wedge-shaped consolidation |

| **Bullish Pennant** | Bull flag with symmetric triangle consolidation |

| **Bearish Normal** | Standard bear flag with parallel consolidation |### Problem Statement| **Student Name** | Rozsa |

| **Bearish Wedge** | Bear flag with wedge-shaped consolidation |

| **Bearish Pennant** | Bear flag with symmetric triangle consolidation |



---This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.| **Aiming for +1 Mark** | Yes || Field | Value |## Project DetailsDeep-learning project for detecting **bull flag** and **bear flag** patterns in financial time series.



## Model Architectures



This project implements **incremental model development** with hyperparameter optimization across multiple architecture families.Financial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using multiple neural network architectures.



### Baseline Models



#### Statistical Baseline (Non-Neural)### Pattern Classes---|-------|-------|

- **Description:** Trend + volatility-based rule detection using training statistics

- **Features:** Linear regression slope, ATR (Average True Range), consolidation detection

- **Parameters:** 14 (statistics only)

- **Purpose:** Establishes performance floor for comparisonThe model distinguishes between **7 classes**:



#### window_lstm - Window Classification LSTM

- **Description:** Original bidirectional LSTM for window-level classification

- **Task:** Classify entire window (6 classes)| Class | Description |## Solution Description| **Selected Topic** | Bull-flag detector |

- **Architecture:** 2-layer LSTM, 128 hidden units

|-------|-------------|

### LSTM Family (Sequence Labeling)

| **None** | No pattern detected |

All LSTM variants use per-timestep sequence labeling (7 classes) with learning rate scheduling.

| **Bullish Normal** | Standard bull flag with parallel consolidation |

| Version | Description | Hidden | Layers | Dropout | LR | Scheduler |

|---------|-------------|--------|--------|---------|-----|-----------|| **Bullish Wedge** | Bull flag with wedge-shaped consolidation |### Problem Statement| **Student Name** | Rozsa |

| **lstm_v2a** | Baseline config | 128 | 2 | 0.3 | 1e-3 | Cosine |

| **lstm_v2b** | Larger, slower LR | 192 | 3 | 0.4 | 5e-4 | Cosine + Warm Restarts || **Bullish Pennant** | Bull flag with symmetric triangle consolidation |

| **lstm_v2c** | Small, fast | 64 | 2 | 0.2 | 2e-3 | OneCycleLR |

| **Bearish Normal** | Standard bear flag with parallel consolidation |

### Transformer Family

| **Bearish Wedge** | Bear flag with wedge-shaped consolidation |

| Version | Description | d_model | Heads | Layers | Dropout | LR | Scheduler |

|---------|-------------|---------|-------|--------|---------|-----|-----------|| **Bearish Pennant** | Bear flag with symmetric triangle consolidation |This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.| **Aiming for +1 Mark** | Yes |### Project Information## What’s in this repo

| **transformer_v1a** | Baseline | 64 | 4 | 3 | 0.2 | 1e-4 | Cosine |

| **transformer_v1b** | Deeper | 64 | 4 | 4 | 0.3 | 5e-5 | Cosine + Warm Restarts |

| **transformer_v1c** | Wider | 128 | 8 | 2 | 0.2 | 2e-4 | OneCycleLR |

---

### Hierarchical Family



| Version | Description | Hidden | Layers | Dropout | LR | Scheduler |

|---------|-------------|--------|--------|---------|-----|-----------|## Model ArchitecturesFinancial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using multiple neural network architectures.

| **hierarchical_v1a** | Baseline 7-class | 128 | 2 | 0.3 | 5e-4 | Cosine |

| **hierarchical_v1b** | Larger capacity | 192 | 3 | 0.4 | 3e-4 | ReduceLROnPlateau |



### Additional ArchitecturesThis project implements and compares **5 different neural network architectures** with incremental versioning:



| Model | Description | Parameters |

|-------|-------------|------------|

| **cnn1d** | Dilated 1D CNN | ~137K |### lstm_v1 - Basic Bidirectional LSTM### Pattern Types---

| **cnn_lstm** | CNN + LSTM hybrid | ~182K |

- Window-level classification (6 classes)

---

- Bidirectional LSTM (2 layers, 128 hidden units)

## Training Methodology

- Original baseline model

### Learning Rate Scheduling

The model distinguishes between 7 classes:

All models use advanced LR scheduling with warmup:

### lstm_v2 - Sequence Labeling LSTM

- **Cosine Annealing:** Smooth decay to minimum LR

- **Cosine with Warm Restarts:** Periodic LR resets for escaping local minima- Per-timestep sequence labeling (7 classes)

- **OneCycleLR:** Super-convergence schedule for faster training

- **ReduceLROnPlateau:** Adaptive reduction based on validation metrics- Bidirectional LSTM with attention-ready architecture



### Data Pipeline- Focal Loss support for class imbalance| Class | Description |## Solution Description- **Selected Topic**: Bull-flag detector- `data/` – data prep, label parsing, segment extraction, verification tools



- **Window size:** 256 OHLC bars (determined via claspy SuSS + FFT)- **~201K parameters**

- **Normalization:** StandardScaler per sample

- **Augmentation:** Gaussian noise, time warping, scaling|-------|-------------|

- **Class balancing:** Augmentation-based oversampling of minority classes

### lstm_v3 - Minimal LSTM

### Regularization

- Reduced architecture for overfitting tests| **None** | No pattern detected |

- Dropout (0.2-0.4)

- Weight decay (1e-5 to 1e-3)- Designed to validate training pipeline

- Early stopping (patience 30)

- Gradient clipping (max norm 1.0)- **~7.8K parameters**| **Bullish Normal** | Standard bull flag with parallel consolidation |

- Class weight capping (max 5.0x)



---

### cnn1d - 1D Convolutional Network| **Bullish Wedge** | Bull flag with wedge-shaped consolidation |### Problem Statement- **Student Name**: [Enter Your Name Here]- `src/bullflag_detector/` – model + detection utilities

## Project Structure

- Dilated 1D convolutions for large receptive field

```

VITMMA19/- Channel progression: 32 → 64 → 128 → 256| **Bullish Pennant** | Bull flag with symmetric triangle consolidation |

├── data/                    # Data files

│   ├── X_seq.npy           # Sequence inputs (300, 256, 4)- Maintains sequence length for per-timestep predictions

│   ├── Y_seq.npy           # Sequence labels (300, 256)

│   └── raw_data/           # Original XAU gold CSV files- Batch normalization + dropout| **Bearish Normal** | Standard bear flag with parallel consolidation |

├── models/                  # Model implementations

│   ├── baseline/           # Statistical + MLP baselines- **~137K parameters**

│   ├── window_lstm/        # Window classification LSTM

│   ├── lstm_v2/            # Sequence labeling LSTM| **Bearish Wedge** | Bear flag with wedge-shaped consolidation |

│   ├── lstm_v3/            # Minimal LSTM for testing

│   ├── transformer/        # Transformer encoder### transformer - Transformer Encoder

│   ├── cnn1d/              # 1D CNN

│   ├── cnn_lstm/           # CNN-LSTM hybrid- Multi-head self-attention (4 heads, 64 dim)| **Bearish Pennant** | Bear flag with symmetric triangle consolidation |This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.- **Aiming for +1 Mark**: Yes- `models/` – trained weights (if any)

│   └── hierarchical_v1/    # Two-stage classifier

├── src/                     # Training scripts- 3 transformer encoder layers

│   ├── train_incremental_hp.py  # HP optimization training

│   ├── train_all_models.py      # Multi-model comparison- Sinusoidal positional encoding

│   ├── 02-training-seq.py       # Sequence training

│   └── 03-training-hierarchical.py- GELU activation, pre-norm architecture

├── Dockerfile

├── run.sh- **~151K parameters**### Model Architectures

└── requirements.txt

```



---### cnn_lstm - CNN-LSTM Hybrid



## Quick Start- CNN backbone for local feature extraction



### Installation- Bidirectional LSTM for temporal modelingThis project implements and compares **4 different neural network architectures**:Financial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using LSTM-based neural networks with a hierarchical classification approach.



```bash- Combines the best of both approaches

# Using uv (recommended)

uv pip install -r requirements.txt- **~182K parameters**

```



### Train with Hyperparameter Optimization

### hierarchical_v1 - Two-Stage Classifier#### 1. SeqLabelingLSTM (Bidirectional LSTM)

```bash

# Train all model versions with HP tuning- **Stage 1:** Direction classification (None / Bearish / Bullish)

uv run python src/train_incremental_hp.py

- **Stage 2:** Subtype classification (Normal / Wedge / Pennant)- Bidirectional LSTM (2 layers, 128 hidden units)

# Train specific models

uv run python src/train_incremental_hp.py --models baseline lstm_v2a transformer_v1a- Focal Loss for handling class imbalance



# Train with custom epochs- **~533K parameters total**- Per-timestep sequence labeling### Pattern Types### Solution Description## Labeling workflow

uv run python src/train_incremental_hp.py --models lstm_v2b --epochs 200

```



### Train Basic Comparison---- Dropout regularization (0.3)



```bash

# Compare 4 base architectures

uv run python src/train_all_models.py## Training Methodology

```



---

- **Window size:** 256 OHLC bars (determined via claspy SuSS + FFT analysis)#### 2. SeqCNN1D (1D Convolutional Network)

## Docker Instructions

- **Normalization:** StandardScaler per sample (OHLC normalization)

```bash

# Build- **Data augmentation:** Gaussian noise, time warping, scaling (train set only)- Dilated 1D convolutions for large receptive fieldThe model distinguishes between 7 classes:

docker build -t dl-project .

- **Optimizer:** AdamW with cosine annealing scheduler

# Train

docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dl-project train- **Early stopping:** Patience of 30-50 epochs- Channel progression: 32 → 64 → 128 → 256



# Evaluate- **Class weighting:** Capped at 5.0x to prevent over-correction

docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dl-project evaluate

```- Maintains sequence length for per-timestep predictions



------



## Results- Batch normalization + dropout



### Baseline Comparison## Model Comparison Results



| Model | Parameters | Test F1 | Detection Rate | False Alarm || Class | Description |This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts.### 1) Verify existing Label Studio labels

|-------|------------|---------|----------------|-------------|

| Statistical Baseline | 14 | 55.8% | 27.9% | 32.9% |Latest training results (trained on 504 augmented samples, validated on 45, tested on 45):



The statistical baseline achieves high F1 by predicting "None" frequently but has low detection rate.#### 3. SeqTransformer (Transformer Encoder)



### Neural Network Comparison| Model | Parameters | Test Accuracy | F1 (weighted) | Detection Rate | False Alarm |



Results vary by hyperparameter configuration. Run `train_incremental_hp.py` for full comparison.|-------|------------|---------------|---------------|----------------|-------------|- Multi-head self-attention (4 heads, 64 dim)|-------|-------------|



---| **transformer** | 151K | 21.1% | **23.4%** | 93.7% | 84.4% |



## Extra Credit Justification| **lstm_v2** | 201K | 19.2% | 21.0% | 94.4% | 86.8% |- 3 transformer encoder layers



1. **Multiple Model Architectures:** 8+ model configurations across 4 architecture families| **cnn1d** | 137K | 12.7% | 7.8% | 99.6% | 97.4% |

2. **Hyperparameter Optimization:** Systematic HP search with multiple versions per architecture

3. **Learning Rate Scheduling:** Advanced schedulers (Cosine, OneCycleLR, Warm Restarts)| **cnn_lstm** | 182K | 13.1% | 6.1% | 100.0% | 100.0% |- Sinusoidal positional encoding| **None** | No pattern detected |

4. **Statistical Baseline:** Non-neural baseline for performance comparison

5. **Incremental Development:** Versioned models (v2a, v2b, v2c) showing iterative improvement

6. **Data Augmentation Pipeline:** Class-balanced augmentation for minority classes

**Note:** The low accuracy is expected due to:- GELU activation, pre-norm architecture

---

- Small dataset (300 labeled samples)

## Dependencies

- High class imbalance (majority "None" class)| **Bullish Normal** | Standard bull flag with parallel consolidation |

- PyTorch >= 2.0

- NumPy, scikit-learn- Fine-grained 7-class classification task

- matplotlib, tqdm

- claspy (window size selection)#### 4. SeqCNNLSTM (CNN-LSTM Hybrid)



------



## License- CNN backbone for local feature extraction| **Bullish Wedge** | Bull flag with wedge-shaped consolidation |**Problem**: Financial traders use chart patterns like bull/bear flags to identify potential trading opportunities. Manual pattern detection is time-consuming and subjective. This project automates the detection using LSTM-based neural networks.You can verify an exported Label Studio JSON with either:



Coursework for VITMMA19 Deep Learning at BME.## Project Structure


- Bidirectional LSTM for temporal modeling

```

VITMMA19/- Combines the best of both worlds| **Bullish Pennant** | Bull flag with symmetric triangle consolidation |

├── data/                   # Training/validation/test data

│   ├── X_seq.npy          # Sequence input data

│   ├── Y_seq.npy          # Sequence labels

│   ├── metadata_seq.json  # Sample metadata#### 5. HierarchicalClassifier (Two-Stage LSTM)| **Bearish Normal** | Standard bear flag with parallel consolidation |

│   └── raw_data/          # Original CSV files (XAU gold data)

├── models/                 # Model implementations- **Stage 1:** Direction classification (None / Bearish / Bullish)

│   ├── lstm_v1/           # Basic bidirectional LSTM

│   ├── lstm_v2/           # Sequence labeling LSTM- **Stage 2:** Subtype classification (Normal / Wedge / Pennant)| **Bearish Wedge** | Bear flag with wedge-shaped consolidation |

│   ├── lstm_v3/           # Minimal LSTM for testing

│   ├── cnn1d/             # 1D CNN model- Focal Loss for handling class imbalance

│   ├── transformer/       # Transformer encoder

│   ├── cnn_lstm/          # CNN-LSTM hybrid| **Bearish Pennant** | Bear flag with symmetric triangle consolidation |**Model Architecture**: - a Textual TUI: `main.py verify ... --tui`

│   └── hierarchical_v1/   # Two-stage classifier

├── src/                    # Training and evaluation scripts### Training Methodology

│   ├── 01-data-preprocessing.py

│   ├── 02-training-seq.py

│   ├── 03-evaluation-seq.py

│   ├── train_all_models.py  # Multi-model comparison- **Window size:** 256 OHLC bars (determined via claspy SuSS + FFT analysis)

│   └── inference.py

├── notebooks/              # Jupyter notebooks- **Normalization:** StandardScaler per sample (OHLC normalization)### Model Architecture- Bidirectional LSTM (2 layers, 128 hidden units) captures temporal dependencies in both directions- a matplotlib viewer: `main.py verify ...`

├── docs/                   # Documentation

├── tests/                  # Unit tests- **Data augmentation:** Gaussian noise, time warping, scaling (train set only)

├── Dockerfile             # Docker container

├── run.sh                 # Entry script- **Optimizer:** AdamW with cosine annealing scheduler

└── requirements.txt       # Dependencies

```- **Early stopping:** Patience of 30-50 epochs



---- **Class weighting:** Capped at 5.0x to prevent over-correctionThe solution uses a **Hierarchical Two-Stage LSTM Classifier**:- Fully connected classifier head with dropout for regularization



## Quick Start



### Installation### Extra Credit Justification



```bash

# Using uv (recommended)

uv pip install -r requirements.txt1. **Data Acquisition & Labeling Pipeline:** Built a complete labeling workflow using Label Studio with custom export/import scripts for time series data.**Stage 1 - Direction Classification:**- Class-weighted loss function to handle imbalanced label distribution### 2) Export *potential* flags for Label Studio (non-interactive)



# Or using pip

pip install -r requirements.txt

```2. **Multiple Model Architectures:** Implemented and compared 5 different architectures (LSTM, CNN, Transformer, CNN-LSTM, Hierarchical).- Bidirectional LSTM (2 layers, 128 hidden units)



### Training All Models



```bash3. **Advanced Evaluation:** Per-class metrics, confusion matrix, detection rate, and false alarm rate analysis.- Classifies each timestep as: None / Bearish / Bullish

# Train all architectures and compare

uv run python src/train_all_models.py



# Train specific models4. **Window Size Selection:** Used claspy library's SuSS and FFT algorithms for optimal window size determination.- Dropout 0.3 for regularization

uv run python src/train_all_models.py --models lstm_v2 transformer



# Training with custom settings

uv run python src/train_all_models.py --epochs 150 --batch-size 85. **Hierarchical Classification:** Two-stage approach that first detects pattern direction, then classifies subtype.**Training Methodology**:The interactive GUI labeler has been removed.

```



### Training Individual Models

---**Stage 2 - Subtype Classification:**

```bash

# Sequence labeling LSTM

uv run python src/02-training-seq.py

## Docker Instructions- Shared backbone with Stage 1 (frozen during Stage 2 training)- Window size of 256 OHLC bars determined via claspy window size selection (SuSS + FFT)

# Hierarchical classifier

uv run python src/03-training-hierarchical.py

```

### Build- Classifies pattern regions as: Normal / Wedge / Pennant

### Evaluation



```bash

uv run python src/03-evaluation-seq.py```bash- Focal Loss (γ=2.0) for handling class imbalance- Min-max normalization per segmentInstead, use the batch exporter which:

```

docker build -t dl-project .

---

```- Dropout 0.4 for stronger regularization

## Docker Instructions



### Build

### Run- Stratified train/val/test split (70/15/15)

```bash

docker build -t dl-project .

```

Mount your local data directory and capture logs:**Combined Output:** 7-class per-timestep prediction (sequence labeling)

### Run



```bash

# Preprocess data```bash- AdamW optimizer with ReduceLROnPlateau scheduler- scans all CSVs under `data/raw_data/`

docker run -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log dl-project preprocess

docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1

# Train model

docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/log:/app/log dl-project train```### Training Methodology



# Evaluate

docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dl-project evaluate

### Pipeline Modes- Early stopping with patience of 15 epochs- detects *potential* flag candidates

# Run inference

docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models dl-project infer

```

Set the `PIPELINE_MODE` environment variable:- **Window size:** 256 OHLC bars (determined via claspy SuSS + FFT analysis)

---



## Extra Credit Justification

| Mode | Description |- **Normalization:** StandardScaler per sample (OHLC normalization)- writes per-CSV **Label Studio import JSON** files (unlabeled regions)

1. **Data Acquisition & Labeling Pipeline:** Built a complete labeling workflow using Label Studio with custom export/import scripts for time series data.

|------|-------------|

2. **Multiple Model Architectures:** Implemented and compared 6 different architectures (3 LSTM versions, CNN, Transformer, CNN-LSTM, Hierarchical).

| `ALL_MODELS` (default) | Train and compare all 4 model architectures |- **Data augmentation:** Gaussian noise, time warping, scaling (train set only)

3. **Advanced Evaluation:** Per-class metrics, confusion matrix, detection rate, and false alarm rate analysis.

| `HIERARCHICAL` | Train two-stage hierarchical classifier |

4. **Window Size Selection:** Used claspy library's SuSS and FFT algorithms for optimal window size determination.

| `LSTM_V1` | Train basic LSTM model only |- **Optimizer:** AdamW with cosine annealing scheduler**Classes** (6 total):

5. **Hierarchical Classification:** Two-stage approach that first detects pattern direction, then classifies subtype.



6. **Incremental Model Development:** Versioned model iterations (lstm_v1, v2, v3) demonstrating iterative improvement.

Example with specific mode:- **Early stopping:** Patience of 50 epochs

---



## Dependencies

```bash- **Class weighting:** Capped at 5.0x to prevent over-correction1. Bearish Normal - Standard bear flag patternRun:

- PyTorch >= 2.0

- NumPydocker run -e PIPELINE_MODE=ALL_MODELS -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1

- scikit-learn

- matplotlib```

- tqdm

- claspy (window size selection)



------### Evaluation Results2. Bearish Wedge - Bear flag with wedge formation



## License



This project is submitted as coursework for VITMMA19 Deep Learning at BME.## File Structure




```| Metric | Value |3. Bearish Pennant - Bear flag with pennant formation```bash

.

├── src/                              # Source code|--------|-------|

│   ├── 01-data-preprocessing.py      # Data loading, cleaning, preprocessing

│   ├── 02-training.py                # LSTM model training| Stage 1 (Direction) F1 | 57.5% |4. Bullish Normal - Standard bull flag patternuv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import

│   ├── 02-training-seq.py            # Sequence labeling training

│   ├── 03-evaluation.py              # Test set evaluation| Stage 2 (Subtype) F1 | 49.6% |

│   ├── 03-evaluation-seq.py          # Sequence evaluation

│   ├── 03-training-hierarchical.py   # Hierarchical model training| Combined 7-class Accuracy | 65.5% |5. Bullish Wedge - Bull flag with wedge formation```

│   ├── 04-inference.py               # Inference on new data

│   ├── train_all_models.py           # Multi-model training & comparison| Detection Rate | 81.6% |

│   ├── config.py                     # Hyperparameters and paths

│   ├── utils.py                      # Logger and helper functions| False Alarm Rate | 5.2% |6. Bullish Pennant - Bull flag with pennant formation

│   ├── metrics.py                    # Evaluation metrics

│   ├── normalization.py              # OHLC normalization utilities

│   ├── augmentation.py               # Data augmentation functions

│   └── prepare/                      # Data preparation utilities### Extra Credit JustificationOutput example:

│

├── models/                           # Model definitions

│   ├── base.py                       # Abstract base model class

│   ├── seq_lstm/                     # Bidirectional LSTM1. **Data Acquisition & Labeling Pipeline:** Built a complete labeling workflow using Label Studio with custom export/import scripts for time series data, including automated candidate detection.### Extra Credit Justification

│   │   ├── config.py

│   │   └── model.py                  # SeqLabelingLSTM

│   ├── cnn1d/                        # 1D CNN

│   │   ├── config.py2. **Multiple Model Architectures:** Implemented and compared baseline MLP, LSTM, and hierarchical LSTM models with clear architecture separation.- `data/labelstudio_import/XAU_1h_data_limited_potential_flags.json`

│   │   └── model.py                  # CNN1D, SeqCNN1D

│   ├── transformer/                  # Transformer

│   │   ├── config.py

│   │   └── model.py                  # TransformerClassifier, SeqTransformer3. **Advanced Evaluation:** Per-class metrics, confusion matrix, detection rate, and false alarm rate analysis.This project qualifies for extra credit due to:

│   ├── cnn_lstm/                     # CNN-LSTM hybrid

│   │   ├── config.py

│   │   └── model.py                  # CNNLSTM, SeqCNNLSTM

│   ├── hierarchical/                 # Two-stage classifier4. **Window Size Selection:** Used claspy library's SuSS and FFT algorithms for optimal window size determination.Each task contains a proposed `start`/`end` span with `timeserieslabels: []` so you can select one of the 6 classes inside Label Studio.

│   │   ├── config.py

│   │   └── model.py                  # HierarchicalClassifier

│   ├── baseline/                     # MLP baseline

│   └── lstm_v1/                      # Original LSTM5. **Hierarchical Classification:** Two-stage approach that first detects pattern direction, then classifies subtype.1. **Data Acquisition & Labeling Pipeline**: Built a complete labeling workflow using Label Studio with custom export/import scripts for time series data, including automated candidate detection to speed up manual labeling.

│

├── data/                             # Data directory

│   ├── raw_data/                     # OHLC CSV files

│   ├── X_seq.npy                     # Preprocessed sequences---## Notes

│   ├── Y_seq.npy                     # Sequence labels

│   └── metadata_seq.json             # Dataset metadata

│

├── log/                              # Log files## Docker Instructions2. **Incremental Model Development**: Implemented baseline MLP model and LSTM v1 with clear architecture separation in the `models/` directory.

│   └── run.log                       # Training run log

│

├── Dockerfile                        # Container configuration

├── requirements.txt                  # Python dependencies### BuildThis candidate detector is heuristic on purpose. It’s meant to speed up *manual* labeling, not replace it.

├── run.sh                            # Pipeline execution script

└── README.md                         # This file

```

```bash3. **Advanced Evaluation**: Per-class metrics, confusion matrix, and weighted metrics to account for class imbalance.

---

docker build -t dl-project .

## Training Configuration

```4. **Window Size Selection**: Used claspy library's SuSS and FFT algorithms to determine optimal window size rather than arbitrary selection.

### Model Comparison



Run multi-model training:

### Run## Data Preparation

```bash

# Train all models

uv run python src/train_all_models.py

Mount your local data directory and capture logs:The data preparation process involves:

# Train specific models

uv run python src/train_all_models.py --models lstm cnn1d transformer



# Quick test with fewer epochs```bash1. **Raw Data**: OHLC candlestick data for XAU (Gold) at multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h) stored in `data/raw_data/` as CSV files.

uv run python src/train_all_models.py --epochs 50

```docker run -v /absolute/path/to/your/data:/app/data dl-project > log/run.log 2>&1



### Individual Model Training```2. **Labeling**: Patterns were labeled using Label Studio. The verified labels are stored in `data/cleaned_labels_merged.json`.



```bash

# LSTM

uv run python src/02-training-seq.pyExample with the project data:3. **Preprocessing** (`src/01-data-preprocessing.py`):



# Hierarchical   - Loads OHLC CSVs and labeled regions from JSON

uv run python src/03-training-hierarchical.py

``````bash   - Extracts fixed 256-bar windows centered on labeled patterns



### Hyperparametersdocker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1   - Normalizes each window to [0, 1] range



| Parameter | LSTM | CNN1D | Transformer | CNN-LSTM |```   - Saves as `X.npy` (samples, 256, 4) and `Y.npy` (labels)

|-----------|------|-------|-------------|----------|

| Hidden/Channels | 128 | 32→256 | 64 | 32→64 + 64 |

| Layers | 2 | 4 | 3 | 2 CNN + 2 LSTM |

| Dropout | 0.3 | 0.3 | 0.2 | 0.2-0.4 |The container executes the full pipeline:**To prepare data for training**:

| Learning Rate | 0.001 | 0.001 | 0.0005 | 0.001 |

| Epochs | 200 | 200 | 200 | 200 |1. Data preprocessing (`src/01-data-preprocessing.py`)```bash



---2. Model training (`src/02-training.py`)# Data files should be placed in:



## Usage3. Evaluation (`src/03-evaluation.py`)# - data/raw_data/*.csv (OHLC data)



### Training (Docker)4. Inference demo (`src/04-inference.py`)# - data/cleaned_labels_merged.json (labels)



```bash```

# Build and run full pipeline

docker build -t dl-project .---

docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1

```## Docker Instructions



### Training (Local)## File Structure



```bash### Build

# Install dependencies

uv pip install -r requirements.txt```



# Preprocess data.```bash

uv run python src/01-data-preprocessing.py

├── src/                              # Source codedocker build -t dl-project .

# Train all models

uv run python src/train_all_models.py│   ├── 01-data-preprocessing.py      # Data loading, cleaning, preprocessing```



# Or train individual model│   ├── 02-training.py                # Model training with logging

uv run python src/02-training-seq.py

```│   ├── 02-training-seq.py            # Sequence labeling training### Run



### Inference│   ├── 03-evaluation.py              # Test set evaluation with metrics



```bash│   ├── 03-evaluation-seq.py          # Sequence evaluationMount your local data directory and capture logs:

uv run python src/04-inference.py

```│   ├── 03-training-hierarchical.py   # Hierarchical model training



---│   ├── 04-inference.py               # Inference on new data```bash



## Requirements│   ├── config.py                     # Hyperparameters and pathsdocker run -v /absolute/path/to/your/data:/app/data dl-project > log/run.log 2>&1



See `requirements.txt` for full dependencies. Key packages:│   ├── utils.py                      # Logger and helper functions```



- Python 3.10+│   ├── metrics.py                    # Evaluation metrics

- PyTorch 2.3+

- NumPy, Pandas, Scikit-learn│   ├── normalization.py              # OHLC normalization utilitiesExample with the project data:

- Matplotlib, Seaborn (visualization)

- claspy (window size selection)│   ├── augmentation.py               # Data augmentation functions```bash



---│   ├── prepare/                      # Data preparation utilitiesdocker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1



## License│   │   ├── prepare_data.py           # Segment extraction script```



This project is for educational purposes as part of the VITMMA19 Deep Learning course at BME.│   │   ├── prepare_data_sequence.py  # Sequence data preparation


│   │   └── window_size_selection.py  # claspy window analysisThe container executes the full pipeline:

│   └── bullflag_detector/            # Detection and labeling tools1. Data preprocessing

│       ├── flag_detector.py          # Heuristic flag detection2. Model training

│       ├── export_labelstudio.py     # Label Studio export3. Evaluation

│       └── candidate_gui.py          # Candidate verification GUI4. Inference demo

│

├── models/                           # Model definitions## File Structure

│   ├── base.py                       # Abstract base model class

│   ├── best_model.pth                # Best trained model checkpoint```

│   ├── baseline/                     # MLP baseline model.

│   │   ├── config.py                 # Baseline hyperparameters├── src/                          # Source code

│   │   └── model.py                  # MLPBaseline implementation│   ├── 01-data-preprocessing.py  # Data loading, cleaning, preprocessing

│   ├── lstm_v1/                      # Bidirectional LSTM v1│   ├── 02-training.py            # Model training with logging

│   │   ├── config.py                 # LSTM v1 hyperparameters│   ├── 03-evaluation.py          # Test set evaluation with metrics

│   │   └── model.py                  # LSTMv1 implementation│   ├── 04-inference.py           # Inference on new data

│   ├── seq_lstm/                     # Sequence labeling LSTM│   ├── config.py                 # Hyperparameters and paths

│   │   ├── config.py                 # Seq LSTM hyperparameters│   ├── utils.py                  # Logger and helper functions

│   │   └── model.py                  # SeqLabelingLSTM implementation│   ├── prepare/                  # Data preparation utilities

│   ├── hierarchical/                 # Hierarchical 2-stage classifier│   │   ├── prepare_data.py       # Segment extraction script

│   │   ├── config.py                 # Stage 1 & Stage 2 configs│   │   └── window_size_selection.py  # claspy window analysis

│   │   ├── model.py                  # HierarchicalClassifier│   └── bullflag_detector/        # Detection and labeling tools

│   │   ├── stage1_best.pth           # Stage 1 checkpoint│

│   │   └── stage2_best.pth           # Stage 2 checkpoint├── models/                       # Model definitions

│   └── statistical_baseline/         # Non-DL baseline│   ├── base.py                   # Abstract base model class

│       ├── config.py│   ├── baseline/                 # MLP baseline model

│       └── model.py│   │   ├── __init__.py

││   │   ├── config.py             # Baseline hyperparameters

├── notebooks/                        # Jupyter notebooks (EDA, analysis)│   │   └── model.py              # MLPBaseline implementation

││   └── lstm_v1/                  # Bidirectional LSTM v1

├── data/                             # Data directory (mounted in Docker)│       ├── __init__.py

│   ├── raw_data/                     # OHLC CSV files│       ├── config.py             # LSTM v1 hyperparameters

│   │   └── XAU_*.csv                 # Gold price data at various timeframes│       └── model.py              # LSTMv1 implementation

│   ├── cleaned_labels_merged.json    # Verified labels from Label Studio│

│   ├── X_seq.npy                     # Preprocessed sequences (samples, 256, 4)├── notebooks/                    # Jupyter notebooks (analysis)

│   ├── Y_seq.npy                     # Sequence labels (samples, 256)│

│   └── metadata_seq.json             # Dataset metadata├── data/                         # Data directory (mounted in Docker)

││   ├── raw_data/                 # OHLC CSV files

├── log/                              # Log files│   ├── cleaned_labels_merged.json  # Verified labels

│   └── run.log                       # Training run log (Docker output)│   ├── X.npy                     # Preprocessed features

││   ├── Y.npy                     # Preprocessed labels

├── tests/                            # Unit tests│   └── metadata.json             # Dataset metadata

│   ├── test_flag_detector_candidates.py│

│   └── test_candidate_json_writer.py├── log/                          # Log files

││   └── run.log                   # Training run log

├── docs/                             # Documentation│

│   └── requirements.md               # Project requirements├── Dockerfile                    # Container configuration

│├── requirements.txt              # Python dependencies

├── Dockerfile                        # Container configuration├── run.sh                        # Pipeline execution script

├── requirements.txt                  # Python dependencies└── README.md                     # This file

├── pyproject.toml                    # Project configuration```

├── run.sh                            # Pipeline execution script

└── README.md                         # This file## Training Configuration

```

Key hyperparameters (see `src/config.py`):

---

| Parameter | Value | Description |

## Training Configuration|-----------|-------|-------------|

| EPOCHS | 100 | Maximum training epochs |

Key hyperparameters (see `src/config.py` and `models/hierarchical/config.py`):| BATCH_SIZE | 32 | Samples per batch |

| LEARNING_RATE | 0.001 | Initial learning rate |

### General Settings| WINDOW_SIZE | 256 | OHLC bars per sample |

| LSTM_HIDDEN_SIZE | 128 | LSTM hidden dimension |

| Parameter | Value | Description || LSTM_NUM_LAYERS | 2 | Number of LSTM layers |

|-----------|-------|-------------|| EARLY_STOPPING_PATIENCE | 15 | Epochs without improvement |

| WINDOW_SIZE | 256 | OHLC bars per sample |

| BATCH_SIZE | 16 | Samples per batch |## Labeling Tools

| RANDOM_SEED | 42 | Reproducibility seed |

The project includes tools for creating and verifying labels:

### Hierarchical Model

```bash

| Parameter | Stage 1 | Stage 2 |# Export potential flag candidates for Label Studio

|-----------|---------|---------|uv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import

| Hidden Size | 128 | 128 |

| Num Layers | 2 | 2 |# Verify labels with TUI

| Bidirectional | Yes | Yes |python main.py verify --labels data/cleaned_labels_merged.json --tui

| Dropout | 0.3 | 0.4 |```

| Learning Rate | 0.0003 | 0.0003 |
| Epochs | 300 | 200 |
| Patience | 50 | 50 |
| Loss | CrossEntropy | Focal Loss (γ=2.0) |
| Label Smoothing | 0.1 | 0.15 |

---

## Data Preparation

### Raw Data
OHLC candlestick data for XAU (Gold) at multiple timeframes:
- 1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour, 4 hours

### Labeling Workflow

1. **Export candidates for Label Studio:**
   ```bash
   uv run bullflag-export-labelstudio --data-dir data --out-dir data/labelstudio_import
   ```

2. **Label in Label Studio** using the provided XML configuration (see `docs/requirements.md`)

3. **Verify labels:**
   ```bash
   uv run python main.py verify data/cleaned_labels_merged.json data/raw_data --tui
   ```

### Preprocessing

```bash
# Prepare sequence data
uv run python src/01-data-preprocessing.py

# Train hierarchical model
uv run python src/03-training-hierarchical.py

# Evaluate
uv run python src/03-evaluation-seq.py
```

---

## Usage

### Training

```bash
# Full pipeline via Docker
docker build -t dl-project .
docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1

# Local training with uv
uv run python src/02-training-seq.py      # Sequence LSTM
uv run python src/03-training-hierarchical.py  # Hierarchical model
```

### Inference

```bash
uv run python src/04-inference.py
```

---

## Requirements

See `requirements.txt` for full dependencies. Key packages:

- Python 3.10+
- PyTorch 2.3+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)
- claspy (window size selection)

Install with:

```bash
uv pip install -r requirements.txt
```

Or using the pyproject.toml:

```bash
uv pip install -e .
```

---

## License

This project is for educational purposes as part of the VITMMA19 Deep Learning course at BME.
