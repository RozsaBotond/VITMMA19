# Deep Learning Class (VITMMA19) Project Work

## Project Information

| Field                  | Value              |
| ---------------------- | ------------------ |
| **Selected Topic**     | Bull-flag detector |
| **Student Name**       | Rozsa              |
| **Aiming for +1 Mark** | Yes                |

## Solution Description

This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts. This project automates the detection using multiple neural network architectures.

## Project Structure

The repository is structured as follows:

```
VITMMA19/
├── data/
│   ├── raw_data/
│   └── ... (preprocessed data, e.g., X_train.npy, metadata_seq.json)
├── models/
│   ├── baseline/
│   ├── cnn1d/
│   ├── cnn_lstm/
│   ├── hierarchical_v1/
│   ├── lstm_v2/
│   └── transformer/
├── src/
│   ├── data_preprocessing.py       # Data loading, cleaning, preprocessing, splitting
│   ├── training.py                 # Main script for training and comparison
│   ├── evaluation.py               # Script for evaluating a trained model
│   ├── inference.py                # Script for running inference
│   ├── main.py                     # Main CLI entry point
│   ├── utils/                      # Utility functions, config, normalization, augmentation
│   ├── hpo/                        # Hyperparameter optimization scripts
│   └── ... (other modules like models/*)
├── notebooks/
├── log/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── ...
```

- **`src/`**: Contains the core source code for the machine learning pipeline.
  - **`data_preprocessing.py`**: Handles initial data loading, feature extraction, and train/validation/test splitting.
  - **`training.py`**: The main script for training and comparing multiple model architectures using best hyperparameters.
  - **`evaluation.py`**: Used to evaluate a single, already trained model on the test set.
  - **`inference.py`**: Demonstrates how to load a trained model and make predictions on new data.
  - **`main.py`**: The central command-line interface for running different stages of the pipeline.
  - **`utils/`**: Contains helper functions, configuration settings (`config.py`), data normalization (`normalization.py`), and data augmentation (`augmentation.py`) utilities.
  - **`hpo/`**: Contains specialized scripts for hyperparameter optimization for different models.
- **`notebooks/`**: Contains Jupyter notebooks for analysis and experimentation.
- **`log/`**: Contains log files generated during training and evaluation.
- **`tests/`**: Contains unit tests.
- **`Dockerfile`**: Configuration for building the Docker image.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: This file.

## Data Pipeline and Augmentation

### 1. Data Source

The raw data for this project consists of OHLC (Open, High, Low, Close) candlestick data for XAU (Gold) at multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h). This data is stored in CSV files in the `data/raw_data/` directory.

### 2. Labeling

The raw data is labeled using Label Studio, an open-source data labeling tool. The labeled data is exported as a JSON file, which contains information about the start and end times of each pattern, as well as the pattern type. The verified labels are stored in `data/cleaned_labels_merged.json`.

### 3. Data Preprocessing (`src/data_preprocessing.py`)

The `src/data_preprocessing.py` script is the first step in the data pipeline. It takes the raw CSV data and the labeled JSON file as input and performs the following steps:

-   **Segment Extraction:** For each labeled pattern, the script extracts a fixed-size segment of the OHLC data. The `WINDOW_SIZE` parameter in `src/utils/config.py` determines the size of this segment (currently 256). The segment is centered on the labeled pattern, which ensures that the pattern is present in the extracted data, along with some context before and after it (padding).
-   **Normalization (per segment):** Each extracted segment is normalized to a range of [0, 1] using min-max scaling, applied by `src/utils/normalization.py::normalize_window`. This is done on a per-segment basis, helping the model learn the pattern shape regardless of price level.
-   **Output:** The script saves the processed data as two NumPy arrays: `X_seq.npy` (containing the OHLC segments) and `Y_seq.npy` (containing the corresponding labels). It then performs a train/validation/test split and saves these split datasets as `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy`, `X_test.npy`, `Y_test.npy` in the `data/` directory. Metadata including split indices is stored in `metadata_seq.json`.

### 4. Data Augmentation (`src/utils/augmentation.py`)

To increase the size and diversity of the training dataset, data augmentation is applied to the training set within `src/training.py`. The `TimeSeriesAugmenter` class in `src/utils/augmentation.py` performs the following augmentations:

-   **Gaussian Noise:** Adds random noise to the time series data, making the model more robust to noisy input.
-   **Time Warping:** Randomly distorts the time axis of the data, helping the model recognize patterns at different speeds.
-   **Scaling:** Randomly scales the magnitude of the data, making the model more robust to changes in volatility.

The `balance_dataset_with_augmentation` function from `src/utils/augmentation.py` is used to oversample minority classes by creating augmented copies of the samples in those classes. This helps to prevent the model from being biased towards the majority class.

### 5. Data Loading and Batching (within `src/training.py` and `src/evaluation.py`)

The `load_split_data` and `prepare_dataloaders` functions within `src/training.py` (and adapted for `src/evaluation.py`) load the preprocessed and split data from the `data/` directory. These functions perform:

-   **Loading Split Data:** Loads `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy`, `X_test.npy`, `Y_test.npy`.
-   **Normalization (StandardScaler):** A global `OHLCScaler` (from `src/utils/normalization.py`) is fitted on the training data (`X_train`) and then used to transform the training, validation, and test sets. This ensures consistent scaling across all datasets.
-   **Dataset and DataLoader Creation:** The preprocessed (and augmented for training) data is used to create PyTorch `TensorDataset` and `DataLoader` objects, which efficiently load and batch the data during model training and evaluation.

## Docker Instructions

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

### Run

To run the full pipeline within the Docker container, use the following command. You must mount your local `data/` directory to `/app/data` inside the container.

To capture the logs for submission, redirect the output to a file:

```bash
docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1
```

The container's `main.py` entry point is configured to execute the entire pipeline (preprocessing, training, evaluation) based on internal logic or specific commands.

## Local Usage

To use the CLI, ensure the `src` directory is in your `PYTHONPATH`. From the project root, you can set it temporarily: `export PYTHONPATH=$PYTHONPATH:$(pwd)/src`

### Installation (uv)

```bash
# Install dependencies using uv
uv pip install -r requirements.txt
```

### Installation (venv)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Commands via `src/main.py`

All main pipeline stages are run via `src/main.py` followed by the subcommand.

```bash
# Example: Run data preprocessing
uv run python src/main.py preprocess

# Example: Train all configured models
uv run python src/main.py train

# Example: Train specific models (LSTM and Transformer)
uv run python src/main.py train --models lstm_v2 transformer --epochs 100 --log-level INFO

# Example: Evaluate a specific model
uv run python src/main.py evaluate --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth

# Example: Run inference with a specific model
uv run python src/main.py infer --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth
```

### Running Hyperparameter Optimization

The hyperparameter optimization scripts are located in `src/hpo/`. You can run them individually using `uv run python src/hpo/<script_name>.py`.

```bash
# Example: Run HPO for LSTM
uv run python src/hpo/bayesian_hp_lstm.py --n-trials 50 --log-file log/hpo_lstm_run.log
```

## Logging Requirements

The training and evaluation processes produce log files (e.g., `log/run.log`) that capture the following information:

-   **Configuration**: Hyperparameters used (e.g., number of epochs, batch size, learning rate).
-   **Data Processing**: Confirmation of successful data loading and preprocessing steps.
-   **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
-   **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
-   **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
-   **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log files are stored in the `log/` directory.

## Dependencies

See `requirements.txt` for full dependencies. Key packages:

-   Python 3.10+
-   PyTorch 2.3+
-   NumPy, Pandas, Scikit-learn
-   Matplotlib, Seaborn
-   claspy
-   Optuna (for Hyperparameter Optimization)

## License

This project is for educational purposes as part of the VITMMA19 Deep Learning course at BME.
