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
│   └── ... (preprocessed data)
├── models/
│   ├── baseline/
│   ├── cnn1d/
│   ├── cnn_lstm/
│   ├── hierarchical_v1/
│   ├── lstm_v2/
│   └── transformer/
├── src/
│   ├── bullflag_detector/
│   ├── cli/
│   ├── data/
│   ├── prepare/
│   ├── training/
│   └── utils/
├── notebooks/
├── log/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── ...
```

- **`src/`**: Contains the source code for the machine learning pipeline.
  - **`cli/`**: The main command-line interface for interacting with the project.
  - **`data/`**: Scripts for data loading, cleaning, and preprocessing.
  - **`training/`**: Scripts for defining and executing model training.
  - **`utils/`**: Helper functions and utilities, including logging and configuration.
- **`notebooks/`**: Contains Jupyter notebooks for analysis and experimentation.
- **`log/`**: Contains log files.
- **`Dockerfile`**: Configuration for building the Docker image.
- **`requirements.txt`**: Python dependencies.
- **`README.md`**: This file.

## Data Pipeline and Augmentation

### 1. Data Source

The raw data for this project consists of OHLC (Open, High, Low, Close) candlestick data for XAU (Gold) at multiple timeframes (1m, 5m, 15m, 30m, 1h, 4h). This data is stored in CSV files in the `data/raw_data/` directory.

### 2. Labeling

The raw data is labeled using Label Studio, an open-source data labeling tool. The labeled data is exported as a JSON file, which contains information about the start and end times of each pattern, as well as the pattern type. The verified labels are stored in `data/cleaned_labels_merged.json`.

### 3. Data Preprocessing (`src/prepare/prepare_data.py`)

The `prepare_data.py` script is the first step in the data pipeline. It takes the raw CSV data and the labeled JSON file as input and performs the following steps:

-   **Segment Extraction:** For each labeled pattern, the script extracts a fixed-size segment of the OHLC data. The `WINDOW_SIZE` parameter in `src/utils/config.py` determines the size of this segment (currently 256). The segment is centered on the labeled pattern, which ensures that the pattern is present in the extracted data, along with some context before and after it (padding).
-   **Normalization:** Each extracted segment is normalized to a range of [0, 1] using min-max scaling. This is done on a per-segment basis, which helps the model learn the shape of the pattern regardless of the price level.
-   **Output:** The script saves the processed data as two NumPy arrays: `X.npy` (containing the OHLC segments) and `Y.npy` (containing the corresponding labels). It also saves a `metadata.json` file with information about the processed data.

### 4. Data Augmentation (`src/data/augmentation.py`)

To increase the size and diversity of the training dataset, data augmentation is applied to the training set. The `TimeSeriesAugmenter` class in `src/data/augmentation.py` performs the following augmentations:

-   **Gaussian Noise:** Adds random noise to the time series data, which makes the model more robust to noisy input.
-   **Time Warping:** Randomly distorts the time axis of the data, which helps the model learn to recognize patterns at different speeds.
-   **Scaling:** Randomly scales the magnitude of the data, which makes the model more robust to changes in volatility.

The `balance_dataset_with_augmentation` function is used to oversample minority classes by creating augmented copies of the samples in those classes. This helps to prevent the model from being biased towards the majority class.

### 5. Data Loading and Batching

The `prepare_dataloaders` function in `src/training/train_all_models.py` takes the preprocessed and augmented data and prepares it for training. It performs the following steps:

-   **Train/Validation/Test Split:** The data is split into training, validation, and test sets. The split can be done either by using pre-defined indices from the metadata or by creating a new stratified split.
-   **Tensor Conversion:** The NumPy arrays are converted to PyTorch tensors.
-   **Dataset and DataLoader Creation:** The tensors are used to create PyTorch `TensorDataset` and `DataLoader` objects, which are used to efficiently load and batch the data during training.


## Docker Instructions

### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

To capture the logs for submission, redirect the output to a file:

```bash
docker run -v $(pwd)/data:/app/data dl-project > log/run.log 2>&1
```

The container is configured to run the full pipeline (preprocessing, training, evaluation).

## Local Usage

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

### Training

To train the models, you can use the `train_all_models.py` script:

```bash
# With uv, to train all models
uv run python src/training/train_all_models.py

# With a regular venv, to train specific models
python src/training/train_all_models.py --models lstm_v2 transformer
```

## Logging Requirements

The training process produces a log file (`log/run.log`) that captures the following information:

- **Configuration**: Hyperparameters used (e.g., number of epochs, batch size, learning rate).
- **Data Processing**: Confirmation of successful data loading and preprocessing.
- **Model Architecture**: A summary of the model structure with the number of parameters.
- **Training Progress**: Loss and accuracy for each epoch.
- **Validation**: Validation metrics at the end of each epoch.
- **Final Evaluation**: Results on the test set.

## Dependencies

See `requirements.txt` for full dependencies. Key packages:

- Python 3.10+
- PyTorch 2.3+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- claspy

## License

This project is for educational purposes as part of the VITMMA19 Deep Learning course at BME.
