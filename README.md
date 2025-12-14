# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Rózsa Botond

### Solution Description

This project implements a deep learning system for detecting **bull flag** and **bear flag** patterns in financial OHLC (Open, High, Low, Close) time series data. These chart patterns are commonly used in technical analysis to predict price breakouts. This project automates the detection using multiple neural network architectures:

- **LSTM_v2 (SeqLabeling)**: A recurrent neural network designed for sequence labeling, predicting a label for each timestep.
- **Hierarchical LSTM**: A two-stage model that first classifies a broad pattern type (e.g., bullish/bearish) and then refines the classification.
- **Transformer**: A self-attention-based model capable of capturing long-range dependencies in time series data.
- **Statistical Baseline**: A simple statistical model for comparison.

**Iterative Development & Model History**:
The `models/` directory reflects the iterative development process of the project. It contains a history of various architectures experimented with, starting from initial baselines (`baseline`), moving through experimental phases (`cnn1d`, `cnn_lstm`), and culminating in the refined core models (`hierarchical_v1`, `lstm_v2`, `transformer`). This structure allows for transparent benchmarking and highlights the evolution of the solution from simple to complex architectures.

**Hyperparameter Optimization (HPO)**:
To maximize model performance, extensive hyperparameter optimization was conducted using **Optuna**. We employed the **Tree-structured Parzen Estimator (TPE)** sampler, a Bayesian optimization algorithm that efficiently explores the hyperparameter space by modeling the probability of improving the objective function (Validation F1-score).
Key parameters optimized include:

- **Architecture**: `hidden_size`, `num_layers`, `nhead` (Transformer), `dropout` rates.
- **Training**: `learning_rate`, `batch_size`, `weight_decay`.
- **Model-Specific**: `grad_clip` (LSTM), `dim_feedforward` (Transformer).

The optimal hyperparameters found via these search processes were then hardcoded into the final training configuration in `src/training.py` (and `BEST_HPARAMS` dictionary), ensuring the final models achieve the best possible results.

### Data Preparation

The project handles OHLC (Open, High, Low, Close) data for XAU (Gold) across various timeframes. The raw data (CSV files) are located in `data/raw_data/`.

**Preprocessing Steps (`src/data_preprocessing.py`):**

1. **Labeling**: Raw data is labeled using Label Studio (exported to `data/cleaned_labels_merged.json`).
2. **Segment Extraction**: Fixed-size OHLC segments (`WINDOW_SIZE=256`) are extracted, centered on labeled patterns.
3. **Output**: Processed data is saved as `X_seq.npy` (OHLC segments) and `Y_seq.npy` (labels).
4. **Train/Val/Test Split**: Data is split and saved as `X_train.npy`, `Y_train.npy`, etc., in the `data/` directory.

**Augmentation (`src/utils/augmentation.py`):**

- **TimeSeriesAugmenter**: Applies Gaussian noise, time warping, and scaling.
- **balance_dataset_with_augmentation**: Oversamples minority classes during training to address class imbalance.

**Normalization (`src/utils/normalization.py`):**

- **OHLCScaler**: A global scaler is fitted on training data and applied consistently to training, validation, and test sets.

### Logging Requirements

The training process produces a log file that captures the following essential information:

1. **Configuration**: Prints the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2. **Data Processing**: Confirms successful data loading and preprocessing steps.
3. **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4. **Training Progress**: Logs the loss and accuracy (or other relevant metrics) for each epoch.
5. **Validation**: Logs validation metrics at the end of each epoch or at specified intervals.
6. **Final Evaluation**: Results of the evaluation on the test set (e.g., final accuracy, F1-score, detection rate, false alarm rate).

The log file must be uploaded to `log/run.log` to the repository. `src/utils/utils.py` is used to configure the logger so that output is directed to stdout .

## Installation

### Dependencies

See `requirements.txt` for full dependencies. Key packages include:

- Python 3.10+
- PyTorch 2.3+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- claspy
- Optuna (for Hyperparameter Optimization)
- uv (for dependency management and running scripts)

### Local Setup (using `uv`)

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/VITMMA19.git
    cd VITMMA19
    ```

2. **Install `uv`**:
    Follow instructions on the `uv` GitHub page or run:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install dependencies**:
    `uv` will create a virtual environment and install dependencies based on `uv.lock` and `requirements.txt`.

    ```bash
    uv sync
    ```

4. **Set `PYTHONPATH`**:
    Ensure the project root is in your `PYTHONPATH` for correct module imports:

    ```bash
    export PYTHONPATH=$(pwd):$PYTHONPATH
    ```

    (You might want to add this to your shell profile for persistence.)

## Usage

All main pipeline stages are run via `src/main.py` followed by a subcommand. For local execution, it's recommended to use `uv run`.

### Full Pipeline Execution (`run.sh`)

The `run.sh` script is the primary entry point for executing the entire pipeline in sequence. This is the script used by the Docker container.
It performs the following steps automatically:

1. **Preprocessing**: Cleans and prepares the data.
2. **Training**: Trains the configured models (LSTM, Hierarchical, Transformer, Baseline).
3. **Evaluation**: Evaluates the best-performing model on the test set.
4. **Inference**: Demonstrates inference on a sample data point.

To run it locally:

```bash
./run.sh
```

### Running Data Preprocessing

```bash
uv run python src/main.py preprocess
```

### Running Model Training

You can train all configured models or specific ones.

```bash
# Train all configured models (LSTM, Hierarchical, Transformer, Baseline)
uv run python src/main.py train

# Train specific models (e.g., Transformer and Hierarchical LSTM)
uv run python src/main.py train --models transformer hierarchical_v1 --epochs 50 --log-level INFO
```

### Running Model Evaluation

```bash
# Evaluate a specific model (e.g., LSTM_v2)
uv run python src/main.py evaluate --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth
```

# Run inference with a specific model (e.g., LSTM_v2)

```bash
uv run python src/main.py infer --model-key lstm_v2 --checkpoint-path models/lstm_v2/best_model.pth

```

### Running Hyperparameter Optimization (`hpo.sh`)

I provided a dedicated script `hpo.sh` to run the hyperparameter optimization process for all key models. This script executes the Optuna-based Bayesian optimization for the LSTM, Hierarchical LSTM, and Transformer models sequentially. It automatically saves the optimization logs to the `log/` directory.

To run the full HPO suite:

```bash
./hpo.sh

```

You can also run optimization for individual models manually:

```bash
# Run HPO for Hierarchical LSTM (example)
uv run python src/hpo/bayesian_hp_hierarchical.py --n-trials 50 | tee log/hpo_hierarchical_run.log

# Run HPO for Transformer (example)
uv run python src/hpo/bayesian_hp_transformer.py --n-trials 50 | tee log/hpo_transformer_run.log
```

## Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the full pipeline within the Docker container, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

- Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine. This directory should contain `raw_data/` and other processed data if not generated by the container.
- The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
- The container's default command (`/app/run.sh`) executes the entire pipeline (preprocessing, training, evaluation, inference) for all configured models.

## File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
  - `data_preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
  - `training.py`: The main script for defining models, executing training loops, and comparing multiple architectures.
  - `evaluation.py`: Scripts for evaluating trained models on test data and generating metrics.
  - `inference.py`: Script for running models on new, unseen data to generate predictions.
  - `main.py`: The central command-line interface (CLI) for orchestrating pipeline stages.
  - `utils/`: Helper functions and utilities, including `config.py` (hyperparameters and paths), `normalization.py`, `augmentation.py`, and `utils.py` (logger setup).
  - `hpo/`: Contains hyperparameter optimization scripts (`bayesian_hp_hierarchical.py`, `bayesian_hp_transformer.py`, etc.).
  - `models/`: Defines the various model architectures (e.g., `lstm_v2/`, `transformer/`, `hierarchical_v1/`, `baseline/`).
- **`notebooks/`**: Contains Jupyter notebooks for analysis and experimentation.
  - `visualize_segments.ipynb`: Notebook for visualizing training data segments for each class.
- **`log/`**: Contains log files.
  - `run.log`: Example log file showing the output of a successful training run.
  - `comparison_results_*.json`: JSON files with detailed comparison results.
  - `hpo_*.log`: Logs from hyperparameter optimization runs.
- **Root Directory**:
  - `Dockerfile`: Configuration file for building the Docker image.
  - `pyproject.toml`: Project configuration for `uv`.
  - `uv.lock`: Locked dependencies for `uv`.
  - `requirements.txt`: List of Python dependencies (used by `uv`).
  - `run.sh`: Shell script to execute the full pipeline.
  - `hpo.sh`: Shell script to execute the hyperparameter optimization suite.
  - `README.md`: This project documentation.

## License

This project is for educational purposes as part of the VITMMA19 Deep Learning course at BME.
