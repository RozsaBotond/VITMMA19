"""Bayesian Hyperparameter Optimization using Optuna.

Fine-tunes LSTM v2e (best performing model) using Bayesian optimization.
Uses Tree-structured Parzen Estimator (TPE) sampler with pruning.

Usage:
    python src/hpo/bayesian_hp_lstm.py --n_trials 50
    python src/hpo/bayesian_hp_lstm.py --n_trials 100 --timeout 3600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)

# Add project root to path
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.utils.config import AppConfig
from src.utils.utils import setup_logger, set_seed, get_device
from src.utils.normalization import OHLCScaler
from src.utils.augmentation import (
    TimeSeriesAugmenter,
    balance_dataset_with_augmentation,
)
from src.models.lstm_v2.model import SeqLabelingLSTM

logger = setup_logger("bayesian_hp_lstm")


def load_and_prepare_data(
    config: AppConfig, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare data for training."""

    X = np.load(config.data_dir / "X_seq.npy")
    Y = np.load(config.data_dir / "Y_seq.npy")
    logger.info(f"Loaded data: X={X.shape}, Y={Y.shape}")

    n_samples = len(X)
    dominant_labels = [
        np.argmax(np.bincount(y[y > 0])) if (y > 0).any() else 0 for y in Y
    ]
    dominant_labels = np.array(dominant_labels)

    indices = np.arange(n_samples)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=dominant_labels
    )
    temp_labels = dominant_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    scaler = OHLCScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    augmenter = TimeSeriesAugmenter(seed=42)
    X_train_aug, Y_train_aug = balance_dataset_with_augmentation(
        X_train_norm, Y_train, augmenter=augmenter, seed=42
    )
    logger.info(f"Augmented: {len(X_train_aug)} train samples")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_aug, dtype=torch.float32),
            torch.tensor(Y_train_aug, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.long),
        ),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_norm, dtype=torch.float32),
            torch.tensor(Y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
    )

    return train_loader, val_loader, test_loader


class Objective:
    """Optuna objective function for LSTM hyperparameter optimization."""

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        max_epochs: int = 100,
    ):
        self.config = config
        self.device = device
        self.max_epochs = max_epochs

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate a single trial."""

        hparams = {
            "hidden_size": trial.suggest_categorical(
                "hidden_size", [32, 48, 64, 96, 128, 192, 256]
            ),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.05),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
            "scheduler": trial.suggest_categorical(
                "scheduler", ["cosine", "one_cycle", "reduce_on_plateau", "step"]
            ),
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0, step=0.5),
            "use_class_weights": trial.suggest_categorical(
                "use_class_weights", [False, True]
            ),
            "class_weight_scale": (
                trial.suggest_float("class_weight_scale", 0.1, 1.0, step=0.1)
                if trial.params.get("use_class_weights")
                else 1.0
            ),
        }

        train_loader, val_loader, _ = load_and_prepare_data(
            self.config, hparams["batch_size"]
        )

        model = SeqLabelingLSTM(
            input_size=4,
            hidden_size=hparams["hidden_size"],
            num_layers=hparams["num_layers"],
            num_classes=7,
            dropout=hparams["dropout"],
            bidirectional=hparams["bidirectional"],
        ).to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        if hparams["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.max_epochs, eta_min=1e-6
            )
        elif hparams["scheduler"] == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=hparams["learning_rate"],
                epochs=self.max_epochs,
                steps_per_epoch=len(train_loader),
            )
        elif hparams["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
            )

        criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        patience_counter = 0
        patience = 15

        for epoch in range(self.max_epochs):
            model.train()
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.reshape(-1, 7), Y_batch.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
                optimizer.step()
                if hparams["scheduler"] == "one_cycle":
                    scheduler.step()

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    logits = model(X_batch.to(self.device))
                    preds = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
                    val_preds.extend(preds)
                    val_targets.extend(Y_batch.cpu().numpy().flatten())

            val_f1 = f1_score(val_targets, val_preds, average="weighted")

            if hparams["scheduler"] in ["cosine", "step"]:
                scheduler.step()
            elif hparams["scheduler"] == "reduce_on_plateau":
                scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HP Optimization with Optuna for LSTM"
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = AppConfig()
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=HyperbandPruner(
            min_resource=10, max_resource=args.epochs, reduction_factor=3
        ),
    )

    objective = Objective(config, device, args.epochs)

    study.optimize(
        objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()