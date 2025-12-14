"""Bayesian Hyperparameter Optimization for Transformer (SeqTransformer) using Optuna.

Usage:
    python src/hpo/bayesian_hp_transformer.py --n-trials 50
    python src/hpo/bayesian_hp_transformer.py --n-trials 100 --epochs 80
"""

from __future__ import annotations

import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


from src.utils.config import AppConfig
from src.utils.utils import setup_logger, get_device
from src.utils.normalization import OHLCScaler
from src.utils.augmentation import (
    TimeSeriesAugmenter,
    balance_dataset_with_augmentation,
)
from models.transformer.model import SeqTransformer

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Please install optuna: pip install optuna")
    sys.exit(1)


logger = logging.getLogger(__name__)


def load_and_prepare_data(config: AppConfig, batch_size: int = 16):
    X = np.load(config.data_dir / "X_seq.npy")
    Y = np.load(config.data_dir / "Y_seq.npy")
    # Stratified split based on dominant label
    n_samples = len(X)
    dominant_labels = []
    for y in Y:
        non_zero_mask = y > 0
        if non_zero_mask.sum() > 0:
            unique, counts = np.unique(y[non_zero_mask], return_counts=True)
            dominant_labels.append(unique[np.argmax(counts)])
        else:
            dominant_labels.append(0)
    dominant_labels = np.array(dominant_labels)
    from sklearn.model_selection import train_test_split

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

    scaler = OHLCScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    augmenter = TimeSeriesAugmenter(seed=42)
    X_train_aug, Y_train_aug = balance_dataset_with_augmentation(
        X_train_norm, Y_train, augmenter=augmenter, seed=42
    )

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

    return train_loader, val_loader


class Objective:
    def __init__(self, config: AppConfig, device, max_epochs=80):
        self.config = config
        self.device = device
        self.max_epochs = max_epochs

    def __call__(self, trial):
        hparams = {
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 6),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward", [128, 256, 512]
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
            "learnable_pe": trial.suggest_categorical("learnable_pe", [False, True]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }

        train_loader, val_loader = load_and_prepare_data(
            self.config, hparams['batch_size']
        )

        model = SeqTransformer(
            input_size=4,
            num_classes=7,
            seq_len=256,
            d_model=hparams["d_model"],
            nhead=hparams["nhead"],
            num_encoder_layers=hparams["num_encoder_layers"],
            dim_feedforward=hparams["dim_feedforward"],
            dropout=hparams["dropout"],
            learnable_pe=hparams["learnable_pe"],
            max_len=512,
        ).to(self.device)
        optimizer = AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()
        best_val_f1 = 0.0
        patience_counter = 0
        patience = 10
        for epoch in range(self.max_epochs):
            model.train()
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                logits = logits.view(-1, 7)
                Y_batch = Y_batch.view(-1)
                loss = criterion(logits, Y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    logits = model(X_batch)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
                    val_preds.extend(preds)
                    val_targets.extend(Y_batch.cpu().numpy().flatten())
            val_f1 = f1_score(val_targets, val_preds, average="weighted")
            scheduler.step()
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        return best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    config = AppConfig()
    device = get_device()
    setup_logger(
        log_file=Path("log/hp_optim_transformer.log"), level=logging.INFO, name=__name__
    )

    objective = Objective(config, device, max_epochs=args.epochs)
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(), pruner=MedianPruner()
    )
    study.optimize(objective, n_trials=args.n_trials)

    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_value}")
    logger.info("  Params: ")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    with open(
        f"bayesian_optim_transformer_{str(Path().cwd()).split('/')[-1]}_{study.best_trial.number}.json",
        "w",
    ) as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    main()
