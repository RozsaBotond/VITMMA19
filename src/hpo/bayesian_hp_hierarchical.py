"""Bayesian hyperparameter optimization for hierarchical LSTM model.

Uses Optuna to find optimal hyperparameters for both Stage 1 and Stage 2.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Remove the sys.path.insert as it causes E402 and rely on external PYTHONPATH
# project_root = Path(__file__).resolve().parents[2]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

from models.hierarchical_v1.model import HierarchicalClassifier
from src.utils.augmentation import balance_dataset_with_augmentation
from src.utils.config import AppConfig
from src.utils.utils import setup_logger, set_seed, get_device


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def convert_labels_to_hierarchical(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 7-class labels to hierarchical labels."""
    Y_stage1 = np.zeros_like(Y)
    Y_stage2 = np.zeros_like(Y)

    # This assumes a specific mapping, consider moving to a config or utils
    LABEL_7CLASS_TO_STAGE1 = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
    LABEL_7CLASS_TO_STAGE2 = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2}

    for old_class, new_class in LABEL_7CLASS_TO_STAGE1.items():
        Y_stage1[Y == old_class] = new_class
    for old_class, new_class in LABEL_7CLASS_TO_STAGE2.items():
        Y_stage2[Y == old_class] = new_class
    return Y_stage1, Y_stage2


class HierarchicalBayesianOptimizer:
    """Bayesian hyperparameter optimization for hierarchical LSTM."""

    def __init__(
        self,
        data_dir: str = "data",
        epochs: int = 80,
        patience: int = 15,
    ):
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load data
        logger.info("\n" + "=" * 60)
        logger.info("Loading data...")
        logger.info("=" * 60)

        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        Y_test = np.load(os.path.join(data_dir, "Y_test.npy"))

        logger.info(f"Loaded data: X={X_train.shape}, Y={Y_train.shape}")

        # Augment training data
        X_train_aug, Y_train_aug = balance_dataset_with_augmentation(X_train, Y_train)
        logger.info(f"Augmented: {len(X_train_aug)} train samples")

        # Convert to hierarchical labels
        Y_train_s1, Y_train_s2 = convert_labels_to_hierarchical(Y_train_aug)
        Y_val_s1, Y_val_s2 = convert_labels_to_hierarchical(Y_val)
        Y_test_s1, Y_test_s2 = convert_labels_to_hierarchical(Y_test)

        # Store data
        self.X_train = torch.FloatTensor(X_train_aug)
        self.Y_train_s1 = torch.LongTensor(Y_train_s1)
        self.Y_train_s2 = torch.LongTensor(Y_train_s2)
        self.Y_train_7class = torch.LongTensor(Y_train_aug)

        self.X_val = torch.FloatTensor(X_val)
        self.Y_val_s1 = torch.LongTensor(Y_val_s1)
        self.Y_val_s2 = torch.LongTensor(Y_val_s2)
        self.Y_val_7class = torch.LongTensor(Y_val)

        self.X_test = torch.FloatTensor(X_test)
        self.Y_test_s1 = torch.LongTensor(Y_test_s1)
        self.Y_test_s2 = torch.LongTensor(Y_test_s2)
        self.Y_test_7class = torch.LongTensor(Y_test)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""

        # Shared hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        scheduler_type = trial.suggest_categorical(
            "scheduler", ["cosine", "one_cycle", "step"]
        )
        grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0, 3.0])

        # Stage 1 hyperparameters
        s1_hidden = trial.suggest_categorical("s1_hidden_size", [64, 96, 128, 192])
        s1_layers = trial.suggest_int("s1_num_layers", 1, 3)
        s1_dropout = trial.suggest_float("s1_dropout", 0.1, 0.5, step=0.05)
        s1_bidirectional = trial.suggest_categorical("s1_bidirectional", [True, False])

        # Stage 2 hyperparameters
        s2_hidden = trial.suggest_categorical("s2_hidden_size", [64, 96, 128, 192])
        s2_layers = trial.suggest_int("s2_num_layers", 1, 3)
        s2_dropout = trial.suggest_float("s2_dropout", 0.1, 0.5, step=0.05)
        s2_bidirectional = trial.suggest_categorical("s2_bidirectional", [True, False])

        # Class weights option
        use_class_weights = trial.suggest_categorical(
            "use_class_weights", [True, False]
        )
        if use_class_weights:
            class_weight_scale = trial.suggest_float(
                "class_weight_scale", 0.2, 1.0, step=0.2
            )
        else:
            class_weight_scale = 0.0

        # Create model configs
        stage1_config = {
            "input_size": 4,
            "hidden_size": s1_hidden,
            "num_layers": s1_layers,
            "num_classes": 3,
            "dropout": s1_dropout,
            "bidirectional": s1_bidirectional,
        }

        stage2_config = {
            "input_size": 4,
            "hidden_size": s2_hidden,
            "num_layers": s2_layers,
            "num_classes": 3,
            "dropout": s2_dropout,
            "bidirectional": s2_bidirectional,
        }

        # Create model
        model = HierarchicalClassifier(
            stage1_config=stage1_config,
            stage2_config=stage2_config,
        ).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            self.X_train, self.Y_train_s1, self.Y_train_s2, self.Y_train_7class
        )
        val_dataset = TensorDataset(
            self.X_val, self.Y_val_s1, self.Y_val_s2, self.Y_val_7class
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss functions with optional class weights
        if use_class_weights:
            # Calculate class weights for stage 1
            s1_counts = torch.bincount(self.Y_train_s1.flatten(), minlength=3).float()
            s1_weights = 1.0 / (s1_counts + 1e-6)
            s1_weights = s1_weights / s1_weights.sum() * 3
            s1_weights = 1.0 + (s1_weights - 1.0) * class_weight_scale
            s1_weights = s1_weights.to(self.device)

            # Calculate class weights for stage 2
            s2_counts = torch.bincount(self.Y_train_s2.flatten(), minlength=3).float()
            s2_weights = 1.0 / (s2_counts + 1e-6)
            s2_weights = s2_weights / s2_weights.sum() * 3
            s2_weights = 1.0 + (s2_weights - 1.0) * class_weight_scale
            s2_weights = s2_weights.to(self.device)

            criterion_s1 = nn.CrossEntropyLoss(weight=s1_weights)
            criterion_s2 = nn.CrossEntropyLoss(weight=s2_weights)
        else:
            criterion_s1 = nn.CrossEntropyLoss()
            criterion_s2 = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif scheduler_type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate * 10,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )

        # Training loop
        current_best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()

            for X_batch, Y_s1, Y_s2, Y_7class in train_loader:
                X_batch = X_batch.to(self.device)
                Y_s1 = Y_s1.to(self.device)
                Y_s2 = Y_s2.to(self.device)

                optimizer.zero_grad()

                # Forward pass - get both stage outputs
                s1_logits, s2_logits = model(X_batch, return_both=True)

                # Compute losses
                s1_logits_flat = s1_logits.reshape(-1, 3)
                s2_logits_flat = s2_logits.reshape(-1, 3)
                Y_s1_flat = Y_s1.reshape(-1)
                Y_s2_flat = Y_s2.reshape(-1)

                loss_s1 = criterion_s1(s1_logits_flat, Y_s1_flat)
                loss_s2 = criterion_s2(s2_logits_flat, Y_s2_flat)
                loss = loss_s1 + loss_s2  # Combined loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if scheduler_type == "one_cycle":
                    scheduler.step()

            # Validate - using combined 7-class predictions
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, Y_s1, Y_s2, Y_7class in val_loader:
                    X_batch = X_batch.to(self.device)

                    # Get combined 7-class predictions
                    output = model(X_batch)  # Returns combined logits
                    preds = output.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds.flatten())
                    all_labels.extend(Y_7class.numpy().flatten())

            val_f1 = f1_score(
                all_labels, all_preds, average="weighted", zero_division=0
            )

            # Report for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Update scheduler
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "step":
                scheduler.step()

            # Early stopping
            if val_f1 > current_best_val_f1:
                current_best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return current_best_val_f1

    def run_optimization(
        self,
        n_trials: int = 100,
        save_dir: str = "models",
    ):
        """Run Bayesian optimization."""

        logger.info("\n" + "=" * 60)
        logger.info(f"Starting Hierarchical Bayesian Optimization ({n_trials} trials)")
        logger.info("=" * 60)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name="hierarchical_bayesian_optimization",
        )

        # Optimize with progress bar
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Get best trial
        best_trial = study.best_trial
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best validation F1: {best_trial.value:.4f}")
        logger.info("Best parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # Train final model with best hyperparameters and evaluate on test
        self._train_final_model(best_trial.params, save_dir)

    def _train_final_model(self, params: Dict, save_dir: str):
        """Train final model with best parameters and evaluate on test set."""

        logger.info("\n" + "=" * 60)
        logger.info("Training final model with best parameters...")
        logger.info("=" * 60)

        # Extract parameters
        stage1_config = {
            "input_size": 4,
            "hidden_size": params["s1_hidden_size"],
            "num_layers": params["s1_num_layers"],
            "num_classes": 3,
            "dropout": params["s1_dropout"],
            "bidirectional": params["s1_bidirectional"],
        }

        stage2_config = {
            "input_size": 4,
            "hidden_size": params["s2_hidden_size"],
            "num_layers": params["s2_num_layers"],
            "num_classes": 3,
            "dropout": params["s2_dropout"],
            "bidirectional": params["s2_bidirectional"],
        }

        model = HierarchicalClassifier(
            stage1_config=stage1_config,
            stage2_config=stage2_config,
        ).to(self.device)

        batch_size = params["batch_size"]

        train_dataset = TensorDataset(
            self.X_train, self.Y_train_s1, self.Y_train_s2, self.Y_train_7class
        )
        test_dataset = TensorDataset(
            self.X_test, self.Y_test_s1, self.Y_test_s2, self.Y_test_7class
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Loss and optimizer
        use_class_weights = params["use_class_weights"]
        if use_class_weights:
            class_weight_scale = params.get("class_weight_scale", 0.5)
            s1_counts = torch.bincount(self.Y_train_s1.flatten(), minlength=3).float()
            s1_weights = 1.0 / (s1_counts + 1e-6)
            s1_weights = s1_weights / s1_weights.sum() * 3
            s1_weights = 1.0 + (s1_weights - 1.0) * class_weight_scale
            s1_weights = s1_weights.to(self.device)

            s2_counts = torch.bincount(self.Y_train_s2.flatten(), minlength=3).float()
            s2_weights = 1.0 / (s2_counts + 1e-6)
            s2_weights = s2_weights / s2_weights.sum() * 3
            s2_weights = 1.0 + (s2_weights - 1.0) * class_weight_scale
            s2_weights = s2_weights.to(self.device)

            criterion_s1 = nn.CrossEntropyLoss(weight=s1_weights)
            criterion_s2 = nn.CrossEntropyLoss(weight=s2_weights)
        else:
            criterion_s1 = nn.CrossEntropyLoss()
            criterion_s2 = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        scheduler_type = params["scheduler"]
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        elif scheduler_type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=params["learning_rate"] * 10,
                epochs=self.epochs,
                steps_per_epoch=len(train_loader),
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )

        grad_clip = params["grad_clip"]

        # Training
        for epoch in range(self.epochs):
            model.train()

            for X_batch, Y_s1, Y_s2, Y_7class in train_loader:
                X_batch = X_batch.to(self.device)
                Y_s1 = Y_s1.to(self.device)
                Y_s2 = Y_s2.to(self.device)

                optimizer.zero_grad()
                s1_logits, s2_logits = model(X_batch, return_both=True)

                loss_s1 = criterion_s1(s1_logits.reshape(-1, 3), Y_s1.reshape(-1))
                loss_s2 = criterion_s2(s2_logits.reshape(-1, 3), Y_s2.reshape(-1))
                loss = loss_s1 + loss_s2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if scheduler_type == "one_cycle":
                    scheduler.step()

            if scheduler_type in ["cosine", "step"]:
                scheduler.step()

        # Evaluate on test set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, Y_s1, Y_s2, Y_7class in test_loader:
                X_batch = X_batch.to(self.device)
                output = model(X_batch)
                preds = output.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(Y_7class.numpy().flatten())

        test_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        test_acc = accuracy_score(all_labels, all_preds)

        # Calculate detection rate and false alarm rate
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        pattern_mask = all_labels > 0
        detection_rate = (
            np.mean(all_preds[pattern_mask] > 0) if pattern_mask.sum() > 0 else 0
        )

        none_mask = all_labels == 0
        false_alarm = np.mean(all_preds[none_mask] > 0) if none_mask.sum() > 0 else 0

        logger.info("\nTest Results:")
        logger.info(f"  Test F1: {test_f1:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.4f}")
        logger.info(f"  Detection Rate: {detection_rate:.4f}")
        logger.info(f"  False Alarm Rate: {false_alarm:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian HP optimization for hierarchical LSTM"
    )
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-dir", type=str, default="models")
    args = parser.parse_args()

    optimizer = HierarchicalBayesianOptimizer(
        data_dir=args.data_dir,
        epochs=args.epochs,
        patience=args.patience,
    )

    optimizer.run_optimization(
        n_trials=args.n_trials,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
