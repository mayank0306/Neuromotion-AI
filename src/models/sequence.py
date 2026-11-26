"""
Optional small 1D-CNN sequence classifier on 60-frame angle sequences for REHAB24-6.

Kept optional behind the `--run` CLI flag. Uses the same per-exercise
Leave-One-Person-Out (LOPO) cross-validation and output CSV schema as
src/models/baseline.py so results can be compared directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data.rehab24_dataset import Rehab24Dataset
from src.data.splits import get_person_splits
from src.features.angles import compute_joint_angles


class AngleSequenceCNN(nn.Module):
    """Compact 1D-CNN for 60-frame joint-angle sequences."""

    def __init__(self, in_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, in_channels)
        Returns:
            Logits of shape (batch,)
        """
        feat = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(feat).squeeze(-1)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_single_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train AngleSequenceCNN with early stopping and return (y_pred, y_prob) for X_test."""
    _set_seed(seed)

    # Standardize per channel using training fold stats only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 1), keepdims=True), 1e-5)
    X_tr_norm = ((X_train - mean) / std).astype(np.float32)
    X_te_norm = ((X_test - mean) / std).astype(np.float32)

    # Internal 85/15 train/val carve-out for early stopping
    n_tr = len(X_tr_norm)
    perm = np.random.RandomState(seed).permutation(n_tr)
    val_count = max(1, int(0.15 * n_tr)) if n_tr >= 6 else 0

    if val_count > 0:
        val_idx, tr_idx = perm[:val_count], perm[val_count:]
        X_tr_sub, y_tr_sub = X_tr_norm[tr_idx], y_train[tr_idx]
        X_val_sub, y_val_sub = X_tr_norm[val_idx], y_train[val_idx]
    else:
        X_tr_sub, y_tr_sub = X_tr_norm, y_train
        X_val_sub, y_val_sub = X_tr_norm, y_train

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_sub), torch.from_numpy(y_tr_sub.astype(np.float32))),
        batch_size=min(batch_size, max(1, len(X_tr_sub))),
        shuffle=True,
    )

    model = AngleSequenceCNN(in_channels=X_train.shape[2], dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val_sub))
            val_loss = float(criterion(val_logits, torch.from_numpy(y_val_sub.astype(np.float32))).item())

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_te_norm))
        y_prob = torch.sigmoid(test_logits).numpy()
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return y_pred, y_prob


def evaluate_sequence_model(
    samples: Sequence[Dict[str, Any]],
    output_dir: Path = Path("results"),
    epochs: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate AngleSequenceCNN per exercise with LOPO CV and save sequence_metrics.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_ids = sorted({int(s["exercise_id"]) for s in samples})
    metric_rows: List[Dict[str, Any]] = []

    for ex_id in exercise_ids:
        ex_samples = [s for s in samples if int(s["exercise_id"]) == ex_id]
        angle_seqs = np.stack(
            [compute_joint_angles(s["world_landmarks"])[0] for s in ex_samples],
            axis=0,
        ).astype(np.float32)
        y = np.array([int(s["correctness"]) for s in ex_samples], dtype=np.int64)
        folds = get_person_splits(ex_samples, n_folds=None, always_train_persons=(10,))

        fold_accs, fold_f1s, fold_aurocs = [], [], []
        all_y_true, all_y_pred, all_y_prob = [], [], []

        for fold in folds:
            y_pred, y_prob = _train_single_fold(
                X_train=angle_seqs[fold.train_indices],
                y_train=y[fold.train_indices],
                X_test=angle_seqs[fold.test_indices],
                epochs=epochs,
                seed=seed,
            )
            y_test = y[fold.test_indices]
            fold_accs.append(float(accuracy_score(y_test, y_pred)))
            fold_f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
            if len(np.unique(y_test)) == 2:
                fold_aurocs.append(float(roc_auc_score(y_test, y_prob)))

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

        cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        pooled_auroc = (
            float(roc_auc_score(all_y_true, all_y_prob))
            if len(set(all_y_true)) == 2
            else float("nan")
        )

        metric_rows.append(
            {
                "exercise_id": ex_id,
                "model": "1d_cnn",
                "n_reps": len(ex_samples),
                "n_folds": len(folds),
                "accuracy": float(np.mean(fold_accs)),
                "accuracy_std": float(np.std(fold_accs)),
                "f1": float(np.mean(fold_f1s)),
                "f1_std": float(np.std(fold_f1s)),
                "auroc": float(np.mean(fold_aurocs)) if fold_aurocs else pooled_auroc,
                "auroc_std": float(np.std(fold_aurocs)) if fold_aurocs else 0.0,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "sequence_metrics.csv", index=False)
    return metrics_df


def main() -> None:
    """CLI entry point for optional 1D-CNN sequence model."""
    parser = argparse.ArgumentParser(description="Optional 1D-CNN sequence model for REHAB24-6.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required flag to execute sequence model training/evaluation.",
    )
    parser.add_argument(
        "--segmentation-csv",
        type=Path,
        default=Path("datasets/rehab24_6/segmentation/Segmentation.csv"),
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=Path("datasets/rehab24_6/landmarks"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.run:
        print(
            "Sequence model is optional and disabled by default. "
            "Pass `--run` to train and evaluate AngleSequenceCNN."
        )
        return

    dataset = Rehab24Dataset(
        segmentation_csv=args.segmentation_csv,
        landmarks_dir=args.landmarks_dir,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples loaded from {args.landmarks_dir}.")

    df = evaluate_sequence_model(
        samples=dataset.samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
