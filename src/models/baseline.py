"""
Baseline per-exercise correctness classifiers for REHAB24-6.

Trains Random Forest, Standardized Logistic Regression, and a Majority-Class
baseline separately for each exercise_id using Leave-One-Person-Out (LOPO)
cross-validation (with Person 10 pinned to the training set).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.rehab24_dataset import Rehab24Dataset
from src.data.splits import get_person_splits
from src.features.summary import build_feature_matrix


def _make_estimator(model_name: str, seed: int) -> Any:
    """Instantiate a scikit-learn classifier by name."""
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown model name: {model_name}")


def _extract_feature_importances(estimator: Any) -> Optional[np.ndarray]:
    """Extract normalized feature importances from RF or absolute coefficients from LogReg."""
    if isinstance(estimator, RandomForestClassifier):
        return estimator.feature_importances_
    if isinstance(estimator, Pipeline) and "clf" in estimator.named_steps:
        clf = estimator.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_[0])
            total = float(np.sum(coef))
            return coef / total if total > 0 else coef
    return None


def evaluate_baseline_on_samples(
    samples: Sequence[Dict[str, Any]],
    output_dir: Path = Path("results"),
    models: Sequence[str] = ("random_forest", "logreg", "majority"),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run per-exercise LOPO cross-validation on preloaded Rehab24Dataset samples.

    Args:
        samples: Sequence of sample dicts from Rehab24Dataset.
        output_dir: Directory where baseline_metrics.csv and feature_importance.csv are saved.
        models: Which models to evaluate ('random_forest', 'logreg', 'majority').
        seed: Fixed random seed for reproducibility.

    Returns:
        Tuple of (metrics_df, importance_df).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_ids = sorted({int(s["exercise_id"]) for s in samples})
    metric_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    for ex_id in exercise_ids:
        ex_samples = [s for s in samples if int(s["exercise_id"]) == ex_id]
        X, feature_names = build_feature_matrix(ex_samples, include_subtype=True)
        y = np.array([int(s["correctness"]) for s in ex_samples], dtype=np.int64)
        folds = get_person_splits(ex_samples, n_folds=None, always_train_persons=(10,))

        for model_name in models:
            fold_accs: List[float] = []
            fold_f1s: List[float] = []
            fold_aurocs: List[float] = []
            fold_importances: List[np.ndarray] = []

            all_y_true: List[int] = []
            all_y_pred: List[int] = []
            all_y_prob: List[float] = []

            for fold in folds:
                X_train, y_train = X[fold.train_indices], y[fold.train_indices]
                X_test, y_test = X[fold.test_indices], y[fold.test_indices]

                if len(np.unique(y_train)) < 2:
                    # Fallback if synthetic/tiny fold has only one training class
                    est = DummyClassifier(strategy="most_frequent")
                else:
                    est = _make_estimator(model_name, seed=seed)

                est.fit(X_train, y_train)
                y_pred = est.predict(X_test)

                if hasattr(est, "predict_proba"):
                    probs = est.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        y_prob = probs[:, 1]
                    else:
                        pos_val = int(est.classes_[0])
                        y_prob = np.full(len(X_test), float(pos_val), dtype=np.float64)
                else:
                    y_prob = y_pred.astype(np.float64)

                fold_accs.append(float(accuracy_score(y_test, y_pred)))
                fold_f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))
                if len(np.unique(y_test)) == 2:
                    fold_aurocs.append(float(roc_auc_score(y_test, y_prob)))

                imp = _extract_feature_importances(est)
                if imp is not None:
                    fold_importances.append(imp)

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
                    "model": model_name,
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

            # Save top 10 feature importances for Random Forest (and LogReg if run)
            if fold_importances and model_name == "random_forest":
                mean_imp = np.mean(np.stack(fold_importances, axis=0), axis=0)
                top_idx = np.argsort(mean_imp)[::-1][:10]
                for rank, idx in enumerate(top_idx, start=1):
                    importance_rows.append(
                        {
                            "exercise_id": ex_id,
                            "model": model_name,
                            "rank": rank,
                            "feature": feature_names[idx],
                            "importance": float(mean_imp[idx]),
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    importance_df = pd.DataFrame(importance_rows)

    metrics_path = output_dir / "baseline_metrics.csv"
    importance_path = output_dir / "feature_importance.csv"
    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    return metrics_df, importance_df


def print_metrics_summary(metrics_df: pd.DataFrame) -> None:
    """Print per-exercise metrics table and overall mean across exercises per model."""
    print("\n=== Baseline LOPO Metrics (Per Exercise) ===")
    display_cols = [
        "exercise_id",
        "model",
        "n_reps",
        "accuracy",
        "accuracy_std",
        "f1",
        "f1_std",
        "auroc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    print(metrics_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Overall Mean Across Exercises ===")
    summary = (
        metrics_df.groupby("model")[["accuracy", "f1", "auroc"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    """CLI entry point for baseline training and LOPO evaluation."""
    parser = argparse.ArgumentParser(description="Train and evaluate REHAB24-6 baseline classifiers.")
    parser.add_argument(
        "--segmentation-csv",
        type=Path,
        default=Path("datasets/rehab24_6/segmentation/Segmentation.csv"),
        help="Path to Segmentation.csv.",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=Path("datasets/rehab24_6/landmarks"),
        help="Directory containing extracted .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write baseline_metrics.csv and feature_importance.csv.",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "logreg", "both"],
        default="both",
        help="Which classifier(s) to run alongside the majority-class baseline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset = Rehab24Dataset(
        segmentation_csv=args.segmentation_csv,
        landmarks_dir=args.landmarks_dir,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples loaded from {args.landmarks_dir}. "
            "Extract landmarks first before running baseline training."
        )

    models_to_run: List[str] = []
    if args.model in ("rf", "both"):
        models_to_run.append("random_forest")
    if args.model in ("logreg", "both"):
        models_to_run.append("logreg")
    models_to_run.append("majority")

    metrics_df, _ = evaluate_baseline_on_samples(
        samples=dataset.samples,
        output_dir=args.output_dir,
        models=models_to_run,
        seed=args.seed,
    )
    print_metrics_summary(metrics_df)


if __name__ == "__main__":
    main()
