"""
End-to-end tests on fake REHAB24-6 data for angles, summary features,
subject-independent splits, and baseline classification.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.rehab24_dataset import Rehab24Dataset
from src.data.splits import get_person_splits
from src.features.angles import compute_joint_angles
from src.features.summary import build_feature_matrix
from src.models.baseline import evaluate_baseline_on_samples
from src.models.sequence import evaluate_sequence_model
from tests.fake_data import build_fake_rehab24_dataset


def test_angles_in_range_and_no_nans(tmp_path: Path):
    """Verify computed angles stay in [0, 180] and contain no NaNs after processing."""
    csv_path, lm_dir, _ = build_fake_rehab24_dataset(tmp_path, seed=42)
    ds = Rehab24Dataset(segmentation_csv=csv_path, landmarks_dir=lm_dir, target_frames=60)

    assert len(ds) > 0
    for sample in ds.samples:
        assert sample["world_landmarks"].shape == (60, 33, 3)
        assert np.all(np.isfinite(sample["world_landmarks"]))
        angles, angle_names = compute_joint_angles(sample["world_landmarks"])
        assert angles.shape == (60, len(angle_names))
        assert np.all(np.isfinite(angles))
        assert float(np.min(angles)) >= 0.0
        assert float(np.max(angles)) <= 180.0

    X, feat_names = build_feature_matrix(ds.samples, include_subtype=True)
    assert X.shape == (len(ds), len(feat_names))
    assert np.all(np.isfinite(X))


def test_splits_disjoint_and_person_10_pinned(tmp_path: Path):
    """Verify no person appears in both train and test, and Person 10 is never in test."""
    csv_path, lm_dir, _ = build_fake_rehab24_dataset(tmp_path, seed=42)
    ds = Rehab24Dataset(segmentation_csv=csv_path, landmarks_dir=lm_dir)
    folds = get_person_splits(ds.samples, n_folds=None, always_train_persons=(10,))

    assert len(folds) == 4  # persons 1, 2, 3, 4
    for fold in folds:
        assert 10 in fold.train_persons
        assert 10 not in fold.test_persons
        assert set(fold.train_persons).isdisjoint(set(fold.test_persons))
        assert set(fold.train_indices).isdisjoint(set(fold.test_indices))


def test_baseline_end_to_end_on_fake_data(tmp_path: Path):
    """Verify baseline RF, LogReg, and majority-class run end-to-end on fake data."""
    csv_path, lm_dir, _ = build_fake_rehab24_dataset(tmp_path, seed=42)
    ds = Rehab24Dataset(segmentation_csv=csv_path, landmarks_dir=lm_dir)
    out_dir = tmp_path / "results"

    metrics_df, imp_df = evaluate_baseline_on_samples(
        samples=ds.samples,
        output_dir=out_dir,
        models=("random_forest", "logreg", "majority"),
        seed=42,
    )

    assert (out_dir / "baseline_metrics.csv").exists()
    assert (out_dir / "feature_importance.csv").exists()
    assert set(metrics_df["exercise_id"]) == {1, 2}
    assert set(metrics_df["model"]) == {"random_forest", "logreg", "majority"}
    assert len(imp_df) == 20  # top 10 per exercise * 2 exercises


def test_optional_sequence_model_on_fake_data(tmp_path: Path):
    """Verify optional 1D-CNN runs end-to-end on fake data and writes sequence_metrics.csv."""
    csv_path, lm_dir, _ = build_fake_rehab24_dataset(tmp_path, seed=42)
    ds = Rehab24Dataset(segmentation_csv=csv_path, landmarks_dir=lm_dir)
    out_dir = tmp_path / "results"

    seq_df = evaluate_sequence_model(
        samples=ds.samples,
        output_dir=out_dir,
        epochs=2,
        seed=42,
    )
    assert (out_dir / "sequence_metrics.csv").exists()
    assert set(seq_df["exercise_id"]) == {1, 2}
    assert set(seq_df["model"]) == {"1d_cnn"}
