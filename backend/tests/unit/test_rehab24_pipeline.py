"""
Unit tests for REHAB24-6 data pipeline (src/data/).
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repo root is on sys.path so `src.data` imports cleanly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.extract_landmarks import match_videos_by_prefix
from src.data.rehab24_dataset import (
    Rehab24Dataset,
    interpolate_undetected_frames,
    resample_sequence,
)
from src.data.splits import get_person_splits


def test_match_videos_by_prefix(tmp_path: Path):
    """Verify video_id prefix matching against Camera17 filenames."""
    (tmp_path / "PM_000_cam17.mp4").touch()
    (tmp_path / "PM_001_cam17.mp4").touch()
    (tmp_path / ".gitkeep").touch()

    matched = match_videos_by_prefix(tmp_path, ["PM_000", "PM_001", "PM_999"])
    assert len(matched) == 2
    assert matched[0][0] == "PM_000" and matched[0][1].name == "PM_000_cam17.mp4"
    assert matched[1][0] == "PM_001" and matched[1][1].name == "PM_001_cam17.mp4"


def test_interpolate_and_resample():
    """Verify linear interpolation of undetected frames and resampling to 60 frames."""
    lms = np.zeros((10, 33, 4), dtype=np.float32)
    for t in range(10):
        lms[t, :, :] = float(t)
    # Corrupt frame 4 and mark undetected
    lms[4, :, :] = 999.0
    det = np.ones(10, dtype=bool)
    det[4] = False

    interp = interpolate_undetected_frames(lms, det)
    assert np.allclose(interp[4], 4.0, atol=1e-5)

    resampled = resample_sequence(interp, target_frames=60)
    assert resampled.shape == (60, 33, 4)
    assert np.isclose(resampled[0, 0, 0], 0.0)
    assert np.isclose(resampled[-1, 0, 0], 9.0)


def test_rehab24_dataset_filtering_and_loading(tmp_path: Path):
    """Verify mocap_erroneous and >20% undetected filtering and 60-frame output."""
    csv_path = tmp_path / "Segmentation.csv"
    lm_dir = tmp_path / "landmarks"
    lm_dir.mkdir()

    csv_content = (
        "video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;"
        "cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;"
        "extra_person_in_cam17;extra_person_in_cam18;correctness\n"
        "PM_000;1;1;1;1;20;front;0;right arm;0;0;0;1\n"
        "PM_000;2;1;1;21;40;front;1;right arm;0;0;0;0\n"
        "PM_000;3;2;10;41;60;profile;0;none;0;0;0;1\n"
    )
    csv_path.write_text(csv_content, encoding="utf-8")

    # Video has 60 frames; rep 1 has 10% undetected (kept), rep 3 has 50% undetected (dropped)
    rng = np.random.default_rng(42)
    lms = rng.random((60, 33, 4), dtype=np.float32)
    det = np.ones(60, dtype=bool)
    det[2:4] = False  # 2/20 = 10% undetected in rep 1 (frames 1..20 -> idx 0..19)
    det[40:50] = False  # 10/20 = 50% undetected in rep 3 (frames 41..60 -> idx 40..59)
    np.savez_compressed(lm_dir / "PM_000.npz", landmarks=lms, detected=det, fps=np.float32(30.0))

    ds = Rehab24Dataset(segmentation_csv=csv_path, landmarks_dir=lm_dir, target_frames=60)
    assert ds.stats.total_csv_rows == 3
    assert ds.stats.dropped_mocap_erroneous == 1
    assert ds.stats.dropped_undetected_gt_20pct == 1
    assert ds.stats.loaded_reps == 1

    sample = ds[0]
    assert sample["landmarks"].shape == (60, 33, 4)
    assert sample["exercise_id"] == 1
    assert sample["person_id"] == 1
    assert sample["correctness"] == 1
    assert sample["cam17_orientation"] == "front"


def test_person_splits_keep_person_10_in_train():
    """Verify LOPO and grouped splits never put Person 10 in test and never overlap."""
    samples = [{"person_id": pid} for pid in [1, 1, 2, 2, 3, 3, 10, 10]]

    lopo_folds = get_person_splits(samples, n_folds=None, always_train_persons=(10,))
    assert len(lopo_folds) == 3  # persons 1, 2, 3
    for fold in lopo_folds:
        assert 10 in fold.train_persons
        assert 10 not in fold.test_persons
        assert set(fold.train_persons).isdisjoint(set(fold.test_persons))
        assert set(fold.train_indices).isdisjoint(set(fold.test_indices))

    grouped_folds = get_person_splits(samples, n_folds=2, always_train_persons=(10,))
    assert len(grouped_folds) == 2
    for fold in grouped_folds:
        assert 10 in fold.train_persons
        assert 10 not in fold.test_persons
