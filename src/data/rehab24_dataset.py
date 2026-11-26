"""
REHAB24-6 repetition dataset loader.

Loads repetition segments from semicolon-separated Segmentation.csv,
slices per-video BlazePose .npz landmarks and world_landmarks,
filters/interpolates undetected frames, and resamples each repetition
to a fixed length (default 60 frames).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Tracks repetition loading and drop reasons."""

    total_csv_rows: int = 0
    dropped_mocap_erroneous: int = 0
    dropped_missing_npz: int = 0
    dropped_empty_slice: int = 0
    dropped_undetected_gt_20pct: int = 0
    loaded_reps: int = 0

    @property
    def total_dropped(self) -> int:
        return (
            self.dropped_mocap_erroneous
            + self.dropped_missing_npz
            + self.dropped_empty_slice
            + self.dropped_undetected_gt_20pct
        )


def interpolate_undetected_frames(
    landmarks: np.ndarray, detected: np.ndarray
) -> np.ndarray:
    """
    Linearly interpolate landmarks (or world_landmarks) across frames where
    detected is False or coordinates contain NaNs.

    Args:
        landmarks: Float array of shape (T, 33, C) where C is 3 or 4.
        detected: Boolean array of shape (T,).

    Returns:
        Interpolated float32 array of shape (T, 33, C) with no NaNs.
    """
    if len(landmarks) == 0:
        return landmarks.astype(np.float32, copy=False)

    out = landmarks.astype(np.float32, copy=True)
    all_idx = np.arange(len(landmarks))

    for joint_idx in range(out.shape[1]):
        for coord_idx in range(out.shape[2]):
            series = out[:, joint_idx, coord_idx]
            valid_mask = detected & np.isfinite(series)
            valid_idx = np.flatnonzero(valid_mask)
            if len(valid_idx) == 0:
                out[:, joint_idx, coord_idx] = 0.0
            elif len(valid_idx) < len(series):
                out[:, joint_idx, coord_idx] = np.interp(
                    all_idx, valid_idx, series[valid_idx]
                )
    return out


def resample_sequence(landmarks: np.ndarray, target_frames: int = 60) -> np.ndarray:
    """
    Resample a landmark sequence of length T to a fixed target_frames length.

    Args:
        landmarks: Float array of shape (T, 33, C).
        target_frames: Output sequence length (default 60).

    Returns:
        Resampled float32 array of shape (target_frames, 33, C).
    """
    t_len = len(landmarks)
    channels = landmarks.shape[2] if landmarks.ndim == 3 else 4
    if t_len == 0:
        return np.zeros((target_frames, 33, channels), dtype=np.float32)
    if t_len == target_frames:
        return landmarks.astype(np.float32, copy=False)
    if t_len == 1:
        return np.repeat(landmarks.astype(np.float32), target_frames, axis=0)

    src_x = np.arange(t_len, dtype=np.float64)
    dst_x = np.linspace(0.0, t_len - 1, target_frames, dtype=np.float64)
    out = np.empty((target_frames, landmarks.shape[1], channels), dtype=np.float32)

    for joint_idx in range(landmarks.shape[1]):
        for coord_idx in range(channels):
            out[:, joint_idx, coord_idx] = np.interp(
                dst_x, src_x, landmarks[:, joint_idx, coord_idx]
            )
    return out


def load_segmentation_rows(segmentation_csv: Path) -> List[Dict[str, str]]:
    """Load rows from semicolon-separated Segmentation.csv."""
    with segmentation_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


class Rehab24Dataset(Dataset):
    """
    PyTorch Dataset for REHAB24-6 repetitions.

    Each sample is a dictionary containing:
    - landmarks: float32 array of shape (target_frames, 33, 4)
    - world_landmarks: float32 array of shape (target_frames, 33, 3)
    - exercise_id: int
    - person_id: int
    - correctness: int (1 = correct, 0 = incorrect)
    - cam17_orientation: str
    - exercise_subtype: str
    - duration_sec: float (computed from original rep length / fps before resampling)
    """

    def __init__(
        self,
        segmentation_csv: Path = Path("datasets/rehab24_6/segmentation/Segmentation.csv"),
        landmarks_dir: Path = Path("datasets/rehab24_6/landmarks"),
        target_frames: int = 60,
        max_undetected_ratio: float = 0.20,
        one_based_frames: bool = True,
    ) -> None:
        """
        Initialize the REHAB24-6 repetition dataset.

        Args:
            segmentation_csv: Path to semicolon-separated Segmentation.csv.
            landmarks_dir: Directory containing per-video .npz landmark files.
            target_frames: Fixed number of frames to resample each rep to.
            max_undetected_ratio: Drop reps with undetected fraction strictly above this.
            one_based_frames: True if CSV first_frame/last_frame are 1-based.
        """
        self.segmentation_csv = Path(segmentation_csv)
        self.landmarks_dir = Path(landmarks_dir)
        self.target_frames = target_frames
        self.max_undetected_ratio = max_undetected_ratio
        self.one_based_frames = one_based_frames

        self.stats = DatasetStats()
        self.samples: List[Dict[str, Any]] = []
        self._npz_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

        self._load_dataset()

    def _get_video_arrays(
        self, video_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Load and cache (landmarks, world_landmarks, detected, fps) for a video_id."""
        if video_id in self._npz_cache:
            return self._npz_cache[video_id]

        npz_path = self.landmarks_dir / f"{video_id}.npz"
        if not npz_path.exists():
            return None

        with np.load(npz_path) as data:
            landmarks = data["landmarks"].astype(np.float32)
            if "world_landmarks" in data:
                world_landmarks = data["world_landmarks"].astype(np.float32)
            else:
                world_landmarks = landmarks[:, :, :3].copy()
            detected = data["detected"].astype(bool)
            fps = float(data["fps"]) if "fps" in data and float(data["fps"]) > 0 else 30.0

        self._npz_cache[video_id] = (landmarks, world_landmarks, detected, fps)
        return landmarks, world_landmarks, detected, fps

    def _slice_bounds(self, first_frame: int, last_frame: int, num_frames: int) -> Tuple[int, int]:
        """Convert CSV frame bounds into 0-based Python slice [start, end)."""
        if self.one_based_frames:
            start = max(0, first_frame - 1)
            end = min(num_frames, last_frame)
        else:
            start = max(0, first_frame)
            end = min(num_frames, last_frame + 1)
        return start, end

    def _load_dataset(self) -> None:
        """Parse Segmentation.csv, filter/interpolate/resample reps, and record drop stats."""
        rows = load_segmentation_rows(self.segmentation_csv)
        self.stats.total_csv_rows = len(rows)

        for row in rows:
            if int(row["mocap_erroneous"]) == 1:
                self.stats.dropped_mocap_erroneous += 1
                continue

            video_id = row["video_id"].strip()
            arrays = self._get_video_arrays(video_id)
            if arrays is None:
                self.stats.dropped_missing_npz += 1
                continue

            landmarks_full, world_full, detected_full, fps = arrays
            start, end = self._slice_bounds(
                int(row["first_frame"]),
                int(row["last_frame"]),
                len(landmarks_full),
            )
            if end <= start:
                self.stats.dropped_empty_slice += 1
                continue

            rep_lms = landmarks_full[start:end]
            rep_world = world_full[start:end]
            rep_det = detected_full[start:end]
            original_len = len(rep_lms)

            undetected_ratio = float(1.0 - np.mean(rep_det))
            if undetected_ratio > self.max_undetected_ratio:
                self.stats.dropped_undetected_gt_20pct += 1
                continue

            rep_lms = interpolate_undetected_frames(rep_lms, rep_det)
            rep_world = interpolate_undetected_frames(rep_world, rep_det)

            rep_lms = resample_sequence(rep_lms, target_frames=self.target_frames)
            rep_world = resample_sequence(rep_world, target_frames=self.target_frames)

            duration_sec = float(original_len / fps) if fps > 0 else float(original_len / 30.0)

            self.samples.append(
                {
                    "landmarks": rep_lms,
                    "world_landmarks": rep_world,
                    "exercise_id": int(row["exercise_id"]),
                    "person_id": int(row["person_id"]),
                    "correctness": int(row["correctness"]),
                    "cam17_orientation": row["cam17_orientation"].strip(),
                    "exercise_subtype": row.get("exercise_subtype", "").strip(),
                    "duration_sec": duration_sec,
                }
            )
            self.stats.loaded_reps += 1

        self._npz_cache.clear()

        if self.stats.dropped_undetected_gt_20pct > 0:
            logger.info(
                "Dropped %d reps with > %.0f%% undetected frames.",
                self.stats.dropped_undetected_gt_20pct,
                self.max_undetected_ratio * 100.0,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
