"""
Summary feature extraction over repetition joint-angle sequences.

Computes per-angle statistics (min, max, mean, std, ROM, relative frame
positions of min and max), original repetition duration in seconds,
and left/right ROM symmetry for paired joints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.features.angles import ANGLE_CONFIG, compute_joint_angles

PAIRED_JOINTS: Tuple[str, ...] = ("elbow", "shoulder", "hip", "knee", "ankle")


def _encode_subtype(subtype: str) -> Tuple[float, float]:
    """Encode exercise_subtype ('left arm', 'right arm', etc.) into numeric side flags."""
    s = (subtype or "").strip().lower()
    is_left = 1.0 if "left" in s else 0.0
    is_right = 1.0 if "right" in s else 0.0
    return is_left, is_right


def extract_rep_summary(
    angles: np.ndarray,
    angle_names: Sequence[str],
    duration_sec: float,
    exercise_subtype: str = "",
    include_subtype: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute summary features for a single repetition's angle sequence.

    Args:
        angles: Array of shape (num_frames, num_angles) in degrees.
        angle_names: Ordered names matching columns of angles.
        duration_sec: Repetition duration in seconds (computed before resampling).
        exercise_subtype: Optional subtype string (e.g. 'left arm', 'right arm').
        include_subtype: Whether to append encoded subtype side indicators.

    Returns:
        Tuple of (1D float32 feature vector, list of feature names).
    """
    t_len = angles.shape[0]
    denom = float(max(1, t_len - 1))

    features: List[float] = []
    names: List[str] = []
    rom_by_name: Dict[str, float] = {}

    for col_idx, name in enumerate(angle_names):
        col = angles[:, col_idx].astype(np.float64)
        if not np.all(np.isfinite(col)):
            col = np.nan_to_num(col, nan=0.0, posinf=180.0, neginf=0.0)

        min_val = float(np.min(col))
        max_val = float(np.max(col))
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        rom_val = max_val - min_val
        min_pos = float(np.argmin(col)) / denom
        max_pos = float(np.argmax(col)) / denom

        rom_by_name[name] = rom_val

        features.extend([min_val, max_val, mean_val, std_val, rom_val, min_pos, max_pos])
        names.extend(
            [
                f"{name}_min",
                f"{name}_max",
                f"{name}_mean",
                f"{name}_std",
                f"{name}_rom",
                f"{name}_min_pos",
                f"{name}_max_pos",
            ]
        )

    # Original repetition duration
    features.append(float(duration_sec))
    names.append("duration_sec")

    # Left/right ROM symmetry (absolute difference of left and right ROM)
    for joint in PAIRED_JOINTS:
        left_rom = rom_by_name.get(f"left_{joint}", 0.0)
        right_rom = rom_by_name.get(f"right_{joint}", 0.0)
        features.append(abs(left_rom - right_rom))
        names.append(f"{joint}_rom_symmetry_diff")

    if include_subtype:
        is_left, is_right = _encode_subtype(exercise_subtype)
        features.extend([is_left, is_right])
        names.extend(["subtype_is_left", "subtype_is_right"])

    return np.array(features, dtype=np.float32), names


def build_feature_matrix(
    samples: Sequence[Dict[str, Any]],
    angle_config: Dict[str, Any] = ANGLE_CONFIG,
    include_subtype: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a sequence of Rehab24Dataset samples into a 2D summary feature matrix.

    Args:
        samples: Sequence of sample dicts containing 'world_landmarks',
            'duration_sec', and optionally 'exercise_subtype'.
        angle_config: Angle configuration dict passed to compute_joint_angles.
        include_subtype: Whether to include encoded exercise_subtype features.

    Returns:
        Tuple of (feature_matrix of shape (N, D), feature_names of length D).
    """
    if not samples:
        return np.zeros((0, 0), dtype=np.float32), []

    rows: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None

    for sample in samples:
        world_lms = sample["world_landmarks"]
        angles, angle_names = compute_joint_angles(world_lms, config=angle_config)
        vec, names = extract_rep_summary(
            angles=angles,
            angle_names=angle_names,
            duration_sec=float(sample.get("duration_sec", len(world_lms) / 30.0)),
            exercise_subtype=str(sample.get("exercise_subtype", "")),
            include_subtype=include_subtype,
        )
        if feature_names is None:
            feature_names = names
        rows.append(vec)

    matrix = np.stack(rows, axis=0).astype(np.float32)
    return matrix, (feature_names or [])
