"""
3D joint-angle feature extraction from MediaPipe world_landmarks.

Computes per-frame joint angles (in degrees) at the middle joint of three
3D points, plus trunk lean against the vertical axis, with NaN handling
and temporal smoothing (Savitzky-Golay filter).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import savgol_filter

# Single config dict containing MediaPipe 33-landmark indices and angle definitions.
ANGLE_CONFIG: Dict[str, Any] = {
    "landmark_indices": {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    },
    "three_point_angles": {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
        "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
        "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
        "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
    },
    # In MediaPipe world_landmarks, +Y points downward (gravity), so upward is (0, -1, 0).
    "vertical_up_axis": (0.0, -1.0, 0.0),
    "smoothing": {
        "window_length": 7,
        "polyorder": 2,
    },
}


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute angles in degrees between pairs of 3D vectors (T, 3).
    Returns NaN for frames with non-finite coordinates or zero-norm vectors.
    """
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = n1 * n2

    valid = np.isfinite(denom) & (denom > 1e-7) & np.all(np.isfinite(v1), axis=-1) & np.all(np.isfinite(v2), axis=-1)
    angles = np.full((len(v1),), np.nan, dtype=np.float64)

    if np.any(valid):
        dots = np.sum(v1[valid] * v2[valid], axis=-1)
        cos_theta = np.clip(dots / denom[valid], -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(cos_theta))
    return angles


def _interpolate_nan_1d(series: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in a 1D array; fills all-NaN arrays with 0.0."""
    finite_mask = np.isfinite(series)
    if np.all(finite_mask):
        return series
    valid_idx = np.flatnonzero(finite_mask)
    if len(valid_idx) == 0:
        return np.zeros_like(series, dtype=np.float64)
    all_idx = np.arange(len(series))
    return np.interp(all_idx, valid_idx, series[valid_idx])


def _smooth_angle_series(series: np.ndarray, window_length: int = 7, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing (or moving average fallback for short clips) and clip to [0, 180]."""
    t_len = len(series)
    if t_len == 0:
        return series.astype(np.float32)

    clean = _interpolate_nan_1d(series)

    # Adjust window_length to odd integer <= t_len
    win = min(window_length, t_len)
    if win % 2 == 0:
        win -= 1

    if win >= 3 and win > polyorder:
        smoothed = savgol_filter(clean, window_length=win, polyorder=polyorder, mode="interp")
    elif t_len >= 3:
        kernel = np.ones(3, dtype=np.float64) / 3.0
        padded = np.pad(clean, (1, 1), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    else:
        smoothed = clean

    return np.clip(smoothed, 0.0, 180.0).astype(np.float32)


def compute_joint_angles(
    world_landmarks: np.ndarray,
    config: Dict[str, Any] = ANGLE_CONFIG,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute smoothed 3D joint angles per frame from MediaPipe world_landmarks.

    Args:
        world_landmarks: Array of shape (num_frames, 33, 3) in meters.
        config: Angle and landmark index configuration dictionary.

    Returns:
        Tuple of:
        - angles: float32 array of shape (num_frames, num_angles) in [0.0, 180.0]
        - angle_names: ordered list of angle names
    """
    if world_landmarks.ndim != 3 or world_landmarks.shape[1] < 33 or world_landmarks.shape[2] < 3:
        raise ValueError(
            f"Expected world_landmarks shape (num_frames, 33, 3), got {world_landmarks.shape}"
        )

    lm_idx: Dict[str, int] = config["landmark_indices"]
    three_pt: Dict[str, Tuple[str, str, str]] = config["three_point_angles"]
    smooth_cfg: Dict[str, int] = config.get("smoothing", {"window_length": 7, "polyorder": 2})
    win_len = int(smooth_cfg.get("window_length", 7))
    poly = int(smooth_cfg.get("polyorder", 2))

    num_frames = world_landmarks.shape[0]
    angle_names: List[str] = list(three_pt.keys()) + ["trunk_lean"]
    angles_out = np.zeros((num_frames, len(angle_names)), dtype=np.float32)

    if num_frames == 0:
        return angles_out, angle_names

    # 1. Three-point joint angles
    for col_idx, (angle_name, (p1_name, v_name, p3_name)) in enumerate(three_pt.items()):
        p1 = world_landmarks[:, lm_idx[p1_name], :3]
        pv = world_landmarks[:, lm_idx[v_name], :3]
        p3 = world_landmarks[:, lm_idx[p3_name], :3]
        raw_angles = _angle_between_vectors(p1 - pv, p3 - pv)
        angles_out[:, col_idx] = _smooth_angle_series(raw_angles, win_len, poly)

    # 2. Trunk lean: angle of (shoulder_mid - hip_mid) against vertical_up_axis
    ls = world_landmarks[:, lm_idx["left_shoulder"], :3]
    rs = world_landmarks[:, lm_idx["right_shoulder"], :3]
    lh = world_landmarks[:, lm_idx["left_hip"], :3]
    rh = world_landmarks[:, lm_idx["right_hip"], :3]

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)
    trunk_vec = shoulder_mid - hip_mid
    vertical_vec = np.broadcast_to(
        np.array(config["vertical_up_axis"], dtype=np.float64),
        trunk_vec.shape,
    )
    raw_trunk = _angle_between_vectors(trunk_vec, vertical_vec)
    angles_out[:, -1] = _smooth_angle_series(raw_trunk, win_len, poly)

    return angles_out, angle_names
