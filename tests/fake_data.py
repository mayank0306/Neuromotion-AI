"""
Synthetic REHAB24-6 data generator for unit and end-to-end pipeline testing.

Generates fake per-video .npz files (landmarks, world_landmarks, detected, fps)
and a semicolon-separated Segmentation.csv / DataFrame matching the real schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_fake_rehab24_dataset(
    root_dir: Path,
    seed: int = 42,
) -> Tuple[Path, Path, pd.DataFrame]:
    """
    Build fake .npz landmark files and a Segmentation.csv DataFrame in root_dir.

    Includes:
    - Multiple persons (1, 2, 3, 4, and 10)
    - Multiple exercises (1 and 2) with exercise_subtype ('left arm', 'right arm')
    - Both correctness classes (0 and 1)
    - Some undetected frames (<= 20% in valid reps, plus NaNs in world_landmarks)

    Args:
        root_dir: Temporary directory where segmentation/Segmentation.csv and
            landmarks/*.npz will be written.
        seed: Fixed random seed.

    Returns:
        Tuple of (segmentation_csv_path, landmarks_dir, segmentation_df).
    """
    rng = np.random.default_rng(seed)
    seg_dir = root_dir / "segmentation"
    lm_dir = root_dir / "landmarks"
    seg_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    persons = [1, 2, 3, 4, 10]
    exercises = [1, 2]
    frames_per_rep = 30

    vid_idx = 0
    for pid in persons:
        for ex_id in exercises:
            video_id = f"PM_{vid_idx:03d}"
            vid_idx += 1

            # 4 reps per (person, exercise) -> 2 correct, 2 incorrect (person 10 gets 2 reps)
            n_reps = 2 if pid == 10 else 4
            total_frames = n_reps * frames_per_rep + 10

            # Generate anatomically plausible moving 3D world landmarks (meters)
            t_axis = np.linspace(0.0, 2.0 * np.pi, total_frames, dtype=np.float32)
            landmarks = rng.uniform(0.2, 0.8, size=(total_frames, 33, 4)).astype(np.float32)
            world_lms = np.zeros((total_frames, 33, 3), dtype=np.float32)

            # Base skeleton in meters
            world_lms[:, 11] = [-0.2, -0.4, 0.0]  # left_shoulder
            world_lms[:, 12] = [0.2, -0.4, 0.0]   # right_shoulder
            world_lms[:, 23] = [-0.15, 0.0, 0.0]  # left_hip
            world_lms[:, 24] = [0.15, 0.0, 0.0]   # right_hip
            world_lms[:, 25] = [-0.15, 0.4, 0.0]  # left_knee
            world_lms[:, 26] = [0.15, 0.4, 0.0]   # right_knee
            world_lms[:, 27] = [-0.15, 0.8, 0.0]  # left_ankle
            world_lms[:, 28] = [0.15, 0.8, 0.0]   # right_ankle
            world_lms[:, 31] = [-0.15, 0.85, 0.15]  # left_foot_index
            world_lms[:, 32] = [0.15, 0.85, 0.15]   # right_foot_index

            # Animate elbows and wrists
            world_lms[:, 13, 0] = -0.35
            world_lms[:, 13, 1] = -0.2 + 0.15 * np.sin(t_axis)
            world_lms[:, 14, 0] = 0.35
            world_lms[:, 14, 1] = -0.2 + 0.15 * np.sin(t_axis)
            world_lms[:, 15, 0] = -0.45
            world_lms[:, 15, 1] = -0.05 + 0.25 * np.cos(t_axis)
            world_lms[:, 16, 0] = 0.45
            world_lms[:, 16, 1] = -0.05 + 0.25 * np.cos(t_axis)

            detected = np.ones((total_frames,), dtype=bool)
            # Inject 2 undetected frames with NaN world_landmarks in rep 1 (<= 20%)
            detected[5] = False
            detected[12] = False
            world_lms[5] = np.nan
            world_lms[12] = np.nan

            np.savez_compressed(
                lm_dir / f"{video_id}.npz",
                landmarks=landmarks,
                world_landmarks=world_lms,
                detected=detected,
                fps=np.float32(30.0),
            )

            for rep_num in range(1, n_reps + 1):
                first_f = (rep_num - 1) * frames_per_rep + 1
                last_f = rep_num * frames_per_rep
                correctness = 1 if rep_num % 2 == 1 else 0
                subtype = "right arm" if rep_num % 2 == 1 else "left arm"

                rows.append(
                    {
                        "video_id": video_id,
                        "repetition_number": rep_num,
                        "exercise_id": ex_id,
                        "person_id": pid,
                        "first_frame": first_f,
                        "last_frame": last_f,
                        "cam17_orientation": "front",
                        "mocap_erroneous": 0,
                        "exercise_subtype": subtype,
                        "lights_on": 1,
                        "extra_person_in_cam17": 0,
                        "extra_person_in_cam18": 0,
                        "correctness": correctness,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = seg_dir / "Segmentation.csv"
    df.to_csv(csv_path, sep=";", index=False)
    return csv_path, lm_dir, df
