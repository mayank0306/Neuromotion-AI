"""
PyTorch models and dataset splitting logic for NeuroMotion.
Shared between training pipelines and backend inference.

Two models:
1. YogaClassifierMLP: Kaggle Yoga 5-Class classifier (5 classes).
2. RehabQualityMLP: Multi-task model with UI-PRMD classification head (correct/incorrect)
   and KIMORE regression head (continuous clinician quality score 0-100).
"""

from typing import List, Tuple, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


class YogaClassifierMLP(nn.Module):
    """
    MLP for 5-class Yoga pose classification on Kaggle Yoga 5-Class dataset.
    Takes normalized kinematic features from FeatureEngineering.
    """

    YOGA_CLASSES = ["warrior_ii", "downward_dog", "tree_pose", "goddess_pose", "plank"]

    def __init__(self,
                 input_features: int = 32,
                 hidden_layers: List[int] = None,
                 num_classes: int = 5,
                 dropout_rate: float = 0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers: List[nn.Module] = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input feature tensor of shape (batch_size, input_features)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)


class RehabQualityMLP(nn.Module):
    """
    Multi-task MLP for Rehabilitation Movement Quality Assessment.
    Shared representation with:
    1. Classification head: UI-PRMD correct vs incorrect form (2 classes)
    2. Regression head: KIMORE continuous clinician quality score (0-100)
    """

    def __init__(self,
                 input_features: int = 32,
                 hidden_layers: List[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        # Shared feature representation
        layers: List[nn.Module] = []
        prev_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared_backbone = nn.Sequential(*layers)

        # Head 1: UI-PRMD classification (0: incorrect, 1: correct)
        self.classifier_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        # Head 2: KIMORE regression (continuous quality score 0-100)
        self.regression_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Multiplied by 100 in forward
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input feature tensor of shape (batch_size, input_features)
        Returns:
            Tuple of (classification_logits, quality_score_0_to_100)
        """
        features = self.shared_backbone(x)
        logits = self.classifier_head(features)
        quality_score = self.regression_head(features) * 100.0
        return logits, quality_score.squeeze(-1)


# ---------------------------------------------------------------------------
# Subject-Based Dataset Splitting (Point 8)
# Splits by subject, never by frame or rep!
# ---------------------------------------------------------------------------

def split_ui_prmd_subjects(subjects: List[int], fold: int = 0) -> Dict[str, List[int]]:
    """
    Split UI-PRMD subjects into rotating 5 train / 2 validation / 3 test folds.
    Strictly splits by subject ID to prevent data leakage across frames or reps.

    Args:
        subjects: List of unique subject IDs (e.g., [1, 2, ..., 10])
        fold: Fold index (0 to 4 for rotating combinations)

    Returns:
        Dict with 'train', 'validation', and 'test' subject ID lists
    """
    sorted_subjects = sorted(list(set(subjects)))
    num_subjects = len(sorted_subjects)
    if num_subjects < 10:
        # Fallback for subsets
        train_count = max(1, int(0.5 * num_subjects))
        val_count = max(1, int(0.2 * num_subjects))
        return {
            "train": sorted_subjects[:train_count],
            "validation": sorted_subjects[train_count:train_count + val_count],
            "test": sorted_subjects[train_count + val_count:]
        }

    # Rotate subjects based on fold
    shift = (fold * 2) % num_subjects
    rotated = sorted_subjects[shift:] + sorted_subjects[:shift]

    train_subjects = rotated[0:5]
    val_subjects = rotated[5:7]
    test_subjects = rotated[7:10]

    return {
        "train": sorted(train_subjects),
        "validation": sorted(val_subjects),
        "test": sorted(test_subjects),
    }


def split_kimore_stratified(subject_records: List[Dict[str, Any]],
                            train_ratio: float = 0.70,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            seed: int = 42) -> Dict[str, List[str]]:
    """
    Split KIMORE subjects ~70/15/15 stratified by clinical diagnosis group.
    Groups: 'healthy', 'stroke', 'parkinsons', 'back_pain'.
    Splits by subject, never by frame or rep.

    Args:
        subject_records: List of dicts with 'subject_id' and 'diagnosis'
        train_ratio: Proportion for train (~0.70)
        val_ratio: Proportion for validation (~0.15)
        test_ratio: Proportion for test (~0.15)
        seed: Random seed

    Returns:
        Dict with 'train', 'validation', and 'test' subject ID lists
    """
    rng = np.random.RandomState(seed)

    # Group subjects by diagnosis
    diagnosis_groups: Dict[str, List[str]] = {}
    for rec in subject_records:
        sub_id = str(rec["subject_id"])
        diag = str(rec.get("diagnosis", "healthy")).lower()
        diagnosis_groups.setdefault(diag, []).append(sub_id)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for diag, sub_ids in sorted(diagnosis_groups.items()):
        unique_subs = sorted(list(set(sub_ids)))
        rng.shuffle(unique_subs)
        n = len(unique_subs)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in val and test if group has >= 3 subjects
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        n_test = n - n_train - n_val

        train_ids.extend(unique_subs[:n_train])
        val_ids.extend(unique_subs[n_train:n_train + n_val])
        test_ids.extend(unique_subs[n_train + n_val:])

    return {
        "train": sorted(train_ids),
        "validation": sorted(val_ids),
        "test": sorted(test_ids),
    }


def split_yoga_dataset(train_filenames: List[str],
                       test_filenames: List[str],
                       val_ratio: float = 0.20,
                       seed: int = 42) -> Dict[str, List[str]]:
    """
    Kaggle Yoga 5-Class split:
    Uses the provided train/test split with an 80/20 train/val carve-out from the training set.

    Args:
        train_filenames: Official training split image filenames
        test_filenames: Official test split image filenames
        val_ratio: Proportion carved out of training set for validation (0.20)
        seed: Random seed

    Returns:
        Dict with 'train', 'validation', and 'test' filename lists
    """
    rng = np.random.RandomState(seed)
    shuffled_train = list(train_filenames)
    rng.shuffle(shuffled_train)

    val_count = int(round(len(shuffled_train) * val_ratio))
    val_files = shuffled_train[:val_count]
    carved_train_files = shuffled_train[val_count:]

    return {
        "train": sorted(carved_train_files),
        "validation": sorted(val_files),
        "test": sorted(test_filenames),
    }


# ---------------------------------------------------------------------------
# Reference Means and Statistics Persistence (Point 6)
# Computes & persists training-set class means for inference deviations
# ---------------------------------------------------------------------------

DEFAULT_REFERENCE_STATS = {
    # Rehabilitation exercises (angles in degrees: mean, std, target ROM)
    "shoulder-abduction": {
        "left_shoulder_angle": {"mean": 92.5, "std": 6.8, "target": 90.0},
        "right_shoulder_angle": {"mean": 91.8, "std": 7.1, "target": 90.0},
        "left_elbow_angle": {"mean": 174.0, "std": 5.2, "target": 180.0},
        "right_elbow_angle": {"mean": 173.5, "std": 5.0, "target": 180.0},
        "trunk_angle": {"mean": 4.5, "std": 2.1, "target": 0.0},
        "shoulder_level_symmetry": {"mean": 0.96, "std": 0.03, "target": 1.0},
    },
    "shoulder-flexion": {
        "left_shoulder_angle": {"mean": 168.0, "std": 8.5, "target": 175.0},
        "right_shoulder_angle": {"mean": 167.5, "std": 8.0, "target": 175.0},
        "left_elbow_angle": {"mean": 175.0, "std": 4.5, "target": 180.0},
        "right_elbow_angle": {"mean": 174.8, "std": 4.8, "target": 180.0},
        "trunk_angle": {"mean": 6.0, "std": 2.5, "target": 0.0},
    },
    "elbow-flexion": {
        "left_elbow_angle": {"mean": 52.0, "std": 6.5, "target": 45.0},
        "right_elbow_angle": {"mean": 53.0, "std": 6.2, "target": 45.0},
        "left_shoulder_angle": {"mean": 15.0, "std": 4.0, "target": 10.0},
        "right_shoulder_angle": {"mean": 14.5, "std": 3.8, "target": 10.0},
    },
    "knee-extension": {
        "left_knee_angle": {"mean": 172.0, "std": 5.5, "target": 175.0},
        "right_knee_angle": {"mean": 171.5, "std": 5.8, "target": 175.0},
        "left_hip_angle": {"mean": 95.0, "std": 6.0, "target": 90.0},
        "right_hip_angle": {"mean": 94.5, "std": 6.2, "target": 90.0},
    },
    "hip-abduction": {
        "left_hip_angle": {"mean": 42.0, "std": 5.0, "target": 45.0},
        "right_hip_angle": {"mean": 41.5, "std": 5.2, "target": 45.0},
        "trunk_angle": {"mean": 5.0, "std": 2.2, "target": 0.0},
    },
    "sit-to-stand": {
        "left_knee_angle": {"mean": 175.0, "std": 6.0, "target": 180.0},
        "right_knee_angle": {"mean": 174.5, "std": 6.2, "target": 180.0},
        "left_hip_angle": {"mean": 172.0, "std": 5.5, "target": 180.0},
        "right_hip_angle": {"mean": 171.8, "std": 5.7, "target": 180.0},
        "knee_angle_symmetry": {"mean": 0.95, "std": 0.04, "target": 1.0},
    },
    # Yoga poses (supporting module)
    "warrior_ii": {
        "left_knee_angle": {"mean": 94.0, "std": 6.2, "target": 90.0},
        "right_knee_angle": {"mean": 172.0, "std": 5.1, "target": 175.0},
        "left_shoulder_angle": {"mean": 88.5, "std": 5.5, "target": 90.0},
        "right_shoulder_angle": {"mean": 89.0, "std": 5.2, "target": 90.0},
        "trunk_angle": {"mean": 3.0, "std": 1.8, "target": 0.0},
    },
    "downward_dog": {
        "left_hip_angle": {"mean": 72.0, "std": 6.5, "target": 70.0},
        "right_hip_angle": {"mean": 71.5, "std": 6.8, "target": 70.0},
        "left_shoulder_angle": {"mean": 162.0, "std": 7.0, "target": 165.0},
        "right_shoulder_angle": {"mean": 161.5, "std": 7.2, "target": 165.0},
    },
    "tree_pose": {
        "left_knee_angle": {"mean": 175.0, "std": 4.5, "target": 180.0},
        "right_knee_angle": {"mean": 48.0, "std": 6.0, "target": 45.0},
        "trunk_angle": {"mean": 2.5, "std": 1.5, "target": 0.0},
        "overall_stability": {"mean": 0.91, "std": 0.05, "target": 1.0},
    },
    "goddess_pose": {
        "left_knee_angle": {"mean": 95.0, "std": 6.5, "target": 90.0},
        "right_knee_angle": {"mean": 94.5, "std": 6.8, "target": 90.0},
        "left_shoulder_angle": {"mean": 89.0, "std": 5.0, "target": 90.0},
        "right_shoulder_angle": {"mean": 89.5, "std": 5.2, "target": 90.0},
    },
    "plank": {
        "trunk_angle": {"mean": 88.0, "std": 3.5, "target": 90.0},
        "left_elbow_angle": {"mean": 175.0, "std": 4.0, "target": 180.0},
        "right_elbow_angle": {"mean": 174.5, "std": 4.2, "target": 180.0},
        "left_knee_angle": {"mean": 176.0, "std": 3.8, "target": 180.0},
        "right_knee_angle": {"mean": 175.5, "std": 4.0, "target": 180.0},
    }
}


def load_reference_stats(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load persisted reference statistics for inference corrections."""
    if file_path and file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read reference stats file, using defaults", error=str(e))
    return DEFAULT_REFERENCE_STATS


def save_reference_stats(stats: Dict[str, Any], file_path: Path) -> None:
    """Save reference statistics calculated during training."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
