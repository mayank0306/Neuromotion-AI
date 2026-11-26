"""Joint-angle and repetition summary feature extraction for REHAB24-6."""

from src.features.angles import (
    ANGLE_CONFIG,
    compute_joint_angles,
)
from src.features.summary import (
    build_feature_matrix,
    extract_rep_summary,
)

__all__ = [
    "ANGLE_CONFIG",
    "compute_joint_angles",
    "build_feature_matrix",
    "extract_rep_summary",
]
