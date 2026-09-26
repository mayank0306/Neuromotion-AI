"""
Feature engineering module for biomechanical calculations.
Provides shared geometric calculations for all movement analysis modules.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineering:
    """
    Shared feature engineering for biomechanical calculations.

    Provides methods to calculate:
    - Joint angles from 3D coordinates
    - Segment lengths
    - Angular velocities and accelerations
    - Symmetry metrics
    - Range of motion
    - Stability metrics
    """

    @staticmethod
    def calculate_angle(point1: Tuple[float, float, float],
                       point2: Tuple[float, float, float],
                       point3: Tuple[float, float, float]) -> float:
        """
        Calculate angle between three points (point2 is the vertex).

        Args:
            point1: First point (x, y, z)
            point2: Vertex point (x, y, z)
            point3: Third point (x, y, z)

        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Create vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    @staticmethod
    def calculate_distance(point1: Tuple[float, float, float],
                          point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point (x, y, z)
            point2: Second point (x, y, z)

        Returns:
            Distance
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def calculate_vector(point1: Tuple[float, float, float],
                        point2: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate vector from point1 to point2.

        Args:
            point1: Start point (x, y, z)
            point2: End point (x, y, z)

        Returns:
            Vector as numpy array
        """
        return np.array(point2) - np.array(point1)

    @staticmethod
    def calculate_angle_change(angle1: float, angle2: float, dt: float) -> float:
        """
        Calculate angular velocity.

        Args:
            angle1: Previous angle in degrees
            angle2: Current angle in degrees
            dt: Time difference in seconds

        Returns:
            Angular velocity in degrees/second
        """
        if dt <= 0:
            return 0.0
        return (angle2 - angle1) / dt

    @staticmethod
    def calculate_symmetry(left_value: float, right_value: float) -> float:
        """
        Calculate symmetry metric between left and right values.

        Args:
            left_value: Left side measurement
            right_value: Right side measurement

        Returns:
            Symmetry score (0-1, where 1 is perfect symmetry)
        """
        if left_value == 0 and right_value == 0:
            return 1.0
        max_val = max(abs(left_value), abs(right_value))
        if max_val == 0:
            return 1.0
        return 1.0 - (abs(left_value - right_value) / max_val)

    @staticmethod
    def calculate_range_of_motion(values: List[float]) -> float:
        """
        Calculate range of motion from a list of values.

        Args:
            values: List of measurements (e.g., joint angles over time)

        Returns:
            Range of motion (max - min)
        """
        if not values:
            return 0.0
        return float(max(values) - min(values))

    @staticmethod
    def calculate_stability(values: List[float]) -> float:
        """
        Calculate stability metric (inverse of coefficient of variation).

        Args:
            values: List of measurements over time

        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(values) < 2:
            return 1.0

        mean_val = np.mean(values)
        if mean_val == 0:
            return 1.0 if np.std(values) == 0 else 0.0

        cv = np.std(values) / abs(mean_val)
        # Convert to stability score (0-1)
        return max(0.0, min(1.0, 1.0 / (1.0 + cv)))

    @staticmethod
    def normalize_angle(angle: float, reference_mean: float, reference_std: float) -> float:
        """
        Normalize angle deviation from reference mean.

        Args:
            angle: Measured angle
            reference_mean: Mean angle from training data
            reference_std: Standard deviation from training data

        Returns:
            Normalized deviation (z-score clipped to reasonable range)
        """
        if reference_std == 0:
            return 0.0
        z_score = (angle - reference_mean) / reference_std
        # Clip to +/- 3 standard deviations
        return np.clip(z_score, -3.0, 3.0)

    @staticmethod
    def extract_landmarks_by_name(landmarks: List[Tuple[float, float, float, float]],
                                 landmark_names: List[str],
                                 target_names: List[str]) -> Dict[str, Tuple[float, float, float]]:
        """
        Extract specific landmarks by name from MediaPipe landmark list.

        Args:
            landmarks: List of (x, y, z, visibility) tuples
            landmark_names: List of landmark names in order
            target_names: List of target landmark names to extract

        Returns:
            Dictionary mapping landmark name to (x, y, z) tuple
        """
        result = {}
        landmark_dict = dict(zip(landmark_names, landmarks))

        for name in target_names:
            if name in landmark_dict:
                lm = landmark_dict[name]
                result[name] = (lm[0], lm[1], lm[2])  # x, y, z (ignore visibility for now)
            else:
                logger.warning(f"Landmark {name} not found in landmark list")

        return result