"""
Feature engineering module for biomechanical calculations.
Provides shared geometric calculations and kinematic feature extraction for all movement analysis modules.
Reused identically between training and inference.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineering:
    """
    Shared feature engineering for biomechanical calculations.

    Provides methods to calculate:
    - Joint angles from 3D coordinates (with zero-division protection)
    - Segment lengths and Euclidean distances
    - Angular velocities and accelerations
    - Bilateral symmetry metrics (0-1 scale)
    - Range of motion across movement sequences
    - Stability metrics (CV-based and postural sway)
    - Normalized fixed-length feature vectors for ML models
    """

    FEATURE_NAMES = [
        "left_elbow_angle_mean",
        "left_elbow_angle_rom",
        "right_elbow_angle_mean",
        "right_elbow_angle_rom",
        "left_shoulder_angle_mean",
        "left_shoulder_angle_rom",
        "right_shoulder_angle_mean",
        "right_shoulder_angle_rom",
        "left_hip_angle_mean",
        "left_hip_angle_rom",
        "right_hip_angle_mean",
        "right_hip_angle_rom",
        "left_knee_angle_mean",
        "left_knee_angle_rom",
        "right_knee_angle_mean",
        "right_knee_angle_rom",
        "trunk_angle_mean",
        "trunk_angle_rom",
        "head_tilt_mean",
        "shoulder_level_symmetry",
        "hip_level_symmetry",
        "elbow_angle_symmetry",
        "shoulder_angle_symmetry",
        "hip_angle_symmetry",
        "knee_angle_symmetry",
        "step_width_mean",
        "step_width_rom",
        "max_elbow_velocity",
        "max_shoulder_velocity",
        "max_knee_velocity",
        "trunk_stability",
        "overall_stability",
    ]
    NUM_FEATURES = len(FEATURE_NAMES)  # 32 normalized kinematic features

    @staticmethod
    def calculate_angle(point1: Tuple[float, float, float],
                       point2: Tuple[float, float, float],
                       point3: Tuple[float, float, float]) -> float:
        """
        Calculate angle between three points (point2 is the vertex).
        Includes protection against division by zero and numerical instability.

        Args:
            point1: First point (x, y, z)
            point2: Vertex point (x, y, z)
            point3: Third point (x, y, z)

        Returns:
            Angle in degrees (0.0 to 180.0)
        """
        p1 = np.array(point1, dtype=np.float64)
        p2 = np.array(point2, dtype=np.float64)
        p3 = np.array(point3, dtype=np.float64)

        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        norm_product = norm1 * norm2

        if norm_product < 1e-7:
            return 0.0

        cos_angle = np.dot(v1, v2) / norm_product
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return float(np.degrees(angle))

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
        p1 = np.array(point1, dtype=np.float64)
        p2 = np.array(point2, dtype=np.float64)
        return float(np.linalg.norm(p1 - p2))

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
        return np.array(point2, dtype=np.float64) - np.array(point1, dtype=np.float64)

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
        return float((angle2 - angle1) / dt)

    @staticmethod
    def calculate_angular_acceleration(velocity1: float, velocity2: float, dt: float) -> float:
        """
        Calculate angular acceleration.

        Args:
            velocity1: Previous angular velocity in degrees/second
            velocity2: Current angular velocity in degrees/second
            dt: Time difference in seconds

        Returns:
            Angular acceleration in degrees/second^2
        """
        if dt <= 0:
            return 0.0
        return float((velocity2 - velocity1) / dt)

    @staticmethod
    def calculate_symmetry(left_value: float, right_value: float) -> float:
        """
        Calculate symmetry metric between left and right values.
        Returns 1.0 for perfect symmetry, down to 0.0 for extreme asymmetry.

        Args:
            left_value: Left side measurement
            right_value: Right side measurement

        Returns:
            Symmetry score (0.0 to 1.0)
        """
        abs_l = abs(left_value)
        abs_r = abs(right_value)
        max_val = max(abs_l, abs_r)
        if max_val < 1e-7:
            return 1.0
        diff = abs(abs_l - abs_r)
        return float(max(0.0, min(1.0, 1.0 - (diff / max_val))))

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
            Stability score (0.0 to 1.0, higher is more stable)
        """
        if len(values) < 2:
            return 1.0

        vals = np.array(values, dtype=np.float64)
        std_val = float(np.std(vals))
        mean_val = float(np.mean(vals))

        if abs(mean_val) < 1e-7:
            return 1.0 if std_val < 1e-7 else 0.0

        cv = std_val / abs(mean_val)
        return float(max(0.0, min(1.0, 1.0 / (1.0 + cv))))

    @staticmethod
    def normalize_angle(angle: float, reference_mean: float, reference_std: float) -> float:
        """
        Normalize angle deviation from reference mean (z-score clipped to [-3, 3]).

        Args:
            angle: Measured angle
            reference_mean: Mean angle from training data
            reference_std: Standard deviation from training data

        Returns:
            Normalized deviation z-score
        """
        if reference_std <= 1e-7:
            return 0.0
        z_score = (angle - reference_mean) / reference_std
        return float(np.clip(z_score, -3.0, 3.0))

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
                result[name] = (float(lm[0]), float(lm[1]), float(lm[2]))
            else:
                logger.warning(f"Landmark {name} not found in landmark list")

        return result

    def extract_frame_features(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Extract core geometric and joint angle features from a single frame.

        Args:
            landmarks_dict: Dictionary mapping landmark names to (x, y, z) tuples

        Returns:
            Dictionary of calculated joint angles and spatial metrics
        """
        def get_lm(name: str) -> Tuple[float, float, float]:
            return landmarks_dict.get(name, (0.0, 0.0, 0.0))

        # Upper body landmarks
        ls = get_lm("left_shoulder")
        rs = get_lm("right_shoulder")
        le = get_lm("left_elbow")
        re = get_lm("right_elbow")
        lw = get_lm("left_wrist")
        rw = get_lm("right_wrist")
        lear = get_lm("left_ear")
        rear = get_lm("right_ear")

        # Lower body landmarks
        lh = get_lm("left_hip")
        rh = get_lm("right_hip")
        lk = get_lm("left_knee")
        rk = get_lm("right_knee")
        la = get_lm("left_ankle")
        ra = get_lm("right_ankle")

        # Joint angles
        left_elbow_angle = self.calculate_angle(ls, le, lw)
        right_elbow_angle = self.calculate_angle(rs, re, rw)
        left_shoulder_angle = self.calculate_angle(lh, ls, le)
        right_shoulder_angle = self.calculate_angle(rh, rs, re)
        left_hip_angle = self.calculate_angle(ls, lh, lk)
        right_hip_angle = self.calculate_angle(rs, rh, rk)
        left_knee_angle = self.calculate_angle(lh, lk, la)
        right_knee_angle = self.calculate_angle(rh, rk, ra)

        # Trunk angle (spine relative to vertical axis)
        mid_shoulder = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0, (ls[2] + rs[2]) / 2.0)
        mid_hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0, (lh[2] + rh[2]) / 2.0)
        # Angle of spine vector against vertical axis (0, 1, 0)
        spine_top = (mid_hip[0], mid_hip[1] - 1.0, mid_hip[2])
        trunk_angle = self.calculate_angle(spine_top, mid_hip, mid_shoulder)

        # Head tilt angle
        head_tilt = abs(lear[1] - rear[1]) * 100.0  # Normalized percentage

        # Symmetries
        shoulder_level_symmetry = self.calculate_symmetry(ls[1], rs[1])
        hip_level_symmetry = self.calculate_symmetry(lh[1], rh[1])
        elbow_angle_symmetry = self.calculate_symmetry(left_elbow_angle, right_elbow_angle)
        shoulder_angle_symmetry = self.calculate_symmetry(left_shoulder_angle, right_shoulder_angle)
        hip_angle_symmetry = self.calculate_symmetry(left_hip_angle, right_hip_angle)
        knee_angle_symmetry = self.calculate_symmetry(left_knee_angle, right_knee_angle)

        # Step width
        step_width = self.calculate_distance(la, ra)

        return {
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_shoulder_angle": left_shoulder_angle,
            "right_shoulder_angle": right_shoulder_angle,
            "left_hip_angle": left_hip_angle,
            "right_hip_angle": right_hip_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "trunk_angle": trunk_angle,
            "head_tilt": head_tilt,
            "shoulder_level_symmetry": shoulder_level_symmetry,
            "hip_level_symmetry": hip_level_symmetry,
            "elbow_angle_symmetry": elbow_angle_symmetry,
            "shoulder_angle_symmetry": shoulder_angle_symmetry,
            "hip_angle_symmetry": hip_angle_symmetry,
            "knee_angle_symmetry": knee_angle_symmetry,
            "step_width": step_width,
        }

    def extract_sequence_features(self,
                                 landmark_frames: List[Dict[str, Tuple[float, float, float]]],
                                 fps: float = 30.0) -> Dict[str, float]:
        """
        Extract temporal, geometric, ROM, and stability features across a sequence of frames.
        Used identically for video inference and training pipelines.

        Args:
            landmark_frames: Sequence of landmark dictionaries per frame
            fps: Video frames per second

        Returns:
            Dictionary containing 32 summary kinematic features
        """
        if not landmark_frames:
            # Return zeroes if empty
            return {name: 0.0 for name in self.FEATURE_NAMES}

        # Extract frame-by-frame metrics
        frame_metrics: List[Dict[str, float]] = [
            self.extract_frame_features(frame) for frame in landmark_frames
        ]

        def get_series(metric_name: str) -> List[float]:
            return [m[metric_name] for m in frame_metrics]

        dt = 1.0 / max(fps, 1.0)
        num_frames = len(landmark_frames)

        # Joint angle series
        le_series = get_series("left_elbow_angle")
        re_series = get_series("right_elbow_angle")
        ls_series = get_series("left_shoulder_angle")
        rs_series = get_series("right_shoulder_angle")
        lh_series = get_series("left_hip_angle")
        rh_series = get_series("right_hip_angle")
        lk_series = get_series("left_knee_angle")
        rk_series = get_series("right_knee_angle")
        trunk_series = get_series("trunk_angle")
        head_series = get_series("head_tilt")
        step_series = get_series("step_width")

        # Angular velocities
        def calc_max_velocity(series: List[float]) -> float:
            if len(series) < 2:
                return 0.0
            vels = [abs(self.calculate_angle_change(series[i], series[i + 1], dt))
                    for i in range(len(series) - 1)]
            return float(max(vels)) if vels else 0.0

        max_elbow_vel = max(calc_max_velocity(le_series), calc_max_velocity(re_series))
        max_shoulder_vel = max(calc_max_velocity(ls_series), calc_max_velocity(rs_series))
        max_knee_vel = max(calc_max_velocity(lk_series), calc_max_velocity(rk_series))

        # Stabilities
        trunk_stab = self.calculate_stability(trunk_series)
        elbow_stab = (self.calculate_stability(le_series) + self.calculate_stability(re_series)) / 2.0
        knee_stab = (self.calculate_stability(lk_series) + self.calculate_stability(rk_series)) / 2.0
        overall_stab = float(trunk_stab * 0.4 + elbow_stab * 0.3 + knee_stab * 0.3)

        return {
            "left_elbow_angle_mean": float(np.mean(le_series)),
            "left_elbow_angle_rom": self.calculate_range_of_motion(le_series),
            "right_elbow_angle_mean": float(np.mean(re_series)),
            "right_elbow_angle_rom": self.calculate_range_of_motion(re_series),
            "left_shoulder_angle_mean": float(np.mean(ls_series)),
            "left_shoulder_angle_rom": self.calculate_range_of_motion(ls_series),
            "right_shoulder_angle_mean": float(np.mean(rs_series)),
            "right_shoulder_angle_rom": self.calculate_range_of_motion(rs_series),
            "left_hip_angle_mean": float(np.mean(lh_series)),
            "left_hip_angle_rom": self.calculate_range_of_motion(lh_series),
            "right_hip_angle_mean": float(np.mean(rh_series)),
            "right_hip_angle_rom": self.calculate_range_of_motion(rh_series),
            "left_knee_angle_mean": float(np.mean(lk_series)),
            "left_knee_angle_rom": self.calculate_range_of_motion(lk_series),
            "right_knee_angle_mean": float(np.mean(rk_series)),
            "right_knee_angle_rom": self.calculate_range_of_motion(rk_series),
            "trunk_angle_mean": float(np.mean(trunk_series)),
            "trunk_angle_rom": self.calculate_range_of_motion(trunk_series),
            "head_tilt_mean": float(np.mean(head_series)),
            "shoulder_level_symmetry": float(np.mean(get_series("shoulder_level_symmetry"))),
            "hip_level_symmetry": float(np.mean(get_series("hip_level_symmetry"))),
            "elbow_angle_symmetry": float(np.mean(get_series("elbow_angle_symmetry"))),
            "shoulder_angle_symmetry": float(np.mean(get_series("shoulder_angle_symmetry"))),
            "hip_angle_symmetry": float(np.mean(get_series("hip_angle_symmetry"))),
            "knee_angle_symmetry": float(np.mean(get_series("knee_angle_symmetry"))),
            "step_width_mean": float(np.mean(step_series)),
            "step_width_rom": self.calculate_range_of_motion(step_series),
            "max_elbow_velocity": max_elbow_vel,
            "max_shoulder_velocity": max_shoulder_vel,
            "max_knee_velocity": max_knee_vel,
            "trunk_stability": trunk_stab,
            "overall_stability": overall_stab,
        }

    def compute_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert sequence features into a standardized 1D numpy array of length NUM_FEATURES (32).
        Guarantees identical ordering and normalization between training and inference.

        Args:
            features_dict: Extracted feature dictionary

        Returns:
            1D numpy array of shape (32,)
        """
        vec = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        for idx, feat_name in enumerate(self.FEATURE_NAMES):
            vec[idx] = float(features_dict.get(feat_name, 0.0))
        return vec
