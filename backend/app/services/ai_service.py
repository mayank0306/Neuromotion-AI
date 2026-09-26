"""AI service for pose estimation and analysis."""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import structlog
from datetime import datetime

from app.core.config import settings
from app.schemas.movement import AnalysisResponse, AnalysisRequest, PoseLandmarkSchema
from app.exceptions import AppException
from app.analysis.feature_engineering import FeatureEngineering

logger = structlog.get_logger(__name__)


class AIService:
    """
    AI service for pose estimation and movement analysis.

    Uses MediaPipe BlazePose for real-time pose detection and
    specialized models for different movement analysis modules.
    """

    def __init__(self):
        self.pose_estimator = None
        self.feature_engineering = FeatureEngineering()
        self._loaded = False

        # MediaPipe pose landmark names (33 landmarks)
        self.LANDMARK_NAMES = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky_finger",
            "right_pinky_finger",
            "left_index_finger",
            "right_index_finger",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]

    async def load_model(self):
        """Load AI models (lazy initialization)."""
        if self._loaded:
            return

        logger.info("Loading AI models...")

        try:
            # Lazy import to avoid startup overhead
            import mediapipe as mp

            self.pose_estimator = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
                min_tracking_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
            )

            self._loaded = True
            logger.info("AI models loaded successfully")

        except ImportError:
            logger.warning("MediaPipe not available, using mock model")
            self._loaded = True  # Mark as loaded even if using mock

    async def estimate_pose(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Estimate pose from an image frame.

        Args:
            image: RGB image as numpy array.

        Returns:
            List of (x, y, z, visibility) tuples for each landmark.
        """
        if not self._loaded:
            await self.load_model()

        try:
            import cv2

            # Convert to RGB if needed
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = self.pose_estimator.process(image_rgb)

            if results.pose_landmarks:
                landmarks = [
                    (lm.x, lm.y, lm.z, lm.visibility)
                    for lm in results.pose_landmarks.landmark
                ]
                return landmarks

            return None

        except Exception as e:
            logger.error("Pose estimation failed", error=str(e))
            return None

    def _convert_to_landmark_dict(self, landmarks: List[Tuple[float, float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """
        Convert MediaPipe landmarks to dictionary by name.

        Args:
            landmarks: List of (x, y, z, visibility) tuples

        Returns:
            Dictionary mapping landmark name to (x, y, z) tuple
        """
        landmark_dict = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            if i < len(landmarks):
                lm = landmarks[i]
                landmark_dict[name] = (lm[0], lm[1], lm[2])  # x, y, z (ignore visibility for calculations)
        return landmark_dict

    async def analyze_yoga_pose(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Analyze yoga pose using classification approach.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples

        Returns:
            Analysis results for yoga pose
        """
        # TODO: Implement actual yoga pose classification model
        # For now, return placeholder with measured/estimated tags

        # Calculate some basic angles for demonstration
        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        left_elbow = landmarks_dict.get("left_elbow", (0, 0, 0))
        left_wrist = landmarks_dict.get("left_wrist", (0, 0, 0))

        elbow_angle = self.feature_engineering.calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Mock classification result
        pose_name = "warrior_ii"  # This would come from actual model
        confidence = 0.85

        # Calculate metrics
        metrics = {
            "left_elbow_angle": MetricWithTag(
                value=elbow_angle,
                is_measured=True,
                description="Left elbow angle in degrees"
            ),
            "pose_confidence": MetricWithTag(
                value=confidence,
                is_measured=False,
                description="Model confidence in pose classification"
            )
        }

        # Calculate overall score (0-100)
        overall_score = confidence * 100

        # Score breakdown
        score_breakdown = {
            "pose_accuracy": confidence * 100,
            "angle_accuracy": 80.0,  # Placeholder
            "symmetry": 75.0,  # Placeholder
            "stability": 85.0   # Placeholder
        }

        # Feedback
        feedback = [
            f"Detected pose: {pose_name.replace('_', ' ').title()} with {confidence:.0%} confidence",
            f"Left elbow angle: {elbow_angle:.1f}°"
        ]

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": {}
        }

    async def analyze_rehab_movement(self, landmarks_dict: Dict[str, Tuple[float, float, float]], exercise_type: str) -> Dict[str, Any]:
        """
        Analyze rehabilitation movement using quality score prediction.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples
            exercise_type: Type of rehab exercise being performed

        Returns:
            Analysis results for rehab movement
        """
        # TODO: Implement actual rehab quality model (classification + regression)
        # For now, return placeholder with measured/estimated tags

        # Calculate some basic metrics for demonstration
        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        right_shoulder = landmarks_dict.get("right_shoulder", (0, 0, 0))
        left_elbow = landmarks_dict.get("left_elbow", (0, 0, 0))
        right_elbow = landmarks_dict.get("right_elbow", (0, 0, 0))

        # Calculate shoulder symmetry
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])  # y-coordinate difference
        shoulder_symmetry = self.feature_engineering.calculate_symmetry(
            left_shoulder[1], right_shoulder[1]
        )

        # Calculate elbow angles
        left_elbow_angle = self.feature_engineering.calculate_angle(
            left_shoulder, left_elbow,
            landmarks_dict.get("left_wrist", (0, 0, 0))
        ) if "left_wrist" in landmarks_dict else 0

        right_elbow_angle = self.feature_engineering.calculate_angle(
            right_shoulder, right_elbow,
            landmarks_dict.get("right_wrist", (0, 0, 0))
        ) if "right_wrist" in landmarks_dict else 0

        elbow_symmetry = self.feature_engineering.calculate_symmetry(left_elbow_angle, right_elbow_angle)

        # Mock quality score (0-100)
        # In reality, this would come from a trained model
        base_score = 85.0
        symmetry_bonus = (shoulder_symmetry + elbow_symmetry) * 10
        quality_score = min(100.0, base_score + symmetry_bonus)

        # Calculate metrics
        metrics = {
            "left_shoulder_y": MetricWithTag(
                value=left_shoulder[1],
                is_measured=True,
                description="Left shoulder Y coordinate"
            ),
            "right_shoulder_y": MetricWithTag(
                value=right_shoulder[1],
                is_measured=True,
                description="Right shoulder Y coordinate"
            ),
            "shoulder_symmetry": MetricWithTag(
                value=shoulder_symmetry,
                is_measured=True,
                description="Shoulder height symmetry (0-1)"
            ),
            "left_elbow_angle": MetricWithTag(
                value=left_elbow_angle,
                is_measured=True,
                description="Left elbow angle in degrees"
            ),
            "right_elbow_angle": MetricWithTag(
                value=right_elbow_angle,
                is_measured=True,
                description="Right elbow angle in degrees"
            ),
            "elbow_symmetry": MetricWithTag(
                value=elbow_symmetry,
                is_measured=True,
                description="Elbow angle symmetry (0-1)"
            ),
            "quality_score": MetricWithTag(
                value=quality_score,
                is_measured=False,
                description="Predicted movement quality score (0-100)"
            )
        }

        # Score breakdown based on configurable weights
        angle_accuracy = min(100.0, 100.0 - abs(left_elbow_angle - 90.0))  # Example: target 90°
        score_breakdown = {
            "angle_accuracy": angle_accuracy * 0.4,  # 40% weight
            "symmetry": ((shoulder_symmetry + elbow_symmetry) / 2) * 100 * 0.2,  # 20% weight
            "stability": 85.0 * 0.25,  # 25% weight (placeholder)
            "range_of_motion": 80.0 * 0.15  # 15% weight (placeholder)
        }

        overall_score = sum(score_breakdown.values())

        # Feedback
        feedback = [
            f"Exercise: {exercise_type.replace('-', ' ').title()}",
            f"Quality Score: {quality_score:.1f}/100"
        ]

        if shoulder_symmetry < 0.8:
            feedback.append("Try to keep shoulders level during the movement")
        if elbow_symmetry < 0.8:
            feedback.append("Work on keeping both elbows moving symmetrically")

        # Explanations for low scores
        explanations = {}
        if shoulder_symmetry < 0.7:
            explanations["shoulder_symmetry"] = "Asymmetric shoulder height detected - may indicate compensatory movement"
        if elbow_symmetry < 0.7:
            explanations["elbow_symmetry"] = "Asymmetric elbow movement - consider focusing on equal bilateral movement"

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations
        }

    async def analyze_posture(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Analyze posture using geometric calculations.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples

        Returns:
            Analysis results for posture
        """
        # Calculate key posture metrics
        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        right_shoulder = landmarks_dict.get("right_shoulder", (0, 0, 0))
        left_ear = landmarks_dict.get("left_ear", (0, 0, 0))
        right_ear = landmarks_dict.get("right_ear", (0, 0, 0))
        left_hip = landmarks_dict.get("left_hip", (0, 0, 0))
        right_hip = landmarks_dict.get("right_hip", (0, 0, 0))

        # Shoulder level
        shoulder_symmetry = self.feature_engineering.calculate_symmetry(
            left_shoulder[1], right_shoulder[1]
        )

        # Head position (ear to shoulder)
        left_head_tilt = abs(left_ear[1] - left_shoulder[1]) if left_ear and left_shoulder else 0
        right_head_tilt = abs(right_ear[1] - right_shoulder[1]) if right_ear and right_shoulder else 0

        # Pelvic level
        pelvic_symmetry = self.feature_engineering.calculate_symmetry(
            left_hip[1], right_hip[1]
        )

        # Spinal alignment (simplified)
        shoulder_midpoint_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_midpoint_y = (left_hip[1] + right_hip[1]) / 2
        spinal_vertical_alignment = abs(shoulder_midpoint_y - hip_midpoint_y)

        # Mock overall score
        overall_score = (
            shoulder_symmetry * 0.3 +
            (1.0 - min(left_head_tilt, right_head_tilt) * 5) * 0.2 +  # Head tilt penalty
            pelvic_symmetry * 0.3 +
            (1.0 - min(spinal_vertical_alignment * 10, 1.0)) * 0.2  # Spinal alignment
        ) * 100

        # Calculate metrics
        metrics = {
            "left_shoulder_y": MetricWithTag(
                value=left_shoulder[1],
                is_measured=True,
                description="Left shoulder Y coordinate"
            ),
            "right_shoulder_y": MetricWithTag(
                value=right_shoulder[1],
                is_measured=True,
                description="Right shoulder Y coordinate"
            ),
            "shoulder_symmetry": MetricWithTag(
                value=shoulder_symmetry,
                is_measured=True,
                description="Shoulder level symmetry (0-1)"
            ),
            "left_head_tilt": MetricWithTag(
                value=left_head_tilt,
                is_measured=True,
                description="Left ear to shoulder vertical distance"
            ),
            "right_head_tilt": MetricWithTag(
                value=right_head_tilt,
                is_measured=True,
                description="Right ear to shoulder vertical distance"
            ),
            "pelvic_symmetry": MetricWithTag(
                value=pelvic_symmetry,
                is_measured=True,
                description="Pelvic level symmetry (0-1)"
            ),
            "spinal_alignment": MetricWithTag(
                value=spinal_vertical_alignment,
                is_measured=True,
                description="Vertical alignment between shoulders and hips"
            ),
            "posture_score": MetricWithTag(
                value=overall_score,
                is_measured=False,
                description="Overall posture score (0-100)"
            )
        }

        # Score breakdown
        score_breakdown = {
            "shoulder_level": shoulder_symmetry * 100 * 0.3,
            "head_position": (1.0 - min(left_head_tilt, right_head_tilt) * 5) * 100 * 0.2,
            "pelvic_level": pelvic_symmetry * 100 * 0.3,
            "spinal_alignment": (1.0 - min(spinal_vertical_alignment * 10, 1.0)) * 100 * 0.2
        }

        # Feedback
        feedback = [
            f"Posture Score: {overall_score:.1f}/100"
        ]

        if shoulder_symmetry < 0.8:
            feedback.append("Work on keeping shoulders level")
        if left_head_tilt > 0.05 or right_head_tilt > 0.05:  # 5cm threshold
            feedback.append("Try to keep head centered over shoulders")
        if pelvic_symmetry < 0.8:
            feedback.append("Focus on keeping hips level")

        # Explanations
        explanations = {}
        if shoulder_symmetry < 0.7:
            explanations["shoulder_level"] = "Uneven shoulder height may indicate muscle imbalance or compensatory patterns"
        if left_head_tilt > 0.1 or right_head_tilt > 0.1:
            explanations["head_position"] = "Excessive head tilt may indicate neck strain or poor posture habits"

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations
        }

    async def analyze_gait(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Analyze gait using geometric calculations.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples

        Returns:
            Analysis results for gait
        """
        # For single frame analysis, we can only assess static measures
        # For temporal analysis, we would need a sequence of poses

        left_ankle = landmarks_dict.get("left_ankle", (0, 0, 0))
        right_ankle = landmarks_dict.get("right_ankle", (0, 0, 0))
        left_knee = landmarks_dict.get("left_knee", (0, 0, 0))
        right_knee = landmarks_dict.get("right_knee", (0, 0, 0))
        left_hip = landmarks_dict.get("left_hip", (0, 0, 0))
        right_hip = landmarks_dict.get("right_hip", (0, 0, 0))
        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        right_shoulder = landmarks_dict.get("right_shoulder", (0, 0, 0))

        # Step width (distance between ankles)
        step_width = self.feature_engineering.calculate_distance(left_ankle, right_ankle)

        # Leg symmetry (knee heights)
        left_knee_height = left_knee[1] if left_knee else 0
        right_knee_height = right_knee[1] if right_knee else 0
        knee_symmetry = self.feature_engineering.calculate_symmetry(left_knee_height, right_knee_height)

        # Hip symmetry
        left_hip_height = left_hip[1] if left_hip else 0
        right_hip_height = right_hip[1] if right_hip else 0
        hip_symmetry = self.feature_engineering.calculate_symmetry(left_hip_height, right_hip_height)

        # Shoulder symmetry
        left_shoulder_y = left_shoulder[1] if left_shoulder else 0
        right_shoulder_y = right_shoulder[1] if right_shoulder else 0
        shoulder_symmetry = self.feature_engineering.calculate_symmetry(left_shoulder_y, right_shoulder_y)

        # Mock overall score
        overall_score = (
            knee_symmetry * 0.3 +
            hip_symmetry * 0.3 +
            shoulder_symmetry * 0.2 +
            (1.0 - min(step_width * 5, 1.0)) * 0.2  # Ideal step width penalty
        ) * 100

        # Calculate metrics
        metrics = {
            "left_ankle_y": MetricWithTag(
                value=left_ankle[1],
                is_measured=True,
                description="Left ankle Y coordinate"
            ),
            "right_ankle_y": MetricWithTag(
                value=right_ankle[1],
                is_measured=True,
                description="Right ankle Y coordinate"
            ),
            "step_width": MetricWithTag(
                value=step_width,
                is_measured=True,
                description="Distance between ankles (step width)"
            ),
            "left_knee_y": MetricWithTag(
                value=left_knee[1],
                is_measured=True,
                description="Left knee Y coordinate"
            ),
            "right_knee_y": MetricWithTag(
                value=right_knee[1],
                is_measured=True,
                description="Right knee Y coordinate"
            ),
            "knee_symmetry": MetricWithTag(
                value=knee_symmetry,
                is_measured=True,
                description="Knee height symmetry (0-1)"
            ),
            "left_hip_y": MetricWithTag(
                value=left_hip[1],
                is_measured=True,
                description="Left hip Y coordinate"
            ),
            "right_hip_y": MetricWithTag(
                value=right_hip[1],
                is_measured=True,
                description="Right hip Y coordinate"
            ),
            "hip_symmetry": MetricWithTag(
                value=hip_symmetry,
                is_measured=True,
                description="Hip height symmetry (0-1)"
            ),
            "left_shoulder_y": MetricWithTag(
                value=left_shoulder[1],
                is_measured=True,
                description="Left shoulder Y coordinate"
            ),
            "right_shoulder_y": MetricWithTag(
                value=right_shoulder[1],
                is_measured=True,
                description="Right shoulder Y coordinate"
            ),
            "shoulder_symmetry": MetricWithTag(
                value=shoulder_symmetry,
                is_measured=True,
                description="Shoulder height symmetry (0-1)"
            ),
            "gait_score": MetricWithTag(
                value=overall_score,
                is_measured=False,
                description="Overall gait score (0-100)"
            )
        }

        # Score breakdown
        score_breakdown = {
            "step_symmetry": knee_symmetry * 100 * 0.3,
            "hip_level": hip_symmetry * 100 * 0.3,
            "shoulder_level": shoulder_symmetry * 100 * 0.2,
            "step_width": (1.0 - min(step_width * 5, 1.0)) * 100 * 0.2
        }

        # Feedback
        feedback = [
            f"Gait Score: {overall_score:.1f}/100",
            f"Step Width: {step_width:.2f}m"
        ]

        if knee_symmetry < 0.8:
            feedback.append("Work on symmetrical knee movement")
        if hip_symmetry < 0.8:
            feedback.append("Focus on keeping hips level during walking")
        if step_width < 0.1:  # Too narrow
            feedback.append("Try to widen your stance slightly")
        elif step_width > 0.3:  # Too wide
            feedback.append("Try to narrow your stance slightly")

        # Explanations
        explanations = {}
        if knee_symmetry < 0.7:
            explanations["knee_symmetry"] = "Asymmetric knee movement may indicate pain, weakness, or protective movement patterns"
        if hip_symmetry < 0.7:
            explanations["hip_level"] = "Uneven hip height may suggest pelvic tilt or leg length discrepancy"

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations
        }

    async def analyze_balance(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Analyze balance using geometric calculations.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples

        Returns:
            Analysis results for balance
        """
        # For balance, we primarily look at center of mass and sway
        # Using ankle, knee, hip, and shoulder positions to estimate stability

        left_ankle = landmarks_dict.get("left_ankle", (0, 0, 0))
        right_ankle = landmarks_dict.get("right_ankle", (0, 0, 0))
        left_knee = landmarks_dict.get("left_knee", (0, 0, 0))
        right_knee = landmarks_dict.get("right_knee", (0, 0, 0))
        left_hip = landmarks_dict.get("left_hip", (0, 0, 0))
        right_hip = landmarks_dict.get("right_hip", (0, 0, 0))
        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        right_shoulder = landmarks_dict.get("right_shoulder", (0, 0, 0))

        # Base of support (ankle width)
        base_width = self.feature_engineering.calculate_distance(left_ankle, right_ankle)

        # Center of mass approximation (midpoint between shoulders and hips)
        left_com_x = (left_shoulder[0] + left_hip[0]) / 2
        right_com_x = (right_shoulder[0] + right_hip[0]) / 2
        com_x = (left_com_x + right_com_x) / 2

        # Midpoint of base of support
        base_midpoint_x = (left_ankle[0] + right_ankle[0]) / 2

        # Lateral stability (how far COM is from base midpoint)
        lateral_stability = abs(com_x - base_midpoint_x)

        # Vertical alignment (ankle to shoulder)
        left_vertical_alignment = abs(left_ankle[0] - left_shoulder[0])
        right_vertical_alignment = abs(right_ankle[0] - right_shoulder[0])
        avg_vertical_alignment = (left_vertical_alignment + right_vertical_alignment) / 2

        # Weight distribution (heel/toe - simplified using ankle Y position)
        left_ankle_y = left_ankle[1] if left_ankle else 0
        right_ankle_y = right_ankle[1] if right_ankle else 0
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2

        # Mock overall score
        overall_score = (
            (1.0 - min(lateral_stability * 10, 1.0)) * 0.4 +  # Lateral stability
            (1.0 - min(avg_vertical_alignment * 5, 1.0)) * 0.3 +  # Vertical alignment
            min(base_width * 3, 1.0) * 0.2 +  # Base of support adequacy
            (1.0 - min(abs(avg_ankle_y) * 10, 1.0)) * 0.1  # Ankle position
        ) * 100

        # Calculate metrics
        metrics = {
            "left_ankle_x": MetricWithTag(
                value=left_ankle[0],
                is_measured=True,
                description="Left ankle X coordinate"
            ),
            "right_ankle_x": MetricWithTag(
                value=right_ankle[0],
                is_measured=True,
                description="Right ankle X coordinate"
            ),
            "base_width": MetricWithTag(
                value=base_width,
                is_measured=True,
                description="Distance between ankles (base of support)"
            ),
            "left_com_x": MetricWithTag(
                value=left_com_x,
                is_measured=True,
                description="Left side center of mass estimate X"
            ),
            "right_com_x": MetricWithTag(
                value=right_com_x,
                is_measured=True,
                description="Right side center of mass estimate X"
            ),
            "com_x": MetricWithTag(
                value=com_x,
                is_measured=True,
                description="Overall center of mass estimate X"
            ),
            "base_midpoint_x": MetricWithTag(
                value=base_midpoint_x,
                is_measured=True,
                description="Midpoint of base of support X"
            ),
            "lateral_stability": MetricWithTag(
                value=lateral_stability,
                is_measured=True,
                description="Lateral stability (distance from COM to base midpoint)"
            ),
            "left_vertical_alignment": MetricWithTag(
                value=left_vertical_alignment,
                is_measured=True,
                description="Left ankle-shoulder vertical alignment"
            ),
            "right_vertical_alignment": MetricWithTag(
                value=right_vertical_alignment,
                is_measured=True,
                description="Right ankle-shoulder vertical alignment"
            ),
            "avg_vertical_alignment": MetricWithTag(
                value=avg_vertical_alignment,
                is_measured=True,
                description="Average vertical alignment"
            ),
            "left_ankle_y": MetricWithTag(
                value=left_ankle_y,
                is_measured=True,
                description="Left ankle Y coordinate (heel/toe indicator)"
            ),
            "right_ankle_y": MetricWithTag(
                value=right_ankle_y,
                is_measured=True,
                description="Right ankle Y coordinate (heel/toe indicator)"
            ),
            "balance_score": MetricWithTag(
                value=overall_score,
                is_measured=False,
                description="Overall balance score (0-100)"
            )
        }

        # Score breakdown
        score_breakdown = {
            "lateral_stability": (1.0 - min(lateral_stability * 10, 1.0)) * 100 * 0.4,
            "vertical_alignment": (1.0 - min(avg_vertical_alignment * 5, 1.0)) * 100 * 0.3,
            "base_of_support": min(base_width * 3, 1.0) * 100 * 0.2,
            "ankle_position": (1.0 - min(abs(avg_ankle_y) * 10, 1.0)) * 100 * 0.1
        }

        # Feedback
        feedback = [
            f"Balance Score: {overall_score:.1f}/100",
            f"Base Width: {base_width:.2f}m"
        ]

        if lateral_stability > 0.05:  # 5cm threshold
            feedback.append("Work on keeping your center of mass over your base of support")
        if base_width < 0.1:
            feedback.append("Try widening your stance slightly for better stability")
        elif base_width > 0.3:
            feedback.append("Your stance may be too wide - try narrowing it slightly")

        # Explanations
        explanations = {}
        if lateral_stability > 0.1:
            explanations["lateral_stability"] = "Excessive lateral sway may indicate balance deficits or proprioceptive issues"
        if base_width < 0.05:
            explanations["base_of_support"] = "Very narrow base of support reduces stability and increases fall risk"

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations
        }

    async def analyze_flexibility(self, landmarks_dict: Dict[str, Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Analyze flexibility using geometric calculations.

        Args:
            landmarks_dict: Dictionary of landmark names to (x, y, z) tuples

        Returns:
            Analysis results for flexibility
        """
        # For flexibility, we look at range of motion at various joints
        # Since we only have a single frame, we assess current position relative to norms

        left_shoulder = landmarks_dict.get("left_shoulder", (0, 0, 0))
        right_shoulder = landmarks_dict.get("right_shoulder", (0, 0, 0))
        left_elbow = landmarks_dict.get("left_elbow", (0, 0, 0))
        right_elbow = landmarks_dict.get("right_elbow", (0, 0, 0))
        left_wrist = landmarks_dict.get("left_wrist", (0, 0, 0))
        right_wrist = landmarks_dict.get("right_wrist", (0, 0, 0))
        left_hip = landmarks_dict.get("left_hip", (0, 0, 0))
        right_hip = landmarks_dict.get("right_hip", (0, 0, 0))
        left_knee = landmarks_dict.get("left_knee", (0, 0, 0))
        right_knee = landmarks_dict.get("right_knee", (0, 0, 0))
        left_ankle = landmarks_dict.get("left_ankle", (0, 0, 0))
        right_ankle = landmarks_dict.get("right_ankle", (0, 0, 0))

        # Shoulder flexion (approximated by arm angle relative to torso)
        # Simplified: using Y coordinate difference between wrist and shoulder
        left_shoulder_flexion = abs(left_wrist[1] - left_shoulder[1]) if left_wrist and left_shoulder else 0
        right_shoulder_flexion = abs(right_wrist[1] - right_shoulder[1]) if right_wrist and right_shoulder else 0

        # Hip flexion (knee height relative to hip)
        left_hip_flexion = abs(left_knee[1] - left_hip[1]) if left_knee and left_hip else 0
        right_hip_flexion = abs(right_knee[1] - right_hip[1]) if right_knee and right_hip else 0

        # Ankle dorsiflexion (toe up/down relative to foot)
        # Simplified using ankle Y position
        left_ankle_dorsi = left_ankle[1] if left_ankle else 0
        right_ankle_dorsi = right_ankle[1] if right_ankle else 0

        # Trunk flexibility (approximated by shoulder to hip angle)
        left_trunk_flex = self.feature_engineering.calculate_angle(
            left_shoulder, left_hip,
            landmarks_dict.get("left_knee", (0, 0, 0))
        ) if left_shoulder and left_hip and landmarks_dict.get("left_knee") else 0
        right_trunk_flex = self.feature_engineering.calculate_angle(
            right_shoulder, right_hip,
            landmarks_dict.get("right_knee", (0, 0, 0))
        ) if right_shoulder and right_hip and landmarks_dict.get("right_knee") else 0

        # Mock overall score based on flexibility measures
        # In reality, these would be compared to normative values
        shoulder_score = min(left_shoulder_flexion, right_shoulder_flexion) * 2  # Arbitrary scaling
        hip_score = min(left_hip_flexion, right_hip_flexion) * 2
        ankle_score = (1.0 - min(abs(left_ankle_dorsi) * 5, 1.0)) * 50 + \
                     (1.0 - min(abs(right_ankle_dorsi) * 5, 1.0)) * 50
        trunk_score = min(left_trunk_flex, right_trunk_flex) * 2

        overall_score = (shoulder_score + hip_score + ankle_score + trunk_score) / 4
        overall_score = min(100.0, max(0.0, overall_score))

        # Calculate metrics
        metrics = {
            "left_shoulder_flexion": MetricWithTag(
                value=left_shoulder_flexion,
                is_measured=True,
                description="Left shoulder flexion estimation"
            ),
            "right_shoulder_flexion": MetricWithTag(
                value=right_shoulder_flexion,
                is_measured=True,
                description="Right shoulder flexion estimation"
            ),
            "left_hip_flexion": MetricWithTag(
                value=left_hip_flexion,
                is_measured=True,
                description="Left hip flexion estimation"
            ),
            "right_hip_flexion": MetricWithTag(
                value=right_hip_flexion,
                is_measured=True,
                description="Right hip flexion estimation"
            ),
            "left_ankle_position": MetricWithTag(
                value=left_ankle_dorsi,
                is_measured=True,
                description="Left ankle position (dorsi/plantar flexion)"
            ),
            "right_ankle_position": MetricWithTag(
                value=right_ankle_dorsi,
                is_measured=True,
                description="Right ankle position (dorsi/plantar flexion)"
            ),
            "left_trunk_flexion": MetricWithTag(
                value=left_trunk_flex,
                is_measured=True,
                description="Left trunk flexion estimation"
            ),
            "right_trunk_flexion": MetricWithTag(
                value=right_trunk_flex,
                is_measured=True,
                description="Right trunk flexion estimation"
            ),
            "flexibility_score": MetricWithTag(
                value=overall_score,
                is_measured=False,
                description="Overall flexibility score (0-100)"
            )
        }

        # Score breakdown
        score_breakdown = {
            "shoulder_flexibility": min(shoulder_score, 100) * 0.25,
            "hip_flexibility": min(hip_score, 100) * 0.25,
            "ankle_flexibility": min(ankle_score, 100) * 0.25,
            "trunk_flexibility": min(trunk_score, 100) * 0.25
        }

        # Feedback
        feedback = [
            f"Flexibility Score: {overall_score:.1f}/100"
        ]

        if shoulder_score < 30:
            feedback.append("Consider shoulder mobility exercises")
        if hip_score < 30:
            feedback.append("Hip flexibility may benefit from stretching routines")
        if ankle_score < 40:
            feedback.append("Ankle mobility work could improve movement quality")
        if trunk_score < 30:
            feedback.append("Core and spinal flexibility exercises may be beneficial")

        # Explanations
        explanations = {}
        if shoulder_score < 25:
            explanations["shoulder_flexibility"] = "Limited shoulder mobility may affect overhead movements and reaching"
        if hip_score < 25:
            explanations["hip_flexibility"] = "Restricted hip motion may impact walking, squatting, and transitional movements"
        if ankle_score < 30:
            explanations["ankle_flexibility"] = "Ankle mobility limitations can affect balance and gait patterns"

        return {
            "overall_score": overall_score,
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations
        }

    async def analyze_form(
        self, poses: List[List[Tuple[float, float, float, float]]]
    ) -> dict:
        """
        Analyze form from a sequence of poses.

        Args:
            poses: List of pose landmarks (each pose is list of (x,y,z,visibility) tuples)

        Returns:
            Form analysis results with scores and feedback.
        """
        # For now, use the first pose for analysis
        # In future versions, we could analyze temporal patterns
        if not poses or len(poses) == 0:
            return {
                "overall_score": 0.0,
                "score_breakdown": {},
                "metrics": {},
                "feedback": ["No pose data available for analysis"],
                "explanations": {},
                "processing_time_ms": 0,
                "exercise_type": "unknown"
            }

        # Use first pose for analysis (could be enhanced to use multiple poses)
        first_pose = poses[0]
        landmarks_dict = self._convert_to_landmark_dict(first_pose)

        # Determine exercise type from context (this would come from request in real implementation)
        # For now, default to rehab assessment as the flagship module
        exercise_type = "rehab_assessment"

        # Route to appropriate analysis method
        if exercise_type == "yoga":
            return await self.analyze_yoga_pose(landmarks_dict)
        elif exercise_type.startswith("rehab") or exercise_type == "rehab_assessment":
            # Extract specific exercise type from context
            specific_exercise = "shoulder-abduction"  # Would come from request
            return await self.analyze_rehab_movement(landmarks_dict, specific_exercise)
        elif exercise_type == "posture":
            return await self.analyze_posture(landmarks_dict)
        elif exercise_type == "gait":
            return await self.analyze_gait(landmarks_dict)
        elif exercise_type == "balance":
            return await self.analyze_balance(landmarks_dict)
        elif exercise_type == "flexibility":
            return await self.analyze_flexibility(landmarks_dict)
        else:
            # Default to rehab assessment
            return await self.analyze_rehab_movement(landmarks_dict, "shoulder-abduction")
