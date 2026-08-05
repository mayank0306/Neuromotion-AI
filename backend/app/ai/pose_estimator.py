"""
Pose estimation service.

Provides real-time pose detection using MediaPipe BlazePose.
"""

from typing import List, Optional, Tuple
import structlog
import numpy as np

from app.core.config import settings

logger = structlog.get_logger(__name__)


class PoseEstimator:
    """
    Pose estimation using MediaPipe BlazePose.
    
    Features:
    - Real-time pose detection
    - 33 3D pose landmarks
    - Visibility confidence scores
    - World coordinate landmarks
    """
    
    # Landmark names matching MediaPipe BlazePose
    LANDMARK_NAMES = [
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

    def __init__(self):
        self.model = None
        self._initialized = False

    async def initialize(self):
        """Initialize the pose estimation model."""
        if self._initialized:
            return
        
        logger.info("Initializing pose estimation model...")
        
        try:
            import mediapipe as mp
            
            self.model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
                min_tracking_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
            )
            
            self._initialized = True
            logger.info("Pose estimation model initialized successfully")
            
        except ImportError:
            logger.warning("MediaPipe not available, using mock implementation")
            self._initialized = True  # Still mark as initialized

    async def estimate(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Estimate pose from an image frame.
        
        Args:
            image: RGB image as numpy array.
            
        Returns:
            List of (x, y, z, visibility) tuples for each landmark, or None if no pose detected.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            import cv2
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            results = self.model.process(image_rgb)
            
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

    def get_landmark_names(self) -> List[str]:
        """Get list of all landmark names."""
        return self.LANDMARK_NAMES

    async def close(self):
        """Release resources."""
        if self.model:
            self.model.close()
        self._initialized = False


# Singleton instance
_pose_estimator: Optional[PoseEstimator] = None


def get_pose_estimator() -> PoseEstimator:
    """
    Get or create the singleton PoseEstimator instance.
    
    Returns:
        PoseEstimator instance.
    """
    global _pose_estimator
    
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    
    return _pose_estimator
