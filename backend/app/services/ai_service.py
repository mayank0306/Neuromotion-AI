"""AI service for pose estimation and analysis."""

from typing import Optional, List, Tuple
import numpy as np
import structlog

from app.core.config import settings
from app.schemas.movement import AnalysisResponse, AnalysisRequest
from app.exceptions import AppException

logger = structlog.get_logger(__name__)


class AIService:
    """
    AI service for pose estimation and movement analysis.
    
    Uses MediaPipe BlazePose for real-time pose detection and
    custom models for form analysis.
    """
    
    def __init__(self):
        self.pose_estimator = None
        self._loaded = False

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

    async def analyze_form(
        self, poses: List[List[Tuple[float, float, float, float]]]
    ) -> dict:
        """
        Analyze form from a sequence of poses.
        
        Args:
            poses: List of pose landmarks.
            
        Returns:
            Form analysis results with scores and feedback.
        """
        # Placeholder for actual form analysis logic
        return {
            "overall_score": 85.0,
            "feedback": [
                "Great alignment in downward dog",
                "Try lifting your left heel slightly in warrior II"
            ],
            "improvements": ["Increase your hold time by 5 seconds"],
            "safety_notes": [],
        }
