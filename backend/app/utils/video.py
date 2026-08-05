"""Video processing utilities."""

from typing import Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class VideoProcessor:
    """
    Video processing utilities for movement analysis.
    
    Features:
    - Frame extraction
    - Video compression
    - Format conversion
    """
    
    def __init__(self):
        self.cap = None

    def extract_frames(
        self,
        video_path: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
    ) -> list:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file.
            frame_interval: Extract every N frames.
            max_frames: Maximum number of frames to extract.
            
        Yields:
            Numpy arrays representing video frames.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            extracted = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    yield frame
                    extracted += 1
                    
                    if max_frames and extracted >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            logger.error("Frame extraction failed", error=str(e))

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video file metadata.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            Dictionary with video metadata.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error("Failed to get video info", error=str(e))
            return {}
