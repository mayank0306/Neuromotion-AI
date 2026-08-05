"""Image processing utilities."""

from typing import Tuple
import structlog

logger = structlog.get_logger(__name__)


class ImageProcessor:
    """
    Image processing utilities for pose estimation.
    
    Features:
    - Image normalization
    - Resize operations
    - Rotation correction
    """
    
    @staticmethod
    def normalize_image(image: bytes, target_size: Tuple[int, int] = (256, 256)):
        """
        Normalize image for AI model input.
        
        Args:
            image: Raw image bytes.
            target_size: Target dimensions (width, height).
            
        Returns:
            Normalized numpy array.
        """
        try:
            import cv2
            import numpy as np
            
            # Decode image
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Resize
            resized = cv2.resize(img, target_size)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error("Image normalization failed", error=str(e))
            raise

    @staticmethod
    def base64_to_image(base64_string: str):
        """
        Convert base64 string to numpy image array.
        
        Args:
            base64_string: Base64-encoded image.
            
        Returns:
            Numpy array representing the image.
        """
        import base64
        import numpy as np
        import cv2
        
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        # Decode
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
