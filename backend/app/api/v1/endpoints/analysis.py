"""Movement analysis API endpoints."""

import base64
import numpy as np
import cv2
from time import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.api.deps import get_current_user_dependency, get_ai_service
from app.schemas.movement import AnalysisRequest, AnalysisResponse, PoseLandmarkSchema

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_movement(
    request: AnalysisRequest,
    current_user=Depends(get_current_user_dependency),
    ai_service=Depends(get_ai_service),
):
    """
    Analyze movement from video or image.

    - **video_url**: URL to a video file (alternative to image_data)
    - **image_data**: Base64 encoded image (alternative to video_url)
    - **exercise_type**: Type of exercise being analyzed
    - **user_id**: User identifier for personalized analysis
    """
    start_time = time()

    # Extract pose data from request
    poses = []

    if request.image_data:
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            image_data = request.image_data
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]

            # Decode base64
            image_bytes = base64.b64decode(image_data)

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                # Estimate pose
                pose_landmarks = await ai_service.estimate_pose(image)
                if pose_landmarks:
                    poses.append(pose_landmarks)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image data: {str(e)}"
            )

    # TODO: Handle video_url case (would require video processing)
    # For now, we'll focus on image-based analysis

    if not poses:
        raise HTTPException(
            status_code=400,
            detail="No pose data available for analysis"
        )

    # Perform analysis
    analysis_result = await ai_service.analyze_form(poses)

    processing_time_ms = int((time() - start_time) * 1000)

    # Add exercise type to result
    analysis_result["exercise_type"] = request.exercise_type
    analysis_result["processing_time_ms"] = processing_time_ms

    return AnalysisResponse(
        analysis_id=uuid4(),
        **analysis_result
    )


@router.post("/upload-video", response_model=dict)
async def upload_video(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user_dependency),
):
    """
    Upload a video file for analysis.

    Returns a URL that can be used in the analyze endpoint.
    """
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="File must be a video"
        )

    # Save to storage (placeholder)
    file_id = f"video_{current_user.id}_{int(time())}"
    # TODO: Implement actual file storage
    video_url = f"https://storage.example.com/{file_id}.mp4"

    return {
        "video_url": video_url,
        "message": "Video uploaded successfully"
    }
