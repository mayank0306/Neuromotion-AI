"""Movement analysis API endpoints."""

import base64
import os
import tempfile
import numpy as np
import cv2
from time import time
from uuid import uuid4
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from app.api.deps import get_current_user_dependency, get_ai_service
from app.core.config import settings
from app.schemas.movement import AnalysisRequest, AnalysisResponse, PoseLandmarkSchema
from app.services.ai_service import AIService

router = APIRouter()


@router.post("/video", response_model=AnalysisResponse)
async def analyze_video_upload(
    file: UploadFile = File(...),
    exercise_type: str = Form("rehab_assessment"),
    exercise_id: Optional[str] = Form("shoulder-abduction"),
    current_user=Depends(get_current_user_dependency),
    ai_service: AIService = Depends(get_ai_service),
):
    """
    Primary V1 Endpoint: Analyze movement from recorded video upload.

    - **file**: Uploaded video file (e.g. mp4, mov, avi, webm)
    - **exercise_type**: Movement assessment type ('rehab_assessment', 'yoga', 'posture', 'gait', 'balance', 'flexibility')
    - **exercise_id**: Exercise identifier (e.g. 'shoulder-abduction', 'knee-extension', 'sit-to-stand')
    """
    start_time = time()

    # Validate video format
    content_type = file.content_type or ""
    filename = file.filename or ""
    is_video = (
        content_type.startswith("video/") or
        filename.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))
    )
    if not is_video:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a recorded video (mp4, mov, avi, webm)"
        )

    # Save to temporary file for OpenCV processing
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_path = temp_video.name
        content = await file.read()
        temp_video.write(content)

    try:
        # Run full video assessment pipeline
        analysis_result = await ai_service.analyze_video(
            video_path=temp_path,
            exercise_type=exercise_type,
            exercise_id=exercise_id or "shoulder-abduction",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video analysis failed: {str(e)}"
        )
    finally:
        # Clean up temporary video file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    processing_time_ms = int((time() - start_time) * 1000)
    analysis_result["exercise_type"] = exercise_type
    analysis_result["processing_time_ms"] = processing_time_ms

    return AnalysisResponse(
        analysis_id=uuid4(),
        **analysis_result
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_movement(
    request: AnalysisRequest,
    current_user=Depends(get_current_user_dependency),
    ai_service: AIService = Depends(get_ai_service),
):
    """
    Analyze movement.
    V1 requirement: Recorded video upload only.
    Real-time single frame image scaffolding is gated behind ENABLE_REALTIME_ANALYSIS feature flag.
    """
    start_time = time()

    if request.video_url:
        # Process recorded video from provided path / URL
        if not os.path.exists(request.video_url):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video file not found at path: {request.video_url}"
            )

        analysis_result = await ai_service.analyze_video(
            video_path=request.video_url,
            exercise_type=request.exercise_type,
            exercise_id=request.exercise_id or "shoulder-abduction",
        )
        processing_time_ms = int((time() - start_time) * 1000)
        analysis_result["exercise_type"] = request.exercise_type
        analysis_result["processing_time_ms"] = processing_time_ms
        return AnalysisResponse(
            analysis_id=uuid4(),
            **analysis_result
        )

    if request.image_data:
        # Check feature flag for real-time / image analysis
        if not settings.ENABLE_REALTIME_ANALYSIS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Real-time / single-frame image analysis is disabled in V1 (ENABLE_REALTIME_ANALYSIS=False). V1 accepts recorded video upload only via POST /api/v1/analysis/video.",
            )

        # Decode base64 image if feature flag is active
        try:
            image_data = request.image_data
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]

            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            poses = []
            if image is not None:
                pose_landmarks = await ai_service.estimate_pose(image)
                if pose_landmarks:
                    poses.append(pose_landmarks)

            if not poses:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No pose detected in provided image frame",
                )

            analysis_result = await ai_service.analyze_form(
                poses,
                exercise_type=request.exercise_type,
                exercise_id=request.exercise_id or "shoulder-abduction",
            )
            processing_time_ms = int((time() - start_time) * 1000)
            analysis_result["exercise_type"] = request.exercise_type
            analysis_result["processing_time_ms"] = processing_time_ms

            return AnalysisResponse(
                analysis_id=uuid4(),
                **analysis_result
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process image frame: {str(e)}",
            )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Please provide a recorded video for analysis (video_url or upload via /api/v1/analysis/video)."
    )


@router.post("/upload-video", response_model=dict)
async def upload_video(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user_dependency),
):
    """
    Upload a video file for future analysis.
    """
    content_type = file.content_type or ""
    filename = file.filename or ""
    is_video = (
        content_type.startswith("video/") or
        filename.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))
    )
    if not is_video:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )

    # In production, save to object storage or media volume
    file_id = f"video_{current_user.id}_{int(time())}"
    video_url = f"/media/{file_id}_{filename}"

    return {
        "video_url": video_url,
        "message": "Video uploaded successfully. Submit to /api/v1/analysis/analyze or /api/v1/analysis/video for movement assessment."
    }
