"""
Unit tests for Movement Analysis Endpoints.
Verifies:
- V1 recorded video upload handling
- ENABLE_REALTIME_ANALYSIS feature flag enforcement
- AnalysisResponse schema structure with measured/estimated tags and clinical quality score
"""

import pytest
import numpy as np
import cv2
import io
import tempfile
import os
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from app.core.config import settings
from app.schemas.movement import MetricWithTag, AnalysisResponse
from app.services.ai_service import AIService


@pytest.fixture
def dummy_video_file():
    """Create a temporary valid MP4 video with a few black frames for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    # Write 5 frames of 64x64 video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_feature_flag_blocks_realtime_image():
    """Verify that when ENABLE_REALTIME_ANALYSIS is False, image_data is rejected."""
    # Ensure flag is False
    with patch.object(settings, "ENABLE_REALTIME_ANALYSIS", False):
        from app.api.v1.endpoints.analysis import analyze_movement
        from app.schemas.movement import AnalysisRequest
        from fastapi import HTTPException

        req = AnalysisRequest(
            image_data="data:image/jpeg;base64,dGVzdA==",
            exercise_type="rehab_assessment"
        )
        mock_user = MagicMock()
        mock_ai = MagicMock(spec=AIService)

        with pytest.raises(HTTPException) as exc_info:
            await analyze_movement(request=req, current_user=mock_user, ai_service=mock_ai)

        assert exc_info.value.status_code == 400
        assert "ENABLE_REALTIME_ANALYSIS=False" in exc_info.value.detail


@pytest.mark.asyncio
async def test_ai_service_analyze_video_pipeline(dummy_video_file):
    """Verify full video processing pipeline with AIService."""
    ai_service = AIService()

    # Analyze recorded video for rehabilitation movement
    result = await ai_service.analyze_video(
        video_path=dummy_video_file,
        exercise_type="rehab_assessment",
        exercise_id="shoulder-abduction",
    )

    assert "overall_score" in result
    assert "score_breakdown" in result
    assert "metrics" in result
    assert "clinical_quality_score" in result
    assert "movement_correct" in result

    # Check weights in breakdown
    assert "angle_accuracy" in result["score_breakdown"]
    assert "symmetry" in result["score_breakdown"]
    assert "stability" in result["score_breakdown"]
    assert "range_of_motion" in result["score_breakdown"]

    # Verify tagging
    assert result["metrics"]["left_shoulder_angle"].tag == "measured"
    assert result["metrics"]["clinical_quality_score"].tag == "estimated"
