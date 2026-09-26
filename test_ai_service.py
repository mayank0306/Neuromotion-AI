"""
Test script for AI service.
"""

import sys
import os
# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock

from app.services.ai_service import AIService
from app.schemas.movement import AnalysisRequest


async def test_ai_service():
    """Test AI service methods."""
    print("Testing AI Service")
    print("=" * 30)

    # Create AI service instance
    ai_service = AIService()

    # Test landmark conversion
    mock_landmarks = [
        (0.1, 0.2, 0.3, 0.9),  # nose
        (0.4, 0.5, 0.6, 0.8),  # left_shoulder
        (0.7, 0.5, 0.6, 0.8),  # right_shoulder
        (0.4, 0.3, 0.2, 0.7),  # left_elbow
        (0.5, 0.3, 0.2, 0.7),  # right_elbow
    ]

    # Extend to 33 landmarks (MediaPipe standard)
    while len(mock_landmarks) < 33:
        mock_landmarks.append((0.0, 0.0, 0.0, 0.0))

    landmark_dict = ai_service._convert_to_landmark_dict(mock_landmarks)
    print(f"Converted {len(landmark_dict)} landmarks")
    assert "nose" in landmark_dict
    assert "left_shoulder" in landmark_dict

    # Test yoga pose analysis (will use placeholder)
    yoga_result = await ai_service.analyze_yoga_pose(landmark_dict)
    print(f"Yoga analysis score: {yoga_result['overall_score']:.2f}")
    assert "overall_score" in yoga_result
    assert "metrics" in yoga_result
    assert "feedback" in yoga_result

    # Test rehab analysis
    rehab_result = await ai_service.analyze_rehab_movement(landmark_dict, "shoulder-abduction")
    print(f"Rehab analysis score: {rehab_result['overall_score']:.2f}")
    assert "overall_score" in rehab_result
    assert "score_breakdown" in rehab_result
    assert "metrics" in rehab_result

    # Test posture analysis
    posture_result = await ai_service.analyze_posture(landmark_dict)
    print(f"Posture analysis score: {posture_result['overall_score']:.2f}")
    assert "overall_score" in posture_result

    # Test gait analysis
    gait_result = await ai_service.analyze_gait(landmark_dict)
    print(f"Gait analysis score: {gait_result['overall_score']:.2f}")
    assert "overall_score" in gait_result

    # Test balance analysis
    balance_result = await ai_service.analyze_balance(landmark_dict)
    print(f"Balance analysis score: {balance_result['overall_score']:.2f}")
    assert "overall_score" in balance_result

    # Test flexibility analysis
    flex_result = await ai_service.analyze_flexibility(landmark_dict)
    print(f"Flexibility analysis score: {flex_result['overall_score']:.2f}")
    assert "overall_score" in flex_result

    print("\nAll AI service tests passed!")


if __name__ == "__main__":
    asyncio.run(test_ai_service())