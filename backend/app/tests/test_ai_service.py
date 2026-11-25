"""
Test script for AI service.
"""

import sys
import os
import asyncio
import numpy as np

# Add the backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.services.ai_service import AIService
from app.schemas.movement import MetricWithTag


async def test_ai_service():
    """Test AI service methods."""
    print("Testing AI Service")
    print("=" * 30)

    ai_service = AIService()

    # Test landmark conversion
    mock_landmarks = [
        (0.1, 0.2, 0.3, 0.9),  # nose
        (0.4, 0.5, 0.6, 0.8),  # left_shoulder
        (0.7, 0.5, 0.6, 0.8),  # right_shoulder
        (0.4, 0.3, 0.2, 0.7),  # left_elbow
        (0.5, 0.3, 0.2, 0.7),  # right_elbow
    ]

    while len(mock_landmarks) < 33:
        mock_landmarks.append((0.0, 0.0, 0.0, 0.0))

    landmark_dict = ai_service._convert_to_landmark_dict(mock_landmarks)
    assert "nose" in landmark_dict
    assert "left_shoulder" in landmark_dict

    # Test yoga pose analysis (supporting module)
    yoga_result = await ai_service.analyze_yoga_pose(landmark_dict)
    assert "overall_score" in yoga_result
    assert "metrics" in yoga_result
    assert "feedback" in yoga_result

    # Test rehab analysis (flagship module)
    rehab_result = await ai_service.analyze_rehab_movement(landmark_dict, "shoulder-abduction")
    assert "overall_score" in rehab_result
    assert "score_breakdown" in rehab_result
    assert "metrics" in rehab_result
    assert "clinical_quality_score" in rehab_result
    assert "movement_correct" in rehab_result
    # Verify measured vs estimated tagging
    for metric_name, metric_obj in rehab_result["metrics"].items():
        assert isinstance(metric_obj, MetricWithTag)
        assert metric_obj.tag in ("measured", "estimated")

    # Test posture analysis
    posture_result = await ai_service.analyze_posture(landmark_dict)
    assert "overall_score" in posture_result
    assert posture_result["metrics"]["shoulder_symmetry"].tag == "measured"
    assert posture_result["metrics"]["postural_stability_score"].tag == "estimated"

    # Test gait analysis
    gait_result = await ai_service.analyze_gait(landmark_dict)
    assert "overall_score" in gait_result
    assert gait_result["metrics"]["step_width"].tag == "measured"

    # Test balance analysis
    balance_result = await ai_service.analyze_balance(landmark_dict)
    assert "overall_score" in balance_result
    assert balance_result["metrics"]["trunk_sway_angle"].tag == "measured"
    assert balance_result["metrics"]["overall_balance_score"].tag == "estimated"

    # Test flexibility analysis
    flex_result = await ai_service.analyze_flexibility(landmark_dict)
    assert "overall_score" in flex_result
    assert flex_result["metrics"]["shoulder_range_of_motion"].tag == "measured"

    print("\nAll AI service tests passed!")


if __name__ == "__main__":
    asyncio.run(test_ai_service())
