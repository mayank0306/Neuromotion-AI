"""
Test script for feature engineering module.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.analysis.feature_engineering import FeatureEngineering
import numpy as np


def test_feature_engineering():
    """Test all feature engineering methods."""
    fe = FeatureEngineering()

    print("Testing Feature Engineering Module")
    print("=" * 40)

    # Test angle calculation
    p1 = (0, 0, 0)
    p2 = (1, 0, 0)
    p3 = (1, 1, 0)
    angle = fe.calculate_angle(p1, p2, p3)
    print(f"Angle calculation: {angle:.2f}° (expected: 90.00°)")
    assert abs(angle - 90.0) < 0.1, f"Expected 90°, got {angle}°"

    # Test distance calculation
    dist = fe.calculate_distance((0, 0, 0), (3, 4, 0))
    print(f"Distance calculation: {dist:.2f} (expected: 5.00)")
    assert abs(dist - 5.0) < 0.1, f"Expected 5.0, got {dist}"

    # Test vector calculation
    vec = fe.calculate_vector((1, 2, 3), (4, 6, 8))
    expected_vec = np.array([3, 4, 5])
    print(f"Vector calculation: {vec} (expected: {expected_vec})")
    assert np.allclose(vec, expected_vec), f"Expected {expected_vec}, got {vec}"

    # Test angular velocity
    omega = fe.calculate_angle_change(30.0, 60.0, 2.0)  # 30° to 60° in 2 seconds
    print(f"Angular velocity: {omega:.2f}°/s (expected: 15.00°/s)")
    assert abs(omega - 15.0) < 0.1, f"Expected 15.0°/s, got {omega}°/s"

    # Test symmetry
    sym = fe.calculate_symmetry(10.0, 10.0)
    print(f"Symmetry (equal): {sym:.2f} (expected: 1.00)")
    assert abs(sym - 1.0) < 0.1, f"Expected 1.0, got {sym}"

    sym = fe.calculate_symmetry(10.0, 5.0)
    print(f"Symmetry (10 vs 5): {sym:.2f} (expected: 0.50)")
    assert abs(sym - 0.5) < 0.1, f"Expected 0.5, got {sym}"

    # Test range of motion
    values = [10, 20, 15, 25, 30]
    rom = fe.calculate_range_of_motion(values)
    print(f"Range of motion: {rom:.2f} (expected: 20.0)")
    assert rom == 20.0, f"Expected 20.0, got {rom}"

    # Test stability
    stable_values = [10.0, 10.1, 9.9, 10.0, 10.05]
    unstable_values = [0.0, 10.0, 0.0, 10.0, 0.0]
    stability_stable = fe.calculate_stability(stable_values)
    stability_unstable = fe.calculate_stability(unstable_values)
    print(f"Stability (stable): {stability_stable:.2f} (expected: high)")
    print(f"Stability (unstable): {stability_unstable:.2f} (expected: low)")
    assert stability_stable > stability_unstable, "Stable sequence should have higher stability"

    # Test angle normalization
    normalized = fe.normalize_angle(100.0, 90.0, 5.0)  # 100° vs mean 90°, std 5°
    print(f"Angle normalization: {normalized:.2f} (expected: 2.00)")
    assert abs(normalized - 2.0) < 0.1, f"Expected 2.0, got {normalized}"

    # Test landmark extraction
    landmark_names = ["nose", "left_shoulder", "right_shoulder", "left_elbow"]
    landmarks = [
        (0.1, 0.2, 0.3, 0.9),  # nose
        (0.4, 0.5, 0.6, 0.8),  # left_shoulder
        (0.7, 0.5, 0.6, 0.8),  # right_shoulder
        (0.4, 0.3, 0.2, 0.7),  # left_elbow
    ]
    extracted = fe.extract_landmarks_by_name(landmarks, landmark_names, ["nose", "left_elbow"])
    print(f"Landmark extraction: {len(extracted)} landmarks extracted")
    assert "nose" in extracted, "Nose landmark should be extracted"
    assert "left_elbow" in extracted, "Left elbow landmark should be extracted"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_feature_engineering()