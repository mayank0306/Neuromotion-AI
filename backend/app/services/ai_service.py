"""
AI service for pose estimation and movement analysis.
NeuroMotion: AI-Assisted Rehabilitation Movement Assessment.
Flagship Module: Rehab Movement Quality Assessment.
Supporting Module: Yoga Pose Classification.
Deterministic Geometric Modules: Posture, Gait, Balance, Flexibility.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
import cv2
import structlog
from datetime import datetime

from app.core.config import settings
from app.schemas.movement import (
    AnalysisResponse,
    AnalysisRequest,
    PoseLandmarkSchema,
    MetricWithTag,
)
from app.exceptions import AppException
from app.analysis.feature_engineering import FeatureEngineering
from app.analysis.models import (
    RehabQualityMLP,
    YogaClassifierMLP,
    load_reference_stats,
    DEFAULT_REFERENCE_STATS,
)

logger = structlog.get_logger(__name__)


class AIService:
    """
    AI service for pose estimation and movement analysis.

    Uses pretrained MediaPipe BlazePose for real-time / video pose detection,
    a unified FeatureEngineering layer, and two PyTorch MLPs:
    1. RehabQualityMLP (Flagship: UI-PRMD classification head + KIMORE regression head)
    2. YogaClassifierMLP (Supporting: Kaggle 5-Class Yoga pose classification)
    Deterministic geometry for Posture, Gait, Balance, and Flexibility.
    """

    def __init__(self):
        self.pose_estimator = None
        self.feature_engineering = FeatureEngineering()
        self._loaded = False

        # PyTorch models
        self.rehab_model: Optional[RehabQualityMLP] = None
        self.yoga_model: Optional[YogaClassifierMLP] = None

        # Persisted reference statistics for angle deviation calculations
        ref_file = Path(__file__).resolve().parents[1] / "data" / "reference_stats.json"
        self.reference_stats: Dict[str, Any] = load_reference_stats(ref_file)

        # MediaPipe pose landmark names (33 landmarks)
        self.LANDMARK_NAMES = [
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

        self._init_models()

    def _init_models(self) -> None:
        """Initialize and load PyTorch model checkpoints if available."""
        models_dir = Path(__file__).resolve().parents[1] / "ai-models"
        rehab_ckpt = models_dir / "rehab-assessment" / "rehab_quality_mlp.pt"
        yoga_ckpt = models_dir / "yoga-classifier" / "yoga_classifier_mlp.pt"

        # 1. Flagship Rehab Quality Model
        self.rehab_model = RehabQualityMLP(input_features=self.feature_engineering.NUM_FEATURES)
        if rehab_ckpt.exists():
            try:
                ckpt = torch.load(rehab_ckpt, map_location=torch.device("cpu"), weights_only=True)
                self.rehab_model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded RehabQualityMLP checkpoint", path=str(rehab_ckpt))
            except Exception as e:
                logger.warning("Could not load rehab checkpoint, using initialized weights", error=str(e))
        self.rehab_model.eval()

        # 2. Supporting Yoga Classifier Model
        self.yoga_model = YogaClassifierMLP(input_features=self.feature_engineering.NUM_FEATURES, num_classes=5)
        if yoga_ckpt.exists():
            try:
                ckpt = torch.load(yoga_ckpt, map_location=torch.device("cpu"), weights_only=True)
                self.yoga_model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded YogaClassifierMLP checkpoint", path=str(yoga_ckpt))
            except Exception as e:
                logger.warning("Could not load yoga checkpoint, using initialized weights", error=str(e))
        self.yoga_model.eval()

    async def load_model(self):
        """Load MediaPipe pose estimator (lazy initialization)."""
        if self._loaded:
            return

        logger.info("Loading MediaPipe BlazePose...")
        try:
            import mediapipe as mp

            self.pose_estimator = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
                min_tracking_confidence=settings.MODEL_CONFIDENCE_THRESHOLD,
            )
            self._loaded = True
            logger.info("MediaPipe BlazePose loaded successfully")
        except ImportError:
            logger.warning("MediaPipe not available, using mock mode")
            self._loaded = True

    async def estimate_pose(self, image: np.ndarray) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Estimate 33 pose landmarks from an image frame using MediaPipe BlazePose.

        Args:
            image: RGB image as numpy array

        Returns:
            List of (x, y, z, visibility) tuples
        """
        if not self._loaded:
            await self.load_model()

        if self.pose_estimator is None:
            # Fallback mock landmarks for testing environments
            return [(0.0, 0.0, 0.0, 1.0) for _ in range(33)]

        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = self.pose_estimator.process(image_rgb)
            if results.pose_landmarks:
                return [
                    (lm.x, lm.y, lm.z, lm.visibility)
                    for lm in results.pose_landmarks.landmark
                ]
            return None
        except Exception as e:
            logger.error("Pose estimation failed", error=str(e))
            return None

    def _convert_to_landmark_dict(self, landmarks: List[Tuple[float, float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """Convert list of (x, y, z, visibility) tuples into dictionary keyed by landmark name."""
        landmark_dict = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            if i < len(landmarks):
                lm = landmarks[i]
                landmark_dict[name] = (float(lm[0]), float(lm[1]), float(lm[2]))
            else:
                landmark_dict[name] = (0.0, 0.0, 0.0)
        return landmark_dict

    def _normalize_input_frames(self, input_data: Union[Dict[str, Tuple[float, float, float]],
                                                        List[Dict[str, Tuple[float, float, float]]],
                                                        List[List[Tuple[float, float, float, float]]]]) -> List[Dict[str, Tuple[float, float, float]]]:
        """Normalize various input formats into a list of landmark dictionary frames."""
        if isinstance(input_data, dict):
            return [input_data]
        elif isinstance(input_data, list) and len(input_data) > 0:
            first = input_data[0]
            if isinstance(first, dict):
                return input_data
            elif isinstance(first, list):
                return [self._convert_to_landmark_dict(frame) for frame in input_data]
        return [{name: (0.0, 0.0, 0.0) for name in self.LANDMARK_NAMES}]

    async def process_video_frames(self, video_path: str, max_frames: int = 150) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Process uploaded video file: extract frames with OpenCV and detect poses with BlazePose.

        Args:
            video_path: Local path to recorded video file
            max_frames: Maximum frames to process

        Returns:
            List of detected landmark frames
        """
        if not self._loaded:
            await self.load_model()

        frames_landmarks: List[Dict[str, Tuple[float, float, float]]] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Could not open video file", video_path=video_path)
            return frames_landmarks

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // max_frames) if frame_count > max_frames else 1

        idx = 0
        while cap.isOpened() and len(frames_landmarks) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                pose = await self.estimate_pose(frame)
                if pose is not None:
                    frames_landmarks.append(self._convert_to_landmark_dict(pose))
            idx += 1

        cap.release()
        logger.info("Processed video frames", total_extracted=len(frames_landmarks), video_path=video_path)
        return frames_landmarks

    # -----------------------------------------------------------------------
    # Flagship Module: Rehabilitation Movement Quality Assessment
    # -----------------------------------------------------------------------

    async def analyze_rehab_movement(self,
                                     landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                            List[Dict[str, Tuple[float, float, float]]]],
                                     exercise_id: str = "shoulder-abduction") -> Dict[str, Any]:
        """
        Flagship module: Rehab Movement Quality Assessment.
        Predicts continuous quality score (via KIMORE regression head),
        evaluates correct/incorrect form (via UI-PRMD classification head),
        and explains which joint/phase caused a low score using deviations from persisted class means.
        """
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)
        feat_vector = self.feature_engineering.compute_feature_vector(seq_features)

        # Run through dual-head PyTorch MLP
        with torch.no_grad():
            x_tensor = torch.from_numpy(feat_vector).unsqueeze(0)
            logits, raw_quality_score = self.rehab_model(x_tensor)
            probs = torch.softmax(logits, dim=-1)
            is_correct = bool(torch.argmax(probs, dim=-1).item() == 1)
            form_confidence = float(probs[0, 1 if is_correct else 0].item())
            clinical_quality_score = float(raw_quality_score.item())

        # Configurable scoring formula constants:
        # Angle accuracy 0.4, symmetry 0.2, stability 0.25, ROM 0.15
        weights = settings.SCORING_WEIGHTS

        # Retrieve persisted reference means for this exercise
        exercise_key = exercise_id.lower().replace("_", "-")
        ref_dict = self.reference_stats.get(exercise_key, self.reference_stats.get("shoulder-abduction", {}))

        # Deviation analysis & joint/phase explanations
        explanations: Dict[str, str] = {}
        deviations: Dict[str, float] = {}
        angle_accuracies: List[float] = []

        for metric_name, ref_info in ref_dict.items():
            if isinstance(ref_info, dict) and "mean" in ref_info:
                target_mean = ref_info["mean"]
                target_std = max(1e-3, ref_info.get("std", 5.0))
                measured_val = seq_features.get(f"{metric_name}_mean", seq_features.get(metric_name, target_mean))

                dev = float(measured_val - target_mean)
                deviations[metric_name] = dev

                # Calculate joint angle accuracy (0-100)
                joint_acc = max(0.0, 100.0 - (abs(dev) / target_std) * 15.0)
                angle_accuracies.append(joint_acc)

                # If deviation is significant, generate clinical explanation
                if abs(dev) > 12.0:
                    direction = "above" if dev > 0 else "below"
                    readable_name = metric_name.replace("_", " ").title()
                    explanations[metric_name] = (
                        f"{readable_name} measured at {measured_val:.1f}°, which is {abs(dev):.1f}° {direction} "
                        f"the target mean ({target_mean:.1f}°). Check range of motion and joint alignment."
                    )

        mean_angle_accuracy = float(np.mean(angle_accuracies)) if angle_accuracies else 85.0

        # Bilateral symmetry score
        sym_metrics = [
            seq_features.get("shoulder_level_symmetry", 0.95),
            seq_features.get("hip_level_symmetry", 0.95),
            seq_features.get("elbow_angle_symmetry", 0.90),
            seq_features.get("shoulder_angle_symmetry", 0.90),
            seq_features.get("knee_angle_symmetry", 0.90),
        ]
        symmetry_score = float(np.mean(sym_metrics) * 100.0)

        # Stability score
        stability_score = float(seq_features.get("overall_stability", 0.85) * 100.0)

        # Range of motion score
        rom_values = [
            seq_features.get("left_shoulder_angle_rom", 40.0),
            seq_features.get("right_shoulder_angle_rom", 40.0),
            seq_features.get("left_elbow_angle_rom", 30.0),
            seq_features.get("right_elbow_angle_rom", 30.0),
        ]
        mean_rom = float(np.mean(rom_values))
        rom_score = min(100.0, max(20.0, mean_rom * 1.5))

        # Weighted composite score formula
        score_breakdown = {
            "angle_accuracy": mean_angle_accuracy * weights.get("angle_accuracy", 0.4),
            "symmetry": symmetry_score * weights.get("symmetry", 0.2),
            "stability": stability_score * weights.get("stability", 0.25),
            "range_of_motion": rom_score * weights.get("range_of_motion", 0.15),
        }
        overall_composite = sum(score_breakdown.values())

        # Blend composite score with KIMORE regression head score
        final_quality_score = float(0.6 * overall_composite + 0.4 * clinical_quality_score)

        # Structure metrics with measured vs estimated tags
        metrics: Dict[str, MetricWithTag] = {
            "left_shoulder_angle": MetricWithTag(
                value=float(seq_features.get("left_shoulder_angle_mean", 90.0)),
                tag="measured",
                unit="degrees",
                reference_mean=ref_dict.get("left_shoulder_angle", {}).get("mean"),
                deviation=deviations.get("left_shoulder_angle"),
                description="Mean left shoulder abduction/flexion angle",
            ),
            "right_shoulder_angle": MetricWithTag(
                value=float(seq_features.get("right_shoulder_angle_mean", 90.0)),
                tag="measured",
                unit="degrees",
                reference_mean=ref_dict.get("right_shoulder_angle", {}).get("mean"),
                deviation=deviations.get("right_shoulder_angle"),
                description="Mean right shoulder abduction/flexion angle",
            ),
            "left_elbow_angle": MetricWithTag(
                value=float(seq_features.get("left_elbow_angle_mean", 170.0)),
                tag="measured",
                unit="degrees",
                reference_mean=ref_dict.get("left_elbow_angle", {}).get("mean"),
                deviation=deviations.get("left_elbow_angle"),
                description="Mean left elbow angle",
            ),
            "right_elbow_angle": MetricWithTag(
                value=float(seq_features.get("right_elbow_angle_mean", 170.0)),
                tag="measured",
                unit="degrees",
                reference_mean=ref_dict.get("right_elbow_angle", {}).get("mean"),
                deviation=deviations.get("right_elbow_angle"),
                description="Mean right elbow angle",
            ),
            "trunk_angle": MetricWithTag(
                value=float(seq_features.get("trunk_angle_mean", 0.0)),
                tag="measured",
                unit="degrees",
                description="Mean spinal trunk vertical inclination",
            ),
            "bilateral_symmetry": MetricWithTag(
                value=symmetry_score,
                tag="measured",
                unit="score (0-100)",
                description="Measured bilateral movement symmetry",
            ),
            "range_of_motion": MetricWithTag(
                value=rom_score,
                tag="measured",
                unit="score (0-100)",
                description="Measured dynamic range of motion",
            ),
            "stability_index": MetricWithTag(
                value=stability_score,
                tag="estimated",
                unit="score (0-100)",
                description="Estimated stability derived from postural sway and angular CV",
            ),
            "clinical_quality_score": MetricWithTag(
                value=clinical_quality_score,
                tag="estimated",
                unit="score (0-100)",
                description="Continuous quality score predicted by KIMORE regression head",
            ),
            "movement_form_classification": MetricWithTag(
                value=1.0 if is_correct else 0.0,
                tag="estimated",
                unit="binary (1=correct, 0=incorrect)",
                description="UI-PRMD form correctness classification",
            ),
        }

        # Clinical feedback
        feedback = [
            f"Exercise: {exercise_id.replace('-', ' ').title()}",
            f"Overall Quality Score: {final_quality_score:.1f}/100",
            f"Clinician-Equivalent Rating: {clinical_quality_score:.1f}/100 ({'Optimal Execution' if is_correct else 'Compensatory Patterns Detected'})",
        ]
        if explanations:
            feedback.append(f"Identified {len(explanations)} joint deviations requiring correction.")
            for exp in explanations.values():
                feedback.append(f"• {exp}")
        else:
            feedback.append("Excellent form! Movement closely matches reference clinical kinematics.")

        return {
            "overall_score": round(final_quality_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
            "clinical_quality_score": round(clinical_quality_score, 1),
            "movement_correct": is_correct,
        }

    # -----------------------------------------------------------------------
    # Supporting Module: Yoga Pose Classification
    # -----------------------------------------------------------------------

    async def analyze_yoga_pose(self,
                                landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                       List[Dict[str, Tuple[float, float, float]]]]) -> Dict[str, Any]:
        """
        Supporting module: Yoga pose classification using Kaggle Yoga 5-Class MLP.
        """
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)
        feat_vector = self.feature_engineering.compute_feature_vector(seq_features)

        # Run YogaClassifierMLP
        with torch.no_grad():
            x_tensor = torch.from_numpy(feat_vector).unsqueeze(0)
            logits = self.yoga_model(x_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_class_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, pred_class_idx].item())

        predicted_pose = YogaClassifierMLP.YOGA_CLASSES[pred_class_idx]
        ref_dict = self.reference_stats.get(predicted_pose, {})

        # Compute deviations from reference pose mean
        explanations = {}
        deviations = {}
        for metric_name, ref_info in ref_dict.items():
            if isinstance(ref_info, dict) and "mean" in ref_info:
                target_mean = ref_info["mean"]
                measured_val = seq_features.get(f"{metric_name}_mean", seq_features.get(metric_name, target_mean))
                dev = float(measured_val - target_mean)
                deviations[metric_name] = dev
                if abs(dev) > 15.0:
                    explanations[metric_name] = (
                        f"{metric_name.replace('_', ' ').title()} is {measured_val:.1f}°, "
                        f"deviating by {dev:+.1f}° from standard {predicted_pose.replace('_', ' ').title()}."
                    )

        overall_score = float(confidence * 100.0)
        score_breakdown = {
            "angle_accuracy": 85.0 * 0.4,
            "symmetry": float(seq_features.get("shoulder_level_symmetry", 0.95) * 100 * 0.2),
            "stability": float(seq_features.get("overall_stability", 0.90) * 100 * 0.25),
            "range_of_motion": 80.0 * 0.15,
        }

        metrics = {
            "pose_confidence": MetricWithTag(
                value=round(confidence * 100.0, 1),
                tag="estimated",
                unit="percent",
                description=f"Model confidence in {predicted_pose.replace('_', ' ').title()}",
            ),
            "left_elbow_angle": MetricWithTag(
                value=round(float(seq_features.get("left_elbow_angle_mean", 170.0)), 1),
                tag="measured",
                unit="degrees",
                description="Left elbow angle",
            ),
            "right_elbow_angle": MetricWithTag(
                value=round(float(seq_features.get("right_elbow_angle_mean", 170.0)), 1),
                tag="measured",
                unit="degrees",
                description="Right elbow angle",
            ),
            "trunk_angle": MetricWithTag(
                value=round(float(seq_features.get("trunk_angle_mean", 0.0)), 1),
                tag="measured",
                unit="degrees",
                description="Trunk inclination relative to vertical",
            ),
            "stability": MetricWithTag(
                value=round(float(seq_features.get("overall_stability", 0.90) * 100.0), 1),
                tag="estimated",
                unit="score (0-100)",
                description="Estimated pose stability",
            ),
        }

        feedback = [
            f"Detected Pose: {predicted_pose.replace('_', ' ').title()} ({confidence:.0%} confidence)",
            f"Overall Form Score: {overall_score:.1f}/100",
        ]
        for exp in explanations.values():
            feedback.append(f"• {exp}")

        return {
            "overall_score": round(overall_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
        }

    # -----------------------------------------------------------------------
    # Deterministic Geometric Modules: Posture, Gait, Balance, Flexibility
    # -----------------------------------------------------------------------

    async def analyze_posture(self,
                              landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                     List[Dict[str, Tuple[float, float, float]]]]) -> Dict[str, Any]:
        """Deterministic geometric posture analysis on landmarks. No ML models used."""
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)

        shoulder_level = float(seq_features.get("shoulder_level_symmetry", 0.95))
        hip_level = float(seq_features.get("hip_level_symmetry", 0.95))
        trunk_dev = float(seq_features.get("trunk_angle_mean", 3.0))
        head_tilt = float(seq_features.get("head_tilt_mean", 2.0))

        # Geometric posture scores
        shoulder_score = shoulder_level * 100.0
        hip_score = hip_level * 100.0
        spine_score = max(0.0, 100.0 - trunk_dev * 5.0)
        head_score = max(0.0, 100.0 - head_tilt * 4.0)

        # Composite score
        weights = settings.SCORING_WEIGHTS
        score_breakdown = {
            "angle_accuracy": spine_score * weights.get("angle_accuracy", 0.4),
            "symmetry": ((shoulder_score + hip_score) / 2.0) * weights.get("symmetry", 0.2),
            "stability": float(seq_features.get("overall_stability", 0.95) * 100.0) * weights.get("stability", 0.25),
            "range_of_motion": 90.0 * weights.get("range_of_motion", 0.15),
        }
        overall_score = sum(score_breakdown.values())

        metrics = {
            "shoulder_symmetry": MetricWithTag(
                value=round(shoulder_score, 1),
                tag="measured",
                unit="symmetry score (0-100)",
                description="Measured bilateral shoulder levelness",
            ),
            "pelvic_symmetry": MetricWithTag(
                value=round(hip_score, 1),
                tag="measured",
                unit="symmetry score (0-100)",
                description="Measured bilateral hip levelness",
            ),
            "spinal_alignment_angle": MetricWithTag(
                value=round(trunk_dev, 1),
                tag="measured",
                unit="degrees",
                description="Measured vertical trunk angle deviation",
            ),
            "head_tilt_offset": MetricWithTag(
                value=round(head_tilt, 1),
                tag="measured",
                unit="percentage",
                description="Measured bilateral ear height difference",
            ),
            "postural_stability_score": MetricWithTag(
                value=round(overall_score, 1),
                tag="estimated",
                unit="score (0-100)",
                description="Derived composite posture quality score",
            ),
        }

        explanations = {}
        feedback = [f"Overall Posture Score: {overall_score:.1f}/100"]
        if shoulder_level < 0.85:
            explanations["shoulder_symmetry"] = "Shoulder asymmetry detected; one shoulder is elevated."
            feedback.append("• Keep shoulders level and relaxed.")
        if trunk_dev > 8.0:
            explanations["spinal_alignment"] = f"Trunk inclined at {trunk_dev:.1f}°; align spine vertically."
            feedback.append("• Focus on upright spinal alignment.")

        return {
            "overall_score": round(overall_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
        }

    async def analyze_gait(self,
                           landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                  List[Dict[str, Tuple[float, float, float]]]]) -> Dict[str, Any]:
        """Deterministic geometric gait analysis on landmarks. No ML models used."""
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)

        step_width = float(seq_features.get("step_width_mean", 0.25))
        knee_sym = float(seq_features.get("knee_angle_symmetry", 0.90))
        hip_sym = float(seq_features.get("hip_angle_symmetry", 0.90))
        knee_rom = float(seq_features.get("left_knee_angle_rom", 45.0) + seq_features.get("right_knee_angle_rom", 45.0)) / 2.0

        angle_acc = float(min(100.0, max(0.0, 100.0 - abs(step_width - 0.20) * 100.0)))
        sym_score = float((knee_sym + hip_sym) / 2.0 * 100.0)
        stab_score = float(seq_features.get("overall_stability", 0.85) * 100.0)
        rom_score = float(min(100.0, (knee_rom / 60.0) * 100.0))

        weights = settings.SCORING_WEIGHTS
        score_breakdown = {
            "angle_accuracy": angle_acc * weights.get("angle_accuracy", 0.4),
            "symmetry": sym_score * weights.get("symmetry", 0.2),
            "stability": stab_score * weights.get("stability", 0.25),
            "range_of_motion": rom_score * weights.get("range_of_motion", 0.15),
        }
        overall_score = sum(score_breakdown.values())

        metrics = {
            "step_width": MetricWithTag(
                value=round(step_width, 3),
                tag="measured",
                unit="meters (normalized)",
                description="Distance between ankles during stance",
            ),
            "knee_symmetry": MetricWithTag(
                value=round(knee_sym * 100.0, 1),
                tag="measured",
                unit="symmetry score (0-100)",
                description="Measured bilateral knee angle symmetry",
            ),
            "hip_symmetry": MetricWithTag(
                value=round(hip_sym * 100.0, 1),
                tag="measured",
                unit="symmetry score (0-100)",
                description="Measured bilateral hip angle symmetry",
            ),
            "knee_range_of_motion": MetricWithTag(
                value=round(knee_rom, 1),
                tag="measured",
                unit="degrees",
                description="Measured dynamic knee flexion/extension range",
            ),
            "gait_harmony_index": MetricWithTag(
                value=round(overall_score, 1),
                tag="estimated",
                unit="score (0-100)",
                description="Derived composite gait performance index",
            ),
        }

        explanations = {}
        feedback = [f"Overall Gait Quality Score: {overall_score:.1f}/100"]
        if knee_sym < 0.80:
            explanations["knee_symmetry"] = "Significant bilateral knee asymmetry during stride phase."
            feedback.append("• Work on equal stride length on both legs.")

        return {
            "overall_score": round(overall_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
        }

    async def analyze_balance(self,
                             landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                    List[Dict[str, Tuple[float, float, float]]]]) -> Dict[str, Any]:
        """Deterministic geometric balance analysis on landmarks. No ML models used."""
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)

        trunk_stab = float(seq_features.get("trunk_stability", 0.90))
        overall_stab = float(seq_features.get("overall_stability", 0.88))
        trunk_dev = float(seq_features.get("trunk_angle_mean", 2.0))

        angle_acc = max(0.0, 100.0 - trunk_dev * 6.0)
        sym_score = float(seq_features.get("hip_level_symmetry", 0.95) * 100.0)
        stab_score = overall_stab * 100.0
        rom_score = max(0.0, 100.0 - seq_features.get("trunk_angle_rom", 2.0) * 8.0)

        weights = settings.SCORING_WEIGHTS
        score_breakdown = {
            "angle_accuracy": angle_acc * weights.get("angle_accuracy", 0.4),
            "symmetry": sym_score * weights.get("symmetry", 0.2),
            "stability": stab_score * weights.get("stability", 0.25),
            "range_of_motion": rom_score * weights.get("range_of_motion", 0.15),
        }
        overall_score = sum(score_breakdown.values())

        metrics = {
            "trunk_sway_angle": MetricWithTag(
                value=round(trunk_dev, 2),
                tag="measured",
                unit="degrees",
                description="Measured angular deviation of trunk during balance hold",
            ),
            "pelvic_tilt_symmetry": MetricWithTag(
                value=round(sym_score, 1),
                tag="measured",
                unit="symmetry score (0-100)",
                description="Measured bilateral pelvic levelness",
            ),
            "trunk_stability_coefficient": MetricWithTag(
                value=round(trunk_stab * 100.0, 1),
                tag="estimated",
                unit="stability score (0-100)",
                description="Estimated trunk stability derived from movement variance",
            ),
            "overall_balance_score": MetricWithTag(
                value=round(overall_score, 1),
                tag="estimated",
                unit="score (0-100)",
                description="Derived composite balance assessment score",
            ),
        }

        explanations = {}
        feedback = [f"Overall Balance Score: {overall_score:.1f}/100"]
        if trunk_dev > 6.0:
            explanations["postural_sway"] = "High postural sway detected outside baseline stability margins."
            feedback.append("• Engage core stabilizers to minimize postural sway.")

        return {
            "overall_score": round(overall_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
        }

    async def analyze_flexibility(self,
                                 landmarks_input: Union[Dict[str, Tuple[float, float, float]],
                                                        List[Dict[str, Tuple[float, float, float]]]]) -> Dict[str, Any]:
        """Deterministic geometric flexibility analysis on landmarks. No ML models used."""
        frames = self._normalize_input_frames(landmarks_input)
        seq_features = self.feature_engineering.extract_sequence_features(frames)

        shoulder_rom = float(seq_features.get("left_shoulder_angle_rom", 60.0) + seq_features.get("right_shoulder_angle_rom", 60.0)) / 2.0
        hip_rom = float(seq_features.get("left_hip_angle_rom", 50.0) + seq_features.get("right_hip_angle_rom", 50.0)) / 2.0
        knee_rom = float(seq_features.get("left_knee_angle_rom", 50.0) + seq_features.get("right_knee_angle_rom", 50.0)) / 2.0

        angle_acc = min(100.0, (shoulder_rom / 90.0) * 100.0)
        sym_score = float(seq_features.get("shoulder_angle_symmetry", 0.90) * 100.0)
        stab_score = float(seq_features.get("overall_stability", 0.90) * 100.0)
        rom_score = min(100.0, ((shoulder_rom + hip_rom + knee_rom) / 200.0) * 100.0)

        weights = settings.SCORING_WEIGHTS
        score_breakdown = {
            "angle_accuracy": angle_acc * weights.get("angle_accuracy", 0.4),
            "symmetry": sym_score * weights.get("symmetry", 0.2),
            "stability": stab_score * weights.get("stability", 0.25),
            "range_of_motion": rom_score * weights.get("range_of_motion", 0.15),
        }
        overall_score = sum(score_breakdown.values())

        metrics = {
            "shoulder_range_of_motion": MetricWithTag(
                value=round(shoulder_rom, 1),
                tag="measured",
                unit="degrees",
                description="Measured bilateral shoulder active range of motion",
            ),
            "hip_range_of_motion": MetricWithTag(
                value=round(hip_rom, 1),
                tag="measured",
                unit="degrees",
                description="Measured bilateral hip active range of motion",
            ),
            "knee_range_of_motion": MetricWithTag(
                value=round(knee_rom, 1),
                tag="measured",
                unit="degrees",
                description="Measured bilateral knee active range of motion",
            ),
            "flexibility_rating": MetricWithTag(
                value=round(overall_score, 1),
                tag="estimated",
                unit="score (0-100)",
                description="Derived composite flexibility index",
            ),
        }

        explanations = {}
        feedback = [f"Flexibility Rating: {overall_score:.1f}/100"]
        if shoulder_rom < 45.0:
            explanations["shoulder_rom"] = "Shoulder active range of motion is restricted below expected 45° benchmark."
            feedback.append("• Perform gradual shoulder mobility exercises.")

        return {
            "overall_score": round(overall_score, 1),
            "score_breakdown": {k: round(v, 1) for k, v in score_breakdown.items()},
            "metrics": metrics,
            "feedback": feedback,
            "explanations": explanations,
        }

    # -----------------------------------------------------------------------
    # Main Dispatcher
    # -----------------------------------------------------------------------

    async def analyze_form(self,
                           poses_or_landmarks: Union[List[Any], Dict[str, Tuple[float, float, float]]],
                           exercise_type: str = "rehab_assessment",
                           exercise_id: Optional[str] = "shoulder-abduction") -> Dict[str, Any]:
        """
        Analyze form by routing to the appropriate movement analysis module.

        Args:
            poses_or_landmarks: Detected pose list or landmark dictionaries
            exercise_type: 'rehab_assessment' (flagship), 'yoga', 'posture', 'gait', 'balance', 'flexibility'
            exercise_id: Specific exercise ID (e.g. 'shoulder-abduction', 'warrior-ii')

        Returns:
            Dictionary with overall_score, score_breakdown, metrics, feedback, explanations
        """
        frames = self._normalize_input_frames(poses_or_landmarks)

        etype = (exercise_type or "rehab_assessment").lower()
        if etype == "rehab_assessment":
            return await self.analyze_rehab_movement(frames, exercise_id=exercise_id or "shoulder-abduction")
        elif etype == "yoga":
            return await self.analyze_yoga_pose(frames)
        elif etype == "posture":
            return await self.analyze_posture(frames)
        elif etype == "gait":
            return await self.analyze_gait(frames)
        elif etype == "balance":
            return await self.analyze_balance(frames)
        elif etype == "flexibility":
            return await self.analyze_flexibility(frames)
        else:
            return await self.analyze_rehab_movement(frames, exercise_id=exercise_id or "shoulder-abduction")

    async def analyze_video(self,
                            video_path: str,
                            exercise_type: str = "rehab_assessment",
                            exercise_id: Optional[str] = "shoulder-abduction") -> Dict[str, Any]:
        """
        Full V1 video pipeline:
        uploaded video -> OpenCV -> MediaPipe BlazePose -> shared landmark processing -> shared feature engineering -> module scoring.
        """
        frames = await self.process_video_frames(video_path)
        if not frames:
            raise AppException("Failed to extract pose landmarks from uploaded video")
        return await self.analyze_form(frames, exercise_type=exercise_type, exercise_id=exercise_id)
